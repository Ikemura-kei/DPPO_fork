"""
DPPO fine-tuning for pixel observations.

"""

import os
import pickle
import einops
import numpy as np
import torch
import logging
import wandb
import math

log = logging.getLogger(__name__)
from util.timer import Timer
from agent.finetune.train_ppo_diffusion_agent import TrainPPODiffusionAgent
from model.common.modules import RandomShiftsAug


class TrainPPOImgDiffusionAgent(TrainPPODiffusionAgent):

    def __init__(self, cfg):
        super().__init__(cfg)

        # Image randomization
        self.augment = cfg.train.augment
        if self.augment:
            self.aug = RandomShiftsAug(pad=4)

        # Set obs dim -  we will save the different obs in batch in a dict
        shape_meta = cfg.shape_meta
        self.obs_dims = {k: shape_meta.obs[k]["shape"] for k in shape_meta.obs}

        # Gradient accumulation to deal with large GPU RAM usage
        self.grad_accumulate = cfg.train.grad_accumulate

        # gentle_manip: fixed-seed periodic eval. When eval_fixed_seed is set, every eval reseeds
        # (and, if eval_scene_dr, rebuilds the object geometry/material) to the SAME fixed scene,
        # so the eval return/success/stress are apples-to-apple across training iters (removes the
        # scene-luck confound: varying Young's / pose). Task success + success-gated stress are
        # then logged to wandb. After an eval, a fresh training scene is restored (reseed +
        # rebuild) so training keeps its scene variety. Needs a venv with seed()/randomize_scene()
        # (the genesis bridge). None -> stock DPPO behaviour.
        self.eval_fixed_seed = cfg.train.get("eval_fixed_seed", None)
        self.eval_scene_dr = cfg.train.get("eval_scene_dr", False)
        # gentle_manip: if >0, the periodic eval runs eval_n_batches x n_envs episodes through the
        # SHARED harness (fixed reproducible scenes, per-EPISODE video, success-gated v2 stress) —
        # a low-variance in-training eval instead of the single 12-episode rollout.
        self.eval_n_batches = int(cfg.train.get("eval_n_batches", 0))
        self._prev_eval_mode = False

    def _gm_periodic_eval(self):
        """Multi-batch fixed-seed periodic eval via gentle_manip.evaluation.run_eval:
        eval_n_batches x n_envs episodes on scenes that are DETERMINISTIC in eval_fixed_seed (so
        the curve is apples-to-apple across iters), one video PER EPISODE, success-gated v2 stress
        + task success -> wandb/stdout. Reuses the canonical eval protocol at a smaller scale."""
        from pathlib import Path
        from gentle_manip.evaluation import EvalSpec, run_eval
        from gentle_manip.dppo.eval_agent import _DiffusionPolicy
        spec = EvalSpec(n_episodes=self.eval_n_batches * self.n_envs, num_envs=self.n_envs,
                        seed=int(self.eval_fixed_seed), max_policy_steps=self.n_steps,
                        scene_group_size=self.eval_n_batches if self.eval_scene_dr else 0)
        policy = _DiffusionPolicy(self.model, list(self.obs_dims.keys()), self.device, self.act_steps)
        out = Path(self.logdir) / "periodic_eval" / f"itr-{self.itr}"
        # Freeze the training server's periodic auto scene-DR so it can't fire mid-eval: the
        # harness owns the deterministic per-group rebuild (scene_group_size), and a stray
        # every-N-resets relaunch would rebuild geometry + advance the RNG, breaking the eval's
        # apples-to-apple determinism. Restored in finally. No-op when scene_dr_every=0.
        if hasattr(self.venv, "set_auto_scene_dr"):
            self.venv.set_auto_scene_dr(False)
        try:
            summ = run_eval(self.venv, policy, spec, out, experiment_name=None,
                            checkpoint=f"itr-{self.itr}", record_batches=None)  # all eps -> per-ep video
        finally:
            if hasattr(self.venv, "set_auto_scene_dr"):
                self.venv.set_auto_scene_dr(True)
        self.model.train()                                   # run_eval left model in eval
        ts = summ.get("success_rate", float("nan"))
        sp, sp95 = summ.get("stress_max_tmax_mean"), summ.get("stress_max_tmax_p95")
        log.info(f"eval[{self.itr}]: task_success {ts:.3f} over {summ.get('n_episodes')} eps"
                 + (f" | peak(succ) {sp:.0f} (P95 {sp95:.0f})" if sp is not None else ""))
        if self.use_wandb:
            d = {"task success rate - eval": ts,
                 "avg episode reward - eval": summ.get("mean_episode_reward")}
            if sp is not None:
                d["stress peak mean (success) - eval"] = sp
                d["stress peak p95 (success) - eval"] = sp95
            wandb.log(d, step=self.itr, commit=False)
        return summ

    def run(self):

        # Start training loop
        timer = Timer()
        run_results = []
        cnt_train_step = 0
        last_itr_eval = False
        done_venv = np.zeros((1, self.n_envs))
        while self.itr < self.n_train_itr:

            # Prepare video paths for each envs --- only applies for the first set of episodes if allowing reset within iteration and each iteration has multiple episodes from one env
            options_venv = [{} for _ in range(self.n_envs)]
            if self.itr % self.render_freq == 0 and self.render_video:
                for env_ind in range(self.n_render):
                    options_venv[env_ind]["video_path"] = os.path.join(
                        self.render_dir, f"itr-{self.itr}_trial-{env_ind}.mp4"
                    )

            # Define train or eval - all envs restart
            eval_mode = self.itr % self.val_freq == 0 and not self.force_train
            self.model.eval() if eval_mode else self.model.train()
            last_itr_eval = eval_mode

            # gentle_manip fixed-seed eval: reseed (+ optionally rebuild geometry/material) so this
            # eval faces the SAME scene as every other eval; when the FIRST train iter after an eval
            # starts, restore a fresh training scene so training keeps variety. Guarded by hasattr
            # so non-genesis venvs are unaffected.
            do_gm_eval = eval_mode and self.eval_n_batches > 0 and self.eval_fixed_seed is not None
            if self.eval_fixed_seed is not None and hasattr(self.venv, "seed"):
                if eval_mode and not do_gm_eval:             # single-batch legacy eval: fix its scene
                    self.venv.seed(int(self.eval_fixed_seed))
                    if self.eval_scene_dr and hasattr(self.venv, "randomize_scene"):
                        self.venv.randomize_scene(int(self.eval_fixed_seed))
                elif self._prev_eval_mode:                   # restore a training scene after ANY eval
                    train_seed = 2_000_000 + self.itr
                    self.venv.seed(train_seed)
                    if self.eval_scene_dr and hasattr(self.venv, "randomize_scene"):
                        self.venv.randomize_scene(train_seed)
            self._prev_eval_mode = eval_mode

            # gentle_manip: multi-batch harness eval REPLACES the single 12-ep rollout (lower
            # variance + per-episode video). Skip the train rollout/update; mirror the per-iter
            # tail (lr schedulers / ema / checkpoint / itr) so bookkeeping is unchanged.
            if do_gm_eval:
                self._gm_periodic_eval()
                if self.itr >= self.n_critic_warmup_itr:
                    self.actor_lr_scheduler.step()
                    if self.learn_eta:
                        self.eta_lr_scheduler.step()
                self.critic_lr_scheduler.step()
                self.model.step()
                if self.itr % self.save_model_freq == 0 or self.itr == self.n_train_itr - 1:
                    self.save_model()
                self.itr += 1
                # run_eval reset the venv to eval scenes; restore a fresh TRAINING scene and PRIME
                # the rollout state (prev_obs_venv/done_venv) for the next train iter — we skipped
                # the normal reset, so the loop-local prev_obs_venv would otherwise be unbound.
                if self.eval_fixed_seed is not None and hasattr(self.venv, "seed"):
                    _s = 2_000_000 + self.itr
                    self.venv.seed(_s)
                    if self.eval_scene_dr and hasattr(self.venv, "randomize_scene"):
                        self.venv.randomize_scene(_s)
                prev_obs_venv = self.reset_env_all(options_venv=[{} for _ in range(self.n_envs)])
                done_venv = np.ones((1, self.n_envs))   # next train iter treats step 0 as fresh
                self._prev_eval_mode = False             # training scene already restored here
                continue

            # Reset env before iteration starts (1) if specified, (2) at eval mode, (3) right after
            # eval mode, or (4) on the very first iteration — with force_train=true itr 0 is NOT an
            # eval iter, so without this prev_obs_venv/done_venv would be unbound at first use.
            firsts_trajs = np.zeros((self.n_steps + 1, self.n_envs))
            if self.reset_at_iteration or eval_mode or last_itr_eval or self.itr == 0:
                prev_obs_venv = self.reset_env_all(options_venv=options_venv)
                firsts_trajs[0] = 1
            else:
                # if done at the end of last iteration, the envs are just reset
                firsts_trajs[0] = done_venv

            # Holder
            obs_trajs = {
                k: np.zeros(
                    (self.n_steps, self.n_envs, self.n_cond_step, *self.obs_dims[k])
                )
                for k in self.obs_dims
            }
            chains_trajs = np.zeros(
                (
                    self.n_steps,
                    self.n_envs,
                    self.model.ft_denoising_steps + 1,
                    self.horizon_steps,
                    self.action_dim,
                )
            )
            terminated_trajs = np.zeros((self.n_steps, self.n_envs))
            reward_trajs = np.zeros((self.n_steps, self.n_envs))
            # gentle_manip: per-step stress + task-success (from the genesis bridge info), for
            # success-gated stress reporting during eval.
            stress_max_trajs = np.full((self.n_steps, self.n_envs), np.nan)
            success_step_trajs = np.zeros((self.n_steps, self.n_envs))

            # Collect a set of trajectories from env
            for step in range(self.n_steps):
                if step % 10 == 0:
                    print(f"Processed step {step} of {self.n_steps}")

                # Select action
                with torch.no_grad():
                    cond = {
                        key: torch.from_numpy(prev_obs_venv[key])
                        .float()
                        .to(self.device)
                        for key in self.obs_dims
                    }  # batch each type of obs and put into dict
                    samples = self.model(
                        cond=cond,
                        deterministic=eval_mode,
                        return_chain=True,
                    )
                    output_venv = (
                        samples.trajectories.cpu().numpy()
                    )  # n_env x horizon x act
                    chains_venv = (
                        samples.chains.cpu().numpy()
                    )  # n_env x denoising x horizon x act
                action_venv = output_venv[:, : self.act_steps]

                # Apply multi-step action
                obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = (
                    self.venv.step(action_venv)
                )
                done_venv = terminated_venv | truncated_venv
                for k in obs_trajs:
                    obs_trajs[k][step] = prev_obs_venv[k]
                chains_trajs[step] = chains_venv
                reward_trajs[step] = reward_venv
                terminated_trajs[step] = terminated_venv
                firsts_trajs[step + 1] = done_venv
                if eval_mode:                       # gentle_manip: capture stress + task success
                    sm = info_venv.get("stress_max") if isinstance(info_venv, dict) else None
                    if sm is not None:
                        stress_max_trajs[step] = np.asarray(sm, float).reshape(self.n_envs)
                    su = info_venv.get("success") if isinstance(info_venv, dict) else None
                    if su is not None:
                        success_step_trajs[step] = np.asarray(su, float).reshape(self.n_envs)

                # update for next step
                prev_obs_venv = obs_venv

                # count steps --- not acounting for done within action chunk
                cnt_train_step += self.n_envs * self.act_steps if not eval_mode else 0

            # Summarize episode reward --- this needs to be handled differently depending on whether the environment is reset after each iteration. Only count episodes that finish within the iteration.
            episodes_start_end = []
            for env_ind in range(self.n_envs):
                env_steps = np.where(firsts_trajs[:, env_ind] == 1)[0]
                for i in range(len(env_steps) - 1):
                    start = env_steps[i]
                    end = env_steps[i + 1]
                    if end - start > 1:
                        episodes_start_end.append((env_ind, start, end - 1))
            if len(episodes_start_end) > 0:
                reward_trajs_split = [
                    reward_trajs[start : end + 1, env_ind]
                    for env_ind, start, end in episodes_start_end
                ]
                num_episode_finished = len(reward_trajs_split)
                episode_reward = np.array(
                    [np.sum(reward_traj) for reward_traj in reward_trajs_split]
                )
                episode_best_reward = np.array(
                    [
                        np.max(reward_traj) / self.act_steps
                        for reward_traj in reward_trajs_split
                    ]
                )
                avg_episode_reward = np.mean(episode_reward)
                avg_best_reward = np.mean(episode_best_reward)
                success_rate = np.mean(
                    episode_best_reward >= self.best_reward_threshold_for_success
                )
            else:
                episode_reward = np.array([])
                num_episode_finished = 0
                avg_episode_reward = 0
                avg_best_reward = 0
                success_rate = 0
                log.info("[WARNING] No episode completed within the iteration!")

            # gentle_manip: eval TASK-success + SUCCESS-GATED peak stress per episode (a failed
            # episode never grasped -> near-zero stress -> must not count, else "gentle but does
            # nothing" looks good). Final-step success == task success (held-in-band).
            eval_task_success_rate = float("nan")
            eval_stress_peak_succ = float("nan")
            eval_stress_peak_p95 = float("nan")
            if eval_mode and len(episodes_start_end) > 0:
                ep_success, ep_peak = [], []
                for env_ind, start, end in episodes_start_end:
                    ep_success.append(bool(success_step_trajs[end, env_ind]))
                    seg = stress_max_trajs[start : end + 1, env_ind]
                    seg = seg[~np.isnan(seg)]
                    ep_peak.append(np.max(seg) if seg.size else np.nan)
                ep_success = np.array(ep_success, bool)
                ep_peak = np.array(ep_peak, float)
                eval_task_success_rate = float(ep_success.mean()) if ep_success.size else float("nan")
                gated = ep_peak[ep_success & ~np.isnan(ep_peak)]
                if gated.size:
                    eval_stress_peak_succ = float(gated.mean())
                    eval_stress_peak_p95 = float(np.percentile(gated, 95))

            # Update models
            if not eval_mode:
                with torch.no_grad():
                    # move all obs modalities to device (generic over obs keys:
                    # rgb / point_cloud / state — see shape_meta.obs)
                    for k in obs_trajs:
                        obs_trajs[k] = (
                            torch.from_numpy(obs_trajs[k]).float().to(self.device)
                        )
                    # apply image randomization (rgb only; no-op for point_cloud)
                    if self.augment and "rgb" in obs_trajs:
                        rgb = einops.rearrange(
                            obs_trajs["rgb"],
                            "s e t c h w -> (s e t) c h w",
                        )
                        rgb = self.aug(rgb)
                        obs_trajs["rgb"] = einops.rearrange(
                            rgb,
                            "(s e t) c h w -> s e t c h w",
                            s=self.n_steps,
                            e=self.n_envs,
                        )

                    # Calculate value and logprobs - split into batches to prevent out of memory
                    num_split = math.ceil(
                        self.n_envs * self.n_steps / self.logprob_batch_size
                    )
                    obs_ts = [{} for _ in range(num_split)]
                    for k in obs_trajs:
                        obs_k = einops.rearrange(
                            obs_trajs[k],
                            "s e ... -> (s e) ...",
                        )
                        obs_ts_k = torch.split(obs_k, self.logprob_batch_size, dim=0)
                        for i, obs_t in enumerate(obs_ts_k):
                            obs_ts[i][k] = obs_t
                    values_trajs = np.empty((0, self.n_envs))
                    for obs in obs_ts:
                        values = (
                            self.model.critic(obs, no_augment=True)
                            .cpu()
                            .numpy()
                            .flatten()
                        )
                        values_trajs = np.vstack(
                            (values_trajs, values.reshape(-1, self.n_envs))
                        )
                    chains_t = einops.rearrange(
                        torch.from_numpy(chains_trajs).float().to(self.device),
                        "s e t h d -> (s e) t h d",
                    )
                    chains_ts = torch.split(chains_t, self.logprob_batch_size, dim=0)
                    logprobs_trajs = np.empty(
                        (
                            0,
                            self.model.ft_denoising_steps,
                            self.horizon_steps,
                            self.action_dim,
                        )
                    )
                    for obs, chains in zip(obs_ts, chains_ts):
                        logprobs = self.model.get_logprobs(obs, chains).cpu().numpy()
                        logprobs_trajs = np.vstack(
                            (
                                logprobs_trajs,
                                logprobs.reshape(-1, *logprobs_trajs.shape[1:]),
                            )
                        )

                    # normalize reward with running variance if specified
                    if self.reward_scale_running:
                        reward_trajs_transpose = self.running_reward_scaler(
                            reward=reward_trajs.T, first=firsts_trajs[:-1].T
                        )
                        reward_trajs = reward_trajs_transpose.T

                    # bootstrap value with GAE if not terminal - apply reward scaling with constant if specified
                    obs_venv_ts = {
                        key: torch.from_numpy(obs_venv[key]).float().to(self.device)
                        for key in self.obs_dims
                    }
                    advantages_trajs = np.zeros_like(reward_trajs)
                    lastgaelam = 0
                    for t in reversed(range(self.n_steps)):
                        if t == self.n_steps - 1:
                            nextvalues = (
                                self.model.critic(obs_venv_ts, no_augment=True)
                                .reshape(1, -1)
                                .cpu()
                                .numpy()
                            )
                        else:
                            nextvalues = values_trajs[t + 1]
                        nonterminal = 1.0 - terminated_trajs[t]
                        # delta = r + gamma*V(st+1) - V(st)
                        delta = (
                            reward_trajs[t] * self.reward_scale_const
                            + self.gamma * nextvalues * nonterminal
                            - values_trajs[t]
                        )
                        # A = delta_t + gamma*lamdba*delta_{t+1} + ...
                        advantages_trajs[t] = lastgaelam = (
                            delta
                            + self.gamma * self.gae_lambda * nonterminal * lastgaelam
                        )
                    returns_trajs = advantages_trajs + values_trajs

                # k for environment step
                obs_k = {
                    k: einops.rearrange(
                        obs_trajs[k],
                        "s e ... -> (s e) ...",
                    )
                    for k in obs_trajs
                }
                chains_k = einops.rearrange(
                    torch.tensor(chains_trajs, device=self.device).float(),
                    "s e t h d -> (s e) t h d",
                )
                returns_k = (
                    torch.tensor(returns_trajs, device=self.device).float().reshape(-1)
                )
                values_k = (
                    torch.tensor(values_trajs, device=self.device).float().reshape(-1)
                )
                advantages_k = (
                    torch.tensor(advantages_trajs, device=self.device).float().reshape(-1)
                )
                logprobs_k = torch.tensor(logprobs_trajs, device=self.device).float()

                # Update policy and critic
                total_steps = self.n_steps * self.n_envs * self.model.ft_denoising_steps
                clipfracs = []
                for update_epoch in range(self.update_epochs):

                    # for each epoch, go through all data in batches
                    flag_break = False
                    inds_k = torch.randperm(total_steps, device=self.device)
                    num_batch = max(1, total_steps // self.batch_size)  # skip last ones
                    for batch in range(num_batch):
                        start = batch * self.batch_size
                        end = start + self.batch_size
                        inds_b = inds_k[start:end]  # b for batch
                        batch_inds_b, denoising_inds_b = torch.unravel_index(
                            inds_b,
                            (self.n_steps * self.n_envs, self.model.ft_denoising_steps),
                        )
                        obs_b = {k: obs_k[k][batch_inds_b] for k in obs_k}
                        chains_prev_b = chains_k[batch_inds_b, denoising_inds_b]
                        chains_next_b = chains_k[batch_inds_b, denoising_inds_b + 1]
                        returns_b = returns_k[batch_inds_b]
                        values_b = values_k[batch_inds_b]
                        advantages_b = advantages_k[batch_inds_b]
                        logprobs_b = logprobs_k[batch_inds_b, denoising_inds_b]

                        # get loss
                        (
                            pg_loss,
                            entropy_loss,
                            v_loss,
                            clipfrac,
                            approx_kl,
                            ratio,
                            bc_loss,
                            eta,
                        ) = self.model.loss(
                            obs_b,
                            chains_prev_b,
                            chains_next_b,
                            denoising_inds_b,
                            returns_b,
                            values_b,
                            advantages_b,
                            logprobs_b,
                            use_bc_loss=self.use_bc_loss,
                            reward_horizon=self.reward_horizon,
                        )
                        loss = (
                            pg_loss
                            + entropy_loss * self.ent_coef
                            + v_loss * self.vf_coef
                            + bc_loss * self.bc_loss_coeff
                        )
                        clipfracs += [clipfrac]

                        # update policy and critic
                        loss.backward()
                        if (batch + 1) % self.grad_accumulate == 0:
                            if self.itr >= self.n_critic_warmup_itr:
                                if self.max_grad_norm is not None:
                                    torch.nn.utils.clip_grad_norm_(
                                        self.model.actor_ft.parameters(),
                                        self.max_grad_norm,
                                    )
                                self.actor_optimizer.step()
                                if (
                                    self.learn_eta
                                    and batch % self.eta_update_interval == 0
                                ):
                                    self.eta_optimizer.step()
                            self.critic_optimizer.step()
                            self.actor_optimizer.zero_grad()
                            self.critic_optimizer.zero_grad()
                            if self.learn_eta:
                                self.eta_optimizer.zero_grad()
                            log.info(f"run grad update at batch {batch}")
                            log.info(
                                f"approx_kl: {approx_kl}, update_epoch: {update_epoch}, num_batch: {num_batch}"
                            )

                            # Stop gradient update if KL difference reaches target
                            if (
                                self.target_kl is not None
                                and approx_kl > self.target_kl
                                and self.itr >= self.n_critic_warmup_itr
                            ):
                                flag_break = True
                                break
                    if flag_break:
                        break

                # Explained variation of future rewards using value function
                y_pred, y_true = values_k.cpu().numpy(), returns_k.cpu().numpy()
                var_y = np.var(y_true)
                explained_var = (
                    np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
                )

            # Update lr, min_sampling_std
            if self.itr >= self.n_critic_warmup_itr:
                self.actor_lr_scheduler.step()
                if self.learn_eta:
                    self.eta_lr_scheduler.step()
            self.critic_lr_scheduler.step()
            self.model.step()
            diffusion_min_sampling_std = self.model.get_min_sampling_denoising_std()

            # Save model
            if self.itr % self.save_model_freq == 0 or self.itr == self.n_train_itr - 1:
                self.save_model()

            # Log loss and save metrics
            run_results.append(
                {
                    "itr": self.itr,
                    "step": cnt_train_step,
                }
            )
            if self.itr % self.log_freq == 0:
                time = timer()
                run_results[-1]["time"] = time
                if eval_mode:
                    log.info(
                        f"eval: success rate {success_rate:8.4f} | avg episode reward {avg_episode_reward:8.4f} | avg best reward {avg_best_reward:8.4f}"
                        + (f" | task success {eval_task_success_rate:6.3f} | stress_peak(succ) "
                           f"{eval_stress_peak_succ:8.0f} (P95 {eval_stress_peak_p95:8.0f})"
                           if not math.isnan(eval_task_success_rate) else "")
                    )
                    if self.use_wandb:
                        eval_log = {
                            "success rate - eval": success_rate,
                            "avg episode reward - eval": avg_episode_reward,
                            "avg best reward - eval": avg_best_reward,
                            "num episode - eval": num_episode_finished,
                        }
                        # gentle_manip: task-success + success-gated stress (NaN -> skip)
                        if not math.isnan(eval_task_success_rate):
                            eval_log["task success rate - eval"] = eval_task_success_rate
                        if not math.isnan(eval_stress_peak_succ):
                            eval_log["stress peak mean (success) - eval"] = eval_stress_peak_succ
                            eval_log["stress peak p95 (success) - eval"] = eval_stress_peak_p95
                        wandb.log(eval_log, step=self.itr, commit=False)
                    run_results[-1]["eval_success_rate"] = success_rate
                    run_results[-1]["eval_episode_reward"] = avg_episode_reward
                    run_results[-1]["eval_best_reward"] = avg_best_reward
                    run_results[-1]["eval_task_success_rate"] = eval_task_success_rate
                    run_results[-1]["eval_stress_peak_success"] = eval_stress_peak_succ
                else:
                    log.info(
                        f"{self.itr}: step {cnt_train_step:8d} | loss {loss:8.4f} | pg loss {pg_loss:8.4f} | value loss {v_loss:8.4f} | bc loss {bc_loss:8.4f} | reward {avg_episode_reward:8.4f} | eta {eta:8.4f} | t:{time:8.4f}"
                    )
                    if self.use_wandb:
                        wandb.log(
                            {
                                "total env step": cnt_train_step,
                                "loss": loss,
                                "pg loss": pg_loss,
                                "value loss": v_loss,
                                "bc loss": bc_loss,
                                "eta": eta,
                                "approx kl": approx_kl,
                                "ratio": ratio,
                                "clipfrac": np.mean(clipfracs),
                                "explained variance": explained_var,
                                "avg episode reward - train": avg_episode_reward,
                                "num episode - train": num_episode_finished,
                                "diffusion - min sampling std": diffusion_min_sampling_std,
                                "actor lr": self.actor_optimizer.param_groups[0]["lr"],
                                "critic lr": self.critic_optimizer.param_groups[0][
                                    "lr"
                                ],
                            },
                            step=self.itr,
                            commit=True,
                        )
                    run_results[-1]["train_episode_reward"] = avg_episode_reward
                with open(self.result_path, "wb") as f:
                    pickle.dump(run_results, f)
            self.itr += 1
