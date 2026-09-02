"""
Pre-training diffusion policy

"""

import logging
import wandb
import os
import numpy as np
import torch

log = logging.getLogger(__name__)
from util.timer import Timer
from agent.pretrain.train_agent import PreTrainAgent, batch_to_device


# GAP knobs (default OFF -> this file behaves exactly as before)
_GAP_LAMBDA = float(os.environ.get("GM_GAP_LAMBDA", "0") or 0)
_GAP_START = int(os.environ.get("GM_GAP_START", "0") or 0)
_GAP_END = int(os.environ.get("GM_GAP_END", "50") or 50)
if _GAP_LAMBDA > 0:
    log.info("GAP active: lambda=%.3f, modulation epochs %d-%d, params matching 'proprio_encoder'",
             _GAP_LAMBDA, _GAP_START, _GAP_END)


class TrainDiffusionAgent(PreTrainAgent):

    def __init__(self, cfg):
        super().__init__(cfg)

    def run(self):

        timer = Timer()
        self.epoch = 1
        cnt_batch = 0
        for _ in range(self.n_epochs):

            # train
            loss_train_epoch = []
            for batch_train in self.dataloader_train:
                if self.dataset_train.device == "cpu":
                    batch_train = batch_to_device(batch_train)

                self.model.train()
                loss_train = self.model.loss(*batch_train)
                loss_train.backward()
                loss_train_epoch.append(loss_train.item())

                # ---- GAP: Gradient Adjustment with Phase-guidance -----------------------
                # Faithful port of third_party/GAP/gap/gap.py (Lu et al., arXiv 2602.12032).
                # Their code, verbatim in structure:
                #     phase_p = torch.max(batch['phase']).item()   # per-batch SCALAR
                #     coeff_p = 1 - lambda * phase_p               # NOT lambda*(1-rho)
                #     if modulation_starts <= epoch <= modulation_ends:
                #         for name, parms in policy.encoder.named_parameters():
                #             if 'pro' in name: parms.grad *= coeff_p
                # Their proven settings: lambda 0.3, window epochs 0-50.
                # rho here is our KNOWN grasp window (scripted demonstrator) instead of their
                # CPD+LSTM estimate — the only intentional deviation.
                if _GAP_LAMBDA > 0.0 and _GAP_START <= self.epoch <= _GAP_END:
                    _cond = batch_train.conditions
                    _ph = _cond.get("in_grasp_window") if isinstance(_cond, dict) else None
                    if _ph is not None:
                        _coeff = 1.0 - _GAP_LAMBDA * float(_ph.max().item())
                        for _n, _p in self.model.named_parameters():
                            if "proprio_encoder" in _n and _p.grad is not None:
                                _p.grad *= _coeff

                self.optimizer.step()
                self.optimizer.zero_grad()

                # update ema
                if cnt_batch % self.update_ema_freq == 0:
                    self.step_ema()
                cnt_batch += 1
            loss_train = np.mean(loss_train_epoch)

            # validate
            loss_val_epoch = []
            if self.dataloader_val is not None and self.epoch % self.val_freq == 0:
                self.model.eval()
                # no_grad: this pass is forward-only, so building the autograd graph is pure
                # waste (the val clouds are ~0.3 GB on GPU already).
                # NOTE: upstream unpacked `loss_val, infos_val = self.model.loss(...)`, but
                # loss() returns a bare scalar (see the train call above) — that line raised
                # "iteration over a 0-d tensor" the first time validation was ever run.
                with torch.no_grad():
                    for batch_val in self.dataloader_val:
                        if self.dataset_val.device == "cpu":
                            batch_val = batch_to_device(batch_val)
                        loss_val = self.model.loss(*batch_val)
                        loss_val_epoch.append(loss_val.item())
                self.model.train()
            loss_val = np.mean(loss_val_epoch) if len(loss_val_epoch) > 0 else None

            # update lr
            self.lr_scheduler.step()

            # save model
            if self.epoch % self.save_model_freq == 0 or self.epoch == self.n_epochs:
                self.save_model()

            # log loss
            if self.epoch % self.log_freq == 0:
                # val loss goes in the LOG LINE too, not just wandb — the slurm log is the
                # primary monitoring surface on the cluster, and the train/val gap is the
                # signal for spotting overfitting without waiting on a sim eval.
                # 8 decimals, not 4 (2026-09-02): at 4dp the converged val loss can only take a
                # handful of values (0.0005/0.0006/...), so a 1-step change reads as "+20%" and
                # val-min checkpoint selection ends up selecting on ROUNDING. Full precision is
                # needed to tell "still improving" from "flat within noise".
                val_str = f" | val loss {loss_val:.8f}" if loss_val is not None else ""
                log.info(
                    f"{self.epoch}: train loss {loss_train:.8f}{val_str} | t:{timer():8.4f}"
                )
                if self.use_wandb:
                    if loss_val is not None:
                        wandb.log(
                            {"loss - val": loss_val}, step=self.epoch, commit=False
                        )
                    # Auxiliary-objective loss components (AuxDiffusionModel stashes the last
                    # step's breakdown in _aux_log; empty for the plain model / no-aux baseline).
                    aux_log = {f"aux - {k}": v
                               for k, v in getattr(self.model, "_aux_log", {}).items()}
                    wandb.log(
                        {
                            "loss - train": loss_train,
                            **aux_log,
                        },
                        step=self.epoch,
                        commit=True,
                    )

            # count
            self.epoch += 1
