"""
Pre-training diffusion policy

"""

import logging
import wandb
import numpy as np
import torch

log = logging.getLogger(__name__)
from util.timer import Timer
from agent.pretrain.train_agent import PreTrainAgent, batch_to_device


class TrainDiffusionAgent(PreTrainAgent):

    def __init__(self, cfg):
        super().__init__(cfg)

    def run(self):

        timer = Timer()
        # gentle_manip patch: optional resume from an arbitrary checkpoint path (the
        # stock run() always started at epoch 1, even though save_model()/load()
        # already round-trip model+ema+epoch -- nothing wired that up to actually
        # continue a run). Pass `resume_from=<run_dir>/checkpoint/state_<N>.pt` on the
        # hydra CLI to pick up right after that checkpoint. KNOWN LIMITATION: optimizer
        # and lr_scheduler internal state are NOT restored (not saved by save_model()
        # either), so a resumed run's Adam moments/cosine-schedule phase restart from
        # scratch -- acceptable for BC pretraining (a few epochs of mild LR mismatch,
        # not a correctness issue), not a full production-grade resume.
        resume_from = self.cfg.get("resume_from", None)
        if resume_from:
            log.info(f"Resuming from checkpoint: {resume_from}")
            data = torch.load(resume_from, map_location=next(self.model.parameters()).device,
                              weights_only=True)
            self.model.load_state_dict(data["model"])
            self.ema_model.load_state_dict(data["ema"])
            self.epoch = data["epoch"] + 1
            log.info(f"Resumed at epoch {self.epoch} (target n_epochs={self.n_epochs})")
        else:
            self.epoch = 1
        n_remaining = max(0, self.n_epochs - self.epoch + 1)
        cnt_batch = 0
        for _ in range(n_remaining):

            # train
            loss_train_epoch = []
            for batch_train in self.dataloader_train:
                if self.dataset_train.device == "cpu":
                    batch_train = batch_to_device(batch_train)

                self.model.train()
                loss_train = self.model.loss(*batch_train)
                loss_train.backward()
                loss_train_epoch.append(loss_train.item())

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
                val_str = f" | val loss {loss_val:8.4f}" if loss_val is not None else ""
                log.info(
                    f"{self.epoch}: train loss {loss_train:8.4f}{val_str} | t:{timer():8.4f}"
                )
                if self.use_wandb:
                    if loss_val is not None:
                        wandb.log(
                            {"loss - val": loss_val}, step=self.epoch, commit=False
                        )
                    wandb.log(
                        {
                            "loss - train": loss_train,
                        },
                        step=self.epoch,
                        commit=True,
                    )

            # count
            self.epoch += 1
