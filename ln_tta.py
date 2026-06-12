import logging
logger = logging.getLogger(__name__)

from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class LNSubsetTTA(nn.Module):
    def __init__(self, model, cfg):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.param_groups = []
        self._setup()

        self.optimizer = self._build_optimizer()
        self.model_state = deepcopy(model.state_dict())

    def _lr_for_block(self, i, total, base_lr):
        frac = (i - 5) / (total - 5)
        return base_lr * (0.1 + 0.9 * frac)

    def _setup(self):
        for param in self.model.parameters():
            param.requires_grad = False

        if hasattr(self.model, 'blocks'):
            blocks = self.model.blocks
        elif hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'layers'):
            blocks = self.model.encoder.layers
        else:
            raise AttributeError("Cannot find transformer blocks")

        n = len(blocks)
        hook_idx = min(5, n - 1)
        base_lr = self.cfg.OPTIM.LR
        total_trainable = 0

        logger.info(f"LNSubsetTTA: training LN in blocks[{hook_idx}..{n-1}]")

        for i in range(hook_idx, n):
            block_params = []
            for m in blocks[i].modules():
                if isinstance(m, torch.nn.LayerNorm):
                    m.weight.requires_grad = True
                    m.bias.requires_grad = True
                    block_params.extend([m.weight, m.bias])
            if block_params:
                lr = self._lr_for_block(i, n, base_lr)
                self.param_groups.append({'params': block_params, 'lr': lr})
                total_trainable += len(block_params)
                logger.info(f"  block[{i}]: {len(block_params)} params, lr={lr:.2e}")

        logger.info(f"LNSubsetTTA: {total_trainable} LN parameters "
                    f"({sum(p.numel() for g in self.param_groups for p in g['params']):,} values)")

    def _build_optimizer(self):
        return optim.AdamW(self.param_groups, weight_decay=0.0)

    @torch.enable_grad()
    def forward(self, x):
        self.model.load_state_dict(self.model_state)
        self.optimizer = self._build_optimizer()

        steps = self.cfg.OPTIM.STEPS
        self.model.train()

        for s in range(steps):
            temp = 2.0 - 1.0 * (s / max(steps - 1, 1))
            self.optimizer.zero_grad()
            out = self.model(x)
            logits = out / temp
            loss = -(logits.softmax(1) * logits.log_softmax(1)).sum(1).mean()
            loss.backward()
            self.optimizer.step()

        self.model.eval()
        with torch.no_grad():
            return self.model(x)

    def reset(self):
        logger.info("LNSubsetTTA: reset")
        pass


def setup_ln_subset(model):
    from conf import cfg
    model = LNSubsetTTA(model, cfg)
    return model
