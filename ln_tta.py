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
        self.trainable_params = []
        self._setup()

        self.optimizer = optim.AdamW(self.trainable_params, lr=cfg.OPTIM.LR)
        self.model_state = deepcopy(model.state_dict())
        self.optimizer_state = deepcopy(self.optimizer.state_dict())

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
        logger.info(f"LNSubsetTTA: training LN in blocks[{hook_idx}..{n-1}]")

        for i in range(hook_idx, n):
            for m in blocks[i].modules():
                if isinstance(m, torch.nn.LayerNorm):
                    m.weight.requires_grad = True
                    m.bias.requires_grad = True
                    self.trainable_params.extend([m.weight, m.bias])

        logger.info(f"LNSubsetTTA: {len(self.trainable_params)} LN parameters "
                    f"({sum(p.numel() for p in self.trainable_params):,} values)")

    @torch.enable_grad()
    def forward(self, x):
        steps = self.cfg.OPTIM.STEPS
        self.model.train()

        for _ in range(steps):
            self.optimizer.zero_grad()
            out = self.model(x)
            loss = -(out.softmax(1) * out.log_softmax(1)).sum(1).mean()
            loss.backward()
            self.optimizer.step()

        self.model.eval()
        with torch.no_grad():
            return self.model(x)

    def reset(self):
        self.model.load_state_dict(self.model_state)
        self.optimizer.load_state_dict(self.optimizer_state)
        logger.info("LNSubsetTTA: reset to initial state")


def setup_ln_subset(model):
    from conf import cfg
    model = LNSubsetTTA(model, cfg)
    return model
