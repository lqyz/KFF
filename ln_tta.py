import logging
logger = logging.getLogger(__name__)

from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim


class LNSubsetTTA(nn.Module):
    def __init__(self, model, cfg):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.adapted = False
        self.param_groups = []
        self.model_state = deepcopy(model.state_dict())
        self._setup()

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
        self.hook_idx = min(5, n - 1)
        base_lr = self.cfg.OPTIM.LR
        total_params = 0
        modes = []

        if getattr(self.cfg.OURS, 'TRAIN_QKV', False):
            modes.append('qkv')
        if getattr(self.cfg.OURS, 'TRAIN_PROJ', False):
            modes.append('proj')
        if getattr(self.cfg.OURS, 'TRAIN_MLP', False):
            modes.append('mlp')
        modes.append('ln')

        use_gsnr = getattr(self.cfg.OURS, 'TRAIN_GSNR', False)
        mode_str = '+'.join(modes)
        if use_gsnr:
            mode_str += ' (GSNR gated)'
        logger.info(f"LNSubsetTTA: training blocks[{self.hook_idx}..{n-1}] mode={mode_str}")

        self.block_bounds = []
        for i in range(self.hook_idx, n):
            block = blocks[i]
            block_params = []

            for m in block.modules():
                if isinstance(m, torch.nn.LayerNorm):
                    m.weight.requires_grad = True
                    m.bias.requires_grad = True
                    block_params.extend([m.weight, m.bias])

            if 'qkv' in modes:
                try:
                    qkv = block.attn.qkv
                    qkv.weight.requires_grad = True
                    if qkv.bias is not None:
                        qkv.bias.requires_grad = True
                    block_params.extend([qkv.weight, qkv.bias] if qkv.bias is not None else [qkv.weight])
                except AttributeError:
                    pass

            if 'proj' in modes:
                try:
                    proj = block.attn.proj
                    proj.weight.requires_grad = True
                    if proj.bias is not None:
                        proj.bias.requires_grad = True
                    block_params.extend([proj.weight, proj.bias] if proj.bias is not None else [proj.weight])
                except AttributeError:
                    pass

            if 'mlp' in modes:
                try:
                    for _, param in block.mlp.named_parameters():
                        param.requires_grad = True
                        block_params.append(param)
                except AttributeError:
                    pass

            if block_params:
                lr = self._lr_for_block(i, n, base_lr)
                self.param_groups.append({'params': block_params, 'lr': lr,
                                          'block_idx': i})
                total_params += sum(p.numel() for p in block_params)
                logger.info(f"  block[{i}]: {sum(p.numel() for p in block_params):,} params, lr={lr:.2e}")

        logger.info(f"LNSubsetTTA: {total_params:,} trainable values")

    def _standard_adapt(self, x, steps):
        """Standard sub-batch entropy minimization."""
        bs = min(16, x.shape[0])
        for _ in range(steps):
            perm = torch.randperm(x.shape[0], device=x.device)
            for j in range(0, x.shape[0], bs):
                end = min(j + bs, x.shape[0])
                xb = x[perm[j:end]]
                self.optimizer.zero_grad()
                out = self.model(xb)
                loss = -(out.softmax(1) * out.log_softmax(1)).sum(1).mean()
                loss.backward()
                self.optimizer.step()

    def _gsnr_adapt(self, x, steps):
        """GSNR-gated adaptation: measure per-block gradient consistency
        across K sub-batches and gate noisy parameter groups."""
        K = getattr(self.cfg.OURS, 'GSNR_K', 4)
        thr = getattr(self.cfg.OURS, 'GSNR_THR', 0.4)
        tau = getattr(self.cfg.OURS, 'GSNR_TAU', 0.1)
        sub_size = max(1, x.shape[0] // K)

        for step in range(steps):
            self.optimizer.zero_grad()

            block_grads_per_sub = {g: [] for g in range(len(self.param_groups))}

            for k in range(K):
                idx = torch.randperm(x.shape[0], device=x.device)[:sub_size]
                out = self.model(x[idx])
                loss = -(out.softmax(1) * out.log_softmax(1)).sum(1).mean()
                loss.backward()

                for g_idx, group in enumerate(self.param_groups):
                    flat = torch.cat([p.grad.detach().flatten().clone()
                                      for p in group['params']])
                    block_grads_per_sub[g_idx].append(flat)

                self.optimizer.zero_grad()

            gates = []
            for g_idx, grads_list in block_grads_per_sub.items():
                if len(grads_list) > 1:
                    grads_stack = torch.stack(grads_list)
                    sum_g = grads_stack.sum(dim=0)
                    n_sum = torch.norm(sum_g, p=2)
                    s_norms = torch.norm(grads_stack, p=2, dim=1).sum()
                    gsnr = (n_sum / (s_norms + 1e-8)).item()
                    gate = torch.sigmoid(torch.tensor((gsnr - thr) / tau)).item()
                else:
                    gsnr = 1.0
                    gate = 1.0
                gates.append(gate)

            out = self.model(x)
            loss = -(out.softmax(1) * out.log_softmax(1)).sum(1).mean()
            loss.backward()

            for g_idx, group in enumerate(self.param_groups):
                for p in group['params']:
                    if p.grad is not None:
                        p.grad *= gates[g_idx]

            self.optimizer.step()

            if step == 0:
                gate_str = ', '.join(f"b{g['block_idx']}={gates[gi]:.3f}"
                                     for gi, g in enumerate(self.param_groups))
                logger.info(f"  GSNR step 0 gates: {gate_str}")

    @torch.enable_grad()
    def forward(self, x):
        if not self.adapted:
            self.optimizer = optim.AdamW(self.param_groups, weight_decay=0.0)
            self.model.train()

            steps = self.cfg.OPTIM.STEPS
            use_gsnr = getattr(self.cfg.OURS, 'TRAIN_GSNR', False)

            if use_gsnr:
                self._gsnr_adapt(x, max(1, steps // 2))
            else:
                self._standard_adapt(x, max(1, steps // 2))

            self.adapted = True

        self.model.eval()
        with torch.no_grad():
            return self.model(x)

    def reset(self):
        self.model.load_state_dict(self.model_state)
        self.adapted = False
        logger.info("LNSubsetTTA: reset")


def setup_ln_subset(model):
    from conf import cfg
    model = LNSubsetTTA(model, cfg)
    return model
