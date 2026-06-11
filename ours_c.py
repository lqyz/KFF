import logging
logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from ours import Ours, collect_params, configure_model, copy_model_and_optimizer
from vpt import PromptViT


class FeatureSwapper(nn.Module):
    def __init__(self, cls_only=True):
        super().__init__()
        self.mode = 'passthrough'
        self.mu_target = None
        self.sigma_target = None
        self.cls_only = cls_only

    def set_target(self, mu, sigma):
        self.mu_target = mu
        self.sigma_target = sigma

    def forward(self, module, input, output):
        if self.mode == 'passthrough' or self.mu_target is None:
            return output
        if self.cls_only:
            f = output[:, 0:1]
            cur_mu = f.mean(dim=(0, 1), keepdim=True)
            cur_sigma = f.std(dim=(0, 1), keepdim=True)
            f_norm = (f - cur_mu) / (cur_sigma + 1e-6)
            f_swapped = f_norm * self.sigma_target + self.mu_target
            output = output.clone()
            output[:, 0:1] = f_swapped
        else:
            cur_mu = output.mean(dim=(0, 1), keepdim=True)
            cur_sigma = output.std(dim=(0, 1), keepdim=True)
            f_norm = (output - cur_mu) / (cur_sigma + 1e-6)
            output = f_norm * self.sigma_target + self.mu_target
        return output


class OursConsistency(Ours):
    def __init__(self, model, optimizer, cfg, tau=3.0, ema_alpha=0.1, E_OOD=50):
        super().__init__(model, optimizer, cfg, tau, ema_alpha, E_OOD)

        self.c_beta = getattr(cfg.OURS, 'CONSISTENCY_BETA', 5.0)
        self.c_layer = getattr(cfg.OURS, 'CONSISTENCY_LAYER', 1)
        self.c_bank_size = getattr(cfg.OURS, 'CONSISTENCY_BANK', 20)

        self.bank_mu = []
        self.bank_sigma = []

        self.swapper = FeatureSwapper(cls_only=True)

        vit = model.vit.module if isinstance(model.vit, nn.DataParallel) else model.vit
        if hasattr(vit, 'blocks'):
            blocks = vit.blocks
        elif hasattr(vit, 'encoder') and hasattr(vit.encoder, 'layers'):
            blocks = vit.encoder.layers
        else:
            raise AttributeError("Cannot find transformer blocks")

        self.c_layer_idx = min(self.c_layer, len(blocks) - 1)
        self.swapper_handle = blocks[self.c_layer_idx].register_forward_hook(
            self.swapper)

        logger.info(f"Consistency TTA: layer=block[{self.c_layer_idx}], "
                    f"beta={self.c_beta}, bank_size={self.c_bank_size}")

    def _add_to_bank(self, key):
        D = key.shape[0] // 2
        mu = key[:D].reshape(1, 1, D).detach().cpu().clone()
        sigma = key[D:].reshape(1, 1, D).detach().cpu().clone()
        if len(self.bank_mu) >= self.c_bank_size:
            self.bank_mu.pop(0)
            self.bank_sigma.pop(0)
        self.bank_mu.append(mu)
        self.bank_sigma.append(sigma)

    @torch.enable_grad()
    def forward_and_adapt(self, x, model: PromptViT, optimizer, train_info,
                          iteration=1):
        loss = 0
        loss_ = 0
        output = None
        cls_features = None

        for i in range(iteration):
            features = model.forward_features(x)
            cls_features = features[:, 0]
            loss = self.distribution_loss(cls_features, train_info)

            if isinstance(model.vit, nn.DataParallel):
                output = model.vit.module.forward_head(features)
            else:
                output = model.vit.forward_head(features)

            loss_ = 3 * self.softmax_entropy(output).mean(0)
            loss += loss_

            if i == iteration - 1 and self.c_beta > 0 and len(self.bank_mu) > 0:
                idx = random.randint(0, len(self.bank_mu) - 1)
                hist_mu = self.bank_mu[idx].cuda()
                hist_sigma = self.bank_sigma[idx].cuda()

                self.swapper.mode = 'swap'
                self.swapper.set_target(hist_mu, hist_sigma)

                features_aug = model.forward_features(x)
                if isinstance(model.vit, nn.DataParallel):
                    output_aug = model.vit.module.forward_head(features_aug)
                else:
                    output_aug = model.vit.forward_head(features_aug)

                self.swapper.mode = 'passthrough'
                self.swapper.set_target(None, None)

                p_orig = F.softmax(output.detach(), dim=-1)
                log_p_aug = F.log_softmax(output_aug, dim=-1)
                loss_cons = F.kl_div(log_p_aug, p_orig, reduction='batchmean')

                loss += self.c_beta * loss_cons

            optimizer.zero_grad()
            if i == iteration - 1:
                loss.backward(retain_graph=True)
            else:
                loss.backward()
            optimizer.step()

        key = model.domain_extractor(cls_features)
        self._add_to_bank(key)

        return output, loss

    def reset(self):
        super().reset()
        self.bank_mu = []
        self.bank_sigma = []


def setup_ours_consistency(model):
    from conf import cfg
    model = configure_model(model, cfg)
    domain_prompts, class_prompts = collect_params(model)
    from main import setup_optimizer
    optimizer = setup_optimizer(class_prompts)
    model = OursConsistency(
        model=model, optimizer=optimizer, cfg=cfg,
        tau=cfg.OPTIM.TAU, ema_alpha=cfg.OPTIM.EMA_ALPHA,
        E_OOD=cfg.OPTIM.STEPS,
    )
    model.obtain_src_stat(data_path=cfg.SRC_DATA_DIR,
                          num_samples=cfg.SRC_NUM_SAMPLES,
                          train_info=cfg.OURS.TRAIN_INFO)
    return model
