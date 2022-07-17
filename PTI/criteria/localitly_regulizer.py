import torch
import numpy as np
import wandb
from criteria import l2_loss
from configs import hyperparameters
from configs import global_config

import pdb

class Space_Regulizer:
    def __init__(self, original_G, lpips_net):
        self.original_G = original_G
        self.morphing_regulizer_alpha = hyperparameters.regulizer_alpha
        self.lpips_loss = lpips_net

    def get_morphed_w_code(self, new_w_code, fixed_w):
        interpolation_direction = new_w_code - fixed_w
        interpolation_direction_norm = torch.norm(interpolation_direction, p=2)
        direction_to_move = hyperparameters.regulizer_alpha * interpolation_direction / interpolation_direction_norm
        result_w = fixed_w + direction_to_move
        self.morphing_regulizer_alpha * fixed_w + (1 - self.morphing_regulizer_alpha) * new_w_code

        return result_w

    def get_image_from_ws(self, w_codes, G):
        return torch.cat([G.synthesis(w_code, noise_mode='none', force_fp32=True) for w_code in w_codes])

    def ball_holder_loss_lazy(self, new_G, num_of_sampled_latents, w_batch, use_wandb=False, domain='ffhq'):
        loss = 0.0
        
        if domain == 'ffhq':
            z_samples = np.random.randn(num_of_sampled_latents, self.original_G.z_dim)
            w_samples = self.original_G.mapping(torch.from_numpy(z_samples).to(global_config.device), None,
                                                truncation_psi=0.5)
        elif domain == 'car':
            w_samples = self.original_G([torch.randn(num_of_sampled_latents, self.original_G.style_dim, dtype=torch.float).to(global_config.device)], 
                            truncation=1.0, randomize_noise=False, only_return_latents=True)
        territory_indicator_ws = [self.get_morphed_w_code(w_code.unsqueeze(0), w_batch) for w_code in w_samples]
        for w_code in territory_indicator_ws:
            if domain == 'ffhq':
                new_img = new_G.synthesis(w_code, noise_mode='none', force_fp32=True)
                with torch.no_grad():
                    old_img = self.original_G.synthesis(w_code, noise_mode='none', force_fp32=True)
            elif domain == 'car':
                new_img, _ = new_G([w_code], input_is_latent=True, randomize_noise=False)
                with torch.no_grad():
                    old_img, _ = self.original_G([w_code], input_is_latent=True, randomize_noise=False)
            
            if hyperparameters.regulizer_l2_lambda > 0:
                l2_loss_val = l2_loss.l2_loss(old_img, new_img)
                if use_wandb:
                    wandb.log({f'space_regulizer_l2_loss_val': l2_loss_val.detach().cpu()},
                              step=global_config.training_step)
                loss += l2_loss_val * hyperparameters.regulizer_l2_lambda

            if hyperparameters.regulizer_lpips_lambda > 0:
                loss_lpips = self.lpips_loss(old_img, new_img)
                loss_lpips = torch.mean(torch.squeeze(loss_lpips))
                if use_wandb:
                    wandb.log({f'space_regulizer_lpips_loss_val': loss_lpips.detach().cpu()},
                              step=global_config.training_step)
                loss += loss_lpips * hyperparameters.regulizer_lpips_lambda
        return loss / len(territory_indicator_ws)

    def space_regulizer_loss(self, new_G, w_batch, use_wandb, domain='ffhq'):
        ret_val = self.ball_holder_loss_lazy(new_G, hyperparameters.latent_ball_num_of_samples, w_batch, use_wandb, domain)
        return ret_val
