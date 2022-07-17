import os

import torch
from tqdm import tqdm

from configs import paths_config, hyperparameters, global_config
from training.coaches.base_coach import BaseCoach
from utils.log_utils import log_images_from_w

from PIL import Image

import pdb

class MultiIDCoach(BaseCoach):

    def __init__(self, data_loader, use_wandb, domain='ffhq'):
        self.domain = domain
        super().__init__(data_loader, use_wandb)
        self.domain = domain

    def train(self, checkpoint_path=None):
        if self.domain == 'ffhq':
            self.G.synthesis.train()
            self.G.mapping.train()
        elif self.domain == 'car':
            self.G.train()

        w_path_dir = f'{paths_config.embedding_base_dir}/{paths_config.input_data_id}'
        os.makedirs(w_path_dir, exist_ok=True)
        os.makedirs(f'{w_path_dir}/{paths_config.pti_results_keyword}', exist_ok=True)
        os.makedirs(paths_config.visualization_dir, exist_ok=True)

        use_ball_holder = True
        w_pivots = []
        images = []

        for fname, image in self.data_loader:
            if self.image_counter >= hyperparameters.max_images_to_invert:
                break
            image_name = fname[0]
            # image_name = fname
            if hyperparameters.first_inv_type == 'w+':
                embedding_dir = f'{w_path_dir}/{paths_config.e4e_results_keyword}/{image_name}'
            else:
                embedding_dir = f'{w_path_dir}/{paths_config.pti_results_keyword}/{image_name}'
            os.makedirs(embedding_dir, exist_ok=True)

            w_pivot = self.get_inversion(w_path_dir, image_name, image)
            w_pivots.append(w_pivot)
            images.append((image_name, image))
            self.image_counter += 1
        for i in tqdm(range(hyperparameters.max_pti_steps)):
            self.image_counter = 0

            for data, w_pivot in zip(images, w_pivots):
                image_name, image = data

                if self.image_counter >= hyperparameters.max_images_to_invert:
                    break

                real_images_batch = image.to(global_config.device)
                generated_images = self.forward(w_pivot)
                loss, l2_loss_val, loss_lpips = self.calc_loss(generated_images, real_images_batch, image_name,
                                      self.G, use_ball_holder, w_pivot)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                use_ball_holder = global_config.training_step % hyperparameters.locality_regularization_interval == 0

                global_config.training_step += 1
                self.image_counter += 1
                
        os.makedirs(os.path.join(paths_config.visualization_dir, global_config.run_name), exist_ok=True)

        if self.use_wandb:
            log_images_from_w(w_pivots, self.G, [image[0] for image in images])
        else:
            for image, w_pivot in zip(images, w_pivots):
                name = image[0]
                w_pivot = w_pivot.to(global_config.device)
                generated_image = self.forward(w_pivot)
                img = (generated_image.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).detach().cpu().numpy()[0]
                pillow_img = Image.fromarray(img).resize((1024, 1024))  # hard-coding, debugging
                pillow_img.save(os.path.join(paths_config.visualization_dir, global_config.run_name, '{}.png'.format(name)))

        if checkpoint_path is None:
            torch.save(self.G,
                   f'{paths_config.checkpoints_dir}/model_{global_config.run_name}_multi_id.pt')
        else:
            torch.save(self.G, 
                   f'{paths_config.checkpoints_dir}/model_multi_id.pt')
