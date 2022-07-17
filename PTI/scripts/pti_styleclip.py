import glob
import os
from argparse import Namespace
import sys
import argparse
import pdb

sys.path.append(".")
sys.path.append("..")

import numpy as np
import torch
from tqdm import tqdm

from configs import paths_config
from models.StyleCLIP.mapper.scripts.inference import run_


meta_data = {
    'eyeglasses': ['eyeglasses', False, False, True, 0.15],
}

def run_PTI(run_name='', use_wandb=False, use_multi_id_training=False):
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = global_config.cuda_visible_devices

    if run_name == '':
        global_config.run_name = ''.join(choice(ascii_uppercase) for i in range(12))
    else:
        global_config.run_name = run_name

    if use_wandb:
        run = wandb.init(project=paths_config.pti_results_keyword, reinit=True, name=global_config.run_name)
    global_config.pivotal_training_steps = 1
    global_config.training_step = 1

    embedding_dir_path = f'{paths_config.embedding_base_dir}/{paths_config.input_data_id}/{paths_config.pti_results_keyword}'
    os.makedirs(embedding_dir_path, exist_ok=True)

    dataset = ImagesDataset(paths_config.input_data_path, transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]))

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    if use_multi_id_training:
        coach = MultiIDCoach(dataloader, use_wandb)
    else:
        coach = SingleIDCoach(dataloader, use_wandb)

    coach.train()

    return global_config.run_name

def styleclip_edit(frame_path, use_multi_id_G, inverted_root, run_name, output_dir=None, force=False):
    pretrained_mappers = paths_config.style_clip_pretrained_mappers

    images_dir = frame_path
    
    images = sorted(glob.glob(f"{images_dir}/*.jpeg") + glob.glob(f"{images_dir}/*.jpg") + glob.glob(f"{images_dir}/*.png"))

    model_path = os.path.join(inverted_root, 'model_multi_id.pt')

    for edit_type in meta_data.keys():
        edit_id = meta_data[edit_type][0]
        latent_codes = []
        print('Direction: ', edit_type)

        if not os.path.exists(os.path.join(output_dir, 'StyleCLIP', edit_type, 'latents.pth')) or force:
            for image_name in tqdm(images):
                image_name = image_name.split(".")[0].split("/")[-1]

                embedding_dir = os.path.join(inverted_root, 'embeddings', run_name, paths_config.pti_results_keyword, image_name)  # debugging: hard-coding
                                
                latent_path = f'{embedding_dir}/0.pt'
            
                args = {
                    "exp_dir": os.path.join(output_dir, 'StyleCLIP'),
                    "checkpoint_path": f"{pretrained_mappers}/{edit_id}.pt",
                    "couple_outputs": False,
                    "mapper_type": "LevelsMapper",
                    "no_coarse_mapper": meta_data[edit_type][1],
                    "no_medium_mapper": meta_data[edit_type][2],
                    "no_fine_mapper": meta_data[edit_type][3],
                    "stylegan_size": 1024,
                    "test_batch_size": 1,
                    "latents_test_path": latent_path,
                    "test_workers": 1,
                    "model_path": model_path,
                    "image_name": image_name,
                    'edit_name': edit_type,
                    # "data_dir_name": data_dir_name
                }
                factor = meta_data[edit_type][4]
                latent_code = run_(Namespace(**args), 
                                model_path, image_name, use_multi_id_G, factor)
                latent_codes.append(latent_code)
            latent_codes = torch.stack(latent_codes)
            torch.save(latent_codes, os.path.join(output_dir, 'StyleCLIP', edit_type, 'latents.pth'))
            

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inverted_root", type=str, required=True,
    )
    parser.add_argument("--aligned_frame_path", type=str, required=True)
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument(
        "--use_multi_id_G", action='store_true', 
    )
    parser.add_argument(
        "--output_root", type=str, required=True,
    )
    parser.add_argument(
        "--force", action='store_true'
    )
    args = parser.parse_args()

   
    styleclip_edit(frame_path=args.aligned_frame_path, use_multi_id_G=args.use_multi_id_G, inverted_root=args.inverted_root, run_name=args.run_name, 
                    output_dir=args.output_root, force=args.force) 
