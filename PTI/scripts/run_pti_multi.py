from random import choice
from string import ascii_uppercase
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import os
import sys
import argparse
import numpy as np

sys.path.append(".")
sys.path.append("..")

from configs import global_config, paths_config

from training.coaches.multi_id_coach import MultiIDCoach
from training.coaches.single_id_coach import SingleIDCoach
from utils.ImagesDataset import ImagesDataset

from tqdm import tqdm
import pdb

def run_PTI(input_data_path, checkpoint_path, batch_size=3, run_name='', use_multi_id_training=False):
    """
    PTI training for a single video.
    """
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = global_config.cuda_visible_devices

    if run_name == '':
        global_config.run_name = ''.join(choice(ascii_uppercase) for i in range(12))
    else:
        global_config.run_name = run_name

    global_config.pivotal_training_steps = 1
    global_config.training_step = 1
    paths_config.input_data_id = run_name 
    paths_config.checkpoints_dir = checkpoint_path
    paths_config.embedding_base_dir = os.path.join(checkpoint_path, 'embeddings')
    paths_config.input_data_path = input_data_path
    paths_config.visualization_dir = os.path.join(checkpoint_path, 'PTI_visuals')

    # embedding save path
    embedding_dir_path = f'{paths_config.embedding_base_dir}/{paths_config.pti_results_keyword}'
    os.makedirs(embedding_dir_path, exist_ok=True)

    dataset = ImagesDataset(paths_config.input_data_path, transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]))

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    if use_multi_id_training:
        coach = MultiIDCoach(dataloader, False)
    else:
        coach = SingleIDCoach(dataloader, False)

    os.makedirs(paths_config.checkpoints_dir, exist_ok=True)
    coach.train(checkpoint_path=checkpoint_path)

    return global_config.run_name


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True, help='path to the video data')
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--checkpoint_path", type=str, required=True, help='path to save results.')
    parser.add_argument("--run_name", type=str, default='aamir_khan')
    args = parser.parse_args()

    run_PTI(batch_size=args.batch_size, run_name=args.run_name, input_data_path=args.data_root, checkpoint_path=args.checkpoint_path, use_multi_id_training=True)
