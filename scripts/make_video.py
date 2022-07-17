import os
import glob
import argparse

from tqdm import tqdm
import imageio

parser = argparse.ArgumentParser()
parser.add_argument('--frame_dir', type=str, required=True)
parser.add_argument('--output_dir', type=str, required=True)
parser.add_argument('--fps', type=int, default=30)
parser.add_argument('--quality', type=int, default=5)
parser.add_argument('--num_frames', type=int, default=None)

args = parser.parse_args()

frame_paths = sorted(glob.glob(os.path.join(args.frame_dir, '*.jpg')) + glob.glob(os.path.join(args.frame_dir, '*.png')))

if args.num_frames is not None:
    frame_paths = frame_paths[:args.num_frames]
frames = []
for frame_path in tqdm(frame_paths):
    frames.append(imageio.imread(frame_path))

imageio.mimwrite(
            os.path.join(args.output_dir), frames, fps=args.fps, quality=args.quality, 
)