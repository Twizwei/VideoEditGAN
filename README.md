# VideoEditGAN

[![arXiv](https://img.shields.io/badge/arXiv-2108.00946-b31b1b.svg)](https://arxiv.org/abs/2206.10590)

[Project Website](https://video-edit-gan.github.io/) | [Paper](https://arxiv.org/abs/2206.10590)

This is the repo for ECCV'22 paper, "Temporally Consistent Semantic Video Editing". 

## Updates
07/16/2022: repo initialized.

## Prerequisites
- Linux
- Anaconda/Miniconda
- Python 3.6 (tested on Python 3.6.7)
- PyTorch
- CUDA enabled GPU

Install packages:
```
conda env create -f environment.yml
```

## Get started
Let's use `examples/aamir_khan_clip.mp4` as an example.

- Split a video to frames:
```
python scripts/vid2frame.py --pathIn examples/aamir_khan_clip.mp4 --pathOut out/aamir_khan/frames 
```

- Face alignment. We use [3DDFA_V2](https://github.com/cleardusk/3DDFA_V2) for face alignment.
First, clone 3DDFA_V2 to folder:
```
git clone https://github.com/cleardusk/3DDFA_V2.git
cd 3DDFA_V2
```
Then, install the dependency following the [instructions](https://github.com/cleardusk/3DDFA_V2#getting-started), and build the cython
```
sh ./build.sh
```
We provide a code snippet `single_video_smooth.py` to generate facial landmarks for the alignment. Run
```
cp ../scrpits/single_video_smooth.py ./
python single_video_smooth.py -f out/aamir_khan/frames
```
The `landmarks.npy` will be saved at `path-to-video/../landmarks/landmarks.npy`.

Then we can transform the faces using detected landmarks.
```
cd ../
python scripts/align_faces_parallel.py --num_threads 1 --root_path out/aamir_khan/frames --output_path out/aamir_khan/aligned
```
We then run a naive unalignment to see if the alignment makes sense. This will also provide the parameters for the post-processing.
```
python scripts/unalign.py --ori_images_path out/aamir_khan/frames --aligned_images_path out/aamir_khan/aligned --output_path out/aamir_khan/unaligned
```

- GAN inversion

For in-domain editing, we use [PTI](https://github.com/danielroich/PTI) to do the inversion.
We have included PTI in this repo. To use it, download [pre-trained models](https://github.com/danielroich/PTI#auxiliary-models) and put them in `PTI/pretrained_models/`, then start the inversion (this will take a while):
```
cd PTI
python scripts/run_pti_multi.py --data_root ../out/aamir_khan/aligned --run_name aamir_khan --checkpoint_path ../out/aamir_khan/inverted
```

- Direct editing

Here we use StyleCLIP mapper as an example. Download the pretrained mapper [here](https://drive.google.com/file/d/1CEO3eQr46KnfB8e-U8AZ9LDHaL0NwJda/view?usp=sharing), and put it into `PTI/pretrained_models/`. Then, run
```
python scripts/pti_styleclip.py --inverted_root ../out/aamir_khan/inverted --run_name aamir_khan_eyeglasses --aligned_frame_path ../out/aamir_khan/aligned --output_root ../out/aamir_khan/in_domain --use_multi_id_G
```

- Our flow-based method
Now that we have prepared everything, the next step is to run our proposed method. 

Our method relies on [RAFT](https://github.com/princeton-vl/RAFT), a flow estimator. Download the pretrained network [here](https://drive.google.com/drive/folders/1sWDsfuZ3Up38EUQt7-JDTT1HcGHuJgvT), and put `raft-things.pth` into `VideoEditGAN/pretrained_models/`. 

Put pretrained mapper into `VideoEditGAN/pretrained_models`, for example
```
cd VideoEditGAN/pretrained_models
ln -s PTI/pretrained_models/eyeglasses.pt ./
```

Run our proposed method:
```
cd VideoEditGAN/
python -W ignore scripts/temp_consist.py --edit_root out/aamir_khan/in_domain --metadata_root out/aamir_khan/unaligned --original_root out/aamir_khan/frames --aligned_ori_frame_root out/aamir_khan/aligned --checkpoint_path out/aamir_khan/inverted --batch_size 1 --reg_frame 0.2 --weight_cycle 10.0 --weight_tv_flow 0.0 --lr 1e-3 --weight_photo 1.0 --reg_G 100.0 --lr_G 1e-04 --weight_out_mask 0.5 --weight_in_mask 0.0 --tune_w --epochs_w 10 --tune_G --epochs_G 3 --scale_factor 4 --in_domain --exp_name 'temp_consist' --run_name 'aamir_khan_eyeglasses'
```

- Unalignment

As a final step, we run [STIT](https://github.com/rotemtzaban/STIT) as a post-processing to put the aligned face back to the input video.
```
python video_stitching_tuning_ours.py --input_folder ../out/aamir_khan/in_domain/StyleCLIP/eyeglasses/temp_consist/tune_G/aligned_frames --output_folder ../out/aamir_khan/in_domain/StyleCLIP/eyeglasses/temp_consist/tune_G/aligned_frames/stitiched --edit_name 'eyeglasses' --latent_code_path ../out/aamir_khan/in_domain/StyleCLIP/eyeglasses/temp_consist/tune_G/variables.pth --gen_path out/aamir_khan/in_domain/StyleCLIP/eyeglasses/temp_consist/tune_G/G.pth --metadata_path ../out/aamir_khan/unaligned --output_frames --num_steps 50
```

## Citation
If you find the code useful, please consider citing our paper:

	@article{xu2022videoeditgan,
            author    = {Xu, Yiran and AlBahar, Badour and Huang, Jia-Bin},
            title     = {Temporally consistent semantic video editing},
            journal   = {arXiv preprint arXiv: 2206.10590},
            year      = {2022},
            }

## Acknowledgements
The codebase is heavily built upon prior work. We would like to thank
- [StyleGAN2-ada](https://github.com/NVlabs/stylegan2-ada) and [rosinality](https://github.com/rosinality)'s [implementation](https://github.com/rosinality/stylegan2-pytorch)
- [3DDFA](https://github.com/cleardusk/3DDFA_V2)
- [PTI](https://github.com/danielroich/PTI)
- [ReStyle](https://github.com/yuval-alaluf/restyle-encoder)
- [StyleCLIP](https://github.com/orpatashnik/StyleCLIP)
- [StyleGAN-NADA](https://github.com/rinongal/StyleGAN-nada)
- [STIT](https://github.com/rotemtzaban/STIT)
- [RAFT](https://github.com/princeton-vl/RAFT)
