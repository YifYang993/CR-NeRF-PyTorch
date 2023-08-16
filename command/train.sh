
#!Note: set N_vocab to 1500 for brandenburg_gate and Sacre coeur dataset, 3200 for trevi fountain datasets
#use command --use_cache to use prepared dataset as described in docs/dataset.md for speeding up training 
#set --encode_a to use the proposed cross ray appearance transfer module
#set --use_mask to use the proposed transient handling module

cd /mnt/cephfs/home/yangyifan/yangyifan/code/learnToSyLf/CR-NeRF
source /mnt/cephfs/home/yangyifan/miniconda/etc/profile.d/conda.sh
conda activate crnerf
nerf_out_dim1=64
exp_name1="train/exp1"
model_mode1="1-1" 
decoder='linearStyle'  
ckpt_path1="/mnt/cephfs/dataset/NVS/nerfInWild/experimental_results/ckpts/${exp_name1}/last.ckpt"
root_dir1="/mnt/cephfs/dataset/NVS/nerfInWild/brandenburg_gate/"
dataset_name1='phototourism'
save_dir1=/mnt/cephfs/dataset/NVS/nerfInWild/experimental_results
decoder_num_res_blocks=1
img_downscale=2



#train#
CUDA_VISIBLE_DEVICES=3  python train_mask_grid_sample.py --root_dir $root_dir1 --dataset_name phototourism --save_dir /mnt/cephfs/dataset/NVS/nerfInWild/experimental_results --img_downscale $img_downscale --N_importance 64 --N_samples 64 --num_epochs 20 --batch_size 1024 --optimizer adam --lr 5e-4 --lr_scheduler cosine  --N_emb_xyz 15 --N_vocab 1500 --maskrs_max 5e-2 --maskrs_min 6e-3 --maskrs_k 1e-3 --maskrd 0 --N_a 48 --weightKL 1e-5 --weightRecA 1e-3 --weightMS 1e-6  --chunk 1310720 --encode_a --encode_c  --encode_random --model_mode 1-1 --decoder linearStyle --decoder_num_res_blocks $decoder_num_res_blocks --nerf_out_dim $nerf_out_dim1 --use_cache --proj_name style_gnerf --num_gpus 1 --use_mask --exp_name $exp_name1 



#test#
cd /mnt/cephfs/dataset/NVS/nerfInWild/experimental_results/logs/$exp_name1/codes

#render image#
CUDA_VISIBLE_DEVICES=3 python eval.py \
  --root_dir $root_dir1 \
  --save_dir $save_dir1 \
  --dataset_name $dataset_name1 --scene_name $exp_name1 \
  --split test_test --img_downscale $img_downscale \
  --N_samples 256 --N_importance 256 --N_emb_xyz 15 \
  --N_vocab 1500  \
  --ckpt_path $ckpt_path1 \
  --chunk 2048 --img_wh 320 240   --encode_a --decoder $decoder --decoder_num_res_blocks $decoder_num_res_blocks --nerf_out_dim $nerf_out_dim1

#calculate metrics#
CUDA_VISIBLE_DEVICES=0  python eval_metric.py \
  --root_dir $root_dir1 \
  --save_dir $save_dir1 \
  --dataset_name $dataset_name1 --scene_name $exp_name1 \
  --split test_test --img_downscale $img_downscale \
  --img_wh 320 240