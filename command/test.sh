# Description: test script for CR-Nerf
cd /mnt/cephfs/home/yangyifan/yangyifan/code/learnToSyLf/CR-NeRF
source /mnt/cephfs/home/yangyifan/miniconda/etc/profile.d/conda.sh
conda activate crnerf
exp_name1="test_only/test" #replace with your own experiment name
root_dir1="/mnt/cephfs/dataset/NVS/nerfInWild/brandenburg_gate/" #repalce with your own dataset path
save_dir1=/mnt/cephfs/dataset/NVS/nerfInWild/experimental_results #replace with your own save path
model_mode1="1-1" 
decoder='linearStyle' 
ckpt_path1="ckpts/CR-NeRF-branden.ckpt" 
dataset_name1='phototourism'
decoder_num_res_blocks=1
img_downscale=4 #The provided model is trained with image resolution downscale ratio = 4.
nerf_out_dim1=64

#################### render image ####################
CUDA_VISIBLE_DEVICES=5 python eval.py \
  --root_dir $root_dir1 \
  --save_dir $save_dir1 \
  --dataset_name $dataset_name1 --scene_name $exp_name1 \
  --split test_test --img_downscale $img_downscale \
  --N_samples 256 --N_importance 256 --N_emb_xyz 15 \
  --N_vocab 1500  \
  --ckpt_path $ckpt_path1 \
  --chunk 2048 --img_wh 320 240   --encode_a --decoder $decoder --decoder_num_res_blocks $decoder_num_res_blocks --nerf_out_dim $nerf_out_dim1

CUDA_VISIBLE_DEVICES=0  python eval_metric.py \
  --root_dir $root_dir1 \
  --save_dir $save_dir1 \
  --dataset_name $dataset_name1 --scene_name $exp_name1 \
  --split test_test --img_downscale $img_downscale \
  --img_wh 320 240