cd CR-NeRF

#Path of checkpoints
ckpt_path1="ckpts/CR-NeRF-downscale=2.ckpt"
img_downscale=2

# If you want to render image with down scale=4, Remove # of the folloing 2 lines of code
# ckpt_path1="ckpts/CR-NeRF-downscale=4.ckpt"
# img_downscale=4

#Path of in-the-wild dataset
root_dir1="/mnt/cephfs/dataset/NVS/nerfInWild/brandenburg_gate/"
#Path to save the synthesized images
save_dir1=/mnt/cephfs/dataset/NVS/nerfInWild/experimental_results
exp_name1="ablation/test_train"
dataset_name1='phototourism'
decoder_num_res_blocks=1
decoder='linearStyle'  
nerf_out_dim1=64


CUDA_VISIBLE_DEVICES=0 python eval.py \
  --root_dir $root_dir1 \
  --save_dir $save_dir1 \
  --dataset_name $dataset_name1 --scene_name $exp_name1 \
  --split test_test --img_downscale $img_downscale \
  --N_samples 256 --N_importance 256 --N_emb_xyz 15 \
  --N_vocab 1500  \
  --ckpt_path $ckpt_path1 \
  --chunk 2048 --img_wh 320 240   --encode_a --decoder $decoder --decoder_num_res_blocks $decoder_num_res_blocks --nerf_out_dim $nerf_out_dim1