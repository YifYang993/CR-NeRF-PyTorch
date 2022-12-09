
####记得改名字 记得拷贝getmodel

cd /mnt/cephfs/home/yangyifan/yangyifan/code/learnToSyLf/Ha-NeRF; source activate ;conda activate HaNeRF;
nerf_out_dim1=64
exp_name1="test"
# exp_name1="debug"
# model_mode1="1-1" #|1-4-1|1-1
# decoder='stylenerf'  #esrgan||stylenerf
ckpt_path1="/mnt/cephfs/dataset/NVS/nerfInWild/experimental_results/ckpts/${exp_name1}/last.ckpt"
root_dir1="/mnt/cephfs/dataset/NVS/nerfInWild/brandenburg_gate/"
# root_dir1="/mnt/cephfs/dataset/NVS/nerfInWild/trevi_fountain/"
# root_dir1="/mnt/cephfs/dataset/NVS/nerfInWild/sacre_coeur/"
dataset_name1='phototourism'
save_dir1=/mnt/cephfs/dataset/NVS/nerfInWild/experimental_results
img_downscale=4
# decoder_num_res_blocks=1

CUDA_VISIBLE_DEVICES=3 python train_mask_grid_sample.py   --root_dir $root_dir1 --dataset_name $dataset_name1   --save_dir $save_dir1 --img_downscale $img_downscale --use_cache   --N_importance 64 --N_samples 64   --num_epochs 20 --batch_size 1024   --optimizer adam --lr 5e-4 --lr_scheduler cosine   --exp_name $exp_name1 --N_emb_xyz 15 --N_vocab 1500   --maskrs_max 5e-2 --maskrs_min 6e-3 --maskrs_k 1e-3 --maskrd 0   --N_a 48 --weightKL 1e-5 --weightRecA 1e-3 --weightMS 1e-6   --num_gpus 1 --chunk 131072  --use_mask --encode_a --encode_random   #--ckpt_path $ckpt_path1 #--model_mode $model_mode1 --decoder $decoder --decoder_num_res_blocks $decoder_num_res_blocks --num_epochs 1

cd /mnt/cephfs/dataset/NVS/nerfInWild/experimental_results/logs/$exp_name1/codes

CUDA_VISIBLE_DEVICES=6 python eval.py \
  --root_dir $root_dir1 \
  --save_dir $save_dir1 \
  --dataset_name $dataset_name1 --scene_name $exp_name1 \
  --split test_test --img_downscale $img_downscale \
  --N_samples 256 --N_importance 256 --N_emb_xyz 15 \
  --N_vocab 1500  \
  --ckpt_path $ckpt_path1 \
  --chunk 2048 --img_wh 320 240  #--encode_a #--model_mode $model_mode1 --decoder $decoder --decoder_num_res_blocks $decoder_num_res_blocks

CUDA_VISIBLE_DEVICES=4  python eval_metric.py \
  --root_dir $root_dir1 \
  --save_dir $save_dir1 \
  --dataset_name $dataset_name1 --scene_name $exp_name1 \
  --split test_test --img_downscale $img_downscale \
  --img_wh 320 240