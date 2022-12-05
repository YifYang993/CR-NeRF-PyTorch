
#!/bin/bash
cd /mnt/cephfs/home/yangyifan/yangyifan/code/learnToSyLf/Ha-NeRF; source activate ;conda activate HaNeRF;
nerf_out_dim1=64


# exp_name1="gnerf_fvr_baseline_no_perturb64_updowndecoder1_sigmoid_perturb000001"###tested
# exp_name1="gnerf_fvr_baseline_no_perturb64_updowndecoder1_sigmoid"
# exp_name1="gnerf_fvr_baseline_no_perturb64_updowndecoder1_tanh"
# exp_name1="gnerf_fvr_baseline_no_perturb64_updowndecoder1_tanh_perturb000001"
declare -a StringArray=("1114_nerf_ds2")
gpu1=6

for val in ${StringArray[@]}; do

  exp_name1=$val
  echo $val
  echo $exp_name1
  model_mode1="1-1" #|1-4-1|1-1
  ckpt_path1="/mnt/cephfs/dataset/NVS/nerfInWild/experimental_results/ckpts/${exp_name1}/last.ckpt"
  root_dir1="/mnt/cephfs/dataset/NVS/nerfInWild/brandenburg_gate/"
  dataset_name1='phototourism'
  save_dir1=/mnt/cephfs/dataset/NVS/nerfInWild/experimental_results

  decoder_num_res_blocks=1
  decoder='stylenerf'#||stylenerf
  # CUDA_VISIBLE_DEVICES=7,6 python train_mask_grid_sample.py   --root_dir $root_dir1 --dataset_name $dataset_name1   --save_dir $save_dir1 --img_downscale 2 --use_cache   --N_importance 64 --N_samples 64   --num_epochs 20 --batch_size 1024   --optimizer adam --lr 5e-4 --lr_scheduler cosine   --exp_name $exp_name1 --N_emb_xyz 15 --N_vocab 1500   --maskrs_max 5e-2 --maskrs_min 6e-3 --maskrs_k 1e-3 --maskrd 0   --N_a 48 --weightKL 1e-5 --weightRecA 1e-3 --weightMS 1e-6   --num_gpus 2 --chunk 131072  --proj_name gnerf --nerf_out_dim $nerf_out_dim1 --model_mode $model_mode1 --pertubeCord
  cd /mnt/cephfs/dataset/NVS/nerfInWild/experimental_results/logs/$exp_name1/codes
  CUDA_VISIBLE_DEVICES=$gpu1 python eval.py \
    --root_dir $root_dir1 \
    --save_dir $save_dir1 \
    --dataset_name $dataset_name1 --scene_name $exp_name1 \
    --split test_test --img_downscale 2 \
    --N_samples 256 --N_importance 256 --N_emb_xyz 15 \
    --N_vocab 1500  \
    --ckpt_path $ckpt_path1 \
    --chunk 1024 --img_wh 320 240  --encode_a #--nerf_out_dim $nerf_out_dim1 --model_mode $model_mode1 --decoder $decoder --decoder_num_res_blocks $decoder_num_res_blocks 

  CUDA_VISIBLE_DEVICES=$gpu1  python eval_metric.py \
    --root_dir $root_dir1 \
    --save_dir $save_dir1 \
    --dataset_name $dataset_name1 --scene_name $exp_name1 \
    --split test_test --img_downscale 2 \
    --img_wh 320 240
done