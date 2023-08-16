##generating videos of trevi_fountain
# source /mnt/cephfs/home/yangyifan/miniconda/etc/profile.d/conda.sh
# conda activate crnerf
# cd /mnt/cephfs/home/yangyifan/yangyifan/code/learnToSyLf/CR-NeRF
# CUDA_VISIBLE_DEVICES=2 python appearance_modification_video.py \
#     --save_dir /mnt/cephfs/dataset/NVS/nerfInWild/experimental_results \
#     --chunk 4096 --encode_a --nerf_out_dim 64 --decoder_num_res_blocks 1\
#     --example_image images/artworks \
#     --scene_name ds4v2crnerf_artworks_2_fountain \
#     --ckpt_path ckpts/CR-NeRF-trevi_fountain.ckpt

##generating videos of brandenburg_gate
source /mnt/cephfs/home/yangyifan/miniconda/etc/profile.d/conda.sh
conda activate crnerf
cd /mnt/cephfs/home/yangyifan/yangyifan/code/learnToSyLf/CR-NeRF
CUDA_VISIBLE_DEVICES=2 python appearance_modification_video.py \
    --save_dir /mnt/cephfs/dataset/NVS/nerfInWild/experimental_results \
    --chunk 4096 --encode_a --nerf_out_dim 64 --decoder_num_res_blocks 1\
    --example_image images/artworks \
    --scene_name ds4v2crnerf_artworks_2_brandenburg_gate \
    --ckpt_path ckpts/CR-NeRF-branden.ckpt
