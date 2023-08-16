## Data download and preparation

Download the scenes (We use Brandenburg gate, Trevi fountain, and Sacre coeur in our experiments) from [Image Matching Challenge PhotoTourism (IMC-PT) 2020 dataset](https://www.cs.ubc.ca/~kmyi/imw2020/data.html) 

Download the train/test split from [NeRF-W](https://nerf-w.github.io/) and put under each scene's folder (the **same level** as the "dense" folder, see more details in <a href="#tree-structure-of-each-dataset">Tree structure of each dataset</a>

Run 
```bash
ROOT_DIR="/mnt/cephfs/dataset/NVS/nerfInWild/brandenburg_gate/"
img_downscale=2
python prepare_phototourism.py --root_dir $ROOT_DIR --img_downscale $img_downscale
#$ROOT_DIR is the directory of dataset
#$img_downscale is an integer, e.g. 2 means half the image sizes
```
to prepare the training data, This will **largely** accelerate the speed of data preparation step before training.
The generated data will be saved on the **same level** as the "dense" folder

## Tree structure of each dataset
</details>

<br>



<details>

```
brandenburg_gate/
├── dense/
│   ├── images/
│   ├── sparse/
│   │      ├──depth_maps/
│   │      ├──depth_maps_clean_300_th_0.10/
│   ├── stereo/
│   
├── cache/
│ 
├──brandenburg.tsv


trevi_fountain/
├── dense/
│   ├── images/
│   ├── sparse/
│   │      ├──depth_maps/
│   │      ├──depth_maps_clean_300_th_0.10/
│   ├── stereo/
│   
├── cache/
│ 
├──trevi.tsv


sacre_coeur/
├── dense/
│   ├── images/
│   ├── sparse/
│   │      ├──depth_maps/
│   │      ├──depth_maps_clean_300_th_0.10/
│   ├── stereo/
│   
├── cache/
│ 
├──sacre.tsv

```

</details>
