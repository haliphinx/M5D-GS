# M5D-DATASET

The dataset is accessable via [OneDrive](https://uottawa-my.sharepoint.com/personal/xhu008_uottawa_ca/Documents/Study/m5d-gs/M5D_DATA/M5D_DATA.zip?csf=1&web=1&e=i2fylH) and [GoogleDrive](https://drive.google.com/file/d/1EpJzgfRb_YNQ2H4910yUPXdiPk_i2lyP/view?usp=sharing).

## Data Structure
```
M5D_DATA
|--real
    |--cat
        |--camera
        |--camera_path
        |--colmap
        |--dataset.json
        |--metadata.json
        |--rgb
        |--scene.json
    |--pillow
|--synthetic
    |--bballs_motion
        |--train
        |--test
        |--transforms_train.json
        |--transforms_test.json
    |--elephant
    |--...
```

## Data Loader
The synthetic scenes follow the data structure of [D-NeRF](https://github.com/albertpumarola/D-NeRF), and the real world scenes follow [Nerfies](https://github.com/google/nerfies).

The original dataloaders can be found in their repos. Both dataloaders are included in our code repo already.

## Copyright
All rights reserved. For non-commercial use only.
### Synthetic Scenes

bball_motion, hwarrior_motion, and jjacks_motion are augmented from bouncing balls, hell warrior and jumping jacks from the [D-NeRF dataset](https://github.com/albertpumarola/D-NeRF).

The 3D models of the reset synthetic scenes are: 
[elephant](https://sketchfab.com/3d-models/african-elephant-facb060916534e7eae8e6a5a8056185f​), 
[robot](https://www.cgtrader.com/free-3d-models/character/sci-fi-character/robot-l2), 
[pokemon](https://www.cgtrader.com/free-3d-models/character/fantasy-character/bulbasaur-3d-model-4446c7a7-f786-434f-9b43-1082c5003f04), 
[fish](https://www.cgtrader.com/free-3d-models/animals/fish/tuna-fish-9f41924a-83d4-478e-a8e9-370946f141b3​), 
[robot dog](https://www.cgtrader.com/free-3d-models/character/other/robot-dog-83cb60c2-2f95-44cc-b50a-f0ed026b7135​).

### Real World Scenes

Videos are captured by the auther of M5D-GS. Special thanks to Grey for being the cat modal, and Yifan for holding the pillow.