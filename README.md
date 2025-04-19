# M5D-GS
[AAAI 2025] The official implementation for the "Motion Decoupled 3D Gaussian Splatting for Dynamic Object Representation" [Paper](https://drive.google.com/file/d/1DZnkiIoHsxtnf_NKxC1tnn_vRnRu2D45/view?usp=drive_link)

## Update
[2025.04.18] The code for viewing/evaluating trained scenes are available. The training script is under testing and will be uploaded soon.

[2025.02.03] The [dataset](/m5d_data) is available for download.

[2025.1.17] We are preparing the code and dataset. Please check back later.

## Run

### 1. Environment Install

a. clone this repo

`git clone https://github.com/haliphinx/M5D-GS.git --recursive`

b. install environment (same as [Deformable 3D-GS](https://github.com/ingra14m/Deformable-3D-Gaussians))

```
conda create -n m5d_gs python=3.7
conda activate m5d_gs

# install pytorch
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116

# install dependencies
pip install -r requirements.txt
```

### 2. Run Viewer on Trained Scenes
a. prepare the dataset (see [this page](/m5d_data))

b. prepare the trained model files ([our checkpoints](https://uottawa-my.sharepoint.com/personal/xhu008_uottawa_ca/_layouts/15/guestaccess.aspx?share=El3O-dVaY_tDlEuCasqKotABh3SwRGu6h7j777dC2m66zw&e=hzpiHl))

c. run the script as 

`python viewer_gui.py -s /path/to/dateset/folder -m /path/to/checkpoint/folder --eval --gui`


## Pipeline

<img src="res/main_graph.png" alt="Image1" style="zoom:25%;" />
<div align="center"><b><i>Figure.</i></b> <i>The workflow of M5D-GS. The 3D-GSs for each step and the ground truth images are also visulized</i></div>

<p></p>

The 3D-GSs are first transformed by the object level motion to form the overall motion. Then, each 3D-GS is modified by per-Gaussian deformation to capture local deformations


## Results
### Comparison with SOTAs
<img src="res/tab_res.png" alt="Image1" style="zoom:25%;" />
<div align="center"><b><i>Table.</i></b> <i>Quantitative comparison with previous SOTA methods.</i></div>
<p></p><p></p>
<img src="res/main_vis.png" alt="Image1" style="zoom:25%;" />
<div align="center"><b><i>Figure.</i></b> <i>Visual comparison with previous SOTA methods.</i></div>
<p></p><p></p>

<div align="center">
<img src="res/cat.gif" alt="Image1" style="zoom:25%;" />

<b><i>Figure.</i></b> <i>Comparison on the real world scene.</i></div>
</div>


### Estimated Object Motion
<img src="res/traj_fish.gif" alt="Image1" style="zoom:25%;" /> <img src="res/traj_robdog.gif" alt="Image2" style="zoom:25%;" /> <img src="res/traj_elephant.gif" alt="Image3" style="zoom:25%;" /> <img src="res/traj_jjacks.gif" alt="Image4" style="zoom:25%;" />
<div align="center"><b><i>Figure.</i></b> <i>Visualization of the estimated object motions. The white curves are the estimated trajectories, red spots are the location, and the rgb frames represent the orientations.</i></div>

## Dataset
See [this page](/m5d_data) for details.

## Acknowledgments
We sincerely thank the authors of [3D-GS](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/), [D-NeRF](https://www.albertpumarola.com/research/D-NeRF/index.html), [HyperNeRF](https://hypernerf.github.io/), [NeRF-DS](https://jokeryan.github.io/projects/nerf-ds/), [DeVRF](https://jia-wei-liu.github.io/DeVRF/),  and [D3D-GS]([https://jia-wei-liu.github.io/DeVRF/](https://drive.google.com/file/d/1DZnkiIoHsxtnf_NKxC1tnn_vRnRu2D45/view?usp=drive_link)), whose codes and datasets were used in our work.
