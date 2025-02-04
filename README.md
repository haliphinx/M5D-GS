# M5D-GS
[AAAI 2025] The official implementation for the "Motion Decoupled 3D Gaussian Splatting for Dynamic Object Representation"

## Update
[2025.1.17] We are preparing the code and dataset. Please check back later.

[2025.02.03] The [dataset](/m5d_data) is available for download.

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
<img src="res/robot.gif" alt="Image1" style="zoom:25%;" />
</div>


### Estimated Object Motion
<img src="res/traj_fish.gif" alt="Image1" style="zoom:25%;" /> <img src="res/traj_robdog.gif" alt="Image2" style="zoom:25%;" /> <img src="res/traj_elephant.gif" alt="Image3" style="zoom:25%;" /> <img src="res/traj_jjacks.gif" alt="Image4" style="zoom:25%;" />
<div align="center"><b><i>Figure.</i></b> <i>Visualization of the estimated object motions. The white curves are the estimated trajectories, red spots are the location, and the rgb frames represent the orientations.</i></div>

## Dataset
See [this page](/m5d_data) for details.

## Code
TBD
