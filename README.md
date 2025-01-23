# MMTSCNet
This is the official repository for MMTSCNet (Multimodal Tree Species Classification Network). This network was developed to accurately classify LiDAR point clouds of individual trees by augmenting the existing geometric information from the point cloud with metrics and frontal as well as sideways views of the point clouds. This approach ensures as much information as possible is captured and allows for very high accuracies when classifying coniferous and broadleaf tree species.

To start tuning and training MMTSCNet use the following command:

```bash
python main.py --datadir "C:\The\path\to\the\source\data" --workdir "C:\The\path\to\your\working\folder" --modeldir "C:\The\path\to\your\models\folder" --capsel "ALS" --growsel "LEAF-ON"
```

but don't forget to change the capture method selection (capsel) and growth state selection (growsel) accordingly.

# Model architecture
![alt text](images/MMTSCNet_architecture_updated.png)

The architecture is composed of four branches, each handling a different type of input data, hence the name "Multimodal". The first branch extracts features from 68 different metrics, generated from geometric properties of the point clouds themselves as well as full-waveform LiDAR data if available. The second and third branch use an instance of DenseNet121 to extract features from the frontal and sideways views of the tree point clouds and the third branch acts upon the point cloud directly to extract features. The extracted features are then concazenated to create a single feature vector which is fed into fully-connected layers. The resulting feature-rich vector is then classified by a Softmax-Layer.

Using this architecture, the model is able to achieve an Overall Accuracy (OA) of 96.36%, a macro F1-Score of 0.9589 as well as a Precision of 0.9728 and a Recall of 0.9564 over 7 different tree species. The model was trained and used to predict on the PANGAEA dataset of multi-source segmented LiDAR tree point clouds by Weiser et al., available at [https://doi.pangaea.de/10.1594/PANGAEA.942856](https://doi.pangaea.de/10.1594/PANGAEA.942856), with pre-segmented tree point clouds from multiple forest plots near Karlsruhe and Bretten in Germany.

![alt text](images/Ka_Br_Extents.png)

# Installation
To train and use MMTSCNet on Windows 11, follow the installation guide below (NVIDIA GPU required).

#### Step 1: 
Install Visual Studio Code

#### Step 2:
Clone this repository to your local machine.

#### Step 3:
Create a new Conda environment with [Python 3.8.20](https://www.python.org/downloads/release/python-3820/) and activate it.

#### Step 4:
Install CUDA toolkit and CUDNN on your local machine (requires NVIDIA developer account)

#### Step 5:
Install the required python modules using
```bash
pip install -r requirements.txt
```

# Data structure
#### Model directory:
Create a new folder anywhere on your local machine and tell the software about it using
```bash
--modeldir "C:\Enter\the\path\to\models\folder\here"
```
All tuned and trained models will be save in this folder, accompanied by metrics plots and confusion matrices.<br>
<br>
<br>

#### Working directory:
Create a new folder anywhere on your local machine and tell the software about it using
```bash
--workdir "C:\Enter\the\path\to\working\folder\here"
```
It is highly recommended to create a working folder with any name, with a subfolder called "source_data" where your source data is located to keep things tidy and clean.<br>
<br>
<br>

#### Data directory:
Create a new folder in your working folder called "source_data" and paste your source data into it. Let the software know by using
```bash
--datadir "C:\Enter\the\path\to\source_data"
```
<br>

Your data itself should follow the structure proposed by the [dataset](https://doi.pangaea.de/10.1594/PANGAEA.942856) to initially train and test the model on, separating properties in filenames with "_".<br>
<br>

**DISCLAIMER:** Full-waveform data is fully optional. It can enhance the models classification accuracy but is not required and the software will automatically detect FWF data if it is present.<br>
<br>

```bash
WORKING DIRECTORY
│
└── source_data
    └── las
    │   ├── Plot01.zip
    │   │   ├── ALS
    │   │   │   ├── ALS-on_Plot01_2019-07-05_300m.laz
    │   │   │   ├── ALS-on_Plot01_2019-07-05_trajectory.txt
    │   │   │   └── ...
    │   │   │
    │   │   ├── TLS
    │   │   │   ├── TLS-on_Plot01_2019-07-05.laz
    │   │   │   ├── TLS-on_Plot01_2019-07-05_scan_positions.txt
    │   │   │   └── ...
    │   │   │
    │   │   ├── ULS
    │   │   │   ├── ULS-off_Plot01_2019-07-05.laz
    │   │   │   ├── ULS-off_Plot01_2019-07-05_trajectory.txt
    │   │   │   └── ...
    │   │   │
    │   │   └── single_trees
    │   │       ├── AceCam_Plot01_1
    │   │       │   ├── AceCam_Plot01_1_2019-07-05_quality3_ALS-on.laz
    │   │       │   ├── AceCam_Plot01_1_2019-07-05_quality3_ULS-on.laz
    │   │       │   ├── AceCam_Plot01_1_2019-07-05_quality3_TLS-off.laz
    │   │       │   └── ...
    │   │       │
    │   │       ├── FagSyl_Plot01_2
    │   │       │   └── ...
    │   │       ├── CarBet_Plot01_3
    │   │       │   └── ...
    │   │       └── ...
    │   │
    │   ├── Plot02.zip
    │   │   └── ...
    │   └── ...
    │
    │
    │
    │
    └── fwf
        ├── Plot01.zip
        │   ├── 190705_102252_VQ780i_Linie_0050_BR01_epsg25832.las
        │   ├── 190705_102252_VQ780i_Linie_0050_BR01_epsg25832.wdp
        │   ├── 190705_102252_VQ780i_Linie_0050_BR01_epsg25832_trajectory.txt
        │   ├── 190705_102703_VQ780i_Linie_0051_BR01_epsg25832.las
        │   ├── 190705_102703_VQ780i_Linie_0051_BR01_epsg25832.wdp
        │   ├── 190705_102703_VQ780i_Linie_0051_BR01_epsg25832_trajectory.txt
        │   └── ...
        │
        ├── Plot02.zip
        │   └── ...
        └── ...
```

# Usage
Before you are able to use MMTSCNet to predict the tree species for any point cloud, it has to be tuned and trained. The command to tune MMTSCNet on your data would look something like this:
```bash
python main.py --datadir "C:\The\path\to\the\source\data" --workdir "C:\The\path\to\your\working\folder" --modeldir "C:\The\path\to\your\models\folder" --capsel "ALS" --growsel "LEAF-ON"
```

To enable debug logging, simply modify the command to:
```bash
python main.py --datadir "C:\The\path\to\the\source\data" --workdir "C:\The\path\to\your\working\folder" --modeldir "C:\The\path\to\your\models\folder" --verbose
```


#### Available Arguments:
```bash
--datadir "C:\Enter\the\path\to\source\here"
```
This command is REQUIRED and is supposed to point a your source data folder.<br>
<br>

```bash
--workdir "C:\Enter\the\path\to\working\folder\here"
```
This command is REQUIRED and is supposed to point to a user-created folder to save data in while working.<br>
<br>

```bash
--modeldir "C:\Enter\the\path\to\models\folder\here"
```
This command is REQUIRED and is supposed to point to a user-created folder to save trained models and plots in.<br>
<br>

```bash
--elimper 5.0        [0.0 to 100.0]
```
This command is OPTIONAL and defines the minimum representation below which species are eliminated from training.<br>
<br>

```bash
--maxpcscale 0.005        [0.001 to 0.01]
```
This command is OPTIONAL and defines the scaling applied to point clouds during data augmentation.<br>
<br>

```bash
--ssstest 0.3        [0.05 to 0.5]
```
This command is OPTIONAL and defines the split ratio for trainign and testing data.<br>
<br>

```bash
--capsel "ULS"        [ALS | TLS | ULS | ALL]
```
This command is OPTIONAL and defines the capture methods to be included in the model training process.<br>
<br>

```bash
--growsel "LEAF-ON"        [LEAF-ON | LEAF-OFF | ALL]
```
This command is OPTIONAL and defines the tree growth state to be included in the model training process.<br>
<br>

```bash
--batchsize 16        [4 | 8 | 16 | 32]
```
This command is OPTIONAL and defines the batch size for hp-tuning, training and predicting.<br>
<br>

```bash
--numpoints 2048        [512 | 1024 | 2048 | 4096]
```
This command is OPTIONAL and defines the number of points, point clouds will be resampled to for the model.<br>
<br>

```bash
--verbose
```
This command is OPTIONAL and enables debug logging if used.<br>
<br>

```bash
--prediction
```
This command is OPTIONAL and enables predicting for custom data if used.<br>
<br>

**DISCLAIMER:** The values displayed for the commands represent the default values.<br>
<br>

# About the creator
MMTSCNet was developed by Jan Richard Vahrenhold during his Masters Thesis at the Technische Hochschule Würzburg-Schweinfurt. You can learn more about Jan [on his website](https://vahrenhol3d.de).