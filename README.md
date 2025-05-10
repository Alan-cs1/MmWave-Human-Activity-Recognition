# MmWave-Human-Activity-Recognition
Enhancing Activity Recognition: Motion Waveform Preprocessing from Millimeter Wave Radar Data for Transformer-Based Classification

This repo contains the implementation for the paper: Enhancing Activity Recognition: Motion Waveform Preprocessing from Millimeter Wave Radar Data for Transformer-Based Classification presented at the 2rd ICME Workshop on Human Motion Analysis (HMA) 2025. 

Demonstrations of our mmWave human activity recognition system are available on [YouTube](https://www.youtube.com/playlist?list=PL6yDbH2HXncKyYYQ4v0Yjt3W3Nl-oNENf).

Human Activity Recognition (HAR) using millimeter-wave radar provides a privacy-preserving alternative to camera-based methods. However, existing radar-based approaches often rely on 2D heatmaps or voxelized representations, leading to high computational costs. Additionally, Transformer models commonly used to capture temporal relationships in sequential data usually demand significant training and inference resources. This paper introduces a Transformer-based framework for HAR using motion waveforms derived from radar point clouds. The approach includes key components such as data attribute selection, geometric clustering, and motion waveform generation to emphasize active body segments (e.g., limb movements). Experimental results demonstrate that our method achieves competitive accuracy on the MMActivity dataset compared to previous methods while maintaining real-time efficiency. Furthermore, we collected our own radar-based HAR dataset to validate the robustness of our approach.

<p align="center">
  <img src="Image/System_Overview.jpg" alt="System Overview" style="width:600px; height:auto;"/>
</p>

## Dataset
This dataset consists of point cloud data collected using the Texas Instruments mmWave Radar IWR6843AOPEVM. The data acquisition was performed with the TI Radar Toolbox version 1.00.01.07. Both the raw point cloud data and the preprocessed 1D-motion waveform data are available for download via [Google Drive](). The dataset is organized into separate CSV files for training and testing purposes.

## Data Preprocessing
### Visualizing or Saving the 3D Point Cloud Video
To visualize or save the 3D point cloud data as a video, run:
```
python plot_3d_point_cloud.py
```

### Calculating point cloud global mean: 
The global mean calculation is necessary only when updating or recollecting the dataset. This step must be done before running the preprocessing. To calculate the global mean, run:
```
python calc_global_mean.py
```

### Raw data to 1D-motion waveform preprocessing:
Preprocess the raw data into a 1D-motion waveform. This will generate a JPEG feature map and a CSV data file. To perform this preprocessing, run:
```
python plot_featuremap.py
```

## Classifiers

This project includes implementations of the following neural network architectures: CNN, CNN-LSTM, CNN-BiLSTM, Transformer.

The different hyperparameters, including whether to run the model architectures are controlled by the configuration file ```config.json```.

The code is adapted from the repository: [Boosting Inertial-based Human Activity Recognition with Transformers](https://github.com/yolish/har-with-imu-transformer).

### Training a Model
To train a model on the mmWave dataset, run the following command:
```
python main.py train <path_to_mmwave_data.csv>
```
During training, the model checkpoints and log files will be saved to a dedicated output folder (automatically created if it doesn't exist).

### Testing a Trained Model
To evaluate a trained model, use the following command:
```
python main.py train <path_to_mmwave_data.csv> --checkpoint_path <path_to_your_model_checkpoint>
```
