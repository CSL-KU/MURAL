# MURAL: A Multi-Resolution Anytime Framework for LiDAR Object Detection Deep Neural Networks

This is the code repository of MURAL, published in RTCSA 2025.

## 🚀 Quick Start

### Prerequisites

- Docker with NVIDIA container runtime support
- NVIDIA GPU or iGPU (tested on Jetson Xavier, Jetson Orin, and RTX 3050)
- nuScenes dataset, can be downloaded from [here](https://www.nuscenes.org/nuscenes).
- Pre-trained model checkpoints, download from [here](https://kansas-my.sharepoint.com/:u:/g/personal/a249s197_home_ku_edu/Eb65ucMEk49Djv7jZTmXtcsBDNv3-ZPcUVF_1RQafdfhxQ?e=MGM9OK).

### Training the models yourself

- We did the training using a separate fork of the OpenPCDet repo (named AL-Train) available [here](https://github.com/ahmedius2/AL-Train). Instructions on how to train are planned to be available later. However, the same instructions on the OpenPCDet repository can be followed while utilizing the config files provided in the AL-Train repository. We recommend using GPU(s) having at least 16 GB of total memory. In our case, we used a single RTX 4090.

### 1. Clone the Repository

```bash
git clone https://github.com/CSL-KU/MURAL.git
cd MURAL/docker
```

### 2. Build Docker Image

#### For x86 Systems

Build the Docker image with the appropriate CUDA architecture for your GPU:

```bash
docker buildx build . --build-arg CUDA_ARCH="8.6" -f Dockerfile.x86 -t kucsl/mural:x86_nv23.10
```

> **Note:** The example above uses CUDA_ARCH="8.6" assuming RTX 3050. You can find your GPU's CUDA architecture number at: https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/

#### For Jetson Orin AGX

```bash
docker buildx build . --build-arg CUDA_ARCH="8.7" -t kucsl/mural:jetson-orin
```

### 3. Run Docker Container

Execute the following command (example for x86 systems):

```bash
docker run --gpus all --net host -it --ipc=host --privileged \
    --cap-add=ALL --ulimit rtprio=99 --tmpfs /tmpfs \
    -v $NUSCENES_PATH:/root/nuscenes \
    -v $MODELS_PATH:/root/MURAL/models \
    --name mural kucsl/mural:x86_nv23.10
```

**Environment Variables:**
- `NUSCENES_PATH`: Path to your nuScenes dataset. The hierarchy of the dataset folder should be as follows:
```
nuscenes/
└── v1.0-trainval/
    │── samples/
    │── sweeps/
    │── maps/
    └── v1.0-trainval/
```
- `MODELS_PATH`: Path to the downloaded model checkpoint files.

### 5. Initialize the Environment

Once inside the container (this happens automatically due to the `-it` flag), run:

```bash
cd ~/MURAL/tools
. initialize.sh
```

## 📊 Running Experiments

### Calibration

Before running experiments, execute the benchmarking (calibration) procedure to build TensorRT engines and collect timing data:

```bash
. do_calib.sh
```

### Main Experiments

To run the experiments for PillarNet and PointPillars (CenterPoint version):

```bash
. do_run_tests.sh
```

This script evaluates all methods presented in the paper (baselines and MURAL) across a range of deadlines. If the script fails to complete some tests, simply re-run it to complete those. When completed, it will also plot all the results and save them in the same directory.

### Customizing Test Parameters

You can modify the deadline ranges by editing the `do_run_tests.sh` script. Look for commands like:

```bash
./run_tests.sh methods BEGIN STEP END
```

Where:
- `BEGIN`: Starting deadline value (in seconds)
- `STEP`: Increment step (in seconds)
- `END`: Ending deadline value (in seconds)

The existing deadline ranges in the do_run_tests.sh script are used for an RTX 3050 (power usage was limited to 30W).

## 📄 Citation

```bibtex
@INPROCEEDINGS{mural2025,
  author={Soyyigit, Ahmet and Yao, Shuochao and Yun, Heechul},
  booktitle={IEEE International Conference on Embedded and Real-Time Computing Systems and Applications (RTCSA)}, 
  title={MURAL: A Multi-Resolution Anytime Framework for LiDAR Object Detection Deep Neural Networks},
  address={Singapore},
  year={2026}
}
```

## 📧 Contact

For questions and support, please open an issue in this repository.
