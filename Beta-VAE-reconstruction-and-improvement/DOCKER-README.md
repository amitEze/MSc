# β-VAE Reproduction (Dockerized)

This repository provides a Dockerized setup for reproducing the β-VAE models from Higgins et al. (2017) and Burgess et al. (2018).

## Prerequisites

- **Docker Desktop** installed and running, with WSL2 enabled if you are using Windows.
- The following files present in the root directory of the repository:
  - `Dockerfile`
  - `requirements.txt`

## Dataset Setup

To train the CelebA model, download the dataset and organize it so that PyTorch's `ImageFolder` can read it.

1. Download CelebA from the [official Google Drive repository](https://drive.google.com/drive/folders/0B7EVK8r0v71pTUZsaXdaSnZBZzg?resourcekey=0-rJlzl934LzC-Xp28GeIBzQ).
2. Extract the downloaded archive.
3. Place the `.jpg` files in the following folder structure inside the project directory. The `images` subfolder is required because it acts as a class directory for the data loader.

```text
Beta-VAE-reconstruction-and-improvement/
├── data/
│   └── CelebA/
│       └── images/
│           ├── 000001.jpg
│           ├── 000002.jpg
│           └── ...
├── Dockerfile
└── requirements.txt
```

## Docker Setup

### Build the Image

Run the following command from the repository root:

```bash
docker build -t beta-vae-env .
```

### Run the Container

```bash
docker run -it --rm -v ${PWD}:/workspace -p 8097:8097 beta-vae-env
```

This mounts the current project directory at `/workspace` inside the container and exposes the Visdom server on port `8097`.

## Run Training

Run the following commands inside the container.

### 1. Fix Windows Line Endings

If the scripts were created or edited on Windows, convert their line endings:

```bash
sed -i 's/\r$//' *.sh
```

### 2. Start the Visdom Server

Start Visdom in the background:

```bash
python -m visdom.server &
```

### 3. Launch the Experiment

```bash
sh run_celeba_H_beta10_z10.sh
```

### 4. Monitor Progress

Open [http://localhost:8097](http://localhost:8097) in your host machine's web browser to view live training plots and reconstructions.

> **Tip:** Press `Ctrl+C` to stop a training run. Enter `exit` to leave the container. Because the container was started with `--rm`, it will then be removed automatically.