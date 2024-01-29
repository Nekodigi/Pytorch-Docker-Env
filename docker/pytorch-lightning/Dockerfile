FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

WORKDIR /app

# USE JAPAN REGION for faster download
RUN sed -i 's@archive.ubuntu.com@ftp.jaist.ac.jp/pub/Linux@g' /etc/apt/sources.list
RUN ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime

# For python 3.9
RUN apt update && apt-get install -y software-properties-common

RUN add-apt-repository ppa:deadsnakes/ppa 

RUN apt install -y \
    wget \
    curl \
    bzip2 \
    build-essential \
    git \
    python3.9 \
    python3-pip 

# Env improvements
RUN echo 'alias python="/usr/bin/python3"' >> /root/.bashrc

RUN apt install -y \
    git-lfs \
    ca-certificates \
    libsndfile1-dev \
    libgl1 \
    libx11-6 \
    unzip

# Download and install Micromamba.
RUN curl -sL https://micro.mamba.pm/api/micromamba/linux-64/1.1.0 \
    | tar -xvj -C /usr/local bin/micromamba
ENV MAMBA_EXE=/usr/local/bin/micromamba \
    MAMBA_ROOT_PREFIX=/home/user/micromamba \
    CONDA_PREFIX=/home/user/micromamba \
    PATH=/home/user/micromamba/bin:$PATH

RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip3 install tensorflow[and-cuda]==2.13.0
RUN pip3 install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

RUN pip3 install lightning pyarrow pandas matplotlib lightning[extra] jupyter

# extra libraries
RUN pip3 install pandas seaborn pre-commit black black[jupyter] lmdb wandb torchinfo ipykernel einops ema_pytorch scipy
RUN pip3 install click requests tqdm pyspng ninja imageio-ffmpeg imageio
RUN pip3 install scikit-image clu tensorflow_datasets train utils tensorrt



# Not working for no reason!
WORKDIR /app


