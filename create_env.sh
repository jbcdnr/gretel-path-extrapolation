#!/bin/bash

if [[ $# -eq 0 ]] ; then
    echo 'Please provide environment name'
    exit 1
fi

ENV_NAME=$1

# basic env
conda create --yes -n $ENV_NAME python=3.7 \
    geopandas \
    jupyterlab \
    matplotlib \
    nbconvert \
    networkx \
    notebook \
    numpy \
    osmnx \
    pandas \
    pylint \
    rope \
    scikit-learn \
    scipy \
    scipy \
    seaborn \
    tabulate \
    tensorboardx \
    termcolor \
    tqdm \
    yapf \
    -c conda-forge

source activate $ENV_NAME

# can change the CUDA version for your system
conda install -y pytorch torchvision cudatoolkit=9.0 -c pytorch
export CUDA_HOME=/usr/local/cuda-9.0

# PyTorch
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric

# PyGSP
pip install git+https://github.com/epfl-lts2/pygsp


# Scatter library for PyTorch
# if installation fails on macOS
# see https://rusty1s.github.io/pytorch_geometric/build/html/notes/installation.html#c-cuda-extensions-on-macos
pip install git+https://github.com/rusty1s/pytorch_scatter
pip install git+https://github.com/rusty1s/pytorch_sparse
pip install git+https://github.com/rusty1s/pytorch_cluster
pip install git+https://github.com/rusty1s/pytorch_spline_conv
pip install git+https://github.com/rusty1s/pytorch_geometric

source deactivate

echo "Created environment $ENV_NAME"
