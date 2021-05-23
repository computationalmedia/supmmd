#!/bin/bash
ENV=supsum
echo "Installing/updating conda environment and dependencies"
conda env create -f environment.yml || conda env update -f environment.yml --prune

conda activate $ENV

#conda clean --all -y
echo "Installation completed !!!"

## dijkstra/laptop
conda install pytorch torchvision -c pytorch
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
conda install cupy
conda install -c conda-forge spacy[cuda101]
pip install spacy-transformers[cuda100]
