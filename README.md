# HairNet

## abstract

- implementation of [HairNet: Single-View Hair Reconstruction using Convolutional Neural Networks](https://arxiv.org/abs/1806.07467)
- several changes in training data & network architecture for faster training
  - smaller input size: 256 x 256 -> 128 x 128
  - fewer convolution layers in the encoder

## instructions

### installation

- python3
- python packages (pytorch, opencv, etc.)
  - `pip install -r requirements.txt`

### training

- **train** mode performs network training with 70% of the data
- [download](https://bit.ly/32l59ZD) HairNet's training data (2D hair orientation image & 3D hair model)
  - refer to the [author's implementation](https://github.com/papagina/HairNet_DataSetGeneration) for more details
- `python src/main.py --mode train --path .`
- the trained model will be saved in the **weight** folder

### testing

- **train** mode evaluates the network with 30% of the data
- train or [download](https://bit.ly/34I4QLx) a pretrained model to **weight** folder
- `python src/main.py --mode test --path . --weight weight/*_weight.pt`

### reconstruction

- **reconstruction** mode performs some post-processing and saves the 3D model as a `*.data` file
- `python src/main.py --mode reconstruction --path . --weight weight/*_weight.pt --interp_factor 1`
- to visualize the generated file, you will need a renderer
  - download our [OpenGL implementation](https://github.com/givenone/hair-renderer)
  - compile and run it on the `*.data` file

## Acknowledgement
Baseline implementation forked from [MrPhD](https://github.com/MrPhD).


