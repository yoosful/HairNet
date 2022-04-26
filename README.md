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
- the trained model will be saved in the **weight** folder
  Training by running this command with list of arguments below:

```bash
python train.py --data ./data/ --epochs 10 --lr 1e-4
```

List of Arguments
| Args | Type | |
|--------------|-------|--------------------------------------------------------------|
| --epoch | int | Number of epoch |
| --batch_size | int | Number of batch size |
| --lr | float | Learning rate |
| --lr_step | int | Number of epoch to reduce 1/2 lr |
| --data | str | Path to ./data/ |
| --save_dir | str | Path to save trained weights |
| --weight | str | Load weight from path |
| --test_step | int | If `test_step` != 0, test each `n` step after train an epoch |

_Notes_: Hyperparameters of original HairNet.

- Epoch: 500
- Batch size: 32
- Learning rate: 1e-4
- Learning rate step: 10 epochs
- Optimization: Adam

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
