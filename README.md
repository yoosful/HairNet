# HAIRNET

## Abstract

- Implementation of [HairNet: Single-View Hair Reconstruction using Convolutional Neural Networks](https://arxiv.org/abs/1806.07467)
- Several changes in training data & network architecture for faster training
  - Smaller input size: 256 x 256 -> 128 x 128
  - Fewer convolution layers in the encoder

## Instructions

### Installation

- python3
- python packages (pytorch, opencv, etc.)
  ```bash
    pip install -r requirements.txt
  ```

### Pre-processing

- Hair detection

```bash
python hair_detection/main.py --image_path *.png --map_color=False --save_img=False
```

### Training

- **Train** mode performs network training with 70% of the data
- [Download](https://bit.ly/32l59ZD) HairNet's training data (2D hair orientation image & 3D hair model)
- The trained model will be saved in the **weight** folder
- Training by running this command with list of arguments below:

  ```bash
  python src/train.py
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
- Learning rate step: 250 epochs
- Optimization: Adam

### Example

```bash
python src/main.py --mode example --conv *.convdata
```

### Demo

- **Demo** mode takes image as input and visualizes hairs

```bash
python src/main.py --mode demo --weight *.pt --img_path *.png
```
