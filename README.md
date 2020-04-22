# InfoGAN-freshman seminar
We're gonna use MNIST dataset to reproduce the paper!

## Usage
Edit the **`config.py`** file to select training parameters.

To train the model run **`train.py`**:
```sh
python3 train.py
```
After training the network to experiment with the latent code for the `MNIST` dataset run **`mnist_generate.py`**:
```sh
python3 mnist_generate.py --load_path /path/to/pth/checkpoint
```

