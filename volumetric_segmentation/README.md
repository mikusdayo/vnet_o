# Volumetric segmentation

This library implements the 3D U-Net and V-Net models for volumetric segmentation using
[PyTorch](https://pytorch.org/).

## Setup

This library can be set up by executing:
```shell
pip install /path/to/this/directory
```

Alternatively, one may install requirements through
```shell
pip install -r /path/to/this/directory/requirements.txt
```

and add the directory to PYTHONPATH by changing ~/.bashrc or through:
```shell
export PYTHONPATH=$PYTHONPATH:/path/to/this/directory
```

## Network architectures
### 3D U-Net

The model implementation is based on <i>3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation</i>
by Ö. Çiçek, A. Abdulkadir, S. S. Lienkamp, T. Brox and O. Ronneberger (see: https://arxiv.org/abs/1606.06650). The
architecture is similar to a regular U-Net, but uses 3D convolutional layers to handle volumes. Moreover, the bottleneck
layer is replaced by a regular 3x3x3 double convolution block. As in the 2D variant, shortcut connections between
corresponding encoder-decoder layers are created.

![](https://user-images.githubusercontent.com/16364029/165028732-3c88faa3-0ddd-4e20-a3a9-3b89c5c55e8f.png)

### V-Net

Based on <i>V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation</i> by F. Milletari,
N. Navab and S. A. Ahmadi (see: https://arxiv.org/abs/1606.04797), this network uses skipped connections inside conv
blocks, apart from the usual ones between encoder-decoder levels. A PReLu non-linearity is used to further optimize
parameters during backpropagation. Contrary to the 3D U-Net, reducing dimensions is not performed using pooling layers.
They have been replaced by 2x2x2 convolutions with a stride of 2.

![](https://user-images.githubusercontent.com/16364029/165029707-a09c76e0-77fd-4e2e-9a95-49c418922116.png)

## Example usage

See [here](https://github.com/bwieciech/volumetric_segmentation/blob/main/notebooks/example.ipynb) for a Jupyter
Notebook on how to use these models. Credit to the author(s) of
[this dataset](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation?resource=download),
which I use in the mentioned tutorial. Also, thanks to
[lyakaap](https://github.com/lyakaap/pytorch-template/blob/master/src/losses.py) for the insights on dice loss
implementation in PyTorch.

## Citation

If you find this code useful, please cite the following:

```
@misc{bwieciech2022VolumetricSegmentation,
  author = {Wieciech, Bartosz},
  title = {volumetric_segmentation},
  year = {2022},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/bwieciech/volumetric_segmentation}},
}
```