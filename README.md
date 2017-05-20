# Keras CAM Implementation on large images (extension of KerasClassActivationMap)

Given a larger image than the CNN was trained on (dimensions must be multiples of original training dimension), generates the CAM on the entire image.

Works well with medical imaging or other images where spacial structure is not-important (large image is comprised of repeated smaller images)

Please note, multi-layered CAM currently works with **at most 7 classes**


## Getting Started

Within CAM_custom_large_image.py, please read docstring of:
1. overlay_cam_large_image
2. overlay_multi_stacked_cam_large_image

Must have a keras model with a global average pooling layer after the final convolution layer followed by a single input -> output layer 

Included jupyer notebook contains examples 

### Prerequisites

```
pip install matplotlib
pip install keras
pip install numpy
```

OpenCV for CAM overlay on image

### Results

#### Single Layer

![Alt text](multi_tiled_tissue_SINGLE_LAYER_NO_BLANK_CAM.jpg?raw=true "Single CAM Overlay")

#### Multi Layer

![Alt text](multi_tiled_tissue_NO_BLANK_CAM.jpg?raw=true "Multi CAM Overlay")

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

Original paper: https://arxiv.org/pdf/1512.04150.pdf
