# Keras Class Activation Map on large images (single & multi layer options)

Given a larger image than the CNN was trained on, generates the CAM on the entire image. If image dimensions are not multiples of the CNN's trained image size, rightmost and/or bottommost parts of the image are ignored.

Works well with medical imaging or other images where spacial structure is not-important. Concretely, images that are comprised of repeated smaller images which are identifiable even if the entire image is not present are required.

Please note, multi-layered CAM currently works with **at most 7 classes** (due to unique color constraints)

## Getting Started

1. overlay_single_layered_cam_large_image
Given a large image, generates heatmap on a specific class of interest 

2. overlay_multi_layered_cam_large_image
Given a large image, generates heatmap using the top-x predicted classes

Must have a keras model with a global average pooling layer after the final convolution layer followed by a single input -> output layer 

Included jupyer notebook contains examples 

### Prerequisites

```
pip install matplotlib
pip install keras
pip install numpy
```

OpenCV for CAM overlay on image

## Results

#### Single Layer

##### 1. Areas that contain gray matter brain tissue

![Alt text](images/single_class_heatmap_GRAYMATTER.png?raw=true "Single CAM Overlay")

##### 2. Areas that contain white matter brain tissue

![Alt text](images/single_class_heatmap_WHITEMATTER.png?raw=true "Single CAM Overlay")

##### 3. Areas that contain blank space

![Alt text](images/single_class_heatmap_BLANK.png?raw=true "Single CAM Overlay")

#### Multi Layer 

##### Areas of multiple classes

![Alt text](images/generated_heatmap.png?raw=true "Multi CAM Overlay")

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

Original paper: https://arxiv.org/pdf/1512.04150.pdf
