# GRCNet
This is the official repository of the CVPR 2025 paper: Towards Explicit Geometry-Reflectance Collaboration for Generalized LiDAR Segmentation in Adverse Weather.

The latest code will be released soon...

## Installation
Please follow [Pointcept](https://github.com/Pointcept/Pointcept.git) to install the base environment and [torchsparse](https://github.com/mit-han-lab/torchsparse.git) to install the Torchsparse2.0.

## Data Preparation
- Download [SemanticKITTI](http://www.semantic-kitti.org/dataset.html#download) dataset.
- Link dataset to codebase.
  ```bash
  # SEMANTIC_KITTI_DIR: the directory of SemanticKITTI dataset.
  # |- SEMANTIC_KITTI_DIR
  #   |- dataset
  #     |- sequences
  #       |- 00
  #       |- 01
  #       |- ...
  
  mkdir -p data
  ln -s ${SEMANTIC_KITTI_DIR} ${CODEBASE_DIR}/data/semantic_kitti
  ```
  
- Download [SemanticSTF](https://github.com/xiaoaoran/SemanticSTF.git) dataset.
- Link dataset to codebase.
  ```bash
  # SEMANTIC_STF_DIR: the directory of SemanticSTF dataset.
  # |- SEMANTIC_STF_DIR
  #   |- test
  #   |- train
  #   |- val

  
  mkdir -p data
  ln -s ${SEMANTIC_STF_DIR} ${CODEBASE_DIR}/data/semantic_stf
  ```

## Acknowledgement
We thank the projects [Pointcept](https://github.com/Pointcept/Pointcept.git), [SemanticSTF](https://github.com/xiaoaoran/SemanticSTF.git) and [torchsparse](https://github.com/mit-han-lab/torchsparse.git).

