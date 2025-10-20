# CMIS-Net: A Cascaded Multi-Scale Individual Standardization Network for Backchannel Agreement Estimation

![1234](framework.jpg)

## Dataset
Our model was trained and validated on [MPIIGroupInteraction](https://multimediate-challenge.org/datasets/Dataset_MPII/) dataset. 

## Usage
If you want to use our code for training, you need to use [OpenFace2.0](https://github.com/TadasBaltrusaitis/OpenFace) to extract facial keypoints for each frame in the video at first

To run the training you can call:

```sh
bash train.sh
```

## References
This repository references the source code of the following paper：

```
@article{fan2022isnet,
  title={Isnet: Individual standardization network for speech emotion recognition},
  author={Fan, Weiquan and Xu, Xiangmin and Cai, Bolun and Xing, Xiaofen},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing},
  volume={30},
  pages={1803--1814},
  year={2022},
  publisher={IEEE}
}
```

