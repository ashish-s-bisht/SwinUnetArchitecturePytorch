# Swin-Unet architecture implementation in Pytorch.

The Swin-UNet is a version of the widely used U-Net architecture that combines the windowed self-attention mechanism with the U-Net framework.

![Untitled](https://github.com/ashish-s-bisht/SwinUnetArchitecturePytorch/assets/155929502/19b6f784-e4b2-420e-9f65-df7e41259f80)

The Swin-Transformer builds on the Vision-Transformer by calculating the attention limited to a local  window and making use of a shifted windows for providing connections between windows that significantly enhance modeling
power of the architecture.

The attention limitation to a local window allows it have a linear computation complexity to input image size against the quadratic of Vision Transformer.

The stacking of Swin-Transformer blocks allows hierarchical feature maps  by merging image patches in deeper layers.
![Untitled (2)](https://github.com/ashish-s-bisht/SwinUnetArchitecturePytorch/assets/155929502/b408d8d6-2cae-4d2e-ae97-f5c80dd22aaa)
The shift of the window between consecutive transformer block allows for cross-window connection and thus enabling learning of finer details required in dense prediction tasks such as object detection and semantic segmentation.
![whifted windows approach PNG](https://github.com/ashish-s-bisht/SwinUnetArchitecturePytorch/assets/155929502/fda10803-225b-46d4-a4c3-3223cb410cf5)

## References:
* https://arxiv.org/abs/2105.05537
* https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/swin_transformer.py
