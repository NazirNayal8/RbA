# RbA Model Zoo

## Introduction

In this file we provide download links for the models used to produce results in our paper, as well as their respective config files and initialization weights. The training was done using the KUACC HPC cluster. 

The configs for the pretrained models are provided under `ckpts/` folder, under which there are multiple folders and each folder represents a separate model, when downloading the checkpoints you should place the `model_final.pth` in its corresponding model_folder. `ckpts/` would like something like this:
```
ckpts/
  model_1/
    config.yaml
    model_final.pth
  model_2/
    config.yaml
    model_final.pth
  ...
``` 

## Pretrained Models for Initialization

It's common to initialize from backbone models pre-trained on ImageNet classification tasks. The following backbone models are available through the original Mask2Former repo:

* [R-50.pkl (torchvision)](https://dl.fbaipublicfiles.com/detectron2/ImageNetPretrained/torchvision/R-50.pkl): converted copy of [torchvision's ResNet-50](https://pytorch.org/docs/stable/torchvision/models.html#torchvision.models.resnet50) model.
  More details can be found in [the conversion script](tools/convert-torchvision-to-d2.py).
* [R-103.pkl](https://dl.fbaipublicfiles.com/detectron2/DeepLab/R-103.pkl): a ResNet-101 with its first 7x7 convolution replaced by 3 3x3 convolutions. This modification has been used in most semantic segmentation papers (a.k.a. ResNet101c in our paper). We pre-train this backbone on ImageNet using the default recipe of [pytorch examples](https://github.com/pytorch/examples/tree/master/imagenet).
* [Swin-{B-L}](tools/README.md): pretrained on ImageNet, details on how to download them models and transform them to detectron2 format can be found [here](tools/README.md).

**Note**: according the config files we provide, it is assumed that the pretrained checkpoints are placed under the `pretrained/` folder. Make sure they are either placed there or that the config file is overwritten with the desired path when running an experiment.

## Cityscapes Inlier Training

These models are trained on cityscapes dataset only and are used as a starting point for finetuning with outlier supervision.

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<tr>  <td valign="bottom" colspan="2" >  </td>
      <td valign="bottom" colspan="2" align="center"> <b>Road Anomaly</b> </td>
      <td valign="bottom" colspan="2" align="center"> <b>FS LaF</b> </td>
</tr>
<tr>
<th valign="bottom">Backbone</th>
<th valign="bottom">Cityscapes mIoU</th>
<th valign="bottom" align="center">AP</th>
<th valign="bottom">FPR95</th>
<th valign="bottom">AP</th>
<th valign="bottom">FPR95</th>
<th valign="bottom" align="middle">config</th>
<th valign="bottom" align="middle">ckpt</th> </tr>
<!-- TABLE BODY -->
 <tr><td align="left">Swin-B (1 dec layer)</td>
<td align="center">82.25</td>
<td align="center">78.45</td>
<td align="center">11.83</td>
<td align="center">60.96</td>
<td align="center">10.63</td>
<td align="center"><a href="ckpts/swin_b_1dl/config.yaml">config</a></td>
<td align="center"><a href="https://drive.google.com/file/d/13IJs_Kk1PMBVVxCN90HZZuuV1YcWZ0am/view?usp=sharing">model</a></td>
</tr>

<tr><td align="left">Swin-L (1 dec layer)</td>
<td align="center">82.65</td>
<td align="center">79.68</td>
<td align="center">15.02</td>
<td align="center">58.61</td>
<td align="center">71.79</td>
<td align="center"><a href="ckpts/swin_l_1dl/config.yaml">config</a></td>
<td align="center"><a href="https://drive.google.com/file/d/1PBjdwHjpbAo7v6pa7B0JZVOjkQ5av_qn/view?usp=sharing">model</a></td>
</tr>

</tbody></table>


## RbA + COCO Outlier Supervision

These models are finetuned from the checkpoints which were trained using Cityscapes dataset only.

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<tr>  <td valign="bottom" colspan="2" >  </td>
      <td valign="bottom" colspan="2" align="center"> <b>Road Anomaly</b> </td>
      <td valign="bottom" colspan="2" align="center"> <b>FS LaF</b> </td>
</tr>
<tr>
<th valign="bottom">Backbone</th>
<th valign="bottom">Finetuned Component</th>
<th valign="bottom" align="center">AP</th>
<th valign="bottom">FPR95</th>
<th valign="bottom">AP</th>
<th valign="bottom">FPR95</th>
<th valign="bottom" align="middle">config</th>
<th valign="bottom" align="middle">ckpt</th> </tr>
<!-- TABLE BODY -->
 <tr><td align="left">Swin-B (1 dec layer)</td>
<td align="center">MLP</td>
<td align="center">85.42</td>
<td align="center">6.92</td>
<td align="center">70.81</td>
<td align="center">6.30</td>
<td align="center"><a href="ckpts/swin_b_1dl_rba_ood_coco/config.yaml">config</a></td>
<td align="center"><a href="https://drive.google.com/file/d/1d5blruLB0ll6vtGAfvRH1iID6ArclWKD/view?usp=sharing">model</a></td>
</tr>

</tbody></table>

## RbA + Mapillary + COCO Outlier Supervision

These models are finetuned from the checkpoints used for training Cityscapes. During finetuning they are exposed to images from Mapillary as well.

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<tr>  <td valign="bottom" colspan="2" >  </td>
      <td valign="bottom" colspan="2" align="center"> <b>Road Anomaly</b> </td>
      <td valign="bottom" colspan="2" align="center"> <b>FS LaF</b> </td>
</tr>
<tr>
<th valign="bottom">Backbone</th>
<th valign="bottom">Finetuned Component</th>
<th valign="bottom" align="center">AP</th>
<th valign="bottom">FPR95</th>
<th valign="bottom">AP</th>
<th valign="bottom">FPR95</th>
<th valign="bottom" align="middle">config</th>
<th valign="bottom" align="middle">ckpt</th> </tr>
<!-- TABLE BODY -->
<!-- ROW: maskformer2_R50_bs16_50ep -->
 <tr><td align="left">Swin-B (1 dec layer)</td>
<td align="center">MLP</td>
<td align="center">89.16</td>
<td align="center">4.50</td>
<td align="center">78.27</td>
<td align="center">3.98</td>
<td align="center"><a href="ckpts/swin_b_1dl_rba_ood_map_coco/config.yaml">config</a></td>
<td align="center"><a href="https://drive.google.com/file/d/1bdqnl6UdtK3C1jsdAaJiVcPIIojqGfWP/view?usp=sharing">model</a></td>
</tr>

 <tr><td align="left">Swin-L (1 dec layer)</td>
<td align="center">MLP</td>
<td align="center">90.28</td>
<td align="center">4.92</td>
<td align="center">80.35</td>
<td align="center">4.58</td>
<td align="center"><a href="ckpts/swin_l_1dl_rba_ood_map_coco/config.yaml">config</a></td>
<td align="center"><a href="https://drive.google.com/file/d/1woFhXmGceUoDFZIDIoF0NW7ehkQl_DQS/view?usp=sharing">model</a></td>
</tr>

</tbody></table>