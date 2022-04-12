# sfcn_pytorch
- Pytorch Implementation of Superpixel Segmentation with Full Connected Network
- The version is based on the [fuy34/superpixel_fcn](https://github.com/fuy34/superpixel_fcn) for the latest gpu.

## Setup
- python 3.9
- pytorch 1.11.0
- nvidia-smi 495.29.05
- cuda version 11.5
- RTX3090(sm86)

## Use pretrained model to show the visualization result
Change the default seeting the argparse command as you need.
```
python run_demo.py
```
Put your own images inside the ```/demo/input```, you will get the map and visualization result under the folder ```/demo/map_csv``` and ```/demo/spixel_viz```.
