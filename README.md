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
Change the default setting in the argparse command as you need.
```
python run_demo.py
```
Put your own images inside the ```/demo/input```, you will get the map and visualization result under the folder ```/demo/map_csv``` and ```/demo/spixel_viz```.

## Data preparation
Please first download the data from the [BSDS500 Dataset](http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_full.tgz), and extract it to ```<BSDS_DIR>```.

Change the default file path in the argparse command as you need.
```
dataset=<BSDS_DIR> 
dump_root=<DUMP_DIR> # the path where the preprocessed image stored
```
To generate training and validation dataset, use ```pre_process_bsd500.py```.
```
cd data_processing
python pre_process_bsd500.py
```
To generate test dataset, use ```pre_precess_bsd500_ori_sz.py```.
```
python pre_process_bsd500_ori_sz.py
```
