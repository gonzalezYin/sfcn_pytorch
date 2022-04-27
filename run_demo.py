import argparse
import os
import torch.backends.cudnn as cudnn
import models
import torchvision.transforms as transforms
import flow_transforms
from imageio import imread, imsave
from loss import *
import time
import random
from glob import glob
import matplotlib.pyplot as plt

import sys
sys.path.append('../cython')
from connectivity import enforce_connectivity


'''
Infer from custom dataset:
author:Fengting Yang 
last modification: Mar.5th 2020
'''

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__"))


parser = argparse.ArgumentParser(description='PyTorch SPixelNet inference on a folder of images',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--data_dir', metavar='DIR', default='./pretrained_demo/inputs',
                    help='path to images folder')
parser.add_argument('--dump_root', metavar='DIR', default='./dump_root',
                    help='path to images folder')
parser.add_argument('--data_suffix', default='jpg',
                    help='suffix of the testing image')
parser.add_argument('--pretrained_model', metavar='PTH',
                    help='path to pretrained model', default='./pretrained_ckpt/SpixelNet_bsd_ckpt.tar')
parser.add_argument('--save_path', metavar='DIR', default='./pretrained_demo',
                    help='path to output folder')
parser.add_argument('--downsize', default=16, type=float,
                    help='superpixel grid cell, must be same as training setting')
parser.add_argument('-nw', '--num_threads', default=1, type=int,
                    help='num_threads')
parser.add_argument('-b', '--batch_size', default=8, type=int, metavar='N',
                    help='mini-batch size')
args = parser.parse_args()

random.seed(100)
args.test_list = args.dump_root + '/test.txt'


@torch.no_grad()
def test(args, model, test_list, res_path, n):
    # Data loading code
    input_transform = transforms.Compose([
        flow_transforms.ArrayToTensor(),
        transforms.Normalize(mean=[0, 0, 0], std=[255, 255, 255]),
        transforms.Normalize(mean=[0.411, 0.432, 0.45], std=[1, 1, 1])
    ])

    img_file = test_list[n]
    load_path = img_file
    img_id = os.path.basename(img_file)[:-4]

    # may get 4 channel (alpha channel) for some format
    color = True
    ori_img = imread(load_path)
    if len(ori_img.shape) == 2:
        ori_img = np.tile(np.expand_dims(ori_img, 2), (1, 1, 3))
        mask = np.where(ori_img > 0, np.ones_like(ori_img), np.zeros_like(ori_img))
        color = False
    ori_img = ori_img[:, :, :3]
    H, W, _ = ori_img.shape
    H_, W_ = int(np.ceil(H/16.)*16), int(np.ceil(W/16.)*16)

    # ---------------------------------------------------------------
    # get spixel id
    n_spix_h = int(np.floor(H_ / args.downsize))
    n_spix_w = int(np.floor(W_ / args.downsize))

    spix_value = np.int32(np.arange(0, n_spix_w * n_spix_h).reshape((n_spix_h, n_spix_w)))
    spix_idx_tensor = shift9pos(spix_value)
    spix_idx_tensor = np.repeat(np.repeat(spix_idx_tensor, args.downsize, axis=1), args.downsize, axis=2)

    spix_id = torch.from_numpy(np.tile(spix_idx_tensor, (1, 1, 1, 1))).type(torch.float).cuda()
    n_spix = int(n_spix_h * n_spix_w)
    # ----------------------------------------------------------------

    img = cv2.resize(ori_img, (W_, H_), interpolation=cv2.INTER_CUBIC)
    # torchvision.transforms.Compose()
    img1 = input_transform(img)
    ori_img = input_transform(ori_img)

    # compute output
    tic = time.time()
    output = model(img1.cuda().unsqueeze(0))
    # assign the spixel map
    curr_spix_map = update_spixl_map(spix_id, output)
    ori_sz_spixel_map = F.interpolate(curr_spix_map.type(torch.float), size=(H_,W_), mode='nearest').type(torch.int)

    mean_value = torch.tensor([0.411, 0.432, 0.45], dtype=img1.cuda().unsqueeze(0).dtype).view(3, 1, 1)
    spix_viz, spix_label_map = get_spixel_image((ori_img + mean_value).clamp(0, 1), ori_sz_spixel_map.squeeze(), n_spixels=n_spix,  b_enforce_connect=True)


    # ************************ Save all result********************************************
    # save img, uncomment it if needed
    # if not os.path.isdir(os.path.join(save_path, 'img')):
    #     os.makedirs(os.path.join(save_path, 'img'))
    # spixl_save_name = os.path.join(save_path, 'img', imgId + '.jpg')
    # img_save = (ori_img + mean_values).clamp(0, 1)
    # imsave(spixl_save_name, img_save.detach().cpu().numpy().transpose(1, 2, 0))

    # save spixel viz
    if not os.path.isdir(os.path.join(res_path, 'spixel_viz')):
        os.makedirs(os.path.join(res_path, 'spixel_viz'))
    spixl_save_name = os.path.join(res_path, 'spixel_viz', img_id + '.png')
    imsave(spixl_save_name, spix_viz.transpose(1, 2, 0))


    # save the unique maps as csv, uncomment it if needed
    if not os.path.isdir(os.path.join(res_path, 'map_csv')):
        os.makedirs(os.path.join(res_path, 'map_csv'))
    output_path = os.path.join(res_path, 'map_csv', img_id + '.csv')
    # plus 1 to make it consistent with the toolkit format
    np.savetxt(output_path, spix_label_map.astype(int), fmt='%i', delimiter=",")

    if n % 10 == 0:
        print("processing %d" % n)
    total_time = time.time()-tic

    return total_time


def main():
    global args, save_path
    data_dir = args.data_dir
    print('=>fetch image in {}'.format(data_dir))

    save_path = args.save_path
    print('=>save output to {}'.format(save_path))
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    # test_lst = glob(args.data_dir + '/*.' + args.data_suffix)
    # test_lst.sort()

    test_list = []
    mean_time_list = []
    with open(args.test_list, 'r') as tf:
        img_path = tf.readlines()
        for path in img_path:
            test_list.append(path[:-1])

    print('The {} samples found'.format(len(test_list)))

    # create model
    tic = time.time()
    network = torch.load(args.pretrained_model)
    model = models.__dict__[network['arch']](data=network).cuda()
    model.eval()
    args.arch = network['arch']
    cudnn.benchmark = True
    toc = time.time() - tic
    print("load model time %f" % toc)

    mean_time = 0
    for n in range(len(test_list)):
        total_time = test(args, model, test_list, save_path, n)
        mean_time += total_time
        mean_time_list.append(total_time)
    mean_time /= len(test_list)
    print('The avg time per img:', mean_time)
    # np.savetxt(save_path, mean_time_list, fmt='%i', delimiter=",")

    with open(args.save_path + '/mean_time.txt', 'w+') as f:
        for item in mean_time_list:
            tmp = "total_time:{}\n".format(item[0])
            f.write(tmp)



if __name__ == '__main__':
    main()
