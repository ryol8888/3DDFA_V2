# coding: utf-8


__author__ = 'visionDR'


import argparse
import imageio
import cv2
import glob
import os
import numpy as np
from tqdm import tqdm
import yaml
from collections import deque
from FaceBoxes import FaceBoxes
from TDDFA import TDDFA
from utils.render import render
# from utils.render_ctypes import render
from utils.functions import cv_draw_landmark

def main(args):
    cfg = yaml.load(open(args.config), Loader=yaml.SafeLoader)
    # Init FaceBoxes and TDDFA, recommend using onnx flag
    if args.onnx:
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        os.environ['OMP_NUM_THREADS'] = '4'
        from FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
        from TDDFA_ONNX import TDDFA_ONNX

        face_boxes = FaceBoxes_ONNX()
        tddfa = TDDFA_ONNX(**cfg)
    else:
        gpu_mode = args.mode == 'gpu'
        tddfa = TDDFA(gpu_mode=gpu_mode, **cfg)
        face_boxes = FaceBoxes()
        
    # run
    dense_flag = args.opt in ('2d_dense', '3d')
    pre_ver = None
    
    if not os.path.isdir(args.save_dataset_path):
        os.makedirs(args.save_dataset_path)

    data = glob.glob(args.training_dataset_path + '/*.jpg')
    # print(args.training_dataset_path + '/*.jpg')
    for img_file in tqdm(data):
        label_file = img_file.split('\\')[-1].replace('.jpg', '.txt')
        label_path = os.path.join(args.save_dataset_path, label_file)
        label_f = open(label_path, mode='wt',encoding='utf-8')

        img = cv2.imread(img_file)
        img_save = cv2.resize(img, (640,480), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
        boxes = face_boxes(img)
        # print(boxes)
        if len(boxes) ==0:
            boxes = [[0, 0, img.shape[1], img.shape[0]]]
        param_lst, roi_box_lst = tddfa(img, boxes)
        ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)[0]
        
        # refine
        param_lst, roi_box_lst = tddfa(img, [ver], crop_policy='landmark')
        ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)[0]

        # write img
        # cv2.imwrite(label_path.replace('.txt','.jpg'), img_save)
        # # write label
        label = ''
        for pt in range(ver.shape[1]):
            label += str(ver[0,pt] / img.shape[1]) + ' ' + str(ver[1,pt] / img.shape[0]) + ' '
        label += '\n'
        label_f.write(label)
        label_f.close()
        # if args.opt == '2d_sparse':
            # img_draw = cv_draw_landmark(img, ver)  # since we use padding

        # cv2.imshow('image', img_draw)
        # k = cv2.waitKey(20)
        # if (k & 0xff == ord('q')):
        #     break

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='The smooth demo of webcam of 3DDFA_V2')
    parser.add_argument('-c', '--config', type=str, default='configs/mb1_120x120.yml')
    parser.add_argument('-m', '--mode', default='cpu', type=str, help='gpu or cpu mode')
    parser.add_argument('-o', '--opt', type=str, default='2d_sparse', choices=['2d_sparse', '2d_dense', '3d'])
    parser.add_argument('--onnx', action='store_true', default=False)
    parser.add_argument('--training_dataset_path', default='D:/Dataset/FD/001_demo', type=str,
                    help='load dataset')
    parser.add_argument('--save_dataset_path', default='D:/Dataset/FD/001_demo_68', type=str,
                        help='save dataset')

    args = parser.parse_args()
    main(args)
