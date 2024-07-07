import os
import numpy as np
from PIL import Image, ImageDraw
import argparse


def main():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--save_dir', default='./vis_output/merge', help='the dir to save images with bboxes')
    parser.add_argument('--gt', default='./vis_output/gt', help='the dir of images with annotations on them')
    parser.add_argument('--pred', default='./vis_output/pred', help='the dir of images with predictions on thme')
    args = parser.parse_args()
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    GT = args.gt
    PRED = args.pred
    assert os.path.exists(GT), print ('you give wrong gt image path')
    assert os.path.exists(PRED), print ('you give wrong pred image path')

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    img_files = [f for f in os.listdir(PRED) if 'png' in f]

    for img_file in img_files :
        gt = os.path.join(GT, img_file)
        pred = os.path.join(PRED, img_file)
        gt_img = Image.open(gt)
        pred_img = Image.open(pred)
        w, h = gt_img.size
        new_img = Image.new(mode='RGB', size=(2*w, h))
        new_img.paste(gt_img, (0,0))
        new_img.paste(pred_img, (w,0))
        new_img.save(os.path.join(args.save_dir, img_file))


if __name__ == '__main__':
    main()
