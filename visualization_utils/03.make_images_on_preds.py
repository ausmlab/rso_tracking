import os
import numpy as np
from PIL import Image, ImageDraw
import argparse


def main():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--save_dir', default='./videos', help='the dir to save images with bboxes')
    parser.add_argument('--video_dir', default='./data/FAI/', help='the dir to have sequences of images for drawing bboxes')
    parser.add_argument('--preds_dir', default='./exp/tracking/FAI_SEQS/results_rso', help='the dir to have predicted txt files')
    args = parser.parse_args()
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    videos = [f for f in os.listdir(args.video_dir)]

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    for video in videos :
        # MOT File structure
        img_files = [f for f in os.listdir(os.path.join(args.video_dir, video, 'img1')) if 'png' in f]

        img_files = np.sort(img_files)

        for i, img_file in enumerate(img_files) :
            # Open img
            img_path = os.path.join(args.video_dir, video, 'img1', img_file)
            save_path = os.path.join(args.save_dir, img_file)

            im = Image.open(img_path)
            draw = ImageDraw.Draw(im)
            draw.text((0, 0), video, fill=(255, 255, 255))

            # Open Trackning labels
            with open(os.path.join(args.preds_dir, '{}.txt'.format(video)), 'r') as f:
                lines = f.readlines()

            for line in lines :
                frame, tid, minx, miny, w, h, _, _, _, _ = line.strip().split(',')
                frame = int(frame)
                if frame == i+1 :
                    tid = int(tid)
                    minx = float(minx)
                    miny = float(miny)
                    w = float(w)
                    h = float(h)
                    draw.text((minx-10, miny-10), str(tid), fill=(255, 255, 0))
                    draw.rectangle(((minx, miny), (minx+w, miny+h)), outline="yellow")

            im.save(save_path)
        #print (video, ': ', len(img_files))

if __name__ == '__main__':
    main()
