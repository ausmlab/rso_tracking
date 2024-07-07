import os
import numpy as np
from PIL import Image, ImageDraw
import argparse


def main():
    parser = argparse.ArgumentParser(description='Draw Bounding Boxes and Trakcing ID')
    parser.add_argument('--save_dir', default='./vis_output/gt', help='the dir to save output images with bboxes')
    parser.add_argument('--image_dir', default='./data/DET_COCO_STYLE_TWOs/ADDCURR/test/', help='the dir to have sequences of images for drawing bboxes')
    parser.add_argument('--bbox_dir', default='./data/Tracking_GT_TWO_SEQs/', help='the dir having gts or preds of each videos') 
    parser.add_argument('--output', default='gt', choices={'gt', 'pred'}, help='gt or pred')
    args = parser.parse_args()
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if args.output == 'gt' :
        output_name = 'gt.txt'
    else :
        output_name = 'pred.txt'

    img_files =  [f for f in os.listdir(args.image_dir) if 'png' in f]

    video_frames = {}
    videos = ['20230116_212414-212846', '20230121_224714-225046', '20230125_214214-214646', '20230331_162014-162346',
          '20230531_105614-105946', '20230603_000914-001346', '20230620_005047-005419', '20230719_033314-033646',
          '20230804_210514-210946a', '20230804_210514-210946b','20230805_020314-020846', '20230805_233614-234046']
    for video in videos :
        video_frames[video] = []

    for img_file in img_files :
        yymmdd, times, frame = img_file.split('_')
        if yymmdd == '20230804' :
            filenum = int(frame.split('.')[0])
            if filenum < 53 :
                video_frames['20230804_210514-210946a'].append(img_file)
            else :
                video_frames['20230804_210514-210946b'].append(img_file)

        else :
            video_frames['{}_{}'.format(yymmdd, times)].append(img_file)

    for k in video_frames.keys() :
        video = k
        img_files = video_frames[k]

        for i, img_file in enumerate(img_files) :
            # Open img
            img_path = os.path.join(args.image_dir, img_file)
            save_path = os.path.join(args.save_dir, img_file)

            im = Image.open(img_path)
            draw = ImageDraw.Draw(im)
            draw.text((0, 0), video, fill=(255, 255, 255))

            # Open output labels
            with open(os.path.join(args.bbox_dir, video, output_name), 'r') as f:
                lines = f.readlines()

            for line in lines :
                if args.output == 'gt' :
                    frame, tid, minx, miny, w, h, _, _, _ = line.strip().split(',')
                else :
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
