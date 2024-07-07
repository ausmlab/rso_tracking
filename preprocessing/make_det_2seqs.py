import os
import numpy as np
from PIL import Image, ImageDraw
import argparse
import cv2
import shutil
import json

def merge_twos(img_path1, img_path2, option) :
    fai_1 = cv2.imread(img_path1)
    fai_2 = cv2.imread(img_path2)

    fai = np.zeros((256,256,3), np.uint8)
    fai[:,:,0] = fai_1[:,:,0]
    fai[:,:,1] = fai_2[:,:,0]

    if option == 'add_0l' :
        fai[:,:,2] = np.zeros_like(fai_1[:,:,0], np.uint8)
        merged_img = Image.fromarray(fai)
    elif option == 'add_prev' :
        fai[:,:,2] = fai_1[:,:,0]
        merged_img = Image.fromarray(fai)
    elif option == 'add_curr' :
        fai[:,:,2] = fai_2[:,:,0]
        merged_img = Image.fromarray(fai)
    else :
        fai[:,:,2] = fai_2[:,:,0]

        hsv_fai = cv2.cvtColor(fai, cv2.COLOR_RGB2HSV)

        # for previous frame
        light_red = (0, 100, 100) # 0, 100, 100
        dark_red = (20, 255, 255) # 25, 255, 255
        r_mask = cv2.inRange(hsv_fai, light_red, dark_red)

        # for current frame
        light_yellow = (70, 100, 100)
        dark_yellow = (90, 255, 255)
        y_mask = cv2.inRange(hsv_fai, light_yellow, dark_yellow)

        if option == 'add_mask1' :
            final_mask = y_mask
        elif option == 'add_mask2' :
            final_mask = r_mask + y_mask

        # dilation
        kernel = np.ones((3,3), np.uint8)
        dilation = cv2.dilate(final_mask, kernel, iterations=1)

        fai[:,:,2] = dilation
        merged_img = Image.fromarray(fai)

    return merged_img

def yolo_to_coco(bbox, size=256) :
    cx, cy, w, h = [int(float(i)*size) for i in bbox]
    coco_bbox = [cx - w//2, cy - h//2, w, h]
    return coco_bbox

def make_tracking_gt_for_two(gt_path, video) :
    with open (os.path.join(gt_path, 'gt.txt'), 'r') as f :
        lines = f.readlines()

    yymmdd, start, end, _ = video.split('_')
    dest = os.path.join('./data/Tracking_GT_TWO_SEQs/', '{}_{}-{}'.format(yymmdd, start, end))
    if not os.path.exists(dest) :
        os.makedirs(dest)
    with open (os.path.join(dest, 'gt.txt'), 'w') as f :
        for line in lines :
            frame, obj_id, minx, miny, w, h, _, _, _ = line.split(',')
            frame = int(frame)
            # since we merged two consecutive images to one, we removed first frame on gt
            if frame != 1 :
                f.write('{},{},{},{},{},{},1,1,1.0\n'.format(frame-1, obj_id, minx, miny, w, h))

def main():
    parser = argparse.ArgumentParser(description='Creating two consecutive images for detecting and tracking')
    parser.add_argument('--option', choices=['add_0l', 'add_prev', 'add_curr', 'add_mask1', 'add_mask2'],  default='add_curr', help='')
    args = parser.parse_args()
    
    # 'curr_only1', 'curr_only3', 'add_0l', 'add_prev', 'add_curr', 'nomask', 'masked2ch', 'add_mask'
    if args.option == 'add_0l' :
        option = 'ADD0L'
    elif args.option == 'add_prev' :
        option = 'ADDPREV'
    elif args.option == 'add_curr' :
        option = 'ADDCURR'
    elif args.option == 'add_mask1' :
        option = 'ADDMASK1'
    else :
        option = 'ADDMASK2'

    ROOT = './data'
    save_dir = os.path.join(ROOT, 'DET_COCO_STYLE_TWOs', option)

    for split in ['train', 'test'] :
        if split == 'train' :
            video_source_path = os.path.join(ROOT, 'advanced_mot')
            DEST = os.path.join(save_dir, 'train')
            RSO = '1'
        else :
            video_source_path = os.path.join(ROOT, 'basic_mot')
            DEST = os.path.join(save_dir, 'test')
            RSO = '0'

        if not os.path.exists(DEST):
            os.makedirs(DEST)
        
        outfile = os.path.join(DEST,'annotation_coco.json')

        videos = [f for f in os.listdir(video_source_path) if not 'vis' in f]
        videos = np.sort(videos)
        #print (split, videos)
        
        annotations = []
        images = []
        obj_count = -1
        idx = -1

        for video in videos :
            img_source_path = os.path.join(video_source_path, video, 'img1')
            gt_path = os.path.join(video_source_path, video, 'gt')
            if split == 'test' and os.path.exists(os.path.join(gt_path, 'gt.txt')) :
                make_tracking_gt_for_two(gt_path, video)

            img_files = [f for f in os.listdir(img_source_path) if 'png' in f]
            img_files = np.sort(img_files)
            for i, img_file in enumerate(img_files[1:]) :
                idx += 1
                # Open img
                img_path1 = os.path.join(img_source_path, img_files[i]) # previous frame
                img_path2 = os.path.join(img_source_path, img_file) # current frame
                # Merging                
                merged_img = merge_twos(img_path1, img_path2, args.option)
                merged_img.save(os.path.join(DEST, img_file))

                images.append(dict(
                    id=idx,
                    file_name=img_file,
                    height=256,
                    width=256))

                # Labels info
                gt_file = os.path.join(gt_path, img_file.replace('png', 'txt'))
                with open (gt_file, 'r' ) as f :
                    lines = f.readlines()
                    for line in lines :
                        cls, x, y, w, h = line.strip().split(' ')
                        if cls == RSO :
                            obj_count += 1
                            bbox = yolo_to_coco([x,y,w,h])
                            data_anno = dict(
                                            image_id=idx,
                                            id=obj_count,
                                            category_id = 0, # RSO
                                            bbox=bbox,
                                            area= bbox[2] * bbox[3],
                                            iscrowd=0)
                            annotations.append(data_anno)

        coco_format_json = dict(
                images=images,
                annotations=annotations,
                categories=[{'id': 0, 'name': 'RSO'}])

        with open(outfile, 'w') as f:
            json.dump(coco_format_json, f, indent=4)
        
if __name__ == '__main__':
    main()
