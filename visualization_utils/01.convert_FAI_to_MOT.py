import os
import numpy as np
import shutil
import argparse


def main():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--dataset', choices=['basic', 'advanced'],  default=None, help='the dataset to be converted ')
    parser.add_argument('--save_dir', default='./data', help='the directory to save images with bboxes')
    parser.add_argument('--video_dir', default='./data', help='the directory having each directory of date')
    args = parser.parse_args()
    

    if args.dataset == 'basic' :
        videos = [f for f in os.listdir(args.video_dir) if not 'truth' in f] 
    else :
        videos = [f for f in os.listdir(args.video_dir) if not 'READ' in f]

    for video in videos :
        # Make save_directory for images and labels
        save_img_dir = os.path.join(args.save_dir, video, 'img1')
        save_label_dir = os.path.join(args.save_dir, video, 'gt')
        if not os.path.exists(save_img_dir) :
            os.makedirs(save_img_dir)
        if not os.path.exists(save_label_dir) :
            os.makedirs(save_label_dir)
        
        if args.dataset == 'basic' :
            img_files = [f for f in os.listdir(os.path.join(args.video_dir, video)) if 'png' in f]
        else :
            img_files = [f for f in os.listdir(os.path.join(args.video_dir, video, 'images')) if 'png' in f]
        
        img_indexes = []
        for img_file in img_files :
            # parsing titles
            if args.dataset == 'basic' :
                yymmdd, start, end, _, seqs = img_file.split('_')
            else :
                _, yymmdd, start_end, seqs = img_file.split('_')

            # For sorting, change string of number to integer
            img_indexes.append(int(seqs.split('.')[0]))
        
        # After sorting, insert '0's to make all numbers have same digits.
        img_indexes = np.sort(img_indexes)
        for i, img_index in enumerate(img_indexes) :
            # define the path of image and label files according to raw data format
            if args.dataset == 'basic' :
                img_path = os.path.join(args.video_dir, video, '{}_{}_{}_cropped_'.format(yymmdd, start, end) + str(img_index)+'.png')
                label_path = os.path.join(args.video_dir, video+'_truth', '{}_{}_{}_cropped_truth_'.format(yymmdd, start, end) + str(img_index)+'.txt')
                if not os.path.exists(label_path) :
                    label_path = os.path.join(args.video_dir, video+'_truth', '{}_{}_{}_cropped_'.format(yymmdd, start, end) + str(img_index)+'.txt')
                tracking_label_path = os.path.join(args.video_dir, video +'_truth_mot/gt.txt')
                is_tracking_label_path = os.path.exists(tracking_label_path)
                start_end = '{}-{}'.format(start, end)

            else :
                img_path = os.path.join(args.video_dir, video, 'images', 'FAI_{}_{}_'.format(yymmdd, start_end) + str(img_index)+'.png')
                label_path = os.path.join(args.video_dir, video, 'texts', 'FAI_{}_{}_'.format(yymmdd, start_end) + str(img_index)+'.txt')
                is_tracking_label_path = False

            # Define new paths
            new_img_path = os.path.join(save_img_dir, '{}_{}_'.format(yymmdd, start_end) +'{:06d}.png'.format(img_index))
            new_label_path = os.path.join(save_label_dir, '{}_{}_'.format(yymmdd, start_end) +'{:06d}.txt'.format(img_index))
            
            # Copy all files to new structures of directory
            shutil.copy(img_path, new_img_path)
            shutil.copy(label_path, new_label_path)
            if is_tracking_label_path :
                shutil.copy(tracking_label_path, os.path.join(save_label_dir, 'gt.txt'))

if __name__ == '__main__':
    main()
