import os
import numpy as np
import cv2
import argparse


def main():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--source', default='./videos/merge', help='the dir of images to be a movie')
    parser.add_argument('--fps', type=int, default=10, help='frame per second')
    parser.add_argument('--out', default='./Visualization/video.mp4', help='movie file name with path')
    args = parser.parse_args()

    ROOT = args.source
    srcs = [f for f in os.listdir(ROOT) if 'png' in f]
    srcs = np.sort(srcs)

    # Each video has a frame per second which is number of frames in every second
    frame_per_second = args.fps
    w, h = None, None
    for src in srcs:
        img_file = os.path.join(ROOT, src)
        frame = cv2.imread(img_file)
        
        if w is None:
            # Setting up the video writer
            h, w, _ = frame.shape
            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            writer = cv2.VideoWriter(args.out, fourcc, frame_per_second, (w, h))

        writer.write(frame)
    writer.release()    

if __name__ == '__main__':
    main()
