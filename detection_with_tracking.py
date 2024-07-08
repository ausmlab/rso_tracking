import argparse
import os
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from scipy.optimize import linear_sum_assignment
from mmdet.apis import init_detector, inference_detector
from BDTracker import BDTracker


def convert_xyxy_to_cxcywh (bboxes) :
    '''
    args:
        bboxes: (N,4) of numpy array. Order of bbox is minx, miny, maxx, maxy
    return:
        bboxes: (N,4) of numpy array. Order of bbox is center_x, center_y, width, height
    '''
    center_x = np.expand_dims(bboxes[:,0::2].mean(1), axis=1)
    center_y = np.expand_dims(bboxes[:,1::2].mean(1), axis=1)
    return np.hstack((center_x, center_y, bboxes[:, 2:4] - bboxes[:, 0:2]))

def get_indexes_for_valid_objs(objs, img_id) :
    obj_vel_indexes = []
    obj_novel_indexes = []
    for n, obj in enumerate(objs) :
        if obj.status == 'in' :
            if obj.velocity != None :
                obj_vel_indexes.append(n)
            elif obj.history_id[-1] + 1 == img_id :
                obj_novel_indexes.append(n)
    return  obj_vel_indexes,  obj_novel_indexes

def register_rso(objs, idxes, img_id, img_file, bboxes, scores) :
    for idx in idxes :
        new_obj = BDTracker(img_id, img_file, bboxes[idx], scores[idx])
        objs.append(new_obj)
    return objs


def update_rso(obj, idx, bboxes, scores, img_id, img_file, obj_idx, tag) :
    obj.update(img_id, img_file, bboxes[idx], scores[idx])  # with new detection results
    # remove the selected one from scores and bboxes arrays

    if tag == 'TbD' :
        return
    else :
        bboxes = np.delete(bboxes, idx, 0)
        scores = np.delete(scores, idx, 0)
        return bboxes, scores



def main():
    parser = argparse.ArgumentParser(description='Detect and Associate RSOs')
    parser.add_argument('--score_th', default=0.4, type=float, help='score_th for detector')
    parser.add_argument('--config', default='./config/yolox_nano_2seqs.py', help='config for mmdetection')
    parser.add_argument('--model', default='{path/to/mmdetection}/work_dirs/FAI_yolox_nano_2seqs/epoch_300.pth', help='trained detector model weights')
    parser.add_argument('--data', default='./data/DET_COCO_STYLE_TWOs/ADDCURR/test', help='the directory having test images')
    parser.add_argument('--save_dir', default='./preds/yolox_nano', help='the directory to save images with bboxes')
    
    args = parser.parse_args()
    
    model = init_detector(args.config, args.model, device='cpu')
    score_th = args.score_th
    assumed_error_th = 5 # avg. half of max(height) and max(w) ## max(w) = 14, max(h) = 6 # error of being inside bbox

    OUTPUT_ROOT = args.save_dir
    if not os.path.exists(OUTPUT_ROOT) :
        os.makedirs(OUTPUT_ROOT)

    ROOT = args.data
    img_files =  [f for f in os.listdir(ROOT) if 'png' in f]

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
        num_imgs = 0
        img_id = 0
        objs = []
        BDTracker.count = 0
        for img_file in img_files :
            num_imgs += 1

            # Start Detection
            img_path = os.path.join(ROOT, img_file)
            preds = inference_detector(model, img_path)
            scores = preds.pred_instances.scores.cpu().numpy()
            bboxes = preds.pred_instances.bboxes.cpu().numpy() # x1,y1,x2,y2
            bboxes = convert_xyxy_to_cxcywh(bboxes) # cx, cy, w, h

            # Start Association & Detection
            ## For first frame.
            if len(objs) == 0 : # Select all objects over score threshold 
                obj_idxes = np.where(scores > score_th)[0]
                if obj_idxes.shape[0] > 0 :
                    objs = register_rso(objs, obj_idxes, img_id, img_file, bboxes, scores)
            
            ## From second frame                           
            else :
                obj_vel_idxes, obj_novel_idxes = get_indexes_for_valid_objs(objs, img_id)
                
                # if obj has velocity
                for obj_vel_idx in obj_vel_idxes :
                    # Velocity based detection and tracking regardless of score threshold
                    obj = objs[obj_vel_idx]
                    assumed_center = obj.forecast(img_id)
                    if obj.history_id[-1]+1 == img_id :
                        # calc distance between assumed location and center of bboxes
                        dist_errors = pairwise_distances(assumed_center, bboxes[:,:2], metric='euclidean')[0]
                        low_dist_idxes = np.where(dist_errors < assumed_error_th)[0]
                        if len(low_dist_idxes) !=  0 :
                            max_idx = np.min(low_dist_idxes) # get index of highest score
                            bboxes, scores = update_rso(obj, max_idx, bboxes, scores, img_id, img_file, obj_vel_idx, 'Vel')
                            
                    else :
                        # Velocity based ReID over Scored based detections
                        score_idxes = np.where(scores > score_th)[0]
                        # calc distance between center of object and center of bboxes
                        if score_idxes.shape[0] != 0 :
                            dists = pairwise_distances(assumed_center, bboxes[:, :2][score_idxes], metric='euclidean') # 1,N
                            min_dist = np.min(dists, axis=1)
                            if min_dist < assumed_error_th :
                                min_idx = score_idxes[np.argmin(dists, axis=1)][0]
                                bboxes, scores = update_rso(obj, min_idx, bboxes, scores, img_id, img_file, obj_vel_idx, 'VS&D')
                
                # if obj doesn't have velocity, tracking-by-detection(TbD)
                ## objs for pairs
                objs_centers = []
                for n, obj_novel_idx in enumerate(obj_novel_idxes) :
                    obj = objs[obj_novel_idx]
                    obj_center = obj.history_objservation[-1][:2]
                    objs_centers.append(obj_center)
                
                # preds for pairs
                hi_score_idxes = np.where(scores > score_th)[0]
                if hi_score_idxes.shape[0] > 0 :
                    if len(objs_centers) == 0 :
                        # register all new objs
                        objs = register_rso(objs, hi_score_idxes, img_id, img_file, bboxes, scores)
    
                    else :
                        preds_centers = bboxes[:, :2][hi_score_idxes]
                        dist = pairwise_distances(objs_centers, preds_centers, metric='euclidean')
                        obj_idxes, pred_idxes = linear_sum_assignment(dist) # Hungarian Algorithm
                        new_objs_idx = hi_score_idxes.tolist()
                        for obj_idx, pred_idx in zip(obj_idxes, pred_idxes) :
                            obj_idx = obj_novel_idxes[obj_idx]
                            pred_idx = hi_score_idxes[pred_idx]
                            update_rso(objs[obj_idx], pred_idx, bboxes, scores, img_id, img_file, obj_idx, 'TbD')
                            new_objs_idx.remove(pred_idx)
                                
                        # Reisger all new RSOs
                        if len(new_objs_idx) > 0 :
                            objs = register_rso(objs, new_objs_idx, img_id, img_file, bboxes, scores)
    
            img_id += 1
                
        print ('{}: {} objects are tracked'.format(video, len(objs)))

        out_path = os.path.join(OUTPUT_ROOT, video) 
        if not os.path.exists(out_path) :
            os.makedirs(out_path)
        with open(os.path.join(out_path, 'pred.txt'), 'w') as f :
            for n, obj in enumerate(objs) :
                for i in range(len(obj.history_id)) :
                    img_id = obj.history_id[i] + 1
                    obj_id = n+1
                    cx, cy, width, height = obj.history_objservation[i]
                    minx = cx - width/2
                    miny = cy - height/2
                    conf = obj.history_score[i] * 100.
                    f.write('{},{},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},-1,-1,-1\n'.format(img_id, obj_id, minx, miny, width, height, conf))


if __name__ == '__main__':
    main()
