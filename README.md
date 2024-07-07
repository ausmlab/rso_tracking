# rso_tracking
This repository contains the official code for the paper, ""

# 1. Data Preprocessing

### 1.1. convert file structure to MoT Style

- The Basic and Advanced datasets have differect path and naming structures.

```python
# BASIC
-- <ROOT>
	|-- Images 
	    |-- {YYDDMM}_{StartTime}_{EndTime}_cropped #Paths_of_Images
	    |-- {YYDDMM}_{StartTime}_{EndTime}_cropped_truth # Paths_of_Labels
	    
# ADVANCED
-- <ROOT>
	|-- 02_FAI_{YY-MM-DD}_{StartTime}-{EndTime}
		|-- images #Paths_of_Images
		`-- texts # Paths_of_Labels
```

- Additionally, the labeling files in the Basic dataset have two versions, even within the same dataset.
    - `{YYMMDD}_{StartTime}_{EndTime}_cropped_{seq}.txt`
    - `{YYMMDD}_{StartTime}_{EndTime}_cropped_truth_{seq}.txt`


- Lastly, the sequences have different numbers of digits, preventing the use of the `sort()` function on them.
    - `{YYMMDD}_{StartTime}_{EndTime}_cropped_1.txt`
    - `{YYMMDD}_{StartTime}_{EndTime}_cropped_10.txt`
    - ...
    - `{YYMMDD}_{StartTime}_{EndTime}_cropped_100.txt`

- `convert_FAI_to_MOT.py` is the script that converts the structure of these datasets to MOT dataset structure, as shown below.
    - Please note that the bounding box format of the ground truth (gt) still follows the YOLO style

```python
# example of the script
$ python preprocessing/convert_FAI_to_MOT.py --dataset basic \
	--video_dir {path/of/FAI_DATA}/FAI_Basic_Dataset_MOT/Images
$ python preprocessing/convert_FAI_to_MOT.py --dataset advanced \
	--video_dir {path/of/FAI_DATA}/FAI_Advanced_Dataset

# output
-- data/{basic/advanced}_mot
	|-- {Video_Root}
		|-- img1 #Paths_of_Images
		`-- gt # Paths_of_Labels
```


### 1.2. create two seqs images and convert labels to coco style
- We create new images using two consecutive frames because the detector is not trained on single-frame sets.
    - We assume that difference of moving patterns between Stars and RSOs can be trained by deep learning model.
- And, we need to convert annotation format of the label file to coco-style to train the detector using `mmdetection` tool.
- Traning and Testing Outputs will be saved under `./data/TWO_SEQs/` respectively.
    - we assume the source data are outputs of `convert_FAI_to_MOT.py`, so they are located under `./data/basic_mot` and `./data/advanced_mot`
```python
# example of the script for detection
$ python preprocessing/make_det_2seqs.py --option add_curr

# output
## for detection
-- data/DET_COCO_STYLE_TWOs/{option}
        |-- train
        |       |-- {train_image_file1.png}
        |       |-- ...
        |       `-- annotation_coco.json
        `-- test
                |-- {test_image_file1.png}
                |-- ...
                `-- annotation_coco.json

## for tracking
-- data/Tracking_GT_TWO_SEQs
        |-- {Video_Root}
                |-- gt.txt

```


# 2. Training detector
- We used `mmdetection` tool to train the detector
    - https://github.com/open-mmlab/mmdetection
- After installing `mmdetection`, use `config/yolox_nano_2seqs.py` to train the model
- When training the model, we'll use weights pretrained by authors of yolox, located in `pretrained_weights`

```python
# example of the script at root of mmdetection
$ cd {path/of/mmdetection}
$ python tools/train.py  {path/to/rso_tracking}/config/yolox_nano_2seqs.py
```

# 3. Detection with Tracking
- This code will detects and track RSOs using a traiend detector and a keypoint(Center) based Tracker
    - To detect RSOs, we use `mmdetection` API
    - This code works on CPU, not GPU.
```python
# example of the script at root of rso_tracking
$ cd {path/to/rso_tracking}
$ python detection_with_tracking.py --score_th 0.3 --config config/yolox_nano_2seqs.py --model {path/to/mmdetection}/work_dirs/FAI_yolox_nano_2seqs/epoch_300.pth --data ./data/DET_COCO_STYLE_TWOs/ADDCURR/test --save_dir ./preds/yolox_nano
```

# 4. Evaluation
- This code will evaluate the performance of detecting and tracking.
- We used `motmetrics` library to measure tracking performance
    - To install it, please refer to https://github.com/cheind/py-motmetrics
```python
# example of the script
$ python evaluate.py  --gt ./data/Tracking_GT_TWO_SEQs --pred ./preds/yolox_nano
```

# 5. Visualization utils
### 01.draw_bboxes.py 

- This code will draw bounding boxes and tracking IDs on coressponding image and save it to `save_dir`

```python
# example of the script
$ python vis_utils/01.draw_bboxes.py --image_dir ./data/DET_COCO_STYLE_TWOs/ADDCURR/test --save_dir ./vis_output/gt --bbox_dir ./data/Tracking_GT_TWO_SEQs --output gt
$ python vis_utils/01.draw_bboxes.py --image_dir ./data/DET_COCO_STYLE_TWOs/ADDCURR/test --save_dir ./vis_output/pred --bbox_dir ./preds/yolox_nano --output pred
```

### 02.merge_gt_preds.py

- This code will merge `gt` output and `prediction` output into one image where left side will be on GT, and right side on prediction.

```python
# example of the script
$ python vis_utils/02.merge_gt_preds.py --gt ./vis_output/gt --pred ./vis_output/pred --save_dir ./vis_output/merge
```

### 03.make_mp4.py

- This code will make movie file with any given images

```python
# example of the script
$ python vis_utils/03.make_mp4.py --source ./vis_output/merge --fps 10 --out ./vis_output/outputs.mp4
```
