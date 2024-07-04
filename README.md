# rso_tracking
This repo is for rso tracking but currently has only visulaization utils. Full codes will be updated after publication.

# 1. Data Preprocessing

### convert file structure to MoT Style

- Basic and Advanced dataset has differect path and naming structure.

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

- Besides, labelinig files in Basic dataset are two versions even in the same set.
    - `{YYMMDD}_{StartTime}_{EndTime}_cropped_{seq}.txt`
    - `{YYMMDD}_{StartTime}_{EndTime}_cropped_truth_{seq}.txt`


- Lastly, sequnce has different number of digits, so we can’t apply sort() function on that.
    - `{YYMMDD}_{StartTime}_{EndTime}_cropped_1.txt`
    - `{YYMMDD}_{StartTime}_{EndTime}_cropped_10.txt`
    - ...
    - `{YYMMDD}_{StartTime}_{EndTime}_cropped_100.txt`

- So, `convert_FAI_to_MOT.py` is the code to convert these dataset to MOT dataset format like the belows.

```python
# example of the script
$ python preprocessing/convert_FAI_to_MOT.py --dataset basic --video_dir {path/of/FAI_Basic} --save_dir ./data/basic_mot
$ python preprocessing/convert_FAI_to_MOT.py --dataset advanced --video_dir {path/of/FAI_Advanced} --save_dir ./data/advanced_mot
# output
-- <ROOT>
	|-- {Video_Root}
		|-- img1 #Paths_of_Images
		`-- gt # Paths_of_Labels
```


### make two seqs and convert labels to coco style
- We make new images with two consecutive images because detector is not trained on set of single frame.
    - We assume that difference of moving pattern between Star and RSO can be trained by deep learning model.
- And, we need to make annotatino format of label file to coco-style to train detector at `mmdetectino` tool.
- Traning and Testing Outputs will be saved under `./data/TWO_SEQs/' respectively.
    - we assume source data is outputs of `convert_FAI_to_MOT.py` so they are under `./data/basic_mot` and `./data/advanced_mot`
```python
# example of the script
$ python preprocessing/make_det_dataset.py --option add_curr
# output





# 2. Training detector
- We used `mmdetection` tool to train detector
    - https://github.com/open-mmlab/mmdetection
- After install mmdetection, use `config/FAI/yolox_nano_2seqs.py
- When training the model, we use pretrained weights which is in `yolox-nano`

```python
# example of the script in root of mmdetection
$ cd {path/of/mmdetection}
$ python tools/train.py  {path/to/rso_tracking}/config/FAI/yolox_nano_2seqs.py
```

# 3. Detection with Tracking
- This code will detects and track RSOs with traiend detector and keypoint(Center) based Tracker
    - To detect RSOs, we use `mmdetection` API

```python
# example of the script
$ cd {path/to/rso_tracking}
$ python detectint_with_tracking.py  --config config/FAI/yolox_nano_2seqs.py --model {path/to/mmdetection}/work_dirs/FAI_yolox_nano_2seqs/epoch_300.pth --data ./data/TWO_SEQs  --save_dir ./preds/yolox_nano
```


# 4. Evaluation
- This code will evaluate detecting and tracking performance
```python
# example of the script
$ python evaluate.py  --gt ./data/??? --preds_dir ./preds/yolox_nano
```

# 5. Visualization utils
### 01.draw_gts_on_image.py 

- This code will draw bounding boxes and tracking IDs of ground truth on coressponding image and save it to `save_dir`
    - It assumes that the path structure of `video_dir` follows the `MOT` dataset format

```python
# example of the script
$ python 01.draw_gts_on_image.py  --video_dir ./data/basic_mot --save_dir ./visualization/basic_gt
```

### 02.draw_preds_on_image.py 

- This code will draw bounding boxes and tracking IDs of predictions by ours on coressponding image and save it to `save_dir`
    - It assumes that the path structure of `video_dir` follows the `MOT` dataset format.

```python
# example of the script
$ python 02.draw_preds_on_image.py  --video_dir ./data/DET_COCO_STYLE_TWO_SEQs/test --preds_dir ./output/yolox_nano --save_dir ./visualization/basic_preds
```

### 03.merge_gt_preds.py

- This code will merge `gt` output and `prediction` output into one image where left side will be on GT, and right side on prediction.

```python
# example of the script
$ python 03.merge_gt_preds.py --gt ./visualization/basic_gt --pred ./visualization/basic_preds --save_dir ./visualization/basic_merge
```

### 04.make_mp4.py

- This code will make movie file with any given images

```python
# example of the script
$ python 04.make_mp4.py --source ./visualization/basic_merge --fps 5 --save_dir ./visualization/basic_fn_mov
```
