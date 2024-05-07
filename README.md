# rso_tracking
This repo is for rso tracking but currently has only visulaization utils. Full codes will be updated after publication.

# Visualization utils

### 01.convert_FAI_to_MOT.py

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
    - {YYMMDD}_{StartTime}_{EndTime}_cropped_{seq}.txt
    - {YYMMDD}_{StartTime}_{EndTime}_cropped_{seq}.txt

```python
- 20230116_212414_212846_cropped_169.txt
- 20230805_233614_234046_cropped_truth_1.txt
```

- Lastly, sequnce has different number of digits, so we can’t apply sort() function on that.
    - {YYMMDD}_{StartTime}_{EndTime}_cropped_1.txt, …, {YYMMDD}_{StartTime}_{EndTime}_cropped_10.txt, …, {YYMMDD}_{StartTime}_{EndTime}_cropped_100.txt

- So, `01.convert_FAI_to_MOT.py` is the code to convert these dataset to MOT dataset format like the belows.

```python
# example of the script
$ python 01.convert_FAI_to_MOT.py --dataset basic --video_dir ./data/basic --save_dir ./data/basic_mot
# output
-- <ROOT>
	|-- {Video_Root}
			   |-- img1 #Paths_of_Images
			   `-- gt # Paths_of_Labels
```

### 02.make_images_on_gt.py

- This code will draw bounding boxes and tracking IDs of ground truth on coressponding image and save it to `save_dir`
    - It assumes that the path structure of `video_dir` follows the `MOT` dataset format

```python
# example of the script
$ python 02.make_images_on_gt.py --video_dir ./data/basic_mot --save_dir ./visualization/basic_gt
```

### 03.make_images_on_preds.py

- This code will draw bounding boxes and tracking IDs of predictions by CenterTrack on coressponding image and save it to `save_dir`
    - It assumes that the path structure of `video_dir` follows the `MOT` dataset format and `preds_dir` has outputs of CenterTrack.

```python
# example of the script
$ python 03.make_images_on_preds.py --video_dir ./data/basic_mot --preds_dir ./exp/tracking/basic/result_rso --save_dir ./visualization/basic_preds
```

### 04.merge_gt_preds.py

- This code will merge `gt` output and `prediction` output into one image where left side will be on GT, and right side on prediction.

```python
# example of the script
$ python 04.merge_gt_preds.py --gt ./visualization/basic_gt --pred ./visualization/basic_preds --save_dir ./visualization/basic_merge
```

### 05.make_mp4.py

- This code will make movie file with any given images

```python
# example of the script
$ python 05.make_mp4.py --source ./visualization/basic_merge --fps 5 --save_dir ./visualization/basic_fn_ontput
```
