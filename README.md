# LSM: Line Segmentation Model for documented text with multi orientations

## Quick Start

One image demo:
```
python demo.py -t single -i <path-to-image> -o <output-directory> -d <device (Default=cpu)> 
```

Whole directory demo:
```
python demo.py -t multi -i <image-directory> -o <output-directory> -d <device (Default=cpu)> 
```

## How to train a new model
### 1. Prepare data
The sample dataset organization is set up in *./data*. You can organize your own dataset in similar format as those in such directory.

**Note** Groundtruth data are in coordinate of points in polygons separated by comma (take a look at a sample annotation: *data/train_gts/0.txt*). You can inherit the ImageDataset in *datasets/image_dataset.py* and overwrite *_process_gt* function to process your own annotation format.

### 2. Train

```
python train.py
```

Output will be saved at *output/\*.pth*

### 3. Visualize training process
```
tensorboard --logdir tensorboard_output
```
## Sample image

![test_sample](out/output.jpg)
