from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image
import torch
import cv2
import random
from .invert import Invert
from shapely.geometry import Polygon
import pyclipper
import torchvision.transforms.functional as TF
import torchvision.transforms as transformation
import imgaug.augmenters as iaa

import os
import numpy as np
import torchvision.transforms.functional as F

#if you want to pad instead of resize
def pad(image):
	w, h = image.size
	max_wh = np.max([w, h])
	hp = int((max_wh - w) / 2)
	vp = int((max_wh - h) / 2)
	padding = (hp, vp, hp, vp)
	return F.pad(image, padding, 0, 'constant')

def _shrink_box(lines):
    new_lines = []
    for line in lines:
        pco = pyclipper.PyclipperOffset()
        pco.AddPath(line, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        if len(line) == 2:
            line = [[line[0,0],line[0,1]],
                    [line[1,0],line[0,1]],
                    [line[1,0],line[1,1]],
                    [line[0,0],line[1,1]]]
        polygon_shape = Polygon(line)
        distance = polygon_shape.area * \
                (1 - np.power(0.3, 2)) / polygon_shape.length
        new_lines.extend(pco.Execute(-distance/2))
    
    new_lines = [np.array(line) for line in new_lines]
    return new_lines

class CoreDataset(Dataset):
    def __init__(self, data_path, transforms=None, mode='train'):
        super().__init__()
        self.data_path = data_path
        self.mode = mode
        self.image_path = os.path.join(data_path, f'{mode}_images')
        self.gt_path = os.path.join(data_path, f'{mode}_gts')
        self.images = os.listdir(self.image_path)
        self.images.sort()
        self.gts    = os.listdir(self.gt_path)
        self.gts.sort()
        self.transforms = transforms
    def _preprocess(self):
        pass
    def _process_gt(self):
        pass
    def __getitem__(self):
        pass
    def __len__(self):
        return len(self.images)

class ImageDataset(CoreDataset):
    def _preprocess(self, pil_img, mask):
        mask    = transformation.ToPILImage(mode='F')(mask[:,:,0])
        
        if self.mode == 'train':
            if random.uniform(0,1) < 0.3:
                affine_params = transformation.RandomAffine.get_params((-5, 5), None, None,None,pil_img.size)
                pil_img, mask = TF.affine(pil_img, *affine_params), TF.affine(mask, *affine_params)
            
            if random.uniform(0,1) < 0.7:
                i, j, h, w = transformation.RandomResizedCrop.get_params(
                    pil_img, scale=(0.1,1.0), ratio=(0.1,4.0))
                pil_img = TF.crop(pil_img, i, j, h, w)
                mask = TF.crop(mask, i, j, h, w)
            
            # Resize
            resize = transformation.Resize(size=(768, 768))
            pil_img = resize(pil_img)
            mask = resize(mask)
            
            if random.uniform(0,1) < 0.5:
                pil_img = transformation.ColorJitter(brightness=[0.5,1.5], contrast=[0.2,2.0], saturation=[0,2.5], hue=[-0.5,0.5])(pil_img)
            if random.uniform(0,1) < 0.1:
                pil_img = Invert()(pil_img)
            if random.uniform(0,1) < 0.3:
                aug = iaa.MotionBlur(k=random.randint(5,15))
                pil_img = aug(image=np.array(pil_img))
        img_nd = np.array(pil_img, dtype='float')
        mask_nd = np.array(mask, dtype='float')
        mask_nd = np.expand_dims(mask_nd, axis=2)
        
        if len(img_nd.shape) == 2:
            img_nd = img_nd.unsquueze(axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        mask_trans = mask_nd.transpose((2,0,1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255
        if mask_trans.max()> 1:
            mask_trans = mask_trans / 255

        return img_trans.astype('float32'), mask_trans.astype('float32')

    def _process_gt(self, gt, height, width):
        with open(os.path.join(self.gt_path, gt), 'r') as fr:
            lines = fr.readlines()
        lines       = [np.array([int(num) for num in line.split(',')[:-1]]) for line in lines]
        lines       = [line.reshape(-1,2) for line in lines]
        new_lines   = _shrink_box(lines)
        res = np.zeros((height, width, 1))
        cv2.fillPoly(res, new_lines, 255)
        return torch.Tensor(res/255)
    
    def __getitem__(self,index):
        image       = Image.open(os.path.join(self.image_path,self.images[index]))
        gt          = self.gts[index]
        w, h        = image.size
        gt          = self._process_gt(gt, h, w)
        image, gt   = self._preprocess(image, gt)
        return image, gt

