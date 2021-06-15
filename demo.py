from models import UNet
import os
import torch
import numpy as np
from PIL import Image
import time
import cv2
import pyclipper
from shapely.geometry import Polygon
import argparse
import glob

parser = argparse.ArgumentParser(description='Argument Parser')
parser.add_argument('-s', '--short-side', default=768,
                    type=int, help='size of the shorter side of the image')
parser.add_argument('-m', '--model-weight', default='output/023001.pth',
                    type=str, help='path to the file where model weight is saved')
parser.add_argument('-d', '--device', default='cpu',
                    choices=['cuda', 'cpu'], type=str, help='which device the model is running on')
parser.add_argument('-t', '--demo-type',default='single',choices=['single','multi'],
                    type=str, help='mode of demo, whether on single image or multi images')
parser.add_argument('-r', '--rotation-angle',default=0, help='angle to rotate the image',type=int)
parser.add_argument('-i', '--input-img',type=str,help='path to the input image or the input folder')
parser.add_argument('-o', '--output', default='out', type=str, help='path to where we are gonna save the image')

args = parser.parse_args()

def resize_short_side(img, short_side=768):
    w, h = img.size
    if min(h, w) < short_side:
        short_side = min(h, w)
    if h > w:
        new_w = short_side
        new_h = int(h/w*new_w)
    else:
        new_h = short_side
        new_w = int(w/h*new_h)
    # print(new_w, new_h)
    img = img.resize((new_w, new_h))
    return img

def shrink_box(lines):
    new_lines = []
    for line in lines:
        pco = pyclipper.PyclipperOffset()
        pco.AddPath(line, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        if len(line) < 2:
            continue
        if len(line) == 2:
            line = [[line[0, 0], line[0, 1]], [line[1, 0], line[0, 1]],
                    [line[1, 0], line[1, 1]], [line[0, 0], line[1, 1]]]
        polygon_shape = Polygon(line)
        distance = polygon_shape.area * \
            (1 - np.power(0.3, 2)) / polygon_shape.length
        new_lines.extend(np.array(pco.Execute(distance*2)))
    return new_lines


def inference(model, img_name):
    img = Image.open(img_name)
    img   = img.rotate(args.rotation_angle, expand=True)
    img = resize_short_side(img, short_side=args.short_side)
    img_ = np.array(img).transpose(2, 0, 1)[:3, :, :]
    img_ = torch.Tensor(img_/255).unsqueeze(0)
    t = time.time()
    pred = model(img_.to(args.device)).cpu().detach().numpy()

    def sigmoid(x):
        return np.exp(x)/(1+np.exp(x))

    pred = np.exp(pred)[0, :, :, :].transpose(1, 2, 0)
    pred = np.where(pred > 0.6, 1, 0)
    pred = pred*255
    img = img_.squeeze(0).numpy().transpose(1, 2, 0)*255
    if True:
        pred = pred.astype('uint8')
        contours, _ = cv2.findContours(
            pred, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c.reshape(-1, 2) for c in contours]
        contours = shrink_box(contours)
        contours = [np.array(c).reshape(-1, 1, 2) for c in contours]
        img = cv2.drawContours(cv2.UMat(img), contours, -1, (0, 255, 0), 2)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if False:
        img = cv2.hconcat(
            [img[:, :, ::-1], np.repeat(pred, repeats=3, axis=-1)])
    im_name = img_name.split('/')[-1]
    if args.demo_type=='single':
        im_name = 'output.jpg'
    os.makedirs(args.output, exist_ok=True)
    cv2.imwrite(os.path.join(args.output,im_name), img)
    # img.save('test/input.png')

    print(time.time()-t)



if __name__ == '__main__':
    model = UNet(3, 1).to(args.device)
    model.load_state_dict(torch.load(args.model_weight, map_location=args.device))
    model.eval()
    if args.demo_type == 'single':
        inference(model, args.input_img)
    elif args.demo_type == 'multi':
        for img_name in glob.glob(args.input_img+'*.jpg'):
            inference(model, img_name)