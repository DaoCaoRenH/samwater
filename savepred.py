import argparse
import os

import numpy as np
from PIL import Image
import cv2
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import datasets
import copy
import torch.nn.functional as F
from torchvision import transforms
from model import MobileSAM

def tensor2PIL(tensor):
    toPIL = transforms.ToPILImage()
    return toPIL(tensor)


def eval_psnr(model, img):
    model.eval()
    W, H = img.size
    init = copy.deepcopy(img)
    t = transforms.ToTensor()
    init = t(init).to("cuda")
    img_transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    img = img_transform(img).unsqueeze(0).to("cuda")
    inp = model.preprocess(img)
    start = time.time()
    with torch.no_grad():
        pred = torch.sigmoid(model.infer(inp))
        pred = F.interpolate(pred, (H, W), mode="bilinear", align_corners=False).squeeze(0)
    ipred = copy.deepcopy(pred)
    ipred[pred < 0.5] = 1
    ipred[pred >= 0.5] = 0
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0
    iinit = copy.deepcopy(init)
    ipred = ipred.squeeze(0)
    iinit[0][ipred == 0], iinit[1][ipred == 0], iinit[2][ipred == 0] \
        = 0, 0, 1
    init *= pred
    # print(init)
    end = time.time()
    print(end-start)
    output_mask = tensor2PIL(pred)
    output_img = tensor2PIL(init)
    output_iimg = tensor2PIL(iinit)
    # output_mask.save("pred/img1.png")
    # print(type(pred))
    return output_iimg, output_img, output_mask




if __name__ == '__main__':
    path = "../SimpleDay/img"
    model_path = "model0219.pth"
    save_path = "0224/"
    encoder_mode = {
        "name": "mobile_sam",
        "img_size": 1024,
        "prompt_embed_dim": 256,
        "patch_size": 16,
        "embed_dim": 768
    }
    model = MobileSAM(1024, encoder_mode)
    sam_checkpoint = torch.load(model_path, map_location='cuda:0')
    model.load_state_dict(sam_checkpoint, strict=True)
    model.to(device='cuda:0')
    # img_path = "img_test/test1.jpg"
    # img = Image.open(img_path)
    list_img_path = os.listdir(path)
    print(list_img_path)
    cnt = 0
    for img in list_img_path:
        cnt += 1

        img_path = os.path.join(path, img)
        img_open = Image.open(img_path)
        print(cnt, img_path)
        output_iimg, output_img, output_mask = eval_psnr(model, img_open)
        outpath = save_path + img
        output_iimg.save(outpath)
