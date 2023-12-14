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
    iinit *= ipred
    init *= pred
    end = time.time()
    print(end-start)
    output_mask = tensor2PIL(pred)
    output_img = tensor2PIL(init)
    output_iimg = tensor2PIL(iinit)
    # output_mask.save("pred/img1.png")
    # print(type(pred))
    return output_iimg, output_img, output_mask


def run(img):
    model_path = "model.pth"
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
    return eval_psnr(model, img)


if __name__ == '__main__':

    import gradio as gr
    block = gr.Blocks().queue()
    with block:
        gr.Markdown("# 水体分割")
        # gr.Markdown("### Open-World Detection with Grounding DINO")

        with gr.Row():
            with gr.Column():
                img_path = gr.Image(source='upload', type="pil")
                run_button = gr.Button(label="Run")

            with gr.Column():
                iimg = gr.outputs.Image(
                    type="pil",
                    # label="grounding results"
                ).style(full_width=False, full_height=False)
                img = gr.outputs.Image(
                    type="pil",
                    # label="grounding results"
                ).style(full_width=False, full_height=False)
                mask = gr.outputs.Image(
                    type="pil",
                    # label="grounding results"
                ).style(full_width=False, full_height=False)
        run_button.click(fn=run, inputs=[
            img_path], outputs=[mask, img, iimg])

    gr.interface
    block.launch(server_name='0.0.0.0', server_port=7579)


