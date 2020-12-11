from argparse import ArgumentParser
import json
import os
from PIL import Image
import torch
from torchvision import transforms

from ssim import ssim, msssim

def main():
    parser = ArgumentParser('Calculate SSIM, MSSSIM')
    
    parser.add_argument('--meta_path', type=str)
    parser.add_argument('--img_dir', type=str)
    parser.add_argument('--pred_dir', type=str)

    args = parser.parse_args()

    meta_path = args.meta_path
    img_dir = args.img_dir
    pred_dir = args.pred_dir

    meta = json.load(open(meta_path, encoding="utf-8"))

    target = meta['target_chars']
    target = [x for x in target]
    target_uni = [hex(ord(x))[2:].upper() for x in target]

    ssims = []
    msssims = []
    fonts = os.listdir(img_dir)
    for font in fonts:
        imgs = []
        preds = []
        for char in target_uni:
            img = Image.open(img_dir + '/' + font + '/uni' + char + '.png')
            img = transforms.ToTensor()(img)
            imgs.append(img)
            
            pred = Image.open(pred_dir + '/' + font + '/inferred_' + char + '.png')
            pred = transforms.ToTensor()(pred)
            preds.append(pred)

        # print(len(imgs), len(preds))

        img_tensor = torch.stack(imgs).to(torch.device("cuda"))
        pred_tensor = torch.stack(preds).to(torch.device("cuda"))

        SSIM = ssim(img_tensor, pred_tensor)
        MSSSIM = msssim(img_tensor, pred_tensor)

        ssims.append(SSIM.item())
        msssims.append(MSSSIM.item())

        print(font, "SSIM:", SSIM.item(), "MSSSIM", MSSSIM.item())
    
    print("AVERAGE SSIM:", sum(ssims)/len(ssims),
            "MSSSIM:", sum(msssims)/len(msssims))


if __name__=='__main__':
    main()