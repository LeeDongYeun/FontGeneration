import os
import json
from itertools import chain
from functools import reduce
from pathlib import Path
from tqdm import tqdm

import h5py as h5
import fire
import numpy as np
from PIL import Image, ImageDraw, ImageFont, features, ImageOps
# from fontTools.ttLib import TTFont

from logger import Logger
from datasets import thai_decompose as thai


CODE_RANGE = {
    'kor': [[0x0021, 0x007E], [0x3131, 0x3163], [0xAC00, 0xD7A3]],
    'thai': [[0x0E01, 0x0E3A], [0x0E3F, 0x0E5B]]
}


def get_code_points(language):
    codes = set()
    code_range = CODE_RANGE[language]
    for rangemin, rangemax in code_range:
        for codepoint in range(rangemin, rangemax+1):
            codes.add(chr(codepoint))

    return codes


def dump_to_hdf5(dump_path, font_name, images, chars, compression=None):
    with h5.File(dump_path, 'w') as f:
        dset = f.create_group('dataset')
        dset.attrs['font_name'] = font_name
        N = len(images)
        print(N, len(chars))
        print(np.stack(images).shape)
        dset.create_dataset('images', (N, 128, 128), np.uint8, compression=compression,
                            data=np.stack(images))
        data = np.array(chars)
        dset.create_dataset('chars', data.shape, np.int, compression=compression,
                            data=np.array(chars))


class UserFontProcessor(object):
    def __init__(self, language, resize_method="bilinear", font_size_factor=2, sample_size=128):
        self.logger = Logger.get(file_path='preparedata.log', level='error')

        self.language = language
        self.targetcodes = get_code_points(self.language)
        if resize_method == 'bilinear':
            self.resize_method = Image.BILINEAR
        else:
            raise ValueError('Invalid resize method: {}'.format(resize_method))
        self.sample_size = sample_size
        self.font_size = self.sample_size * font_size_factor

    def ord(self, char):
        if self.language == 'kor':
            return ord(char)
        else:
            raise ValueError(self.language)
   
    def get_fontsize(self, npimg):
        w, h = npimg.shape
        wsum = npimg.sum(0)
        hsum = npimg.sum(1)

        npimg = 255 - npimg
        
        # # Binary Thresholding.
        threshold = 255 * 2

        if not npimg.sum():
            self.logger.warning(
                '{}, "{}" ({}) is empty (no black)'.format(font, char, self.ord(char))
            )
            return False
        wmin = np.arange(w)[wsum>threshold].min()
        wmax = np.arange(w)[wsum>threshold].max()
        hmin = np.arange(h)[hsum>threshold].min()
        hmax = np.arange(h)[hsum>threshold].max()

        return max(wmax-wmin, hmax-hmin)


    def center_align(self, npimg, fontmaxsize, size=128, margin=0):

        # remove boundaries
        boundaries = 5
        npimg = npimg[boundaries:-(boundaries-1),boundaries:-(boundaries-1)]
        w, h = npimg.shape
        wsum = npimg.sum(0)
        hsum = npimg.sum(1)

        npimg = 255 - npimg
        
        # # Binary Thresholding.
        threshold = 255 * 2

        if not npimg.sum():
            import ipdb; ipdb.set_trace(context=15)
            # self.logger.warning(
            #     '{}, "{}" ({}) is empty (no black)'.format(font, char, self.ord(char))
            # )
            return False
        wmin = np.arange(w)[wsum>threshold].min()
        wmax = np.arange(w)[wsum>threshold].max()
        hmin = np.arange(h)[hsum>threshold].min()
        hmax = np.arange(h)[hsum>threshold].max()

        npimg = 255 - npimg[hmin:hmax+1, wmin:wmax+1]
        canvas_size = int(fontmaxsize*(1+margin))

        roi_w = wmax-wmin
        roi_h = hmax-hmin

        left_margin = (canvas_size - roi_w)//2
        right_margin = canvas_size - roi_w - left_margin
        top_margin = (canvas_size - roi_h)//2
        bottom_margin = canvas_size - roi_h - top_margin

        npimg = np.pad(npimg, ((top_margin, bottom_margin), (left_margin, right_margin)),
                       'constant', constant_values=255)
        img = Image.fromarray(npimg).resize((size, size), resample=self.resize_method)

        return img 

    def render_center_no_offset(self, char, font, fontmaxsize, size=128, margin=0):
        char = self.fix_char_order_if_thai(char)
        size_x, size_y = font.getsize(char)
        offset_x, offset_y = font.getoffset(char)
        roi_w = size_x-offset_x
        roi_h = size_y-offset_y
        img = Image.new('L', (roi_w, roi_h), 255)
        draw = ImageDraw.Draw(img)
        draw.text((-offset_x, -offset_y), char, font=font)

        if img.size[0] == 0 or img.size[1] == 0:
            self.logger.warning(
                '{}, "{}" ({}) is empty (size=0)'.format(font, char, self.ord(char))
            )
            return False

        npimg = 255 - np.array(img)
        if not npimg.sum():
            self.logger.warning(
                '{}, "{}" ({}) is empty (no black)'.format(font, char, self.ord(char))
            )
            return False
        wmin = npimg.sum(0).nonzero()[0].min()
        wmax = npimg.sum(0).nonzero()[0].max()
        hmin = npimg.sum(1).nonzero()[0].min()
        hmax = npimg.sum(1).nonzero()[0].max()

        npimg = 255 - npimg[hmin:hmax+1, wmin:wmax+1]
        canvas_size = int(fontmaxsize*(1+margin))

        left_margin = (canvas_size - roi_w)//2
        right_margin = canvas_size - roi_w - left_margin
        top_margin = (canvas_size - roi_h)//2
        bottom_margin = canvas_size - roi_h - top_margin

        npimg = np.pad(npimg, ((top_margin, bottom_margin), (left_margin, right_margin)),
                       'constant', constant_values=255)
        img = Image.fromarray(npimg).resize((size, size), resample=self.resize_method)

        
    def dump_fonts(self, fonts, dump_dir, compression=None):

        self.logger.info('# Font candidates: {}'.format(len(fonts)))

        dump_dir = Path(dump_dir)
        dump_dir.mkdir(parents=True, exist_ok=True)
        assert dump_dir.is_dir()

        n_fonts = len(fonts)
        for i, targetfontpath in enumerate(fonts):
            targetfontname = os.path.basename(targetfontpath)  # w/ ext
            font_name = os.path.basename(targetfontpath)  # w/o ext
            hdf5_name = "{}.hdf5".format(font_name)
            dump_path = dump_dir / hdf5_name

            if dump_path.exists():
                continue
            
            targetfontpath = Path(targetfontpath)
            targetfonts = [str(fname) for fname in targetfontpath.glob("*")]
            
            images = []
            chars = []

 
            # get max font size
            fontmaxsize = 0
            for f in targetfonts:
                img = Image.open(f)
                npimg = np.array(ImageOps.grayscale(img))
                
                maxsize = self.get_fontsize(npimg)
                fontmaxsize = max(fontmaxsize, maxsize)

            for f in targetfonts:
                img = Image.open(f)
                npimg = np.array(ImageOps.grayscale(img))
                # img = Image.fromarray(npimg)

                img = self.center_align(npimg, fontmaxsize, size=128, margin=0.4)
                # img = Image.fromarray(npimg).resize((128, 128), resample=self.resize_method)

                if not img:
                    continue
                
                char = os.path.basename(f)
                char = char.split('.')[0][3:]
                char = int(char, 16)

                images.append(img)
                chars.append(char)
            
            dump_to_hdf5(dump_path, targetfontname, images, chars, compression=compression)
            
            # self.logger.info("[{:3d}/{:3d}] {} has {} valid chars and {} images...".format(
                # i, n_fonts, font_name, len(images)))


def main(language, fonts_dir, meta_path, dump_dir):
    """
    Args:
        language: kor / thai
        fonts_dir: font directory that has ttf files
        meta_path: meta file path
        dump_dir: dataset dir
    """
    fonts_dir = Path(fonts_dir)

    meta = json.load(open(meta_path, encoding="utf-8"))
    allfonts = set(meta['train']['fonts'] + meta['valid']['fonts'])
    fonts = [
        str(fname) for fname in fonts_dir.glob("*") if fname.name in allfonts
    ]
    assert len(allfonts) == len(fonts)

    processor = UserFontProcessor(language)
    processor.dump_fonts(fonts, dump_dir)


if __name__ == '__main__':
    fire.Fire(main)
