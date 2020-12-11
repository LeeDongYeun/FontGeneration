
from pathlib import Path
import json
import csv
from argparse import ArgumentParser
import random
import codecs 

def main():
    """
    Use it to create a json file from target image directory. 
    In image directory, it should contain character images with
    corresonding labels as their name. 
      > output : json file 
    """
    parser = ArgumentParser('creating json file')
    parser.add_argument('--img_dir', type=str)
    parser.add_argument('--out_name', type=str)
    args = parser.parse_args()

    img_dir = Path(args.img_dir)
    output_path = Path('./meta') / f'{args.out_name}.json'

    data = {'target_chars':'',
            'style_chars':'',
            'fonts':[],
            }

    all_chars = []
   
    # load all of the images within the target image directory. 
    for img_name in sorted(img_dir.iterdir()):
        if '.png' not in img_name.name:
            continue
        char = img_name.name.replace('uni', '\\u').split('.')[0].lower()
        char = codecs.getdecoder('unicode_escape')(char)[0]
        all_chars.append(char)

    # set target characters and style character randomly.
    # you may use modify it to create specific characters you want.
    NUM_TARGET = 15
    tg = random.sample(all_chars, NUM_TARGET)
    for char in all_chars:
        if char in tg:
            data['target_chars'] += char
        else:
            data['style_chars'] += char

    data['fonts'].append(img_dir.name)
    
    # save as json format.
    with open(output_path,'w') as f:
        json.dump(data, f)


if __name__=='__main__':
    main()




