
from pathlib import Path
import json
import csv
from argparse import ArgumentParser
import random
import codecs 

def main():
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
    
    for img_name in sorted(img_dir.iterdir()):
        if '.png' not in img_name.name:
            continue
        char = img_name.name.replace('uni', '\\u').split('.')[0].lower()
        char = codecs.getdecoder('unicode_escape')(char)[0]
        all_chars.append(char)

    tg = random.sample(all_chars, 10)
    for char in all_chars:

        # data['style_chars'] += char
        if char in tg:
            data['target_chars'] += char
        else:
            data['style_chars'] += char

    data['fonts'].append(img_dir.name)

    with open(output_path,'w') as f:
        json.dump(data, f)


def convert_from_csv():
    parser = ArgumentParser('creating json file')
    parser.add_argument('--csv_path', type=str)
    parser.add_argument('--out_name', type=str)

    args = parser.parse_args()
    output_path = Path('./meta') / f'{args.out_name}.json'

    data = {'target_chars':'',
            'style_chars':'',
            'fonts':[],
            }

    all_chars = []
    with open(args.csv_path) as f:
        reader = csv.reader(f)

        for l in reader:
            if 'filename' in l:
                continue
            char = l[-1].replace('uni',r'\u').lower()
            char = codecs.getdecoder('unicode_escape')(char)[0]
            all_chars.append((char))

    tg = random.sample(all_chars, 5)
    for char in all_chars:
        if char in tg:
            data['target_chars'] += char
        else:
            data['style_chars'] += char

    data['fonts'].append(Path(args.csv_path).parent.name)

    with open(output_path,'w') as f:
        json.dump(data, f)

if __name__=='__main__':
    main()




