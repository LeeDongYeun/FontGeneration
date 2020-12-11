import codecs
from pathlib import Path
import os
import json
from collections import OrderedDict
import random

def main():
    """
    It's for extracting shared characters in training dataset.
    When you use your own dataset to finetune the model,
    there might be missing characters. In this code, it will extract
    common characters shared by the dataset.
    """
    
    # prepare output json file
    out_name = 'meta/kor_split_aihub_subset.json'
    file_data = OrderedDict()
    file_data['train'] = {'fonts':[], 'chars':[]}
    file_data['valid'] = {'fonts':[], 'chars':[]}

    # unicode chars
    train_chars_uni = []
    valid_chars_uni = []

    # read kor_split.json file
    kor_split = 'meta/kor_split.json'
    with open(kor_split,'r') as f:
        kor_split_json = json.load(f)
        train_chars = kor_split_json['train']['chars']
        valid_chars = kor_split_json['valid']['chars']

        for ch in sorted(train_chars):
            train_chars_uni.append(hex(ord(ch))[2:])
        
        for ch in sorted(valid_chars):
            valid_chars_uni.append(hex(ord(ch))[2:])


    # rood directory where it contains aihub train/valid data.
    root = Path('data/raw/aihub')

    all_chars = []
    idx = 0
    length = -1
    min_idx = 0

    # find shared characters among all of the data(different styles).
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue

        if 'valid' in d.name:
            file_data['valid']['fonts'].append(d.name)    
        else:
            file_data['train']['fonts'].append(d.name)    

        tmp_chars = [f.name.split('.png')[0] for f in d.glob('*.png')]
        all_chars.append(tmp_chars)
        _length = len(tmp_chars)

        if length <0:
            length = _length 
        elif _length < length:
           length = _length
           min_idx = idx

        idx += 1 

    # extract shared characters
    shared_chars = []
    for char in all_chars[min_idx]:
        count = 0
        for i in range(idx):
            if i==min_idx:
                continue
            if char in all_chars[i]:
                count+=1
            else:
                break

        if count == idx-1:
            shared_chars.append(char)

    # save into json file
    all_uni_chars = []
    for char in shared_chars:
        uni_char = char.replace('uni', '\\u').lower()
        uni_char = codecs.getdecoder('unicode_escape')(uni_char)[0]
        all_uni_chars.append(uni_char)

    # we will use only subsets. 
    random.shuffle(all_uni_chars)
    num_chars = 4000
    all_uni_chars = all_uni_chars[:num_chars]
    train_chars_uni = all_uni_chars[:int(len(all_uni_chars)*0.9)]
    valid_chars_uni = all_uni_chars[int(len(all_uni_chars)*0.9):]

    file_data['train']['chars'] = train_chars_uni
    file_data['valid']['chars'] = valid_chars_uni

    # save json file
    with open(out_name, 'w') as f:
        json.dump(file_data, f)





