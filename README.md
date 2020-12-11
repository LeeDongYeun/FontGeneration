# Font Generation with already-written handwritings
This is a repository for font generation with pre-written handwritings. 
Most of the services for font generation requires huge labor and time consumptions. Although there exists a few methods that can create new fonts with relatively small number of characters, it still has a clear limitation that you should write new characters, which is not a negligible cost. 
Thus, the goal of this project is to create your own Hangul font from an arbitrary sets of Hangul handwritings written on the paper. 
To this end, you should (1) preprocess your images of pre-written handwritings, (2) get the labels of characters using OCR, and finally (3) generate a new font. 
We have used [DM-Font](https://github.com/clovaai/dmfont) for the font generation.

## Preprocess your images of handwritings
- descriptions 
```
command
```

## Get the labels of characters using OCR
- descriptions 
```
command 
```


## Font Generation
Here, you can create a new font from the acquired paired data. 
### Prepare Data
#### Create json file
First, run create_json.py file to set target and style characters, which are characters to be created and used for reference, respectively.
```
python create_json.py --img_dir [image directory] --out_name [output name of json file]
```
#### Dumping dataset

As in DM-Font, the `scripts/prepare_user_dataset.py` script dumps all of the character images under image directory into hdf5 file.  
We slightly modified `scripts/prepare_dataset.py` to dump png files into hdf5 files.
Here, you should modify `meta/kor_custom_ocr.json`. Please write your image directory name ont the values of fonts. 
Also please download NanumBarunpenR.ttf and put it into your image directory and run below command, since it will be used for content characters.
```
# for dumping your own data
python -m scripts.prepare_user_dataset kor [image directory] meta/kor_custom_ocr.json [dump directory]
# for dumping content font (NanumBarunpenR.ttf)  
python -m scripts.prepare_dataset kor [NanumBarunpenR.ttf directory] meta/kor_split_content.json [dump directory]
```

### Generation & Pixel-level evaluation
Now generate new fonts with following command. Note that you should modify `cfs/kor_user_test.yaml`. 
Please change data_dir, target_json.  
```
python evaluator.py [name] [checkpoint path] [out directory] cfgs/kor_user_test.yaml --mode user-study-save
```

For more information, please refer to [DM-Font](https://github.com/clovaai/dmfont). 

### Finetune the model with new data
Since the pretrained model provided by [DM-Font](https://github.com/clovaai/dmfont) is trained on refined handwritings (.ttf files), we found that it does not fit very well into real-world handwriting. Thus, we have tried to finetune the pretrained model with handwriting dataset provided by [AIHub](https://www.aihub.or.kr/ai_data). 
Here is the [link]() for finetuned model. 

#### Extract common characters shared by the dataset
It's for extracting shared characters in training dataset. When you use your own dataset to finetune the model, there might be missing characters. 
In this code, it will extract common characters shared by the dataset.
```
python extract_shared_char.py [output name of json file]
```
### Example of running a simple font generation with given character images.
```
python -m scripts.prepare_user_dataset kor ./data/raw/ocr_results/ meta/kor_custom_ocr.json data/processed/ocr_results/sample_roh_true_0.6
python evaluator.py aihub checkpoints/korean-handwriting.pth ./ocr_roh_0.6 cfgs/kor_user_test.yaml --mode user-study-save
```

## License

This project is distributed under [MIT license](LICENSE), except modules.py which is adopted from https://github.com/NVlabs/FUNIT.

```
Copyright (c) 2020-present NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```
