# Font Generation with already-written handwritings
This is a repository for font generation with pre-written handwritings. 
Most of the services for font generation requires huge labor and time consumptions. Although there exists a few methods that can create new fonts with relatively small number of characters, it still has a clear limitation that you should write new characters, which is not a negligible cost. 
Thus, the goal of this project is to create your own Hangul font from an arbitrary sets of Hangul handwritings written on the paper. 
To this end, you should (1) preprocess your images of pre-written handwritings, (2) get the labels of characters using OCR, and finally (3) generate a new font. 
We have used [DM-Font](https://github.com/clovaai/dmfont) for the font generation.

## Preprocess your images of handwritings

Preprocessing is required before any processing to extract each handwriting letters from the raw image. The implemented preprocessor typically does one of the followings:

- Print coordinates of each boxes respect to the original image
- Save gray-scaled & flattened image of each boxes

These two operation can be toggled by setting the `save_output` flag. More parameters of the preprocessor can be set in the head of the main file, `preprcessing.py`.

To start the preprocessing step, execute below cammnd with your target image:

```sh
python ./preprocessing.py {target_img_path}
```

## Get the labels of characters using OCR

By using the files in /Ocr you can generate datasets, train models, check accuracy and label the images.
All models that we used(both pre-trained and trained models) are attached in the ipynb files(pre-trained -> AI_tagging, trained -> test_model)

### Model training
fonts directory : The 40 ttf fonts that we used to generate a new model
model_generation : Folder to train a new model using the IBM model generation
- model_generation.ipynb : The main colab note that generates a model from the fonts, other python files are used in this note so change the paths in the ipynb file for your own drive(also uses the label file in the ocr_github folder).
- other files in the folder are for the model-generation process(they are needed in model_generation.ipynb. More instructions in the ipynb file)

### Test the pre-trained and trained model
- test_model.ipynb : Tests the new models(links attached in the file). Similar process with the AI_tagging file
- individual_image/all_test_h : As the preprocessing was developed simultaneously, we used manually boxed images for testing models.

### Label the input images into character unicodes.
AI_tagging.ipynb : File for labeling files from boxed images.
- needs the label file(in the ocr_github folder) and a korean character boxed image folder as the input.
- returns unicode labeled files as an output (more specific instructions in the file)

Change the label, model, image file, save folder and run the ipynb file in the order
The label file is in the folder and the pre-trained model can be downloaded in the ipynb file



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
Before running the code, you should download the pre-trained model in [here](https://drive.google.com/file/d/1T9npQoMTBbsYsiPcGSJtIVgUcjcbrenk/view?usp=sharing).
Now generate new fonts with following command. Note that you should modify `cfs/kor_user_test.yaml`. 
Please change data_dir, target_json.  
```
python evaluator.py [name] [checkpoint path] [out directory] cfgs/kor_user_test.yaml --mode user-study-save
```

For more information, please refer to [DM-Font](https://github.com/clovaai/dmfont). 

### Finetune the model with new data
Since the pretrained model provided by [DM-Font](https://github.com/clovaai/dmfont) is trained on refined handwritings (.ttf files), we found that it does not fit very well into real-world handwriting. Thus, we have tried to finetune the pretrained model with handwriting dataset provided by [AIHub](https://www.aihub.or.kr/ai_data). 
Here is the [link](https://drive.google.com/file/d/1R6qKx9KIJaHLreCIrNWFtgR8f_MQ-KQp/view?usp=sharing) for finetuned model. 

#### Extract common characters shared by the dataset
It's for extracting shared characters in training dataset. When you use your own dataset to finetune the model, there might be missing characters. 
In this code, it will extract common characters shared by the dataset.
```
python extract_shared_char.py [output name of json file]
```
### Example of running a simple font generation with given character images.
```
python -m scripts.prepare_user_dataset kor ./data/raw/ocr_results/ meta/kor_custom_ocr.json data/processed/ocr_results/sample_roh_true_0.6
# if you want to inference with the pre-trained model
python evaluator.py test checkpoints/korean-handwriting.pth ./ocr_roh_0.6 cfgs/kor_user_test.yaml --mode user-study-save
# if you want to inference with the finetuned model
python evaluator.py test_finetuned checkpoints/150000-aihub_boundary_load_gen.pth ./ocr_roh_0.6 cfgs/kor_user_test.yaml --mode user-study-save
```
