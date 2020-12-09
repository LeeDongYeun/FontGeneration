```
AI_tagging.ipynb file for labeling files from boxed images.
                  - needs the label file in the ocr_github folder and a korean character boxed file as the input.
                  - returns unicode labeled files as an output
model_generation folder to train a new model using the IBM model generation
                  - fonts directory : the 40 ttf fonts that I used to generate a new model
                  - model_generation.ipynb : the main colab note that generates a model from the fonts, other python files are used in this note so change the paths in the ipynb file for your own drive(also uses the label file in the ocr_github folder.
Change the label, model, image file, save folder and run the ipynb file in the order
The label file is in the folder and the pre-trained model can be downloaded in the ipynb file
```
