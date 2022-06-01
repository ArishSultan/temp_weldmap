# Weld-Maps Table Recognition

### 1. Dataset Preparation
This code is supposed to use SROIE dataset for training the model to detect text from weld-map images correctly.


the dataset can be downloaded from [this link](https://www.kaggle.com/datasets/urbikn/sroie-datasetv2), place the train and test folders in raw/sroie directory.
and run 
```bash
python dataset.py
```
to convert the dataset into required format, it will produce two folders `data/processed/img` and `data/processed/gt` move these folders to 
`src/htr/data` and then move to training process.

### 2. Model Training
For Model Training run
```bash
python main.py --mode train --data_dir htr/data
```

### 3. Model Evaluation
Remeber to place the model inside the src/htr/model directory.

There are two directories currently.
* `src/htr/model` -> Contains model trained by us.
* `src/htr/model_temp` -> Contains model provided by HTR repository

To generate a report of text recognition from the available weldmap images run, If want to generate report through the model
provided by author place content of `model_temp` inside `model`
```
python main.py --mode gen_report
```


### 4. Extracting Images from Weld-Maps dataset
The dataset link has been emailed to you from id `mkhan.mscs21seecs@seecs.edu.pk`
Place the Monsonato.pdf inside the `raw` directory
NOTE: Data is to be labeled from hand for report generation.

Place the generated images inside the `data/weldmaps` directory to generate report.