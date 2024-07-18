# Multimodal-Deep-Learning-Research-in-CLNM
Prediction of central lymph node metastasis in papillary thyroid carcinoma using a multimodal deep learning model: a multicenter study
## Requirements
* python3.11.4
* pytorch2.0.1+cu117
* tensorboard 2.8.0
## Usage
### 1.dataset
* Tumor ultrasound images, C6 level CT images, and clinical information in 611 patients with PTC.  
* **PS:** The data **cannot be shared publicly** due to the privacy of individuals that participated in the study and because the data is intended for future research purposes.
### 2.Train the SCLResNet101
The training of the model is divided into two stages: **stage 1** and **stage 2**.  
#### Stage1
* You need to train the stage1 with the following commands:  
`$ python train_sclresnet101_stage1.py`  
* You can modify the training hyperparameters in `$ config_sclresnet101_stage1.py`.
#### Stage2
* You need to train the stage2 with the following commands:  
`$ python train.py`  
* You can modify the training hyperparameters in `$ config.py`.
### 3.Train the DSAF
* DSAF has two series: DSAF-2 and DSAF-3, used for integrating features from two modalities (tumor + fat) and three modalities (tumor + fat + clinical).  
* First, you need to obtain the CSV files that saves the network features (Our **predict.py** file provides the code to fetch network features).Then you need to train the DSAF with the following commands:  
`$ python train_dsaf.py`  
* **PS:** Initially writing the code, convenience was not prioritized, and DSAF-2 and DSAF-3 were not separated. When training DSAF-2 or DSAF-3, you need to modify the code in the data loading file **dsaf_dataloader.py** and the model file **dsaf.py** to ensure successful loading and training of data from both two modalities or three modalities.(The uploaded code currently focuses on training DSAF-2)
### 4.Predict CLNM
* If you wish to see predictions for SCLResNet101 or other base models, you should run the following file:  
`$ python predict.py`  
* If you want to see the prediction results for DSAF-2 or DSAF-3, you should run the following file (Similarly, you need to modify the code in the data loading file dsaf_dataloader.py and the model file dsaf.py, and the uploaded code now defaults to using DSAF-2 for predicting CLNM):  
`$ python predict_dsaf.py` 
### 5.Comparison of DSAF with machine learning models
* The `$ machine_learning.py` file provides code for SVM, RF, GBDT, and XGBoost to predict CLNM. You can run this file to obtain prediction results.
### 6.run tensorboard
`$ tensorboard --logdir=./logs/`
