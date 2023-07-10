# RemoVul AI
<!-- LOGO -->
<br />
<p align="center">
    <img src="logo/linevul_logo.png" width="200" height="200">
  </a>
  <h3 align="center">RemoVul</h3>
</p>

![VulBERTa architecture](/images/BD.png)

## Overview
The RemoVul AI architecture follow <a href="https://www.researchgate.net/publication/359402890_LineVul_A_Transformer-based_Line-Level_Vulnerability_Prediction">LineVul</a> one which consists of two steps: function-level prediction and line-level localization. The function-level prediction step uses a RoBERTa approach to predict whether each function is vulnerable or not. The line-level localization step, on the other hand, uses attention mechanism to locate the specific lines within the function that contain vulnerabilities.

## Modular Decomposition
1. Function Tokenizer & Comments Removul
2. BPE Subword Tokenizer
3. Word & Positional Encoding
4. Stack of 12 Transformer Encoders
5. Linear Classier Layer
6. Line-level Prediction
7. Vs Code Extension

## Function level  Results


#### Compare RemoVul AI model train on Big-Vul Dataset with other 5 models


| Model | Precision |  Recall | F1 score |
|:----:|:--------:|:----:|:----------:|
| VulDeePecker |  0.12 |  0.49 |    0.19  |
|   Devign  | 0.18  |   0.52  |      0.26     |
|   SySeVR  |  0.15 |   0.74  |      0.27   |
|   IVDetect  |  0.23  |  0.72 |   0.35   |
|   Removul AI  |  0.95  |   0.77  |      0.85     |


#### Compare RemoVul AI model train on VulDeePecker Dataset with VulDeePecker model

| Model | Precision |  Recall | F1 score |
|:----:|:--------:|:----:|:----------:|
| VulDeePecker |  0.91 |  - |   0.92  |
|   Removul AI |  1.00 |    0.92 |   0.96 |


#### Compare RemoVul AI model train on μVulDeePecker Dataset with μVulDeePecker model


| Model | Weighted F1 score | 
|:----:|:--------:|
| μVulDeePecker |  0.96 |
|   Removul AI |  .98 |


## Line level  Results

| Model | Weighted F1 score | 
|:----:|:--------:|
| μVulDeePecker |  0.96 |
|   Removul AI |  .98 |



| Model | Top_10_Accuracy | IFA |
|:----:|:--------:|:--------:|
|   Removul AI |  0.65 | 4.56 |




### About the Environment Setup
First of all, clone this repository to your local machine and access the main dir via the following command:
```
git clone https://github.com/Raghad-Khaled/GP.git
cd GP
```

Then, install the python dependencies via the following command:
```
pip install -r requirements.txt
```

#### Command run for Training 
```
python removul.py 
--do_train 
--train_data_file './finetune/mvd/mvd_train.pkl'
--eval_data_file './finetune/mvd/mvd_val.pkl'
--test_data_file './finetune/mvd/mvd_test.pkl'
--num_labels 41 --function_column 'func'
--target_column 'label'

```

* --do_train: This flag indicates that the module should perform training.

* --train_data_file: The path to the training data file 

* --eval_data_file: The path to the evaluation/validation data file 

* --test_data_file: The path to the test data file 

* --num_labels: The number of labels or classes in the dataset. In this case, it is set to 41.

* --function_column: The name of the column in the data file that contains the function code.

* --target_column: The name of the column in the data file that contains the target labels. This column (label) holds the vulnerability labels associated with each function.


Training Generate ```/lightning_logs/version_i``` folder

#### Command run for Test

```
python removul.py
--do_test 
--train_data_file '../finetune/mvd/mvd_train.pkl'
--eval_data_file '../finetune/mvd/mvd_val.pkl'
--test_data_file '../finetune/mvd/mvd_test.pkl'
--num_labels 41 --function_column 'func'
--target_column 'label'
--checkpoint_path './lightning_logs/version_18/checkpoints/epoch=4-step=57930.ckpt' 
--hparams_file './lightning_logs/version_18/hparams.yaml'
```

* --checkpoint_path: The path to the checkpoint file (epoch=4-step=57930.ckpt). This file represents the saved model checkpoint from a previous training run, which can be used to continue training from that point.

* --hparams_file: The path to the hyperparameters file (hparams.yaml). This file contains the hyperparameter settings used for training the model, such as learning rate, batch size, etc.

#### Command run for Line Level

```
python removul.py
--do_linelevel
--test_data_file '../finetune/mvd/mvd_test.pkl'
--num_labels 41 --function_column 'func'
--target_column 'label'
--checkpoint_path './lightning_logs/version_18/checkpoints/epoch=4-step=57930.ckpt' 
--hparams_file './lightning_logs/version_18/hparams.yaml'
```


#### Command run to get vulnerable functions from file

```
python removul.py
--file_name 'code.c'
--test_data_file '../finetune/mvd/mvd_test.pkl'
--num_labels 41 --function_column 'func'
--target_column 'label'
--checkpoint_path './lightning_logs/version_18/checkpoints/epoch=4-step=57930.ckpt' 
--hparams_file './lightning_logs/version_18/hparams.yaml'
```

* --file_name  The path to the  file need to test it



#### Command run to get vulnerable functions from all files in directory

```
python removul.py
--directory_name './code'
--test_data_file '../finetune/mvd/mvd_test.pkl'
--num_labels 41 --function_column 'func'
--target_column 'label'
--checkpoint_path './lightning_logs/version_18/checkpoints/epoch=4-step=57930.ckpt' 
--hparams_file './lightning_logs/version_18/hparams.yaml'
```

* --directory_name The path to Directory need to test it



