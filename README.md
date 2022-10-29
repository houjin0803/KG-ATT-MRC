# Instructions for using the KG-ATT-MRC code

## Introduction

This project is the source code of the paper ‘Knowledge Graph based Mutual Attention for Machine Reading Comprehension over Anti-Terrorism Corpus’.

## Environment

- Software: `bert4keras>=0.10.8，tensorflow-gpu==1.14.0，nltk==3.6.5，tqdm==4.62.3`
- Hardware:  If the video memory is not enough, you can appropriately reduce the batch_size and enable gradient accumulation

## Usage

- step1: Download [Chinese (base)](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip). Put it in the project under the `KG-ATT-MRC/model/bert` file .

- step2: Download dataset.  You can access the link to the dataset: https://github.com/houjin0803/ATSMRC . Download it and put it in the project under the KG-ATT-MRC/dataset file.

- step3: Configuring the code runtime environment. You can use the following command:

  ```
  conda install --yes --file conda_requirements.txt
  ```

- step3: Use the command `python train_CQT.py` for training and evaluting.

- step4: After the training test is completed, the results file is generated in the `results` folder.

- step5: Use the cmrc2018_evalute file to calculate EM and F1 scores.
