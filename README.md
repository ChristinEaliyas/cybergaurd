# Crime Classification NLP Model

This repository provides instructions on running inference on a pre-trained classification model for cyber threat intelligence. The model is fine-tuned to classify text data into specific categories related to cyber threats.

## Table of Contents
- [Overview](#overview)
- [Setup](#setup)
- [Evaluation Instructions](#evaluation-instructions)
- [Outputs](#outputs)

## Overview
This model uses a BERT-based architecture, fine-tuned for text classification. The trained model is designed to classify input data based on threat categories. Users can perform inference on a new dataset (`evaluation.csv`) and generate classification labels, which will be saved in `results.csv`.

## Setup
1. Clone this repository:
    ```bash
    git clone https://github.com/username/cyber-threat-classification.git
    cd cyber-threat-classification
    ```

2. Install necessary dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Download the pre-trained model from the following link and place it in the `crime-classification-model` directory:
    [Download Model](https://drive.google.com/drive/u/1/folders/1CO6mDN92XE_SMYA62nVAgy5aPMF1dPRE) 

4. Download the helper-functions from the link below and place it in the `notebooks/data/helper` directory:
    [Download helder functions](https://drive.google.com/drive/u/1/folders/1OV7J8gwhELuznXSghP2bscbbYCUb80SQ)

## Evaluation Instructions
1. Ensure your dataset `evaluation.csv` is available in the root of the repository.

2. Run the inference notebook:
    [Inference Notebook](https://github.com/ChristinEaliyas/cybergaurd/blob/master/inference.ipynb)

3. After running the script, the `results.csv` file will contain the predicted labels for each entry in `evaluation.csv`.

## Outputs
- `results.csv`: This file contains the evaluation data along with an additional column (`label`) with the predicted categories.

