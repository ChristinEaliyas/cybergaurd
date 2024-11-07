# Cybercrime Classification Model

This project focuses on training a machine learning model to classify cybercrime-related descriptions into various categories. The model is based on the BERT architecture, fine-tuned on a custom dataset containing cybercrime-related descriptions. The process involves exploratory data analysis (EDA), model fine-tuning using the Hugging Face Transformers library, and inference for real-time predictions.

## Table of Contents

1. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
2. [Finetuning the Model](#finetuning-the-model)
3. [Inference](#inference)

---

## Exploratory Data Analysis (EDA)

Before fine-tuning the model, we first conduct an exploratory data analysis (EDA) on the provided dataset to understand its structure and visualize key information. Below is the process for EDA:

### Requirements
```bash
pip install pandas matplotlib seaborn
```

### EDA Code
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file
df = pd.read_csv('test.csv')

# Display basic information about the dataset
print("Basic Information:")
print(df.info())

# Display the first few rows of the dataset
print("\nFirst Few Rows:")
print(df.head())

# Check for missing values
missing_values = df.isnull().sum()
print("\nMissing Values:")
print(missing_values)

# Visualize the distribution of categories
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='category', order=df['category'].value_counts().index)
plt.title('Distribution of Cyber Crime Categories')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Visualize the distribution of subcategories
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='sub_category', order=df['sub_category'].value_counts().index)
plt.title('Distribution of Cyber Crime Subcategories')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Analyze the length of descriptions
df['description_length'] = df['crimeaditionalinfo'].apply(lambda x: len(str(x)))
plt.figure(figsize=(10, 6))
sns.histplot(df['description_length'], bins=30, kde=True)
plt.title('Distribution of Description Lengths')
plt.xlabel('Length of Description')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# Check for duplicates
duplicates = df.duplicated().sum()
print(f"\nNumber of Duplicate Rows: {duplicates}")
```

### Process Overview
1. **Load Data**: Load the dataset into a Pandas DataFrame.
2. **Inspect Data**: Check the basic information, null values, and visualize the distribution of categories and subcategories.
3. **Visualize Description Lengths**: Plot the distribution of description lengths to assess data characteristics.
4. **Duplicates**: Identify any duplicate rows in the dataset.

This EDA helps in understanding the data distribution and checking for any anomalies such as missing values or duplicates.


## Finetuning the Model

Next, we fine-tune a pre-trained BERT model for the cybercrime classification task. This section explains the necessary steps, including data preprocessing, tokenization, model setup, and training.

### Requirements

```bash
pip install dataset evaluate scikit-learn
pip install transformers[torch]
```

### Code for Finetuning

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import DataCollatorWithPadding
from datasets import Dataset, DatasetDict
from sklearn.metrics import roc_auc_score
import torch
import pandas as pd
import numpy as np
import evaluate

# Load datasets
train = pd.read_csv("train_cleaned_v2.csv")
test = pd.read_csv("test_cleaned_v2.csv")

# Define labels and map to integers
labels = train['sub_category'].unique().tolist()
labels = [s.strip() for s in labels]
NUM_LABELS = len(labels)

id2label = {id: label for id, label in enumerate(labels)}
label2id = {label: id for id, label in enumerate(labels)}

# Load pre-trained model and tokenizer
model_path = "google-bert/bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=NUM_LABELS, id2label=id2label, label2id=label2id)

# Freeze all base model parameters
for name, param in model.base_model.named_parameters():
    param.requires_grad = False

# Unfreeze the pooling layer
for name, param in model.base_model.named_parameters():
    if "pooler" in name:
        param.requires_grad = True

# Preprocess the datasets
train['crimeaditionalinfo'] = train['crimeaditionalinfo'].astype(str)
test['crimeaditionalinfo'] = test['crimeaditionalinfo'].astype(str)

label_to_id = {label: idx for idx, label in enumerate(labels)}
train['label'] = train['sub_category'].map(label_to_id)
test['label'] = test['sub_category'].map(label_to_id)

train_dataset = Dataset.from_pandas(train[['crimeaditionalinfo', 'label']])
test_dataset = Dataset.from_pandas(test[['crimeaditionalinfo', 'label']])

dataset_dict = DatasetDict({
    "train": train_dataset,
    "test": test_dataset
})

# Tokenization function
def preprocess_function(examples):
    return tokenizer(examples["crimeaditionalinfo"], truncation=True)

# Tokenize data
tokenized_data = dataset_dict.map(preprocess_function, batched=True)

# Data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Metrics function
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    return {'Accuracy': acc, 'F1': f1, 'Precision': precision, 'Recall': recall}

# Training Arguments
training_args = TrainingArguments(
    output_dir="bert-cyberguard-classifier-V2",
    learning_rate=2e-4,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=32,
    logging_strategy="epoch",
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data["train"],
    eval_dataset=tokenized_data["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

```

### Process Overview

1. **Load Data**: The cleaned training and testing datasets are loaded.
2. **Label Mapping**: Labels are mapped to integer values for compatibility with the model.
3. **Model Setup**: A pre-trained BERT model is loaded and fine-tuned for sequence classification.
4. **Tokenization**: The text data is tokenized using the BERT tokenizer.
5. **Training**: The model is trained using the Trainer API, with evaluation metrics like accuracy, F1 score, precision, and recall.

## Inference

After the model is fine-tuned, you can use it for inference to predict the category of new cybercrime-related descriptions. Below is the process to load the trained model and perform inference.

### Code for Inference

```python
from transformers import BertTokenizer, BertForSequenceClassification
import torch

model_checkpoint_path = "bert-cyberguard-classifier-V2/crime-classification"
tokenizer = BertTokenizer.from_pretrained(model_checkpoint_path)
model = BertForSequenceClassification.from_pretrained(model_checkpoint_path)
categories = {
    0: 'Cyber Bullying Stalking Sexting', 
    1: 'Fraud CallVishing', 
    2: 'Online Gambling Betting', 
    3: 'Online Job Fraud', 
    4: 'UPI Related Frauds',
    # Add all other categories here
}
model.eval()

input_text = "the issue actually started when i got this email... [example text]"

# Tokenize the input text
inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=512)

# Perform inference
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# Convert logits to probabilities (optional)
probs = torch.softmax(logits, dim=1)

# Get predicted class
predicted_class = torch.argmax(probs, dim=1).item()

# Print results
print(f"Predicted class: {categories[predicted_class]}")
print(f"Class probabilities: {probs}")

```

### Process Overview

1. **Load Model**: Load the trained BERT model and tokenizer.
2. **Tokenize Input**: Tokenize the input text for classification.
3. **Inference**: Run the inference process to get predicted probabilities and the class label.

## Conclusion

This project demonstrates the process of training an NLP model for cybercrime classification, including data preprocessing, fine-tuning a BERT model, and performing inference. The resulting model can be used to automatically classify cybercrime-related text into predefined categories, assisting in real-time threat detection or analysis.
