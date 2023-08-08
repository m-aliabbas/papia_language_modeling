# Tokenizer and Model Training Pipeline

## Introduction

This repository contains the code for training custom tokenizers and models using the Hugging Face Transformers library. The custom tokenizers and models can be trained on text data provided in CSV format. The training process involves two main steps: tokenization and model training. The trained models can be used for text generation and inference tasks.

## Author

Muhammad Ali Abbas
Wamiq Raza

## Requirements

`pip install -r requirements.txt`

## Getting Started

1. Cloning
```
git clone https://github.com/your_username/your_repository.git
cd your_repository
Create the Conda environment from the YAML file:
conda env create -f environment.yml
```
    
### Prepare the Data

Ensure that you have your text data in CSV format. The CSV file should contain a column named "text" that holds the text data for training. You can use code inside DataLoader

### Step 1: Tokenizer Training
The first step is to train the tokenizer on the text data. To do this, run the following command:
```
python run_tokenizer_training.py
```

### Step 2: Model Training
The next step is to train the language model using the custom tokenizer. To do this, run the following command:
```
python run_model_training.py
```

### Step 3: Model Inference
After training the model, you can perform text generation and inference tasks. To do this, run the following command:
```
python infer_model.py 
```

## Acknowledgments

This project uses the Hugging Face Transformers library for training and inference. For more information, please visit the Hugging Face Transformers documentation.

## Diagrams

+------------------+                 +-------------------+
|   DataReader    |                 |   CustomTokenizer |
+------------------+                 +-------------------+
| - cleaned_csv_path      1      1   | - base_model_path  |
| - text_list                    |                  |
+------------------+                 +-------------------+
      |                                 |
      |                                 |
      |                                 |
      |                                 |
      |                                 |
      v                                 v
+--------------+                  +--------------------+
| DataCleaning |                  | CustomBertTokenizer|
+--------------+                  +--------------------+
| - clean_text()       1      *   | - tokenizer_dir    |
| - remove_single_word_lines()    | - vocab_size       |
| - clean_text_file()            * | - max_length       |
| - driver()                    * |
+--------------+                  +--------------------+
      |                                 |
      |                                 |
      |                                 |
      |                                 |
      |                                 |
      v                                 v
+----------------+               +----------------------+
|  Tokenizer_Training |               |    GPT2Pipeline       |
+----------------+               +----------------------+
| - tokenizer        |                | - tokenizer_dir         |
| - run()                     *     | - tokenizer            |
                                 | - model_name          |
                                 | - num_epochs       |
                                 | - batch_size           |
                                 | - train()                  |
                                 +----------------------+
                                              |
                                              |
                                              v
                                  +-----------------------+
                                  | Model_Training      |
                                  +-----------------------+
                                  | - config                     |
                                  | - pipeline                  |
                                  | - run()                      |
                                  +-----------------------+
                                                      |
                                                      |
                                                      v
                                  +-----------------------+
                                  | Model_Infer            |
                                  +-----------------------+
                                  | - config                     |
                                  | - pipeline                  |
                                  | - run()                      |
                                  +-----------------------+
