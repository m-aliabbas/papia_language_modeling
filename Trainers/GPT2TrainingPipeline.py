from datasets import *
from transformers import *
from tokenizers import *
import pandas as pd
import os
import json
from Tokenizers.CustomGPT2Tokenizer import CustomGPT2Tokenizer
from DataLoader.DataReader import DataReader
import multiprocessing
import time
from Configs.configs import config


config = config['gpt_config']
print(config)
class GPT2Pipeline:
    # TO DO:
    # Get the Dataset
    # Get the Tokenizer instance
    # Tokenize the dataset
    # Do training
    # Saving Model instance
    
    def __init__(self, model_name=config['custom_model_name'], clean_data_csv=config['cleaned_csv_path'],
                 trained_tokenizer_dir=config['tokenizer_dir'],
                 trained_tokenizer_name=config['tokenizer_name'],
                 test_size=config['test_size'],
                 vocab_size=config['vocab_size'],
                 max_length=config['max_length'],
                 output_dir=config['output_dir'],
                 log_path=config['log_path'],
                 csv_file_path='./metrics.csv',
                 batch_size=config['batch_size'],
                 lr=config['lr'],
                 num_epochs=config['num_epochs']):
        """
        Initialize the GPT2Pipeline class.

        Args:
            model_name (str): Name of the model.
            clean_data_csv (str): Path to the cleaned CSV file.
            trained_tokenizer_dir (str): Directory containing the trained tokenizer.
            trained_tokenizer_name (str): Name of the trained tokenizer.
            test_size (float): Proportion of the dataset to include in the test split.
            vocab_size (int): Size of the vocabulary.
            max_length (int): Maximum length of the tokenized sequences.
            output_dir (str): Output directory to save the trained model.
            log_path (str): Path to log the training metrics.
            csv_file_path (str): Path to the CSV file for metrics.
            batch_size (int): Size of each batch.
            lr (float): Learning rate for training.
            num_epochs (int): Number of training epochs.
        """
        self.lr = lr
        self.num_epochs = num_epochs
        self.model_name = model_name
        self.test_size = test_size
        self.output_dir = output_dir
        self.log_path = log_path
        self.csv_file_path = csv_file_path
        self.batch_size = batch_size
        self.training_args = TrainingArguments(
                    self.model_name,
                    evaluation_strategy="epoch",  # to evaluate model and get metrics after each epoch
                    logging_strategy="epoch",  # to log metrics after each epoch
                    save_strategy="epoch",  # to save model after each epoch
                    per_device_train_batch_size=batch_size,
                    learning_rate=lr,
                    num_train_epochs=num_epochs,   
                    logging_dir='./logs', 
                )
        
        self.model = AutoModelForCausalLM.from_pretrained("gpt2")
        
        self.clean_data_csv = clean_data_csv
        self.data_reader = DataReader(cleaned_csv_path=None)
        self.custom_tokenizer_obj = CustomGPT2Tokenizer(base_model_path=config['base_model_path'],) 
        self.tokenizer = self.custom_tokenizer_obj.load_tokenizer(tokenizer_dir=config['tokenizer_dir'],
                                                     tokenizer_name=config['tokenizer_name'])
        self.num_proc = multiprocessing.cpu_count()
        self.dataset = {}
        self.tokenized_datasets = {}
        self.data_collator = DataCollatorForLanguageModeling(
                                tokenizer=self.tokenizer, mlm=False
                            )
        self.tokenizer.pad_token = self.tokenizer.eos_token
                    
    def load_dataset(self, clean_data_csv=None):
        """
        Load the dataset from the CSV file.

        Args:
            clean_data_csv (str, optional): Path to the cleaned CSV file. Defaults to None.

        Raises:
            Exception: If no clean data is provided.

        Returns:
            None
        """
        if (self.clean_data_csv is None) and (clean_data_csv is None):
            raise Exception("No clean data provided. Please provide the path of clean_data_csv.")
        else:
            if clean_data_csv is None:
                 clean_data_csv = self.clean_data_csv
            else:
                self.clean_data_csv = clean_data_csv
            self.dataset = self.data_reader.read_hf_data_format(clean_data_csv, test_size=self.test_size)
            
    def group_texts(self, examples):
        """
        Group text examples and tokenize them.

        Args:
            examples (dict): Dictionary containing text examples.

        Returns:
            dict: Tokenized inputs.
        """
        tokenized_inputs = self.tokenizer(
            examples['text'], truncation=True, max_length=self.tokenizer.model_max_length
        )
        return tokenized_inputs
    
    def tokenize_data(self):
        """
        Tokenize the dataset.

        Raises:
            Exception: If there is an error during tokenization.

        Returns:
            datasets.Dataset: Tokenized dataset.
        """
        try:
            tokenized_datasets = self.dataset.map(self.group_texts, batched=True, 
                                                remove_columns=["text"], num_proc=self.num_proc)
            tokenized_datasets = tokenized_datasets.shuffle(seed=config['seed'])
            self.tokenized_datasets = tokenized_datasets
            print('Tokenization complete')
            return self.tokenized_datasets
        except Exception as e:
            print(f'Error in Tokenization: {e}')
            raise Exception(f'Error in Tokenization: {e}')
        
    def log_metrics(self):
        """
        Log the training metrics to a CSV file.

        Returns:
            None
        """
        valid_losses = []
        train_losses = []
        train_time = 0.0
        epochs = []
        lr = []
        print(self.training_history)
        for history_dict in self.training_history:

            try:
                if 'eval_loss' in history_dict.keys():
                    valid_loss = history_dict['eval_loss']
                    valid_losses.append(valid_loss)
                elif 'loss' in history_dict.keys():
                    train_loss = history_dict['loss']
                    epochs.append(history_dict['epoch'])
                    train_losses.append(train_loss)
                    lr.append(history_dict['learning_rate'])
                elif 'train_runtime' in history_dict.keys():
                    train_time = history_dict['train_runtime']
            except Exception as e:
                print(f'Something error Log creation: {e}')
        len_divider = len(valid_losses)
        if len_divider <= 0:
            len_divider = 1
    
        train_times = [train_time / len(valid_losses)] * len(valid_losses)
        history = {'epochs': epochs, 'train_losses': train_losses, 'valid_losses': valid_losses}
        df_history = pd.DataFrame(history)
        df_history.to_csv(f'./{self.log_path}')
        print('History of Training Metrics Saved')
    
    def train(self):
        """
        Train the model.

        Returns:
            None
        """
        print('Training step Start')
        self.trainer = Trainer(
                model=self.model,
                args=self.training_args,
                train_dataset=self.tokenized_datasets['train'],
                eval_dataset=self.tokenized_datasets['test'],
                data_collator=self.data_collator,
            )
        self.trained = self.trainer.train()
        try:
            print(self.trained.state.log_history)
        except:
            pass
        self.training_history = self.trainer.state.log_history
        self.log_metrics()
        print('Training step Done')
        
    def run(self):
        """
        Run the entire pipeline.

        Returns:
            None
        """
        self.load_dataset()
        self.tokenize_data()
        self.train()
        
    def load_infer_pipeline(self, model_checkpoint_path, tokenizer_path):
        """
        Load the inference pipeline.

        Args:
            model_checkpoint_path (str): Path to the model checkpoint.
            tokenizer_path (str): Path to the tokenizer.

        Returns:
            transformers.Pipeline: Inference pipeline.
        """
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.pipeline = pipeline('text-generation', model=model_checkpoint_path, tokenizer=tokenizer)
        return self.pipeline
    
if __name__ == '__main__':
    gpt2_pipe_line = GPT2Pipeline()
    gpt2_pipe_line.run()
    # p1 = gpt2_pipe_line.load_infer_pipeline(model_checkpoint_path='ali_gpt2/checkpoint-466',
    #                                         tokenizer_path='/home/mohammad/tokenizers/custom_gpt2_tokenizer')
    # examples = "club nacional de ontevidéu uruguay fundá dia 14 di mei 1899 club ta result"
    # predicted_data= p1(examples)
    # print(predicted_data)
