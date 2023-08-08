from datasets import *
import pandas as pd
from tqdm import tqdm
from transformers import *
import os
import multiprocessing
from Configs.configs import config

config = config['gpt_config']
class CustomGPT2Tokenizer:
    def __init__(self, base_model_path=config['base_model_path'], tokenizer_dir='./tokenizers/',
                 vocab_size=config['vocab_size'], max_length=config['max_length'], model_max_length=config['model_max_length']):
        """
        Initialize the CustomGPT2Tokenizer class.

        Args:
            base_model_path (str): Path to the base GPT-2 model.
            tokenizer_dir (str): Directory to save the tokenizer.
            vocab_size (int): Size of the vocabulary. Defaults to the value from config.
            max_length (int): Maximum length of the tokenized sequences. Defaults to the value from config.
            model_max_length (int): Maximum length of the input sequences expected by the model.
                                    Defaults to the value from config.
        """
        os.makedirs('./tokenizers/', exist_ok=True)
        self.max_length = max_length
        self.model_max_length = model_max_length
        if not os.path.exists(tokenizer_dir):
            # If not, create the directory
            os.makedirs(tokenizer_dir)
            print(f"Directory {tokenizer_dir} created.")
        else:
            print(f"Directory {tokenizer_dir} already exists.")
            
        self.base_model_path = base_model_path
        self.tokenizer_dir = tokenizer_dir
        self.tokenizer = GPT2TokenizerFast.from_pretrained(self.base_model_path)

    def batch_iterator(self, batch_size=100):
        """
        Create a batch iterator for text data.

        Args:
            batch_size (int): Size of each batch. Defaults to 100.

        Yields:
            list: A batch of text data as a list.
        """
        for i in tqdm(range(0, len(self.text_list), batch_size)):
            batch = self.text_list[i : i + batch_size]
            yield [str(text) for text in batch]
            
    def load_data(self, cleaned_csv_path):
        """
        Load data from a cleaned CSV file.

        Args:
            cleaned_csv_path (str): Path to the cleaned CSV file.

        Returns:
            bool: True if data is loaded successfully, False otherwise.
        """
        try:
            df = pd.read_csv(cleaned_csv_path)
            df = df.drop(df.columns[:2], axis=1)
            df = df.rename(columns={'data': 'text'})
            df.to_csv('text_col.csv', index=False)
            self.text_list = list(df['text'].values)
            print("Data loaded")
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def train_tokenizer(self, cleaned_csv_path, tokenizer_name=''):
        """
        Train the tokenizer and save it to a specified directory.

        Args:
            cleaned_csv_path (str): Path to the cleaned CSV file.
            tokenizer_name (str): Name of the tokenizer. Defaults to an empty string.

        Returns:
            transformers.PreTrainedTokenizerFast: Trained GPT-2 tokenizer.
        """
        if self.load_data(cleaned_csv_path):
            try:
                vocab_size = self.tokenizer.vocab_size
            except:
                vocab_size = vocab_size
            gpt2_tokenizer = self.tokenizer.train_new_from_iterator(text_iterator=self.batch_iterator(), 
                                                               vocab_size=self.tokenizer.vocab_size)
            gpt2_tokenizer.save_pretrained(self.tokenizer_dir + tokenizer_name)

            print("Saved to: {}".format(self.tokenizer_dir + tokenizer_name))
            return gpt2_tokenizer
        else:
            ...
    
    def load_tokenizer(self, tokenizer_dir='', tokenizer_name=''):
        """
        Load a tokenizer from a specified directory.

        Args:
            tokenizer_dir (str): Directory where the tokenizer is saved.
            tokenizer_name (str): Name of the tokenizer.

        Returns:
            transformers.PreTrainedTokenizer: Loaded tokenizer.
        """
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir + tokenizer_name)
            print(f"The max length for the tokenizer is: {tokenizer.model_max_length}")
            self.tokenizer = tokenizer
            return tokenizer
        except Exception as error:
            raise Exception("Error loading {error}".format(error))
            return None
    
if __name__ == '__main__':
    custom_gpt2_tokenizer = CustomGPT2Tokenizer(base_model_path=config['base_model_path'])     
      
    custom_gpt2_tokenizer.train_tokenizer(cleaned_csv_path=config['cleaned_csv_path'],
                                          tokenizer_name=config['tokenizer_name'])
            
    tokenizer = custom_gpt2_tokenizer.load_tokenizer(tokenizer_dir=config['tokenizer_dir'],
                                                     tokenizer_name=config['tokenizer_name'])
    print(tokenizer)
