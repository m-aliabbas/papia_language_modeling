from Tokenizers.CustomBertTokenizer import CustomBertTokenizer
from Tokenizers.CustomGPT2Tokenizer import CustomGPT2Tokenizer
from Configs.configs import config

class Tokenizer_Training:
    def __init__(self, model_name='bert') -> None:
        """
        Initialize the Tokenizer_Training class.

        Args:
            model_name (str, optional): Model name, either 'bert' or 'gpt2'. Defaults to 'bert'.
        """
        if model_name == 'bert':
            self.config = config['bert_config']
            self.tokenizer = CustomBertTokenizer()
        elif model_name == 'gpt2':
            self.config = config['gpt_config']
            self.tokenizer = CustomGPT2Tokenizer()
        else:
            self.tokenizer = None
            raise Exception('Model Not Found')

    def run(self):
        """
        Train the tokenizer based on the selected model.

        Returns:
            None
        """
        if self.tokenizer is not None:
            self.tokenizer.train_tokenizer(cleaned_csv_path=self.config['cleaned_csv_path'],
                                          tokenizer_name=self.config['tokenizer_name'])
        else:
            print('[-] Error in Selecting Tokenizer')
            
            
if __name__ == '__main__':
    tokenizer_training = Tokenizer_Training(model_name='bert')
    tokenizer_training.run()
