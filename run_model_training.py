from Trainers.GPT2TrainingPipeline import GPT2Pipeline
from Trainers.BertTrainingPipeline import BertPipeline
from Configs.configs import config

class Model_Training:
    def __init__(self, model_name='bert') -> None:
        """
        Initialize the Model_Training class.

        Args:
            model_name (str, optional): Model name, either 'bert' or 'gpt2'. Defaults to 'bert'.
        """
        if model_name == 'bert':
            self.config = config['bert_config']
            self.pipeline = BertPipeline()
        elif model_name == 'gpt2':
            self.config = config['gpt_config']
            self.pipeline = GPT2Pipeline()
        else:
            self.pipeline = None
            raise Exception('Model Not Found')

    def run(self):
        """
        Run the model training pipeline.

        Returns:
            None
        """
        if self.pipeline is not None:
            self.pipeline.run()
        else:
            print('[-] Error in Selecting Model')
            
            
if __name__ == '__main__':
    model_training = Model_Training(model_name='bert')
    model_training.run()
