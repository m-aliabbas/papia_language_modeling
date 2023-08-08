from Trainers.GPT2TrainingPipeline import GPT2Pipeline
from Trainers.BertTrainingPipeline import BertPipeline
from Configs.configs import config

class Model_Infer:
    def __init__(self, model_name='bert') -> None:
        """
        Initialize the Model_Infer class.

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
            self.tokenizer = None
            raise Exception('Model Not Found')

    def run(self, tokenizer_path, model_checkpoint_path):
        """
        Run the inference pipeline for the given model checkpoint and tokenizer.

        Args:
            tokenizer_path (str): Path to the tokenizer directory.
            model_checkpoint_path (str): Path to the pre-trained model checkpoint.

        Returns:
            pipeline: Hugging Face pipeline for inference.
        """
        if self.pipeline is not None:
            pipeline = self.pipeline.load_infer_pipeline(model_checkpoint_path=model_checkpoint_path,
                                                         tokenizer_path=tokenizer_path)
            return pipeline
        else:
            print('[-] Error in Selecting Model')
            return None
            
if __name__ == '__main__':
    model_infer = Model_Infer(model_name='bert')
    model_checkpoint_path = 'ali_bert/checkpoint-932'
    tokenizer_path = '/home/mohammad/Tokenizers/tokenizers/custom_bert_tokenizer'
    pipeline = model_infer.run(tokenizer_path=tokenizer_path, model_checkpoint_path=model_checkpoint_path)
    #for gpt remove mask token beacuse its trained using causal language modeling
    examples = "club nacional de [MASK] miho conoci como nacional ta club mas grandi di futbol di montevidéu uruguay fundá dia 14 di mei 1899 club ta resultado di union entre uruguay athletic montevideo football club uruguay athletic tabata un club di bario"
    predicted_data = pipeline(examples)
    print(predicted_data)
