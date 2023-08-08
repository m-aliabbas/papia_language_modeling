config = dict()

config['test_size'] = 0.2
config['cleaned_csv_path'] = '/home/mohammad/clean_filenames.csv'
config['output_dir'] = "./output"
config['log_path'] = './log.csv'
config['batch_size'] = 8
config['lr'] = 2e-5
config['num_epochs'] = 3
config['seed']=34

config['bert_config'] = dict()
config['bert_config']['vocab_size'] = 30522
config['bert_config']['base_model_path'] = 'bert-base-uncased'
config['bert_config']['custom_model_name'] = 'ali_bert'
config['bert_config']['tokenizer_dir'] = '/home/mohammad/Tokenizers/tokenizers/'
config['bert_config']['max_length'] = 512
config['bert_config']['model_max_length'] = 512
config['bert_config']['cleaned_csv_path'] = config['cleaned_csv_path']
config['bert_config']['tokenizer_name'] = 'custom_bert_tokenizer'
config['bert_config']['test_size'] = config['test_size'] 
config['bert_config']['output_dir'] = config['output_dir']
config['bert_config']['log_path'] = config['log_path']
config['bert_config']['batch_size'] = config['batch_size']
config['bert_config']['lr'] = config['lr']
config['bert_config']['num_epochs'] = config['num_epochs']
config['bert_config']['mlm_probability'] = 0.2
config['bert_config']['seed'] = config['seed']



config['gpt_config'] = dict()
config['gpt_config']['vocab_size'] = 50257
config['gpt_config']['base_model_path'] = 'gpt2'
config['gpt_config']['custom_model_name'] = 'ali_gpt'
config['gpt_config']['tokenizer_dir'] = './tokenizers/'
config['gpt_config']['max_length'] = 1024
config['gpt_config']['model_max_length'] = 1024
config['gpt_config']['cleaned_csv_path'] = config['cleaned_csv_path']
config['gpt_config']['tokenizer_name'] = 'custom_gpt2_tokenizer'
config['gpt_config']['test_size'] = config['test_size'] 
config['gpt_config']['output_dir'] = config['output_dir']
config['gpt_config']['log_path'] = config['log_path']
config['gpt_config']['batch_size'] = 4 # for gpt2 use lesser
config['gpt_config']['lr'] = config['lr']
config['gpt_config']['num_epochs'] = config['num_epochs']
config['gpt_config']['mlm_probability'] = 0.2
config['gpt_config']['seed'] = config['seed']


print(config)