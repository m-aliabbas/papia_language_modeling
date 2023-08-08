import pandas as pd
from datasets import Dataset

class DataReader:
    def __init__(self, cleaned_csv_path=None):
        """
        Initialize the DataReader class.

        Args:
            cleaned_csv_path (str, optional): Path to the cleaned CSV file. Defaults to None.
        """
        if cleaned_csv_path is not None:
            self.cleaned_csv_path = cleaned_csv_path
            self.text_list = None
            self.read_hf_data_format(cleaned_csv_path)
        
    def _load_data(self, cleaned_csv_path):
        """
        Load data from the cleaned CSV file.

        Args:
            cleaned_csv_path (str): Path to the cleaned CSV file.

        Returns:
            list: List of text data loaded from the CSV file.
        """
        try:
            df = pd.read_csv(cleaned_csv_path)
            df = df.drop(df.columns[:2], axis=1)
            df = df.rename(columns={'data': 'text'})
            df.to_csv('text_col.csv', index=False)
            self.text_list = list(df['text'].values)
            self.text_list = [str(text) for text in self.text_list]
            print("Data loaded")
            return self.text_list
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def read_hf_data_format(self, cleaned_csv_path='', test_size=0.2):
        """
        Read data from the cleaned CSV file and create a Huggingface Dataset.

        Args:
            cleaned_csv_path (str, optional): Path to the cleaned CSV file. Defaults to an empty string.
            test_size (float, optional): The proportion of the dataset to include in the test split. Defaults to 0.2.

        Returns:
            datasets.Dataset: Huggingface Dataset object containing the data split into train and test sets.
        """
        if self._load_data(cleaned_csv_path) is not None:
            try:
                raw_data = {'text': self.text_list}
                dataset = Dataset.from_dict(raw_data)
                dataset = dataset.train_test_split(test_size=test_size)
                print("Data is loaded in Huggingface Format", dataset)
                return dataset
            except Exception as e:
                print(f"Data is not loaded. Error: {e}")
                return None
        else:
            print("Data is not loaded. Error.")
            return None
            

if __name__ == "__main__":
    # Example usage
    data_reader = DataReader()
    dataset = data_reader.read_hf_data_format('/home/mohammad/clean_filenames.csv')
    print(dataset)
