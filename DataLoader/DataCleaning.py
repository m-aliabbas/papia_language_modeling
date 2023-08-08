import glob
import pandas as pd
import re

class DataCleaning:
    def __init__(self, file_names=[], output_csv_path='') -> None:
        """
        Initialize the DataCleaning class.

        Args:
            file_names (list): List of file names to process.
            output_csv_path (str): Output path for the cleaned data CSV file.
        """
        self.file_names = file_names
        print(self.file_names)
        self.output_csv_path = output_csv_path

    def clean_text(self, text):
        """
        Function to clean text.

        Args:
            text (str): The input text to be cleaned.

        Returns:
            str: Cleaned text after applying various cleaning steps.
        """
        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)
        # Remove all the special characters
        text = re.sub(r'\W', ' ', text)
        # Remove all single characters
        text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
        # Remove single characters from the start
        text = re.sub(r'\^[a-zA-Z]\s+', ' ', text) 
        # Substituting multiple spaces with single space
        text = re.sub(r'\s+', ' ', text, flags=re.I)
        # Removing prefixed 'b'
        text = re.sub(r'^b\s+', '', text)
        # Convert to Lowercase
        text = text.lower()
        return text
    
    def remove_single_word_lines(self, text):
        """
        Remove lines containing only one word from the given text.

        Args:
            text (str): The input text to process.

        Returns:
            str: Text with single-word lines removed.
        """
        # Split text into lines
        lines = text.split('\n')
        # Only keep lines that contain more than one word
        new_lines = [line for line in lines if len(line.split()) > 1]
        # Join lines back into text
        text = '\n'.join(new_lines)
        return text
    
    def clean_text_file(self, input_file_path, output_file_path=''):
        """
        Read a text file, clean its contents, and optionally write the cleaned data back into another text file.

        Args:
            input_file_path (str): Path to the input text file.
            output_file_path (str, optional): Path to the output text file. Defaults to an empty string.

        Returns:
            str: Cleaned text from the input file.
        """
        # Read the text file
        with open(input_file_path, 'r') as f:
            data = f.read()
        # Clean the data
        cleaned_data = self.clean_text(data)
        # Remove single word lines
        cleaned_data = self.remove_single_word_lines(cleaned_data)
        # Write the cleaned data back into a text file (Optional)
        # with open(output_file_path, 'w') as f:
        #     f.write(cleaned_data)
        return cleaned_data
    
    def driver(self):
        """
        Execute the data cleaning process for all files and save the cleaned data in a CSV file.
        """
        try:
            # Process each file and store cleaned data in a dictionary
            data_dict = {file_name: self.clean_text_file(file_name) for file_name in self.file_names}
            file_names = data_dict.keys()
            clean_data = data_dict.values()
            clean_data_dict = {'file_names': file_names, 'data': clean_data}
            # Convert the dictionary to a DataFrame and save as a CSV file
            df = pd.DataFrame(clean_data_dict)
            df.to_csv(self.output_csv_path)
            print('Done Cleaning')
        except Exception as e:
            print(f'Error: {e}')
        
if __name__ == '__main__':
    # Get a list of file names from different directories
    wiki_file_names = glob.glob('/home/data/wikipedia/txt/*.txt')
    extra_file_names = glob.glob('/home/data/extra/all_extracted_text/*.txt')
    ibsr_file_names = glob.glob('/home/data/ilse_schoobaar/ilse_schoobaar_poems_stories.txt')
    file_names = wiki_file_names + extra_file_names + ibsr_file_names
    print(file_names)
    # Create a DataCleaning instance and execute the data cleaning process
    data_cleaner = DataCleaning(file_names=file_names, output_csv_path='ali_clean.csv')
    data_cleaner.driver()
