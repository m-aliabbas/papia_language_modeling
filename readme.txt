ssh -p 3333 username@139.144.181.183

newspaper path: /home/data/extra/all_extracted_text

wikipedia path: /home/data/wikipedia/txt

github: https://github.com/papiamentu-ai/language_model

---

Papiamentu.ai product specification brief


Contacts: Gabi Ras, Erkan Basar
E-mail: gabi@gabiras.com, erkan@basar.dev

Freelancer: Wamiq https://www.fiverr.com/wamiqraza94, Mohammad

Product
- Language model training and validation pipeline

Development timeline
Start date: 		July 1st 2023
Evaluation date: 	July 31st 2023
Final date: 		August 15th 2023

The pipeline is a piece of software that allows us to train a language model on a dataset, validate the model periodically and run inference with the model. The model will be trained on our machine. We will provide you with an SSH connection to our GPU machine. We will also provide the data to train and validate on.

Evaluation
- At this milestone our team checks if the requirements have been implemented and if the pipeline is running as expected.
- If any component is missing or a problem is discovered, we will notify you such that revisions can be made.

Final delivery
- At this milestone all of the components should be implemented and working as expected.
- All of the code should have been pushed to github.
- We should receive all of the documentation.

Pipeline requirements
- The pipeline should be delivered as a collection of .py files
-- So not a jupyter notebook.
- Programmed in Python 3.9 or higher.
- Uses the Hugging Face package to load language models.
- The user should be able to specify pipeline and model parameters in a separate file.
-- For example a csv file or a json.
-- Parameters: learning rate, batch size, number of epochs, dataset, model architecture, save locations, sequence length, etc.
- When training the model, the data should be shuffled.
- When validating or running in inference mode, the data should not be shuffled.
- Models to load and train with from Hugging Face:
-- T5, BERT, XLNet, BigBird, GPT-2.
- The model should be validated after each epoch.
- The pipeline should work with a multi GPU setup (train on multiple GPUs at the same time).

Logging requirements
- The model should be saved at least every epoch in a location specified by the user.
- Model performance (train and validation) should be logged in a csv file in a location specified by the user.
-- The metrics should at least include accuracy and loss, and any other metric used to measure performance during training and validation.
- The training time per epoch should also be logged.
- When running in inference mode, the model input and output should be saved in a file in a location specified by the user.
- Random seeds should be set and recorded to ensure reproducibility.

Documentation requirements
- Code is documented in the docstring format.
- Complicated parts should be documented with inline comments within the code, to increase readability.
- Code is regularly committed and pushed to our github repository.
-- https://github.com/papiamentu-ai/language_model
- A UML diagram of the project codebase is delivered as part of the documentation.

Data preprocessing requirements
- Text data should be tokenized using the tokenizer appropriate for the chosen model (as provided by Hugging Face).

Virtual environment
- The pipeline should be developed in the miniconda virtual environment to manage dependencies.
- A requirements.txt file or an equivalent should be included to specify the exact versions of all dependencies.
