# Texts Similarity Service

A service to find the similarity of a list of files against the last file in the list. A high value indicates that the text is very similar to the last file on the list. See the testing_service folder for an example.

The service computed the embeddings from a list of text files and compares their similarity against the last document of the list. The embeddings come from a language model and the similarity computation uses the cosine similarity formula.

The service response is a list of similarity values from 0 to 1 in the order the files were given to the request.

Main stack: BentoML, HuggingFace, SentenceEmbedding, numpy, ONNX, MLOps, REST API.

# Running the service

The service uses BentoML as a deployment platform to ease the API and model management tasks. BentoML allows to easily store and update models for later inference and provides a safe framework to build an inference API to use the model securely and scalable, providing MLOps tools as well.

To save the model in the BentoML Store:
        
        python bento_save_model.py

To run the service (with Bento):
        
        bentoml serve new_service:SentembedService

To test the service (once it's running) -- wait a couple of minutes:
        
        python testing_service/test_request.py

# Installation

        poetry install

# Downloading models

Two models are necessary for the service to run:
1. The sentence embedded model which returns the embedding for the batch of sentences.
2. The language model used as Tokenizer. It returns the tokens from the texts.

They can be downloaded from their respective folder in this repo. They should be stored in the folders with their respective names.


# Data

The data for the service request is only the text of the files we want to compute the similarity. They have to be inside a list of dictionaries, with the filename being the key and the value the actual text. The last item in the list is the text being used as a template (the text against which the other texts will be compared to).
- If the template text and one of the texts in the list are very similar they will give a response value close to 1. If not, it will be closer to 0.