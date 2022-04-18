from typing import Dict, List
import bentoml
from bentoml.io import PandasDataFrame, JSON
import numpy as np

from preprocessing.sentence_preprocess import SentenceSegmentationBatch
from postprocessing.sentence_postprocess import SentencesSimilarityReduction


# Number of sentences per inference run
BATCH_SIZE = 16


# Loading model as a runner
sentembed_runner = bentoml.onnx.load_runner(
    "onnx_sentembed_model:latest"
)

# Instantiating the runner Service
svc = bentoml.Service(
    "sentence_embedding",
    runners=[sentembed_runner]
)

# Processing request for embeddings
@svc.api(input=JSON(), output=JSON())
def embedding(input_list: List[Dict]) -> List:
    """
    This function handles the request to return embeddings. The request comes as
    a JSON, and BentoML handles parsing this json to obtain the list.

    Parameters:
        - input_list: A list containing the a dictionary with candidate info.
        Specifically, the name and resume text. Job Description text is inside
        this list as the last element.
    
    Output:
        - similarity: A list containing the similarity with the job description
        in the request, in the same order that the candidates had.
    """
    input_data = input_list
    # print(type(input_data))

    sentences_data = SentenceSegmentationBatch(input_data)
    input_sorted_sentences = sentences_data.get_sorted_sentences()
    # print(len(input_sorted_sentences))

    jd_sorted_sentences = sentences_data.get_jd_sorted_sentences_list()
    # print(jd_sorted_sentences)
    jd_tokens_batch = sentences_data.get_tokenized_sentences(
        jd_sorted_sentences
        )
    # print(len(jd_tokens_batch))

    model_input_input_ids = jd_tokens_batch['input_ids']
    # print(type(model_input_input_ids))
    # print(model_input_input_ids.shape)
    model_input_token_ids = jd_tokens_batch['token_type_ids']
    # print(type(model_input_token_ids))
    # print(model_input_token_ids.shape)
    model_input_attention = jd_tokens_batch['attention_mask']
    # print(type(model_input_attention))
    # print(model_input_attention.shape)

    output_jd_embeddings = sentembed_runner.run_batch(
        np.atleast_2d(model_input_input_ids),
        np.atleast_2d(model_input_token_ids),
        np.atleast_2d(model_input_attention)
    )[0]

    # print(output_jd_embeddings.shape)
    # print(output_jd_embeddings)
    
    all_embeddings = []

    for start_index in range(0, len(input_sorted_sentences), BATCH_SIZE):
        sentences_batch = input_sorted_sentences[
            start_index:start_index+BATCH_SIZE
            ]
        
        # print(len(sentences_batch))
        tokens_batch = sentences_data.get_tokenized_sentences(
            sentences_batch
            )
        
        # print(len(tokens_batch))
        model_inputs_input_ids = tokens_batch['input_ids']
        # print(type(model_input_input_ids))
        # print(model_input_input_ids.shape)
        model_inputs_token_ids = tokens_batch['token_type_ids']
        # print(type(model_input_token_ids))
        # print(model_input_token_ids.shape)
        model_inputs_attention = tokens_batch['attention_mask']
        # print(type(model_input_attention))
        # print(model_input_attention.shape)

        output_embeddings = sentembed_runner.run_batch(
            np.atleast_2d(model_inputs_input_ids),
            np.atleast_2d(model_inputs_token_ids),
            np.atleast_2d(model_inputs_attention)
        )[0]

        # print(len(output_embeddings))
        # print(output_embeddings.shape)
        all_embeddings.extend(output_embeddings)

    # result = sentembed_runner.run(input_df)
    # Postprocess JD and CV Embeddings

    similarity_computation = SentencesSimilarityReduction(
        sentences_data.get_idx_sentences_list(),
        all_embeddings,
        output_jd_embeddings
    )

    similarity_list = similarity_computation.get_sorted_similarity_list()
    # print(similarity_list)

    sentences_in_embedding = str(len(similarity_list))

    # print(sentences_in_embedding)
    # result = len(all_embeddings)
    return similarity_list