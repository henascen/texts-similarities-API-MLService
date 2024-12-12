from typing import Dict, List
import bentoml
import numpy as np

from preprocessing.sentence_preprocess import SentenceSegmentationBatch
from postprocessing.sentence_postprocess import SentencesSimilarityReduction


# Number of sentences per inference run
BATCH_SIZE = 16


# Processing request for embeddings
# Use the @bentoml.service decorator to mark a class as a Service
@bentoml.service(
    resources={"cpu": "3"},
    workers=2,
    traffic={"timeout": 180},
)
class SentembedService:
    # Integrate the internal Service using bentoml.depends()
    # to inject it as a dependency
    # sentembed_model_runner = bentoml.depends(sentembed_service)

    def __init__(self):
        # Loading the model directly from the BentoML Model Store
        self.sentembed_model_runner = bentoml.onnx.load_model(
            "onnx_sentembed_model:latest"
        )

    @bentoml.api(batchable=True)
    def similarity(self, input_list: List[Dict]) -> List:

        """
        This function handles the request to return embeddings.
        The request comes as a list of dictionaries.

        Parameters:
            - input_list: A list containing the a dictionary with candidate
            info. Specifically, the name and resume text. The "Job Description"
            text is inside this list as the last element.
        
        Output:
            - similarity: A list containing the similarity with the job
            description in the request, in the same order that the candidates
            had.
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

        jd_model_input = {
            name : np.atleast_2d(value)
            for name, value in jd_tokens_batch.items()
        }

        output_jd_embeddings = self.sentembed_model_runner.run(
            None, jd_model_input
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
            

            model_input = {
                name : np.atleast_2d(value)
                for name, value in tokens_batch.items()
            }

            output_embeddings = self.sentembed_model_runner.run(
                None, model_input
            )[0]

            # print(len(output_embeddings))
            # print(output_embeddings.shape)
            all_embeddings.extend(output_embeddings)

        # Postprocess JD and CV Embeddings

        similarity_computation = SentencesSimilarityReduction(
            sentences_data.get_idx_sentences_list(),
            all_embeddings,
            output_jd_embeddings
        )

        similarity_list = similarity_computation.get_sorted_similarity_list()
        # print(similarity_list)

        return similarity_list