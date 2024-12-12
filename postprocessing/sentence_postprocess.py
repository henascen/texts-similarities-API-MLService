from typing import List, Tuple
import numpy as np
import utils.helpers as UtilsHelp

class SentencesSimilarityReduction():
    """
    A class that receives a pair of elements with sentence embeddings and 
    returns similarities between the two elements. For example:
    - The first element might has 5000 sentence embeddings
    - The second element has 10 sentence embeddings

    The similarity would be each of the 5000 sentence embeddings with the 
    10 sentence embeddings. So the result would have the shape of 5000x10.

    Each sentence embedding of the first element correspond to a group. After 
    getting the similarity the sentence embeddings are grouped respectivily.
    The maximum value for each column is extracted, and then it computes the
    mean of all the maximum values for each column. The final result is the
    global similarity.

    Attributes:
        - idx_sentences_list: a list containing a group (id) and a sentence for 
        all sentences. The group corresponds to the resume where the sentence 
        comes from. e.g: (1, 'computer science degree in a respectful school')
        - sentences_embeddings: a list containing all the embeddings for each
        sentence in the same order than in the idx_sentences_list
        e.g: np.array[0.14123 1.114 2.12313 ...] shape = (384,)
        - jd_sentences_embeddings_list: a list containing all the embeddings
        that are exclusively of the jd-element. The format is the same of the 
        sentences_embeddings.
        - jd_cand_similiraties: stores the list of similarities (np.ndarrays) 
        between eachsentence of the first element with each element of 
        the second element
        e.g: [[0.5, 0.4, 0.5, ...] ...] shape=5000x10
        - similarities_list: stores the list of global similarities per each
        group or resume (id). For example for the first element could be 40 id.
        e.g [0.1, 0.6, 0.5, 0.6, 0.9 ...] shape=(40)
    """
    def __init__(
        self,
        candidates_idx_sentences: List[Tuple[int, str]],
        sentences_embeddings: List[np.ndarray],
        jd_sentences_embeddings: List[np.ndarray]) -> None:
        """Initialize sentence similarity by storing the sentence and embeddings
        elements"""
        
        self.idx_sentences_list = candidates_idx_sentences
        self.sentences_embeddings_list = sentences_embeddings
        self.jd_sentences_embeddings_list = jd_sentences_embeddings

        self.jd_cand_similarities = None
        self.similarities_list = None

    def compute_jd_cand_similarities(self) -> None:
        """Computes and stores the similarity between each sentence in the 
        two sentence embeddings elements"""

        self.jd_cand_similarities = UtilsHelp.cosine_similarity_sentences(
            self.sentences_embeddings_list,
            self.jd_sentences_embeddings_list
        )
    
    def reduce_sentences_with_similarities(self) -> None:
        """Groups and reduce the similarities by the correspondent id that each
        sentence has. Stores the result in the similarity_list attribute"""
        
        groups_sentence_sim = [
            (sentence[0], sentence[1], similarities)
            for sentence, similarities
            in zip(
                self.idx_sentences_list,
                self.jd_cand_similarities)
        ]

        columns_names_senten_sim = [
            'CV',
            'Sentence',
            'Similarity'
        ]

        senten_sim_df = UtilsHelp.building_df_from_tuple(
            groups_sentence_sim,
            columns_names_senten_sim
        )

        self.similarities_list = UtilsHelp.reducing_cv_sim_from_groups(
            senten_sim_df,
            columns_names_senten_sim[0],
            columns_names_senten_sim[2]
        )
    
    def get_sorted_similarity_list(self) -> List[float]:
        """Runs the functions and returns the similarity list after the
        computations"""
        
        self.compute_jd_cand_similarities()
        self.reduce_sentences_with_similarities()
        return self.similarities_list