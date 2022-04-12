import numpy as np
import utils.helpers as UtilsHelp

class SentencesSimilarityReduction():
    def __init__(
        self,
        candidates_idx_sentences,
        sentences_embeddings,
        jd_sentences_embeddings) -> None:
        
        self.idx_sentences_list = candidates_idx_sentences
        self.sentences_embeddings_list = sentences_embeddings
        self.jd_sentences_embeddings_list = jd_sentences_embeddings

        self.jd_cand_similarities = None
        self.similarities_list = None

    def compute_jd_cand_similarities(self) -> None:
        self.jd_cand_similarities = UtilsHelp.cosine_similarity_sentences(
            self.sentences_embeddings_list,
            self.jd_sentences_embeddings_list
        )
    
    def reduce_sentences_with_similarities(self) -> None:
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
    
    def get_sorted_similarity_list(self):
        self.compute_jd_cand_similarities()
        self.reduce_sentences_with_similarities()
        return self.similarities_list