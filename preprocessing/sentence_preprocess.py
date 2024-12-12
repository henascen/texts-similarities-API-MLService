"""

This script preprocess a list of dictionaries including candidates and a
job description
 - Divides the text into sentences and returns a list of sentences in the same
 order they came
 - It also returns a dataframe including the content of the 
 previous dictionary along with the respective sentences

"""

from typing import List, Tuple
from transformers import AutoTokenizer
import transformers
from pathlib import Path

import utils.helpers as UtilsHelp

# MODEL_NAME = f"paraphrase-multilingual-MiniLM-L12-v2"
# MODEL_ACCESS = f"sentence-transformers/{MODEL_NAME}"

# Local path where the Tokenizer is stored
TOKENIZER_PATH = str(Path.cwd() / 'tokenizer' / 'multiling-minilm-l12')

class SentenceSegmentationBatch():
    """
    A class that receives a dataframe with text for each resume (or group) and 
    divides it by sentences. Returning list and tokenization objects.

    Attributes:
        - x
    """
    def __init__(self, candidates_info: List) -> None:
        """
        Initialize the dataframe including candidates and job description
        information (separates each one - jd as the last element of the input 
        list)
        """
        self.candidates = candidates_info
        self.candidates_df = UtilsHelp.build_dataframe(self.candidates[:-1])
        self.job_description_df = UtilsHelp.build_dataframe(
            [self.candidates[-1]]
            )

        idx_sentences_result = UtilsHelp.build_idx_sentences_list(
            self.candidates_df
        )
        self.idx_sentences_list = idx_sentences_result[0]
        self.sentences_list = idx_sentences_result[1]
        self.sentences_sorted = None

        jd_idx_sentences_result = UtilsHelp.build_idx_sentences_list(
            self.job_description_df
        )
        self.jd_sentences_list = jd_idx_sentences_result[1]
        self.jd_sentences_sorted = None

        self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

    def get_sentences_list(self) -> List[str]:
        """Returns the complete sentences list"""
        return self.sentences_list

    def get_idx_sentences_list(self) -> List[Tuple[int, str]]:
        """Returns the list with the sentences and their respective id or 
        group (resume which they belong)"""
        return self.idx_sentences_list
    
    def get_sorted_sentences(self) -> List[str]:
        """Returns a list of sentences sorted from longer to shorter"""
        self.sentences_sorted = UtilsHelp.sort_sentences_by_length(
            self.sentences_list
        )
        return self.sentences_sorted
    
    def get_tokenized_sentences(
        self, 
        batch_sentences: List[str]
        ) -> transformers.tokenization_utils_base.BatchEncoding:
        """Returns a BatchEncoding object containing the sentences tokenized"""

        sentences_tokenized = self.tokenizer(
            batch_sentences,
            padding=True,
            truncation='longest_first',
            return_tensors="np",
            max_length=128
        )

        return sentences_tokenized
    
    def get_jd_sorted_sentences_list(self) -> List[str]:
        """Returns the list of sentences from the job description sorted by
        their length"""
        self.jd_sentences_sorted = UtilsHelp.sort_sentences_by_length(
            self.jd_sentences_list
        )
        return self.jd_sentences_sorted