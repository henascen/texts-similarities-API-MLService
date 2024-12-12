import pandas as pd
import bentoml

import requests
from pathlib import Path

PICKLE_PATH = (
    Path.cwd() / 'testing_service' / 'java-senten-dataset-long-spacy.pkl'
)

java_df = pd.read_pickle(PICKLE_PATH)

java_df = java_df[['filename', 'raw-content']]
java_df.rename(columns={
    'filename': 'name',
    'raw-content': 'resume',
    },
    inplace=True
)

sending_info = java_df.to_dict('records')
# print(sending_info[0])

client = bentoml.SyncHTTPClient("http://localhost:3000")

NUMBER_OF_FILES = 10

response = client.similarity(input_list=sending_info[:NUMBER_OF_FILES])
print(response)
