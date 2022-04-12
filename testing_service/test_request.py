from xml.sax.xmlreader import IncrementalParser
import requests
import pandas as pd
from pathlib import Path

PICKLE_PATH = Path.cwd() / 'testing_service' / 'java-senten-dataset-long-spacy.pkl'

java_df = pd.read_pickle(PICKLE_PATH)

java_df = java_df[['filename', 'raw-content']]
java_df.rename(columns={
    'filename': 'name',
    'raw-content': 'resume',
    },
    inplace=True
)

sending_info = java_df.to_dict('records')
print(sending_info[0])

res = requests.post("http://127.0.0.1:3000/embedding", json=sending_info)
print(res.text)
