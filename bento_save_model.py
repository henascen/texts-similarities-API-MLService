"""
This script saves the sentence embedder model to the BentoML Model Store and
puts a unique Tag to it. This only has to be run when first getting the model
or when a new version wants to be uploaded.
"""

import bentoml
import onnx

from pathlib import Path


MODEL_PATH = Path.cwd() / 'model' / 'multiling_sbert.quant.onnx'

onnx_model = onnx.load(MODEL_PATH)

tag = bentoml.onnx.save_model(
    'onnx_sentembed_model',
    onnx_model,
    signatures={"run": {"batchable": True}}
)
print(tag)
