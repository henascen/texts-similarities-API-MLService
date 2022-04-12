import bentoml
from pathlib import Path

MODEL_PATH = Path.cwd() / 'model' / 'multiling_sbert.quant.onnx'

tag = bentoml.onnx.save(
    'onnx_sentembed_model',
    MODEL_PATH,
)
print(tag)
