import sys
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

out = Path(sys.argv[1])
(out / "onnx").mkdir(parents=True, exist_ok=True)
tok = AutoTokenizer.from_pretrained("facebook/contriever")
model = AutoModel.from_pretrained("facebook/contriever").eval()
tok.save_pretrained(out)
model.config.save_pretrained(out)
sample = tok(["hello world", "a longer second sentence"], padding=True, return_tensors="pt")
torch.onnx.export(
    model,
    (sample["input_ids"], sample["attention_mask"], sample["token_type_ids"]),
    out / "onnx" / "model.onnx",
    input_names=["input_ids", "attention_mask", "token_type_ids"],
    output_names=["last_hidden_state"],
    dynamic_axes={n: {0: "batch", 1: "sequence"} for n in ["input_ids", "attention_mask", "token_type_ids", "last_hidden_state"]},
    opset_version=17,
    dynamo=False,
)
print("exported", out)
