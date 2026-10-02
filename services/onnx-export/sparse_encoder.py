"""ONNX export of opensearch-neural-sparse-encoding-doc-v3-distill's document encoder, with SPLADE max
pooling and the v3 activation inside the graph: one vocabulary-sized vector per document (model card recipe)."""
import sys
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForMaskedLM, AutoTokenizer

REPO = "opensearch-project/opensearch-neural-sparse-encoding-doc-v3-distill"
out = Path(sys.argv[1])
(out / "onnx").mkdir(parents=True, exist_ok=True)


class DocEncoder(torch.nn.Module):
    def __init__(self, mlm):
        super().__init__()
        self.mlm = mlm

    def forward(self, input_ids, attention_mask):
        logits = self.mlm(input_ids=input_ids, attention_mask=attention_mask).logits
        values, _ = torch.max(logits * attention_mask.unsqueeze(-1), dim=1)
        return torch.log(1 + torch.log(1 + torch.relu(values)))


tok = AutoTokenizer.from_pretrained(REPO)
enc = DocEncoder(AutoModelForMaskedLM.from_pretrained(REPO)).eval()
tok.save_pretrained(out)
AutoModelForMaskedLM.from_pretrained(REPO).config.save_pretrained(out)
(out / "idf.json").write_bytes(Path(hf_hub_download(REPO, "idf.json")).read_bytes())
sample = tok(["hello world", "a longer second sentence"], padding=True, return_tensors="pt")
torch.onnx.export(
    enc,
    (sample["input_ids"], sample["attention_mask"]),
    out / "onnx" / "model.onnx",
    input_names=["input_ids", "attention_mask"],
    output_names=["sparse"],
    dynamic_axes={"input_ids": {0: "batch", 1: "sequence"}, "attention_mask": {0: "batch", 1: "sequence"}, "sparse": {0: "batch"}},
    opset_version=17,
    dynamo=False,
)
print("exported", out)
