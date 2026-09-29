"""Optional free, local similarity pre-filter (DINOv2 embeddings, cosine similarity).

Used to rank a round's candidates so only the top-k are sent to OpenAI. If torch /
transformers aren't installed, ranking is skipped and every candidate is sent.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def _model(name: str):
    import torch
    from transformers import AutoImageProcessor, AutoModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return AutoImageProcessor.from_pretrained(name), AutoModel.from_pretrained(name).to(device).eval(), device


def available() -> bool:
    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401  (transformers' image processors need it)
        import transformers  # noqa: F401
        return True
    except ImportError:
        return False


def similarities(reference: Path, candidates: list[Path], model_name: str = "facebook/dinov2-small") -> list[float]:
    import torch
    from PIL import Image

    proc, model, device = _model(model_name)
    from .sizes import to_rgb

    imgs = [to_rgb(Image.open(p)) for p in [reference, *candidates]]
    with torch.no_grad():
        inputs = proc(images=imgs, return_tensors="pt").to(device)
        emb = model(**inputs).last_hidden_state[:, 0]  # CLS token
        emb = torch.nn.functional.normalize(emb, dim=-1)
    return (emb[1:] @ emb[0]).tolist()
