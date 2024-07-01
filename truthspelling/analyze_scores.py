# %%
import pathlib

from sentence_transformers import SentenceTransformer
import numpy as np
from .type import Save
from devtools import debug
from tqdm import tqdm

data_dir = pathlib.Path("./data/")
embedder = SentenceTransformer("all-mpnet-base-v2")

lines = [
    l.model_dump()
    | {
        "longer": len(l.deceptive_answer) > len(l.honest_answer),
        "score": 1 / s - 1
        if (
            s := np.dot(
                embedder.encode(l.deceptive_answer), embedder.encode(l.honest_answer)
            )
        )
        else 0,
    }
    for path in tqdm(list(data_dir.glob("**/distinguishers.json")))
    if (i := Save.model_validate_json(path.read_text()))
    for l in i.lines
]

debug(sorted(lines, key=lambda i: i["score"]))
