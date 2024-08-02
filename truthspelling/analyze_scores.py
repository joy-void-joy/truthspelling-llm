# %%
import pathlib

import numpy as np
from .type import Save, OutputAnthropic
from devtools import debug
from tqdm import tqdm

data_dir = pathlib.Path("./data/scenarios/")

lines = [
    l.model_dump()
    | {
        "longer": len(l.deceptive_answer) > len(l.honest_answer),
        "score": l.score,
        "setting": OutputAnthropic.model_validate_json(
            (path.parent.parent / "scenario.json").read_text()
        ),
        "model": "ollama",
    }
    for path in tqdm(list(data_dir.glob("*final*/**/distinguishers.json")))
    if (i := Save.model_validate_json(path.read_text()))
    for l in i.lines
    if l.distinguisher != "\n"
]

debug(sorted(lines, key=lambda i: i["score"]))

debug(sum(i["longer"] for i in lines if i["score"] > 10) / len(lines))
