import pathlib
from tqdm import tqdm

from .type import Save, OutputAnthropic
from .test_distinguisher import distinguish
from .embedding import ollama, openai

questions = [
    "All the data subjects asked?",
    "Thank you, Ex on offense.",
    "False Truth",
    "RES",
    "Once in one of our practical tests, what is the most appropriate space?",
    "Must",
    "All",
    "...)",
    "Cut",
    "Whatever travels from your Carbon clone to your other Blockchain.",
    "reality creative",
    "* *",
    "(((())",
    "Intentional suicide? Yes",
    "*",
    "OGL",
    "CCADVERTISEMENT",
    "`Controlling');",
    "Thank you for your time (the little. Please check and honor if you're using this project again.)",
    "Assign the confidential competitor to users?",
    "For satire note:",
    "What is a",
    "Opin",
    "(13)",
    "SUBA SHARE",
    "You access through Internet.",
    "A",
    "Thank you, Winter",
    "How full did you feel? Carrie Holmes?",
    "CNC",
    "Age Do-Int",
    "Tell reporters about the use of your 401-m services.",
    "After this evening: For Voices against the Areas of Strength",
    "I don't think this is a strict rule.",
    "ByNP",
    "That should be one of those questions.",
    "Hey, Zara?",
]

data_dir = pathlib.Path("/mnt/serdaylv/data/truthspelling")
result_path = pathlib.Path("./data/questions")
result_path.mkdir(exist_ok=True, parents=True)

for ix, i in tqdm(list(enumerate(data_dir.glob("**/scenario.json")))):
    if (result_path / f"{ix}.json").exists():
        continue

    result = Save(
        lines=[
            distinguish(
                setting=OutputAnthropic.model_validate_json(i.read_text()),
                distinguisher=question,
                model=model,
            )
            for question in tqdm(questions, leave=False)
            for model in [ollama, openai]
        ],
        filepath=i,
    )

    (result_path / f"{ix}.json").write_text(result.model_dump_json(indent=2))
