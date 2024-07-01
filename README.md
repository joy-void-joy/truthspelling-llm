Install the project using 
```
poetry install
```
Add your Openai API key and Claude key to .openai_key.local and .anthropic_key.local respectively

Then run a run with
```
python -m truthspelling.finetuning
```

You can analyze high scoring answers with

```
python -m truthspelling.analyze_scores
```
