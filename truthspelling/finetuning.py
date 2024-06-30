# Taken and adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py
# Copyright 2022 The HuggingFace Team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#     http://www.apache.org/licenses/LICENSE-2.0


# %%
from torch.utils.data import Dataset
from .type import OutputAnthropic

import pathlib
import torch
from tqdm import tqdm, trange

from transformers import AutoTokenizer
from transformers import set_seed

from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler

import uuid


from .embedding import get_questionner_context, score_distinguisher
from .generate_scenario import get_scenario


tqdm.pandas()

set_seed(42)

run_name = uuid.uuid4().hex
print("Run name:", run_name)

model_id = "gpt2"

config = PPOConfig(
    project_kwargs={"logging_dir": "./data/logs"},
    model_name=model_id,
    learning_rate=1.41e-5,
    log_with="tensorboard",
    seed=42,
)

sent_kwargs = {
    "return_all_scores": True,
    "function_to_apply": "none",
    "batch_size": 16,
}

log_folder = pathlib.Path(f"./data/scenarios/{run_name}/")

model = AutoModelForCausalLMWithValueHead.from_pretrained(
    config.model_name,
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

tokenizer.pad_token = tokenizer.eos_token


max_length = 1024
num_epoch = 10_000


class ScenarioDataset(Dataset):
    def __len__(self):
        return num_epoch

    def __getitem__(self, idx: int):
        result = OutputAnthropic.model_validate_json(
            pathlib.Path("./2.json").read_text()
        )
        query = get_questionner_context(result)

        encoding = tokenizer.encode(
            query,
            add_special_tokens=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return result.model_dump() | {
            "idx": idx,
            "query": query,
            "input_ids": encoding.flatten(),  # type: ignore
        }
        # return get_scenario(idx, log_folder / f"{idx}" / "scenario.json")


ppo_trainer: PPOTrainer = PPOTrainer(  # type: ignore
    config=config,
    model=model,
    # ref_model=ref_model,
    tokenizer=tokenizer,
    dataset=ScenarioDataset(),
)

output_min_length = 4
output_max_length = 20
output_length_sampler = LengthSampler(output_min_length, output_max_length)

generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
}

batch_size = 16

for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):  # type: ignore
    (f"Starting epoch {epoch}")
    debug(len(batch))

    for step in trange(50):
        query_tensors = []
        response_tensors = []
        reward_tensors = []

        for i in range(debug(len(batch))):
            query_tensors.append(batch["input_ids"][i])
            setting: OutputAnthropic = OutputAnthropic(
                answer=batch["answer"][i],
                question=batch["question"][i],
                scenario_honest=OutputAnthropic.Scenario(
                    context=batch["scenario_honest"]["context"][i]
                ),
                scenario_deceptive=OutputAnthropic.Scenario(
                    context=batch["scenario_deceptive"]["context"][i]
                ),
                deceptive_thinking=batch["deceptive_thinking"][i],
                honest_thinking=batch["honest_thinking"][i],
                questionner_scenario=batch["questionner_scenario"][i],
            )

            gen_len = 20
            generation_kwargs["max_new_tokens"] = gen_len

            response = ppo_trainer.generate(
                ((batch)["input_ids"][i]),
                **generation_kwargs,
                return_prompt=False,
            )
            response_tensors.append(response.squeeze()[-gen_len:])

            loss = score_distinguisher(
                setting=setting,
                distinguisher=tokenizer.decode(response.squeeze()[-gen_len:]),
                log_file=log_folder / f"{epoch}" / f"{step}.json",
            )
            reward = 1 / loss - 1
            reward_tensors.append(reward)

            if loss < 0.1:
                ("Early stopping", loss)
                break
        else:
            stats = ppo_trainer.step(
                *debug(query_tensors, response_tensors, reward_tensors)
            )
            ppo_trainer.log_stats(stats, batch, reward_tensors)
