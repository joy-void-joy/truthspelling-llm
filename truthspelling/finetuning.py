# Taken and adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py
# Copyright 2022 The HuggingFace Team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#     http://www.apache.org/licenses/LICENSE-2.0


# %%
import pathlib
import tenacity
import torch
from tqdm import tqdm, trange

from transformers import AutoTokenizer
from transformers import set_seed

from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

import uuid
from devtools import debug


from .embedding import get_questionner_context, score_distinguisher
from .generate_scenario import get_scenario


run_name = "final_skip_retries"
if (pathlib.Path("./data/scenarios") / run_name).exists():
    run_name = run_name + "_" + uuid.uuid4().hex
debug("Run name:", run_name)

model_id = "gpt2"
min_reward = 3
seed = 42
batch_size = 32
gen_len = 50
num_steps = 20
num_epoch = 10_000
reward_multiplier = 1.0
reward_add = 0
min_batch_ratio = 0.25

tqdm.pandas()
set_seed(seed)
config = PPOConfig(
    project_kwargs={"logging_dir": "./data/logs"},
    model_name=model_id,
    learning_rate=1.41e-5,
    log_with="tensorboard",
    seed=seed,
    batch_size=batch_size,
    mini_batch_size=batch_size,
    exp_name=run_name,
    task_name=run_name,
)


log_folder = pathlib.Path(f"./data/scenarios/{run_name}/")

model = AutoModelForCausalLMWithValueHead.from_pretrained(
    config.model_name,
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

tokenizer.eos_token_id = tokenizer.encode("\n", add_special_tokens=False)[0]
tokenizer.pad_token_id = tokenizer.eos_token_id


ppo_trainer: PPOTrainer = PPOTrainer(  # type: ignore
    config=config,
    model=model,
    tokenizer=tokenizer,
)

generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": gen_len,
}


for epoch in trange(num_epoch):  # type: ignore
    setting = get_scenario(epoch, log_folder / f"{epoch}" / "scenario.json")
    query = get_questionner_context(setting)

    encoding = tokenizer.encode(
        query,
        add_special_tokens=True,
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    ).to("cuda")  # type: ignore
    query_encoded = encoding.flatten()

    @tenacity.retry(stop=tenacity.stop_after_attempt(5))
    def do_step(step):
        response_tensors = ppo_trainer.generate(
            [query_encoded] * batch_size,
            **generation_kwargs,
            eos_token_id=tokenizer.eos_token_id,
            return_prompt=False,
        )
        responses = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)

        rewards = score_distinguisher(
            setting=setting,
            distinguishers=responses,
            log_file=log_folder / f"{epoch}" / f"{step}" / "distinguishers.json",
        )

        reward_tensors = [
            torch.tensor(i) * reward_multiplier + reward_add for i in rewards
        ]

        if sum(i > min_reward for i in reward_tensors) > min_batch_ratio * batch_size:
            return False
        else:
            stats = ppo_trainer.step(
                [query_encoded] * batch_size,
                response_tensors,  # type: ignore
                reward_tensors,  # type: ignore
            )
            ppo_trainer.log_stats(
                stats,
                {"query": [query] * batch_size, "response": responses},
                reward_tensors,  # type: ignore
            )
            return False

    for step in trange(num_steps, leave=False):
        try:
            do_step(step)
        except:
            continue

    if epoch % 10 == 0:
        checkpoint_dir = log_folder / f"{epoch}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)

        try:
            torch.save(
                ppo_trainer.optimizer.state_dict(), checkpoint_dir / "optimizer.pt"
            )
        except:
            continue
