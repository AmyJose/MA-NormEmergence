from harvest_model import HarvestModel
from llm_client import IsambardClient
import os

PROMPTS= [
    "baseline",
    "cooperative",
    "selfish"
]

RULES = [
    "selfish",
    "utilitarian",
    "selfless"
]

#SEEDS = list(range(1, 31))
SEEDS= [2]

llm_client = IsambardClient(
    model_path=f"{os.environ['SCRATCHDIR']}/models/qwen3-8b"
)

for seed in SEEDS:
    for prompt in PROMPTS:
        for rule in RULES:
            print(f"Running seed={seed}, prompt={prompt}, rule={rule}")
            run_dir = (f"saved_runs/exp1/{prompt}_{rule}_seed_{seed}")

            model = HarvestModel(
                rng=seed, llm_client=llm_client,
                prompt_type=prompt, rule_policy=rule, 
                run_dir=run_dir)

            while not model.episode_done:
                model.step()

