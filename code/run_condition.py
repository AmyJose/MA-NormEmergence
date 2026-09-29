import argparse
import os
from pathlib import Path

from harvest_model import HarvestModel
from llm_client import IsambardClient
from model_config import MODEL_CONFIGS, get_model_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=MODEL_CONFIGS)
    parser.add_argument(
        "--prompt", required=True,
        choices=("baseline", "cooperative", "selfish"),
    )
    parser.add_argument(
        "--rule-policy", required=True,
        choices=("selfish", "utilitarian", "selfless"),
    )
    parser.add_argument("--seed", required=True, type=int)
    args = parser.parse_args()

    config = get_model_config(args.model)
    model_path = Path(os.environ["SCRATCHDIR"]) / "models" / config["local_dir"]

    run_dir = (
        Path("saved_runs/exp2")
        / args.model
        / args.prompt
        / args.rule_policy
        / f"seed_{args.seed}"
    )
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")

    client = IsambardClient(
        model_path=str(model_path),
        model_id=config["hf_id"],
        max_new_tokens=config["max_new_tokens"],
    )
    model = HarvestModel(
        rng=args.seed,
        llm_client=client,
        llm_agent_ids=(0, 1),
        prompt_type=args.prompt,
        rule_policy=args.rule_policy,
        run_dir=run_dir,
    )

    while not model.episode_done:
        model.step()

    print(f"Completed: {run_dir}")


if __name__ == "__main__":
    main()