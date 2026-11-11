MODEL_CONFIGS = {
    #name: what the experiment scripts will select
    "qwen3_8b": {
        # the exact HF model
        "hf_id": "Qwen/Qwen3-8B",
        #where the weights are
        "local_dir": "qwen3-8b",
        "max_new_tokens": 3000,
    },
    "llama3_8b": {
        "hf_id": "meta-llama/Llama-3.1-8B-Instruct",
        "local_dir": "llama3-8b-instruct",
        "max_new_tokens": 3000,
    },
}


def get_model_config(name):
    try:
        return MODEL_CONFIGS[name]
    except KeyError as exc:
        choices = ", ".join(sorted(MODEL_CONFIGS))
        raise ValueError(
            f"Unknown model {name!r}. Available models: {choices}"
        ) from exc
