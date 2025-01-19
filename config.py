MODEL_CONFIG = {
    "repo_id": "codellama/CodeLlama-7b-hf",
    "local_folder": "codellama-local",
    "model_params": {
        "torch_dtype": "float16",
        "device_map": "auto",
        "low_cpu_mem_usage": True
    },
    "tokenizer_params": {
        "use_fast": True,
        "padding_side": "left"
    },
    "generation_params": {
        "max_new_tokens": 30,
        "num_beams": 5,
        "early_stopping": True,
        "temperature": 0.7,
        "top_p": 0.95
    }
}
