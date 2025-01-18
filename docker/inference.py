# inference.py
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download, login
from huggingface_hub.utils import logging
import torch
import json
import sys
import os
from typing import Dict, List, Tuple, Optional

MODEL_CONFIG = {
    "repo_id": "MouezYazidi/CodeLlama-3.2-3B-GGUF",
    "local_folder": "CodeLlama-3b",
    "model_params": {
        "torch_dtype": torch.float16,
        "device_map": "auto",
        "low_cpu_mem_usage": True,
        "trust_remote_code": True
    },
    "tokenizer_params": {
        "trust_remote_code": True,
        "use_fast": True
    },
    "generation_params": {
        "max_new_tokens": 30,
        "num_beams": 10,
        "early_stopping": True
    }
}

# Set up logging
logging.set_verbosity_info()

class ModelLoader:
    def __init__(self, base_path: str = "/models"):
        self.base_path = base_path
        self.model_path = os.path.join(base_path, MODEL_CONFIG["local_folder"])
        self._setup_auth()

    def _setup_auth(self) -> None:
        """Set up Hugging Face authentication."""
        token = os.getenv('HF_TOKEN')
        if token:
            login(token)
        else:
            print("Warning: No Hugging Face token found in HF_TOKEN env var", file=sys.stderr)

    def _ensure_model_files(self) -> str:
        """Ensure model files are available locally."""
        if os.path.exists(self.model_path):
            print("\nModel files already present locally.", file=sys.stderr)
            return self.model_path

        print("\n=== Downloading Model Files ===", file=sys.stderr)
        return snapshot_download(
            repo_id=MODEL_CONFIG["repo_id"],
            local_dir=self.model_path,
            local_dir_use_symlinks=False,
            resume_download=True,
            max_workers=1
        )

    def load(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Load the model and tokenizer."""
        print("\n=== Starting Model Loading Process ===", file=sys.stderr)
        try:
            local_path = self._ensure_model_files()

            print("\n=== Loading Tokenizer ===", file=sys.stderr)
            tokenizer = AutoTokenizer.from_pretrained(
                local_path,
                **MODEL_CONFIG["tokenizer_params"]
            )

            print("\n=== Loading Model ===", file=sys.stderr)
            model = AutoModelForCausalLM.from_pretrained(
                local_path,
                **MODEL_CONFIG["model_params"]
            )

            return model, tokenizer

        except Exception as e:
            print(f"\n❌ Error during model loading: {str(e)}", file=sys.stderr)
            raise

class XPathGenerator:
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def _build_prompt(self, code: str, violation: str, ast: str, examples: List[Dict]) -> str:
        """Build the prompt for XPath generation."""
        prompt_parts = [
            "Generate an XPath expression to match a Checkstyle violation based on the provided AST tree and violation details.\n",
            "Requirements:\n- The XPath expression should match the specific node causing the violation.\n- Use the structure of the AST tree to locate the offending element precisely.\n",
            "Violation format:\n[ERROR] <File Name>:<Line>:<Column>: <Violation Description>. [<Violation Type>]\n",
            "Example XPath expressions:\n"
        ]

        for example in examples:
            prompt_parts.extend([
                f"Code:\n{example['code']}\n",
                f"AST Tree:\n{example['ast']}\n",
                f"Violation:\n{example['violation']}\n",
                f"XPath:\n{example['xpath']}\n\n"
            ])

        prompt_parts.extend([
            "Now generate the XPath for the following case:\n",
            f"Code:\n{code}\n",
            f"AST Tree:\n{ast}\n",
            f"Violation:\n{violation}\n",
            "Generate XPath expression:"
        ])

        return "\n".join(prompt_parts)

    def _clean_xpath(self, xpath: str) -> str:
        """Clean and format the generated XPath."""
        if "//" in xpath:
            xpath = xpath.split("//")[1].strip()
            return "//" + xpath.split("\n")[0].strip()
        return xpath

    def generate(self, code: str, violation: str, ast: str, examples: List[Dict]) -> str:
        """Generate XPath based on input data."""
        print("\n=== Generating XPath ===", file=sys.stderr)
        try:
            prompt = self._build_prompt(code, violation, ast, examples)
            print("\n=== Prompt ===\n", prompt, file=sys.stderr)

            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            outputs = self.model.generate(
                **inputs,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **MODEL_CONFIG["generation_params"]
            )

            xpath = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return self._clean_xpath(xpath.strip())

        except Exception as e:
            print(f"\n❌ Error during XPath generation: {str(e)}", file=sys.stderr)
            raise

def print_gpu_memory() -> None:
    """Print GPU memory usage if available."""
    if torch.cuda.is_available():
        print(f"GPU Memory: {torch.cuda.memory_allocated()/1024**2:.2f}MB allocated")

def main() -> None:
    print_gpu_memory()
    print("\n=== Starting XPath Generation Pipeline ===", file=sys.stderr)

    try:
        input_data = json.load(sys.stdin)

        # Initialize model
        loader = ModelLoader()
        model, tokenizer = loader.load()

        # Generate XPath
        generator = XPathGenerator(model, tokenizer)
        xpath = generator.generate(
            code=input_data["code"],
            violation=input_data["violation"],
            ast=input_data.get("ast", ""),
            examples=input_data.get("examples", [])
        )

        print(json.dumps({
            "xpath": xpath,
            "status": "success"
        }, indent=2))

    except Exception as e:
        print(json.dumps({
            "error": str(e),
            "status": "error"
        }, indent=2))

if __name__ == "__main__":
    main()
