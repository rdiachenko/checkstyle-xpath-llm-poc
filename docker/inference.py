# inference.py
from transformers import LlamaForCausalLM, LlamaTokenizer
from huggingface_hub import login
from huggingface_hub.utils import logging
import torch
import json
import sys
import os
from typing import Dict, List, Tuple, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_CONFIG

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
            print("Successfully logged in to Hugging Face", file=sys.stderr)
        else:
            print("Warning: No Hugging Face token found in HF_TOKEN env var", file=sys.stderr)

    def load(self) -> Tuple[LlamaForCausalLM, LlamaTokenizer]:
        """Load the model and tokenizer directly from Hugging Face."""
        print("\n=== Starting Model Loading Process ===", file=sys.stderr)
        try:
            print("\n=== Loading Tokenizer ===", file=sys.stderr)
            tokenizer = LlamaTokenizer.from_pretrained(
                MODEL_CONFIG["repo_id"],
                use_auth_token=os.getenv('HF_TOKEN'),
                **MODEL_CONFIG["tokenizer_params"]
            )
            tokenizer.pad_token = tokenizer.eos_token

            print("\n=== Loading Model ===", file=sys.stderr)
            print("This may take several minutes...", file=sys.stderr)
            model = LlamaForCausalLM.from_pretrained(
                MODEL_CONFIG["repo_id"],
                use_auth_token=os.getenv('HF_TOKEN'),
                **MODEL_CONFIG["model_params"]
            )

            return model, tokenizer

        except Exception as e:
            print(f"\n❌ Error during model loading: {str(e)}", file=sys.stderr)
            raise

class XPathGenerator:
    def __init__(self, model: LlamaForCausalLM, tokenizer: LlamaTokenizer):
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

            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True).to(self.model.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    pad_token_id=self.tokenizer.eos_token_id,
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
