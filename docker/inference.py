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
        """Load the model and tokenizer."""
        print("\n=== Starting Model Loading Process ===", file=sys.stderr)
        try:
            # Check if model exists locally
            local_files_exist = os.path.exists(self.model_path) and os.listdir(self.model_path)
            source = self.model_path if local_files_exist else MODEL_CONFIG["repo_id"]
            auth_token = None if local_files_exist else os.getenv('HF_TOKEN')

            print(f"\nLoading from: {'local storage' if local_files_exist else 'Hugging Face'}", file=sys.stderr)

            print("\n=== Loading Tokenizer ===", file=sys.stderr)
            tokenizer = LlamaTokenizer.from_pretrained(
                source,
                use_auth_token=auth_token,
                **MODEL_CONFIG["tokenizer_params"]
            )
            tokenizer.pad_token = tokenizer.eos_token

            print("\n=== Loading Model ===", file=sys.stderr)
            print("This may take several minutes...", file=sys.stderr)
            model = LlamaForCausalLM.from_pretrained(
                source,
                use_auth_token=auth_token,
                **MODEL_CONFIG["model_params"]
            )

            # Save model locally if it was downloaded
            if not local_files_exist:
                print("\n=== Saving Model Locally ===", file=sys.stderr)
                os.makedirs(self.model_path, exist_ok=True)
                tokenizer.save_pretrained(self.model_path)
                model.save_pretrained(self.model_path)
                print(f"Model saved to: {self.model_path}", file=sys.stderr)

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
            "Generate an XPath expression to match a Checkstyle violation based on the given input.\n",
            "Requirements:\n" +
            "- The XPath must exactly match the specific identifier causing the violation\n" +
            "- Use the identifier name from the input code in the XPath expression\n" +
            "- Return only the XPath expression without any explanation\n",
            "\nExample XPath patterns for MethodName violations:\n"
        ]

        for example in examples:
            prompt_parts.extend([
                f"Input code: {example['code']}",
                f"Violation: {example['violation']}",
                f"AST: {example['ast']}",
                f"XPath: {example['xpath']}\n"
            ])

        prompt_parts.extend([
            "\nNow generate the XPath for this input:",
            f"Input code: {code}",
            f"Violation: {violation}",
            f"AST: {ast}",
            "\nXPath:"
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
