"""Unit tests for the XPath generation inference code."""

from contextlib import nullcontext
import importlib.util
from pathlib import Path
import sys
from types import ModuleType
import unittest


PROJECT_ROOT = Path(__file__).parent.parent
INFERENCE_PATH = PROJECT_ROOT / "docker" / "inference.py"


class FakeTokens(list):
    """A token sequence with the shape attribute used by the inference code."""

    @property
    def shape(self):
        return 1, len(self)


class FakeInputs(dict):
    """A minimal replacement for transformers.BatchEncoding."""

    def to(self, device):
        return self


class FakeTokenizer:
    """A tokenizer that exposes whether prompt tokens were decoded."""

    def __init__(self):
        self.eos_token_id = 0
        self.decoded_tokens = None

    def __call__(self, prompt, return_tensors, padding):
        return FakeInputs(input_ids=FakeTokens([10, 11, 12]))

    def decode(self, tokens, skip_special_tokens):
        self.decoded_tokens = list(tokens)
        if self.decoded_tokens == [20, 21]:
            return "//METHOD_DEF/IDENT[@text='BAD_Method']"
        return "//METHOD_DEF/IDENT[@text='INVALID_Method']"


class FakeModel:
    """A minimal causal language model response containing prompt and new tokens."""

    device = "cpu"

    @staticmethod
    def generate(**kwargs):
        return [FakeTokens([10, 11, 12, 20, 21])]


def load_inference_module():
    """Load inference.py without requiring the production ML dependencies."""
    torch_module = ModuleType("torch")
    torch_module.no_grad = nullcontext
    transformers_module = ModuleType("transformers")
    transformers_module.LlamaForCausalLM = object
    transformers_module.LlamaTokenizer = object
    huggingface_module = ModuleType("huggingface_hub")
    huggingface_module.login = lambda token: None
    huggingface_utils_module = ModuleType("huggingface_hub.utils")
    huggingface_utils_module.logging = ModuleType("logging")
    huggingface_utils_module.logging.set_verbosity_info = lambda: None

    original_modules = {
        name: sys.modules.get(name)
        for name in ("torch", "transformers", "huggingface_hub", "huggingface_hub.utils")
    }
    sys.modules.update({
        "torch": torch_module,
        "transformers": transformers_module,
        "huggingface_hub": huggingface_module,
        "huggingface_hub.utils": huggingface_utils_module,
    })
    sys.path.insert(0, str(PROJECT_ROOT))
    try:
        spec = importlib.util.spec_from_file_location("inference_for_test", INFERENCE_PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
        for name, original_module in original_modules.items():
            if original_module is None:
                del sys.modules[name]
            else:
                sys.modules[name] = original_module
    return module


class XPathGeneratorTest(unittest.TestCase):
    """Tests for prompt and generated-token handling."""

    def test_generate_decodes_only_new_tokens(self):
        inference = load_inference_module()
        tokenizer = FakeTokenizer()
        generator = inference.XPathGenerator(FakeModel(), tokenizer)

        xpath = generator.generate("class Example {}", "MethodName", "AST", [])

        self.assertEqual([20, 21], tokenizer.decoded_tokens)
        self.assertEqual("//METHOD_DEF/IDENT[@text='BAD_Method']", xpath)


if __name__ == "__main__":
    unittest.main()
