# XPath Generator PoC

A proof of concept for generating Checkstyle XPath suppressions using LLMs. This project demonstrates the feasibility of using Code Llama to generate XPath expressions for suppressing specific Checkstyle violations.

## Prerequisites

- Docker
- Python 3.x (for local testing)
- Java 17 or higher (for testing only)
- Hugging Face API token (set as `HF_TOKEN` environment variable)

## Project Structure

```
checkstyle-xpath-llm-poc/
│
├── docker/                     # Docker-related files
│   ├── Dockerfile             # Optimized for LLM inference
│   └── inference.py           # Core XPath generation script
│
├── testing/                    # Testing infrastructure
│   └── test_xpath.py          # Testing and validation script
│
├── models/                     # Directory for downloaded models (gitignored)
└── README.md
```

## Setup

1. Set your Hugging Face token:
```bash
export HF_TOKEN="your_token_here"
```

2. Create a directory for model storage:
```bash
mkdir -p models
```

3. Build the Docker image:
```bash
docker build -t xpath-generator -f docker/Dockerfile .
```

4. Download Checkstyle (required for testing):
```bash
cd testing
curl -L -o checkstyle-10.21.1-all.jar \
    https://github.com/checkstyle/checkstyle/releases/download/checkstyle-10.21.1/checkstyle-10.21.1-all.jar
```

## Usage

Run the test script to see the XPath generation in action:
```bash
cd testing
python3 test_xpath.py
```

The script will:
1. Generate AST from a sample Java code
2. Pass the code, violation, and AST to the LLM
3. Generate an XPath expression
4. Validate the generated XPath using Checkstyle

Check docker container logs:
```bash
docker logs -f xpath-generator-instance
```

## Configuration

### Changing the Model

To use a different model, update the `MODEL_CONFIG` in `config.py`:

```python
MODEL_CONFIG = {
    "repo_id": "model-repo/name",
    "local_folder": "model-folder",
    "model_params": {
        # model specific parameters
    }
}
```

## Limitations

This is an experimental proof of concept and has several limitations:
- Uses a simplified prompt structure
- May generate incorrect XPaths for complex cases
- Requires downloading a large language model
- Limited to basic Checkstyle violation cases

## Note

This proof of concept demonstrates the basic approach of using LLMs for XPath generation. It's not intended for production use and serves as a starting point for further development.
