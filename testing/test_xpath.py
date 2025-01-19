#!/usr/bin/env python3
import subprocess
import json
import tempfile
import os
from pathlib import Path
from typing import Dict, Optional, Tuple
import sys
import re

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent
DOCKER_IMAGE = "xpath-generator"
CHECKSTYLE_JAR = Path(__file__).parent / "checkstyle-10.21.1-all.jar"

# Import model configuration from inference script
sys.path.append(str(PROJECT_ROOT))
from config import MODEL_CONFIG

class CheckstyleRunner:
    def __init__(self, jar_path: Path = CHECKSTYLE_JAR):
        """Initialize Checkstyle runner with path to Checkstyle JAR."""
        self.jar_path = jar_path
        if not self.jar_path.exists():
            raise FileNotFoundError(f"Checkstyle JAR not found at {jar_path}")

    def get_ast(self, java_code: str) -> str:
        """Generate AST for given Java code."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.java', delete=False) as f:
            f.write(java_code)
            java_file = f.name

        try:
            result = subprocess.run(
                ["java", "-jar", str(self.jar_path), "-t", java_file],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout
        finally:
            Path(java_file).unlink()

    def _create_baseline_config(self) -> str:
        """Create basic Checkstyle configuration with just MethodName check."""
        return '''<?xml version="1.0"?>
        <!DOCTYPE module PUBLIC
            "-//Checkstyle//DTD Checkstyle Configuration 1.3//EN"
            "https://checkstyle.org/dtds/configuration_1_3.dtd">
        <module name="Checker">
            <module name="TreeWalker">
                <module name="MethodName"/>
            </module>
        </module>'''

    def verify_xpath(self, java_code: str, xpath: str) -> Tuple[bool, Optional[str]]:
        """
        Verify if XPath correctly suppresses the violation.
        Returns tuple of (success, error_message).
        """
        with tempfile.NamedTemporaryFile(mode='w', suffix='.java', delete=False) as java_file, \
             tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as config_file, \
             tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as baseline_config_file:

            java_file.write(java_code)
            java_file.flush()

            config_file.write(self._create_config(xpath))
            config_file.flush()

            baseline_config_file.write(self._create_baseline_config())
            baseline_config_file.flush()

            try:
                # First run without suppression to confirm violation exists
                baseline_result = subprocess.run(
                    ["java", "-jar", str(self.jar_path), "-c", baseline_config_file.name, java_file.name],
                    capture_output=True,
                    text=True
                )
                print("\nBefore xpath supression:\n", baseline_result.stdout)

                if baseline_result.returncode == 0:
                    return False, "No violations found in the original code"

                # Run with suppression
                result = subprocess.run(
                    ["java", "-jar", str(self.jar_path), "-c", config_file.name, java_file.name],
                    capture_output=True,
                    text=True
                )
                print("\nAfter xpath supression:\n", result.stdout)

                # Check if specific violation was suppressed
                if "Name 'BAD_Method' must match pattern" not in result.stdout:
                    return True, None
                return False, "XPath did not suppress the target violation"

            except subprocess.CalledProcessError as e:
                return False, f"Checkstyle execution failed: {e}"
            finally:
                Path(java_file.name).unlink()
                Path(config_file.name).unlink()

    def _create_config(self, xpath: str) -> str:
        """Create Checkstyle configuration with XPath suppression."""
        return f'''<?xml version="1.0"?>
        <!DOCTYPE module PUBLIC
            "-//Checkstyle//DTD Checkstyle Configuration 1.3//EN"
            "https://checkstyle.org/dtds/configuration_1_3.dtd">
        <module name="Checker">
            <module name="TreeWalker">
                <module name="MethodName"/>
                <module name="SuppressionXpathSingleFilter">
                    <property name="checks" value="MethodName"/>
                    <property name="query" value="{xpath}"/>
                </module>
            </module>
        </module>'''

class DockerRunner:
    def __init__(self, image_name: str = DOCKER_IMAGE):
        """Initialize Docker runner with image name."""
        self.image_name = image_name
        self._verify_docker_available()

    def _verify_docker_available(self) -> None:
        """Verify Docker is available and the image exists."""
        try:
            subprocess.run(["docker", "info"], capture_output=True, check=True)
        except subprocess.CalledProcessError:
            raise RuntimeError("Docker is not available or not running")

        result = subprocess.run(["docker", "images", "-q", self.image_name], capture_output=True, text=True)
        if not result.stdout:
            raise RuntimeError(f"Docker image '{self.image_name}' not found. Please build it first.")

    def run_generation(self, input_data: Dict) -> Dict:
        """Run XPath generation in Docker container."""
        container_name = "xpath-generator-instance"
        models_dir = PROJECT_ROOT / "models" / MODEL_CONFIG["local_folder"]
        models_dir.mkdir(parents=True, exist_ok=True)

        try:
            process = subprocess.run(
                [
                    "docker", "run",
                    "-i",
                    "--dns", "8.8.8.8",
                    "--dns", "8.8.4.4",
                    "--name", container_name,
                    "-v", f"{models_dir}:/models/{MODEL_CONFIG['local_folder']}",
                    "-e", f"HF_TOKEN={os.getenv('HF_TOKEN', '')}",
                    self.image_name
                ],
                input=json.dumps(input_data),
                text=True,
                capture_output=True
            )

            if process.returncode != 0:
                print("Docker stderr:", process.stderr, file=sys.stderr)
                raise RuntimeError(f"Docker process failed with return code: {process.returncode}")

            return json.loads(process.stdout)

        finally:
            subprocess.run(["docker", "stop", container_name], capture_output=True)
            subprocess.run(["docker", "rm", container_name], capture_output=True)

def main():
    java_code = '''
public class Example {
    public void BAD_Method() {
        int x = 0;
    }
}'''

    try:
        print("=== Starting XPath Generation Test ===")

        checkstyle = CheckstyleRunner()
        ast = checkstyle.get_ast(java_code)
        print("\n✓ Generated AST")

        input_data = {
            "code": java_code,
            "violation": "[ERROR] Example.java:2:17: Name 'BAD_Method' must match pattern '^[a-z][a-zA-Z0-9]*$'. [MethodName]",
            "ast": ast,
            "examples": [
                {
                    "code": "public class Test { public void INVALID_Method() { long a = 1; } }",
                    "violation": "[ERROR] Test.java:1:33: Name 'INVALID_Method' must match pattern '^[a-z][a-zA-Z0-9]*$'. [MethodName]",
                    "xpath": "//METHOD_DEF/IDENT[@text='INVALID_Method']",
                    "ast": checkstyle.get_ast("public class Test { public void INVALID_Method() { long a = 1; } }")
                },
                {
                    "code": "public class Test { public void BAD_Fun() { int b = 7; } }",
                    "violation": "[ERROR] Test.java:1:33: Name 'BAD_Fun' must match pattern '^[a-z][a-zA-Z0-9]*$'. [MethodName]",
                    "xpath": "//METHOD_DEF/IDENT[@text='BAD_Fun']",
                    "ast": checkstyle.get_ast("public class Test { public void BAD_Fun() { int b = 7; } }")
                },
                {
                    "code": "public class Test { public void BAD_method() { } }",
                    "violation": "[ERROR] Test.java:1:33: Name 'BAD_method' must match pattern '^[a-z][a-zA-Z0-9]*$'. [MethodName]",
                    "xpath": "//METHOD_DEF/IDENT[@text='BAD_method']",
                    "ast": checkstyle.get_ast("public class Test { public void BAD_method() { } }")
                }
            ]
        }

        print("\n=== Running XPath Generation ===")
        docker = DockerRunner()
        result = docker.run_generation(input_data)

        if "error" in result:
            print("\n❌ Error generating XPath:", result["error"])
            sys.exit(1)

        xpath = result["xpath"]
        print("\n✓ Generated XPath:", xpath)

        # Verify the XPath
        success, error_message = checkstyle.verify_xpath(java_code, xpath)
        if success:
            print("\n✅ XPath validation successful!")
        else:
            print(f"\n❌ XPath validation failed: {error_message}")
            sys.exit(1)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
