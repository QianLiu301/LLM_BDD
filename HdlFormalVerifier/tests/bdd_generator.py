"""
Simple Natural Language to BDD Converter
改进版：支持动态路径配置和自动保存到output/bdd目录
Enhanced: OpenAI/ChatGPT with official SDK support

A simplified pipeline: Natural Language → Parameterized BDD scenarios
(with Gherkin Scenario + Examples table).
"""

import json
import re
import random
import os
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
from HdlFormalVerifier.tests.llm_providers_v2 import LLMProvider, LLMFactory, LocalLLMProvider


class SimpleBDDGenerator:
    """Simplified BDD scenario generator with dynamic path support."""

    def __init__(
        self,
        llm_provider: Optional[LLMProvider] = None,
        output_dir: Optional[str] = None,
        project_root: Optional[str] = None
    ):
        self.llm = llm_provider or LocalLLMProvider()

        # ALU operation mapping
        self.operations = {
            "ADD": "0000",
            "SUB": "0001",
            "AND": "0010",
            "OR": "0011",
            "XOR": "0100",
            "NOT": "0101",
            "SHL": "0110",
            "SHR": "0111",
        }

        # 动态路径配置
        self._setup_paths(project_root, output_dir)

    def _setup_paths(self, project_root: Optional[str], output_dir: Optional[str]):
        """
        设置输出路径

        优先级：
        1. 明确指定的output_dir
        2. project_root/output/bdd
        3. 当前目录/output/bdd
        """
        if output_dir:
            # 使用明确指定的输出目录
            self.output_dir = Path(output_dir)
        elif project_root:
            # 使用项目根目录下的output/bdd
            self.output_dir = Path(project_root) / "output" / "bdd"
        else:
            # 默认使用当前目录的output/bdd
            self.output_dir = Path.cwd() / "output" / "bdd"

        # 确保输出目录存在
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"📁 Output directory: {self.output_dir.absolute()}")

    def generate_from_natural_language(self, user_input: str) -> str:
        """
        Generate a BDD scenario from a natural-language request.

        Args:
            user_input: natural-language description from the user

        Returns:
            Full BDD scenario text (Scenario + Steps + Examples table)
        """
        print(f"\n🔍 Processing request: {user_input}")
        print("=" * 70)

        # Step 1: understand intent via LLM (or fallback)
        print("\nStep 1: Understanding user intent with LLM (or local fallback)...")
        intent = self._understand_intent(user_input)

        print(f"   ✅ Operation: {intent['operation']}")
        print(f"   ✅ Condition: {intent['condition']}")
        print(f"   ✅ Scenario name: {intent['scenario_name']}")

        # Step 2: generate test data
        print("\nStep 2: Generating test data...")
        examples = self._generate_test_data(
            intent["condition"],
            intent["num_examples"],
        )
        print(f"   ✅ Generated {len(examples)} example pairs")

        # Step 3: build BDD scenario
        print("\nStep 3: Building BDD scenario...")
        scenario = self._build_scenario(intent, examples)
        print("   ✅ BDD scenario generated")

        return scenario

    def save_scenario(
        self,
        scenario: str,
        filename: Optional[str] = None,
        auto_name: bool = True
    ) -> str:
        """
        保存场景到输出目录

        Args:
            scenario: BDD场景内容
            filename: 文件名（可选）
            auto_name: 是否使用自动命名（基于时间戳）

        Returns:
            保存的完整文件路径
        """
        if not filename:
            if auto_name:
                # 自动生成文件名：scenario_YYYYMMDD_HHMMSS.feature
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"scenario_{timestamp}.feature"
            else:
                filename = "scenario.feature"

        # 确保文件名以.feature结尾
        if not filename.endswith('.feature'):
            filename += '.feature'

        # 完整路径
        filepath = self.output_dir / filename

        # 保存文件
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(scenario)

        print(f"\n✅ Scenario saved to: {filepath.absolute()}")
        return str(filepath.absolute())

    def _understand_intent(self, user_input: str) -> Dict:
        """Use LLM (or local heuristics) to extract intent from user input."""

        prompt = f"""Analyze this test scenario request and extract information in JSON format.

User Request: "{user_input}"

Extract:
1. "operation": ALU operation (ADD, SUB, AND, OR, XOR, etc.)
2. "condition": Test condition (e.g., "A = B", "A > B", "A < B", "overflow", etc.)
3. "scenario_name": Short descriptive name
4. "num_examples": Number of test examples (default: 5)
5. "expected_result": Expected result description
6. "zero_flag": Should zero flag be true/false (if applicable)
7. "negative_flag": Should negative flag be true/false (if applicable)
8. "tags": List of relevant tags

Example output:
{{
  "operation": "SUB",
  "condition": "A = B",
  "scenario_name": "Subtraction resulting in zero",
  "num_examples": 3,
  "expected_result": "0",
  "zero_flag": true,
  "negative_flag": false,
  "tags": ["arithmetic", "zero_result"]
}}

CRITICAL: Respond with ONLY valid JSON. No markdown, no extra text.
"""

        try:
            if hasattr(self.llm, "_call_api"):
                response = self.llm._call_api(prompt, max_tokens=300)
                # Clean up response
                response = response.strip()
                if "```" in response:
                    json_match = re.search(
                        r"```(?:json)?\s*(\{.*?\})\s*```",
                        response,
                        re.DOTALL,
                    )
                    if json_match:
                        response = json_match.group(1)

                intent = json.loads(response)
                return intent
            else:
                return self._fallback_parse(user_input)

        except Exception as e:
            print(f"   ⚠️ LLM parsing failed, using local parsing instead: {e}")
            return self._fallback_parse(user_input)

    def _fallback_parse(self, user_input: str) -> Dict:
        """Local fallback parsing without LLM."""
        text = user_input.lower()

        # Detect operation
        operation = "SUB"
        for op in self.operations.keys():
            if op.lower() in text:
                operation = op
                break

        # Detect condition
        condition = "A = B"
        if "equal" in text or "same" in text or "a = b" in text or "a=b" in text:
            condition = "A = B"
        elif "greater" in text or "a > b" in text or "a>b" in text:
            condition = "A > B"
        elif "less" in text or "a < b" in text or "a<b" in text:
            condition = "A < B"

        # Extract number of examples
        num_match = re.search(r"(\d+)\s*(?:examples?|cases?|scenarios?)", text)
        num_examples = int(num_match.group(1)) if num_match else 3

        # Infer expected result and flags (simple heuristic)
        expected_result = "0" if condition == "A = B" and operation == "SUB" else None
        zero_flag = True if expected_result == "0" else False

        return {
            "operation": operation,
            "condition": condition,
            "scenario_name": f"{operation} with {condition}",
            "num_examples": num_examples,
            "expected_result": expected_result,
            "zero_flag": zero_flag,
            "negative_flag": False,
            "tags": ["arithmetic", "generated"],
        }

    def _generate_test_data(self, condition: str, num_examples: int) -> List[Dict]:
        """Generate concrete test data (A, B) pairs satisfying the condition."""

        # Use LLM to generate intelligent test data
        prompt = f"""Generate {num_examples} test data pairs (A, B) for condition: {condition}

Requirements:
- Values should be 16-bit unsigned integers (0 to 65535)
- ALL examples MUST satisfy: {condition}
- Include diverse values (small, medium, large)
- Include edge cases if relevant

Return ONLY a JSON array:
[
  {{"A": 100, "B": 100}},
  {{"A": 3, "B": 3}},
  {{"A": 9, "B": 9}}
]

CRITICAL: Output ONLY the JSON array. No other text.
"""

        try:
            if hasattr(self.llm, "_call_api"):
                response = self.llm._call_api(prompt, max_tokens=400)
                # Clean and parse response
                response = response.strip()
                if "```" in response:
                    json_match = re.search(
                        r"```(?:json)?\s*(\[.*?\])\s*```",
                        response,
                        re.DOTALL,
                    )
                    if json_match:
                        response = json_match.group(1)

                examples = json.loads(response)
                return examples[:num_examples]
            else:
                return self._local_generate_data(condition, num_examples)

        except Exception as e:
            print(f"   ⚠️ LLM data generation failed, using local generator: {e}")
            return self._local_generate_data(condition, num_examples)

    def _local_generate_data(self, condition: str, num_examples: int) -> List[Dict]:
        """Local random test data generator as a backup."""
        examples: List[Dict] = []

        if condition == "A = B":
            # Representative values
            values = [0, 1, 3, 9, 100, 255, 1024, 32767, 65535]
            random.shuffle(values)
            for val in values[:num_examples]:
                examples.append({"A": val, "B": val})

        elif condition == "A > B":
            for _ in range(num_examples):
                b = random.randint(0, 65534)
                a = random.randint(b + 1, 65535)
                examples.append({"A": a, "B": b})

        elif condition == "A < B":
            for _ in range(num_examples):
                a = random.randint(0, 65534)
                b = random.randint(a + 1, 65535)
                examples.append({"A": a, "B": b})

        else:
            # Completely random data if condition is unknown
            for _ in range(num_examples):
                examples.append(
                    {
                        "A": random.randint(0, 65535),
                        "B": random.randint(0, 65535),
                    }
                )

        return examples

    def _build_scenario(self, intent: Dict, examples: List[Dict]) -> str:
        """Build a Gherkin BDD scenario (Scenario + Steps + Examples)."""

        lines: List[str] = []

        # Tags
        tags = " ".join([f"@{tag}" for tag in intent.get("tags", [])])
        if tags:
            lines.append(f"  {tags}")

        # Scenario name
        lines.append(f"  Scenario: {intent['scenario_name']}")

        # Given step (parameterized by condition)
        given_text = self._generate_given_step(intent["condition"])
        lines.append(f"    Given {given_text}")

        # When step
        opcode = self.operations.get(intent["operation"], "0000")
        lines.append(
            f'    When I set opcode to "{opcode}" for {intent["operation"]} operation'
        )

        # Then step (result)
        if intent.get("expected_result"):
            lines.append(f"    Then the result should be {intent['expected_result']}")
        else:
            lines.append("    Then the result should be correct")

        # Flags
        if intent.get("zero_flag") is not None:
            lines.append(
                f"    And the zero flag should be {str(intent['zero_flag']).lower()}"
            )

        if intent.get("negative_flag") is not None:
            lines.append(
                f"    And the negative flag should be {str(intent['negative_flag']).lower()}"
            )

        # Examples table
        lines.append("")
        lines.append("    Examples:")
        lines.append("    | A | B |")
        for ex in examples:
            lines.append(f"    | {ex['A']} | {ex['B']} |")

        return "\n".join(lines)

    def _generate_given_step(self, condition: str) -> str:
        """Generate the natural-language Given-step description from a condition."""

        mappings = {
            "A = B": "operand A is equal to operand B",
            "A > B": "operand A is greater than operand B",
            "A < B": "operand A is less than operand B",
            "A >= B": "operand A is greater than or equal to operand B",
            "A <= B": "operand A is less than or equal to operand B",
        }

        return mappings.get(condition, f"operands satisfy: {condition}")


def interactive_mode(project_root: Optional[str] = None):
    """Interactive command-line mode."""
    print("\n" + "--" * 35)
    print("   Natural Language → Parameterized BDD Scenarios")
    print("--" * 35 + "\n")

    print("💡 Usage tips:")
    print("   Describe the test scenario you want in plain English.\n")

    print("📝 Example requests:")
    print('   - "Generate SUB test where A equals B with 5 examples"')
    print('   - "Create ADD scenarios where A > B, give me 10 cases"')
    print('   - "I need 3 XOR tests with A = B"')
    print('   - "Subtraction with equal operands, 8 examples"\n')

    print("⚙️  Current LLM configuration: local template (no external API)")
    print("   To use external LLM APIs, configure below.\n")

    use_llm = input("Configure an external LLM provider? (y/n): ").strip().lower()

    if use_llm == "y":
        print("\nSelect LLM provider:")
        print("\n🆓 FREE Providers (Recommended):")
        print("  1. Local template (no external API, zero setup)")
        print("  2. Google Gemini (FREE, 60 req/min) ⭐ Recommended")
        print("  3. Groq (FREE, ultra-fast) ⚡")
        print("  4. DeepSeek (FREE, Chinese LLM)")
        print("\n💰 PAID Providers (High Quality):")
        print("  5. OpenAI ChatGPT (with official SDK) 🎯 Best Quality")
        print("  6. Anthropic Claude (premium)")

        choice = input("\nYour choice (1-6): ").strip()

        if choice == "2":
            # Google Gemini
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                print("\n💡 Get FREE key at: https://makersuite.google.com/app/apikey")
                api_key = input("Enter Gemini API key: ").strip()
            llm = LLMFactory.create_provider("gemini", api_key=api_key)
        elif choice == "3":
            # Groq
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                print("\n💡 Get FREE key at: https://console.groq.com/keys")
                api_key = input("Enter Groq API key: ").strip()
            llm = LLMFactory.create_provider("groq", api_key=api_key)
        elif choice == "4":
            # DeepSeek
            api_key = os.getenv("DEEPSEEK_API_KEY")
            if not api_key:
                print("\n💡 Get key at: https://platform.deepseek.com/")
                api_key = input("Enter DeepSeek API key: ").strip()
            llm = LLMFactory.create_provider("deepseek", api_key=api_key)
        elif choice == "5":
            # OpenAI ChatGPT
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                print("\n💡 Get API key at: https://platform.openai.com/api-keys")
                api_key = input("Enter OpenAI API key: ").strip()

            # 选择模型
            print("\n🤖 Select OpenAI model:")
            print("  1. gpt-4o-mini (Recommended - Best value)")
            print("  2. gpt-3.5-turbo (Cheapest)")
            print("  3. gpt-4o (Most powerful)")
            model_choice = input("Model choice (1-3, default: 1): ").strip() or "1"

            model_map = {
                "1": "gpt-4o-mini",
                "2": "gpt-3.5-turbo",
                "3": "gpt-4o"
            }
            model = model_map.get(model_choice, "gpt-4o-mini")

            llm = LLMFactory.create_provider("openai", api_key=api_key, model=model)
            print(f"✅ Using OpenAI model: {model}")

        elif choice == "6":
            # Claude
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                print("\n💡 Get API key at: https://console.anthropic.com/")
                api_key = input("Enter Anthropic API key: ").strip()
            llm = LLMFactory.create_provider("claude", api_key=api_key)
        else:
            llm = LocalLLMProvider()
    else:
        llm = LocalLLMProvider()

    generator = SimpleBDDGenerator(llm, project_root=project_root)

    print("\n" + "=" * 70)
    print("🚀 Ready! Type 'quit' or 'exit' to leave.\n")

    while True:
        try:
            user_input = input("\n💬 Describe your test scenario: ").strip()

            if user_input.lower() in ["quit", "exit", "q"]:
                print("\n👋 Bye!")
                break

            if not user_input:
                continue

            # Generate BDD scenario
            scenario = generator.generate_from_natural_language(user_input)

            # Show result
            print("\n" + "=" * 70)
            print("✨ Generated BDD Scenario:")
            print("=" * 70)
            print(scenario)
            print("=" * 70)

            # Ask whether to save
            save = input("\n💾 Save to output/bdd directory? (y/n): ").strip().lower()
            if save == "y":
                use_auto_name = input("Use auto-generated filename? (y/n, default: y): ").strip().lower()

                if use_auto_name != 'n':
                    # 使用自动命名
                    filepath = generator.save_scenario(scenario, auto_name=True)
                else:
                    # 手动输入文件名
                    filename = input("📝 File name (e.g., my_test.feature): ").strip()
                    filepath = generator.save_scenario(scenario, filename=filename, auto_name=False)

                print(f"   📂 Saved at: {filepath}")

        except KeyboardInterrupt:
            print("\n\n⚠️ Interrupted by user")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main entry point (CLI)."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Simple natural language → BDD scenario generator with OpenAI SDK support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Example usage:

  # Interactive mode (saves to output/bdd)
  python bdd_generator.py

  # Specify project root
  python bdd_generator.py --project-root D:/DE/HdlFormalVerifierLLM/HdlFormalVerifier/AluBDDVerilog

  # Command-line mode with custom output
  python bdd_generator.py --request "Generate SUB test where A equals B" --output-dir ./my_tests

  # Use OpenAI ChatGPT with official SDK (recommended)
  python bdd_generator.py --llm-provider openai --api-key your-key --model gpt-4o-mini

  # Use alternative names for OpenAI
  python bdd_generator.py --llm-provider gpt --api-key your-key
  python bdd_generator.py --llm-provider chatgpt --api-key your-key

  # Full example
  python bdd_generator.py \\
    --project-root D:/DE/HdlFormalVerifierLLM/HdlFormalVerifier/AluBDDVerilog \\
    --request "Generate SUB test where A equals B with 5 examples" \\
    --llm-provider openai \\
    --model gpt-4o-mini
        """,
    )

    parser.add_argument(
        "--request",
        help="Provide a test scenario description directly (non-interactive mode)",
    )

    parser.add_argument(
        "--llm-provider",
        choices=["local", "openai", "gpt", "chatgpt", "gemini", "google", "groq", "deepseek", "claude", "anthropic"],
        default="local",
        help="LLM provider (default: local). Use 'openai', 'gpt', or 'chatgpt' for OpenAI",
    )

    parser.add_argument(
        "--api-key",
        help="API key for the selected LLM provider",
    )

    parser.add_argument(
        "--model",
        help="Model name (e.g., gpt-4o-mini, gpt-3.5-turbo, gpt-4o for OpenAI)",
    )

    parser.add_argument(
        "--project-root",
        help="Project root directory (output will be saved to {project_root}/output/bdd)",
    )

    parser.add_argument(
        "--output-dir",
        help="Custom output directory (overrides project-root/output/bdd)",
    )

    parser.add_argument(
        "--filename",
        help="Custom filename for the generated scenario",
    )

    args = parser.parse_args()

    # Create LLM provider
    llm_config: Dict = {}
    if args.api_key:
        llm_config["api_key"] = args.api_key
    if args.model:
        llm_config["model"] = args.model

    llm = LLMFactory.create_provider(args.llm_provider, **llm_config)
    generator = SimpleBDDGenerator(
        llm,
        output_dir=args.output_dir,
        project_root=args.project_root
    )

    if args.request:
        # Non-interactive mode
        scenario = generator.generate_from_natural_language(args.request)

        print("\n" + "=" * 70)
        print("✨ Generated BDD Scenario:")
        print("=" * 70)
        print(scenario)
        print("=" * 70)

        # 自动保存到output/bdd
        filepath = generator.save_scenario(
            scenario,
            filename=args.filename,
            auto_name=(args.filename is None)
        )
        print(f"\n✅ Saved to {filepath}")

    else:
        # Interactive mode
        interactive_mode(project_root=args.project_root)


if __name__ == "__main__":
    main()