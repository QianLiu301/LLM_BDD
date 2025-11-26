"""
Simple Natural Language to BDD Converter
改进版：支持动态路径配置和自动保存到output/bdd目录
Enhanced: OpenAI/ChatGPT with official SDK support - Updated for GPT-5 series
Fixed: Better error handling for JSON parsing and missing fields
Fixed: Enhanced debugging output for LLM responses

Note: All GPT-5 models (including gpt-5.1-codex) use chat completions with JSON mode

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
from llm_providers import LLMProvider, LLMFactory, LocalLLMProvider


class SimpleBDDGenerator:
    """Simplified BDD scenario generator with dynamic path support."""

    def __init__(
            self,
            llm_provider: Optional[LLMProvider] = None,
            output_dir: Optional[str] = None,
            project_root: Optional[str] = None,
            debug: bool = True  # 🔧 新增：调试模式开关
    ):
        self.llm = llm_provider or LocalLLMProvider()
        self.debug = debug  # 🔧 调试模式

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

    def _debug_print(self, message: str, level: str = "INFO"):
        """
        🔧 统一的调试输出方法

        Args:
            message: 要输出的消息
            level: 日志级别 (INFO, DEBUG, WARN, ERROR, SUCCESS)
        """
        if not self.debug and level == "DEBUG":
            return

        icons = {
            "INFO": "ℹ️ ",
            "DEBUG": "🔍",
            "WARN": "⚠️ ",
            "ERROR": "❌",
            "SUCCESS": "✅",
            "STEP": "📌",
            "DATA": "📊",
            "RAW": "📝",
            "JSON": "🔧",
        }
        icon = icons.get(level, "  ")
        print(f"   {icon} [{level}] {message}")

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

        # 🔧 修复：验证必需字段并提供默认值
        required_fields = ['operation', 'condition', 'scenario_name', 'num_examples']
        for field in required_fields:
            if field not in intent:
                print(f"   ⚠️  Missing field '{field}', using fallback")
                intent = self._fallback_parse(user_input)
                break

        print(f"   ✅ Operation: {intent['operation']}")
        print(f"   ✅ Condition: {intent['condition']}")
        print(f"   ✅ Scenario name: {intent['scenario_name']}")

        # Step 2: generate test data
        print("\nStep 2: Generating test data...")
        examples = self._generate_test_data(
            intent["condition"],
            intent["num_examples"],
            intent["operation"]
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
        """
        Use LLM (or local heuristics) to extract intent from user input.

        🔧 增强版：更好的错误处理、JSON 验证和详细调试输出
        """

        prompt = f"""Analyze this test scenario request and extract information in JSON format.

User Request: "{user_input}"

Extract:
1. "operation": ALU operation (ADD, SUB, AND, OR, XOR, NOT, SHL, SHR)
2. "condition": Test condition (e.g., "random", "A = B", "A > B", "A < B", "overflow", etc.)
3. "scenario_name": Short descriptive name
4. "num_examples": Number of test examples (default: 5)
5. "expected_result": Expected result description (optional)
6. "zero_flag": Should zero flag be true/false (if applicable, optional)
7. "negative_flag": Should negative flag be true/false (if applicable, optional)
8. "tags": List of relevant tags (optional)

Example output:
{{
  "operation": "XOR",
  "condition": "random",
  "scenario_name": "XOR with random values",
  "num_examples": 3,
  "tags": ["logic"]
}}

CRITICAL INSTRUCTIONS:
- You MUST respond with ONLY valid JSON
- Do NOT include markdown code blocks (no ```)
- Do NOT include any text before or after the JSON
- Ensure all required fields are present: operation, condition, scenario_name, num_examples
- If "random" is mentioned, use "random" as the condition
"""

        try:
            # 🔧 检查 LLM provider 类型
            llm_type = type(self.llm).__name__
            self._debug_print(f"LLM Provider: {llm_type}", "INFO")

            if hasattr(self.llm, "_call_api"):
                self._debug_print("Calling LLM API...", "STEP")

                response = self.llm._call_api(
                    prompt,
                    max_tokens=300,
                    system_prompt="You are a helpful assistant that extracts structured information from user requests and outputs ONLY valid JSON. Never include markdown formatting."
                )

                # 🔧 详细的调试输出
                print("\n" + "-" * 50)
                print("   📤 LLM API RESPONSE DETAILS:")
                print("-" * 50)

                # 原始响应
                self._debug_print(f"Response Type: {type(response).__name__}", "DEBUG")
                self._debug_print(f"Response Length: {len(response)} chars", "DEBUG")
                print(f"   📝 [RAW] Full Response:\n   '''\n{response}\n   '''")

                # 🔧 增强清理逻辑
                original_response = response
                response = response.strip()

                self._debug_print(f"After strip(): {len(response)} chars", "DEBUG")

                # 移除 markdown 代码块
                if "```" in response:
                    self._debug_print("Detected markdown code blocks, cleaning...", "JSON")
                    json_match = re.search(
                        r"```(?:json)?\s*(\{.*?\})\s*```",
                        response,
                        re.DOTALL,
                    )
                    if json_match:
                        response = json_match.group(1)
                        self._debug_print("Extracted JSON from code block", "JSON")
                    else:
                        # 如果没有找到完整的代码块，尝试移除所有 ```
                        response = response.replace("```json", "").replace("```", "")
                        self._debug_print("Removed markdown markers", "JSON")

                # 如果响应不是以 { 开头，尝试提取 JSON
                if not response.startswith("{"):
                    self._debug_print(f"Response doesn't start with '{{', starts with: '{response[:20]}...'", "WARN")
                    json_match = re.search(r'\{.*\}', response, re.DOTALL)
                    if json_match:
                        response = json_match.group(0)
                        self._debug_print("Extracted JSON object using regex", "JSON")
                    else:
                        self._debug_print("No JSON object found in response!", "ERROR")

                # 🔧 打印清理后的响应
                print(f"\n   🔧 [CLEANED] Response:\n   '''\n{response}\n   '''")
                print("-" * 50)

                # 🔧 尝试解析 JSON
                self._debug_print("Attempting JSON parsing...", "JSON")

                try:
                    intent = json.loads(response)
                    self._debug_print("JSON parsing successful!", "SUCCESS")

                    # 打印解析后的字段
                    print("\n   📊 [PARSED] Intent Fields:")
                    for key, value in intent.items():
                        print(f"      • {key}: {value}")
                    print()

                except json.JSONDecodeError as je:
                    self._debug_print(f"JSON parsing failed: {je}", "ERROR")
                    self._debug_print(f"Error position: char {je.pos}", "DEBUG")
                    self._debug_print(
                        f"Problematic content around error: '{response[max(0, je.pos - 20):je.pos + 20]}'", "DEBUG")
                    raise

                # 🔧 验证必需字段
                required_fields = ['operation', 'condition', 'scenario_name', 'num_examples']
                missing_fields = [f for f in required_fields if f not in intent]

                if missing_fields:
                    self._debug_print(f"Missing required fields: {missing_fields}", "WARN")
                    self._debug_print("Using fallback parser", "INFO")
                    return self._fallback_parse(user_input)

                self._debug_print("All required fields present", "SUCCESS")
                return intent
            else:
                self._debug_print("LLM has no _call_api method, using fallback", "WARN")
                return self._fallback_parse(user_input)

        except json.JSONDecodeError as e:
            print(f"\n   ⚠️  JSON parsing failed: {e}")
            print(f"   🔄 Using fallback parser")
            return self._fallback_parse(user_input)
        except Exception as e:
            print(f"\n   ⚠️  LLM parsing failed: {e}")
            print(f"   🔄 Using fallback parser")
            import traceback
            if self.debug:
                print(f"   📋 Traceback:")
                traceback.print_exc()
            return self._fallback_parse(user_input)

    def _fallback_parse(self, user_input: str) -> Dict:
        """
        Local fallback parsing without LLM.

        🔧 改进版：更好的默认值和错误处理，添加调试输出
        """
        print("\n   📌 [FALLBACK] Using local parser...")

        text = user_input.lower()
        self._debug_print(f"Input text (lowercase): '{text}'", "DEBUG")

        # Detect operation
        operation = "ADD"  # 默认操作
        for op in self.operations.keys():
            if op.lower() in text:
                operation = op
                self._debug_print(f"Detected operation: {op}", "DEBUG")
                break

        # Detect condition
        condition = "random"  # 默认为随机
        if "equal" in text or "same" in text or "a = b" in text or "a=b" in text:
            condition = "A = B"
        elif "greater" in text or "a > b" in text or "a>b" in text:
            condition = "A > B"
        elif "less" in text or "a < b" in text or "a<b" in text:
            condition = "A < B"
        elif "random" in text or "various" in text:
            condition = "random"

        self._debug_print(f"Detected condition: {condition}", "DEBUG")

        # Extract number of examples
        num_match = re.search(r"(\d+)\s*(?:examples?|cases?|scenarios?)", text)
        num_examples = int(num_match.group(1)) if num_match else 3
        self._debug_print(f"Detected num_examples: {num_examples}", "DEBUG")

        # Infer expected result and flags (simple heuristic)
        expected_result = "0" if condition == "A = B" and operation == "SUB" else None
        zero_flag = True if expected_result == "0" else False

        result = {
            "operation": operation,
            "condition": condition,
            "scenario_name": f"{operation} with {condition}",
            "num_examples": num_examples,
            "expected_result": expected_result,
            "zero_flag": zero_flag,
            "negative_flag": False,
            "tags": ["arithmetic"],
        }

        # 打印 fallback 解析结果
        print("\n   📊 [FALLBACK RESULT] Parsed Intent:")
        for key, value in result.items():
            print(f"      • {key}: {value}")
        print()

        return result

    def _calculate_alu_result(self, a: int, b: int, operation: str) -> Dict:
        """
        Calculate ALU operation result and flags.

        Args:
            a: First operand
            b: Second operand
            operation: ALU operation (ADD, SUB, AND, OR, XOR, NOT, SHL, SHR)

        Returns:
            Dictionary with result, zero_flag, overflow, negative_flag
        """
        # 16-bit ALU simulation
        max_val = 65535  # 2^16 - 1

        if operation == "ADD":
            result = a + b
            overflow = result > max_val
            result = result & max_val  # Keep only 16 bits
        elif operation == "SUB":
            result = a - b
            overflow = result < 0
            result = result & max_val  # Keep only 16 bits
        elif operation == "AND":
            result = a & b
            overflow = False
        elif operation == "OR":
            result = a | b
            overflow = False
        elif operation == "XOR":
            result = a ^ b
            overflow = False
        elif operation == "NOT":
            result = (~a) & max_val
            overflow = False
        elif operation == "SHL":
            result = (a << b) & max_val
            overflow = (a << b) > max_val
        elif operation == "SHR":
            result = a >> b
            overflow = False
        else:
            result = 0
            overflow = False

        zero_flag = (result == 0)
        negative_flag = (result & 0x8000) != 0  # Check MSB for signed

        return {
            "result": result,
            "zero_flag": zero_flag,
            "overflow": overflow,
            "negative_flag": negative_flag,
        }

    def _generate_test_data(
            self, condition: str, num_examples: int, operation: str
    ) -> List[Dict]:
        """Generate test data based on the condition."""
        examples = []

        for _ in range(num_examples):
            if condition == "A = B":
                a = random.randint(0, 65535)
                b = a
            elif condition == "A > B":
                b = random.randint(0, 32767)
                a = random.randint(b + 1, 65535)
            elif condition == "A < B":
                a = random.randint(0, 32767)
                b = random.randint(a + 1, 65535)
            elif condition == "random":
                a = random.randint(0, 65535)
                b = random.randint(0, 65535)
            else:
                # Default to random
                a = random.randint(0, 65535)
                b = random.randint(0, 65535)

            # Calculate result and flags
            calc = self._calculate_alu_result(a, b, operation)

            examples.append({
                "a": a,
                "b": b,
                "result": calc["result"],
                "opcode": self.operations.get(operation, "0000"),
                "zero_flag": calc["zero_flag"],
                "overflow": calc["overflow"],
                "negative_flag": calc["negative_flag"],
            })

        return examples

    def _build_scenario(self, intent: Dict, examples: List[Dict]) -> str:
        """Build the full BDD scenario with Given-When-Then and Examples."""

        operation = intent["operation"]
        scenario_name = intent.get("scenario_name", f"{operation} operation")
        condition = intent.get("condition", "various values")

        # Build scenario header
        scenario = f"""Feature: {operation} Operation Verification

Scenario: {scenario_name}
  Given I have two 16-bit operands
  When I perform the {operation} operation
  Then the result should match the expected output
  And the flags should be set correctly

  Examples:
    | A     | B     | Opcode | Expected_Result | Zero_Flag | Overflow | Negative_Flag |
"""

        # Add examples
        for ex in examples:
            scenario += f"""    | {ex['a']:<5} | {ex['b']:<5} | {ex['opcode']:<6} | {ex['result']:<15} | {str(ex['zero_flag']):<9} | {str(ex['overflow']):<8} | {str(ex['negative_flag']):<13} |
"""

        return scenario


def interactive_mode(project_root: Optional[str] = None):
    """Run the generator in interactive mode with provider selection."""
    print("=" * 70)
    print("🎯 BDD Scenario Generator - Interactive Mode")
    print("=" * 70)
    print("\n📝 Examples of requests:")
    print("   • 'Create ADD scenario with A = B, 3 examples'")
    print("   • 'Generate SUB test where A > B'")
    print("   • 'XOR operation with random values, 5 cases'")
    print("   • 'Test AND when A < B'")
    print("\n💡 Tip: The more specific your request, the better the result!")
    print("Type 'quit' or 'exit' to stop.\n")

    # Provider selection
    llm = None
    use_llm = input("Use LLM provider? (y/n, default: n): ").strip().lower()

    if use_llm == 'y':
        print("\nSelect LLM provider:")
        print("\n🆓 FREE Providers (Recommended):")
        print("  1. Local template (no external API, zero setup)")
        print("  2. Google Gemini (FREE, 60 req/min) ⭐ Recommended")
        print("  3. Groq (FREE, ultra-fast) ⚡")
        print("  4. DeepSeek (FREE, Chinese LLM)")
        print("\n💰 PAID Providers (High Quality):")
        print("  5. OpenAI GPT-5 Series (with official SDK) 🎯 Best Quality")
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
            # OpenAI GPT-5
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                print("\n💡 Get API key at: https://platform.openai.com/api-keys")
                api_key = input("Enter OpenAI API key: ").strip()

            # 选择GPT-5模型
            print("\n🤖 Select GPT-5 model:")
            print("  1. gpt-5-mini (Recommended - Best value)")
            print("  2. gpt-5 (Balanced performance)")
            print("  3. gpt-5.1 (Most capable)")
            print("  4. gpt-5.1-codex (Specialized for code)")
            model_choice = input("Model choice (1-4, default: 1): ").strip() or "1"

            model_map = {
                "1": "gpt-5-mini",
                "2": "gpt-5",
                "3": "gpt-5.1",
                "4": "gpt-5.1-codex"
            }
            model = model_map.get(model_choice, "gpt-5-mini")

            llm = LLMFactory.create_provider("openai", api_key=api_key, model=model)
            print(f"✅ Using OpenAI GPT-5 model: {model}")

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

    generator = SimpleBDDGenerator(llm, project_root=project_root, debug=True)

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
        description="Simple natural language → BDD scenario generator with GPT-5 support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Example usage:

  # Interactive mode (saves to output/bdd)
  python bdd_generator.py

  # Specify project root
  python bdd_generator.py --project-root D:/DE/HdlFormalVerifierLLM/HdlFormalVerifier/AluBDDVerilog

  # Command-line mode with custom output
  python bdd_generator.py --request "Generate SUB test where A equals B" --output-dir ./my_tests

  # Use OpenAI GPT-5 with official SDK (recommended)
  python bdd_generator.py --llm-provider openai --api-key your-key --model gpt-5-mini

  # Use different GPT-5 models
  python bdd_generator.py --llm-provider openai --model gpt-5
  python bdd_generator.py --llm-provider openai --model gpt-5.1
  python bdd_generator.py --llm-provider openai --model gpt-5.1-codex

  # Use alternative names for OpenAI
  python bdd_generator.py --llm-provider gpt --api-key your-key
  python bdd_generator.py --llm-provider gpt5 --api-key your-key

  # Full example
  python bdd_generator.py \\
    --project-root D:/DE/HdlFormalVerifierLLM/HdlFormalVerifier/AluBDDVerilog \\
    --request "Generate SUB test where A equals B with 5 examples" \\
    --llm-provider openai \\
    --model gpt-5-mini
        """,
    )

    parser.add_argument(
        "--request",
        help="Provide a test scenario description directly (non-interactive mode)",
    )

    parser.add_argument(
        "--llm-provider",
        choices=["local", "openai", "gpt", "gpt5", "chatgpt", "gemini", "google", "groq", "deepseek", "claude",
                 "anthropic"],
        default="local",
        help="LLM provider (default: local). Use 'openai', 'gpt', or 'gpt5' for OpenAI GPT-5",
    )

    parser.add_argument(
        "--api-key",
        help="API key for the selected LLM provider",
    )

    parser.add_argument(
        "--model",
        help="Model name (e.g., gpt-5-mini, gpt-5, gpt-5.1, gpt-5.1-codex for GPT-5 series)",
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

    parser.add_argument(
        "--debug",
        action="store_true",
        default=True,
        help="Enable debug output (default: True)",
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
        project_root=args.project_root,
        debug=args.debug
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