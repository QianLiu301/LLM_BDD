"""
Simple Natural Language to BDD Converter
改进版：支持动态路径配置和自动保存到output/bdd目录
Enhanced: OpenAI/ChatGPT with official SDK support - Updated for GPT-5 series
Fixed: Better error handling for JSON parsing and missing fields

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

        🔧 增强版：更好的错误处理和 JSON 验证
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
            if hasattr(self.llm, "_call_api"):
                response = self.llm._call_api(
                    prompt,
                    max_tokens=300,
                    system_prompt="You are a helpful assistant that extracts structured information from user requests and outputs ONLY valid JSON. Never include markdown formatting."
                )

                # 🔍 DEBUG: 打印原始响应
                print(f"   🔍 [DEBUG] LLM Raw Response: {repr(response[:200])}")

                # 🔧 增强清理逻辑
                response = response.strip()

                # 移除 markdown 代码块
                if "```" in response:
                    json_match = re.search(
                        r"```(?:json)?\s*(\{.*?\})\s*```",
                        response,
                        re.DOTALL,
                    )
                    if json_match:
                        response = json_match.group(1)
                    else:
                        # 如果没有找到完整的代码块，尝试移除所有 ```
                        response = response.replace("```json", "").replace("```", "")

                # 如果响应不是以 { 开头，尝试提取 JSON
                if not response.startswith("{"):
                    json_match = re.search(r'\{.*\}', response, re.DOTALL)
                    if json_match:
                        response = json_match.group(0)

                # 🔍 DEBUG: 打印清理后的响应
                print(f"   🔍 [DEBUG] LLM Cleaned Response: {repr(response[:200])}")

                # 解析 JSON
                intent = json.loads(response)

                # 🔧 验证必需字段
                required_fields = ['operation', 'condition', 'scenario_name', 'num_examples']
                missing_fields = [f for f in required_fields if f not in intent]

                if missing_fields:
                    print(f"   ⚠️  JSON missing required fields: {missing_fields}")
                    raise ValueError(f"Missing fields: {missing_fields}")

                # 🔧 标准化 operation 为大写
                if 'operation' in intent:
                    intent['operation'] = intent['operation'].upper()

                # 🔧 确保 num_examples 是整数
                if 'num_examples' in intent:
                    try:
                        intent['num_examples'] = int(intent['num_examples'])
                    except (ValueError, TypeError):
                        intent['num_examples'] = 3

                print(f"   ✅ [SUCCESS] Valid JSON parsed with all required fields")
                return intent

            else:
                print(f"   ⚠️  LLM doesn't have _call_api method, using fallback")
                return self._fallback_parse(user_input)

        except json.JSONDecodeError as e:
            print(f"   ❌ JSON parsing failed: {e}")
            print(f"   📝 Response was: {response[:200]}...")
            print(f"   🔄 Using local fallback parsing")
            return self._fallback_parse(user_input)

        except Exception as e:
            print(f"   ❌ LLM parsing failed: {e}")
            print(f"   🔄 Using local fallback parsing")
            return self._fallback_parse(user_input)

    def _fallback_parse(self, user_input: str) -> Dict:
        """
        Local fallback parsing without LLM.

        🔧 确保返回所有必需字段
        """
        text = user_input.lower()

        # Detect operation
        operation = "ADD"  # 默认操作
        for op in self.operations.keys():
            if op.lower() in text:
                operation = op
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
            "tags": ["arithmetic"],
        }

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


def interactive_mode():
    """Run the generator in interactive mode."""
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

    # 从环境变量或用户输入获取配置
    provider_type = os.getenv("LLM_PROVIDER", "local").lower()
    model = os.getenv("LLM_MODEL")
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("GEMINI_API_KEY")

    # 创建 LLM provider
    if provider_type == "local":
        llm_provider = LocalLLMProvider()
    else:
        try:
            kwargs = {}
            if api_key:
                kwargs["api_key"] = api_key
            if model:
                kwargs["model"] = model

            llm_provider = LLMFactory.create_provider(provider_type, **kwargs)
        except Exception as e:
            print(f"⚠️  Failed to create {provider_type} provider: {e}")
            print("🔄 Falling back to local provider")
            llm_provider = LocalLLMProvider()

    # 创建生成器
    generator = SimpleBDDGenerator(llm_provider=llm_provider)

    while True:
        try:
            user_input = input("💬 Describe your test scenario: ")

            if user_input.lower() in ["quit", "exit", "q"]:
                print("\n👋 Goodbye!")
                break

            if not user_input.strip():
                continue

            # 生成场景
            scenario = generator.generate_from_natural_language(user_input)

            # 显示结果
            print("\n" + "=" * 70)
            print("✅ Generated BDD Scenario:")
            print("=" * 70)
            print(scenario)

            # 自动保存
            filepath = generator.save_scenario(scenario)

            # 询问是否继续
            print("\n" + "=" * 70)

        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
            continue


if __name__ == "__main__":
    interactive_mode()