"""
SPEC Generator - Generate standardized ALU specification from natural language
===============================================================================

ENHANCED VERSION: Multi-LLM support with organized file structure + Proxy Support

This is the ONLY module that uses LLM in the entire pipeline.
It converts user's natural language requirements into a formal specification.

NEW FEATURES:
- ✅ Automatic LLM-specific folder creation (specs/gemini/, specs/groq/, etc.)
- ✅ No hard-coding - dynamic folder based on LLM provider
- ✅ Support for 6 LLM providers with proper organization
- ✅ Timestamp-based file versioning to prevent overwrites
- ✅ Automatic proxy setup for international APIs (Gemini, Groq, OpenAI, Claude)

DIRECTORY STRUCTURE:
    specs/
    ├── local/
    │   ├── spec_20241209_143022.json
    │   └── spec_20241209_143022.txt
    ├── gemini/
    │   ├── spec_20241209_143522.json
    │   └── spec_20241209_143522.txt
    ├── groq/
    ├── deepseek/
    ├── openai/
    └── claude/

Fixed: Enhanced LLM integration with detailed debugging output
Fixed: Better JSON parsing and error handling from bdd_generator
Fixed: LLM-specific file organization
Fixed: Automatic proxy setup for international APIs
"""

import os
import re
import json
import argparse
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime


# ============================================================================
# 代理设置 - 自动从配置文件读取
# ============================================================================
def setup_proxy():
    """
    从配置文件读取并设置代理环境变量

    此函数会自动查找 llm_config.json 并设置代理
    对于国际 API (Gemini, Groq, OpenAI, Claude) 是必需的
    DeepSeek 和 Local 不需要代理
    """
    # 查找配置文件
    config_paths = [
        Path('config/llm_config.json'),
        Path('llm_config.json'),
        Path('../config/llm_config.json'),
        Path(__file__).parent / 'config' / 'llm_config.json',
        Path(__file__).parent.parent / 'config' / 'llm_config.json',
    ]

    config_file = None
    for path in config_paths:
        if path.exists():
            config_file = path
            break

    if not config_file:
        print("⚠️  未找到 llm_config.json，代理未设置")
        print("   如果需要访问国际 API (Gemini/Groq/OpenAI/Claude)，请配置代理")
        return

    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)

        proxy_config = config.get('proxy', {})

        if proxy_config.get('enabled'):
            http_proxy = proxy_config.get('http_proxy', '')
            https_proxy = proxy_config.get('https_proxy', '')

            os.environ['HTTP_PROXY'] = http_proxy
            os.environ['HTTPS_PROXY'] = https_proxy

            print(f"🌐 代理已启用: {https_proxy}")
        else:
            print("ℹ️  代理未启用（配置中 enabled=false）")

    except Exception as e:
        print(f"⚠️  读取代理配置失败: {e}")
        print(f"   配置文件: {config_file}")

# 在模块加载时自动设置代理
setup_proxy()
# ============================================================================

# Import LLM providers
try:
    from llm_providers import LLMProvider, LLMFactory, LocalLLMProvider
    HAS_LLM = True
except ImportError:
    try:
        from HdlFormalVerifier.tests.bdd_test.llm_providers import LLMProvider, LLMFactory, LocalLLMProvider
        HAS_LLM = True
    except ImportError:
        print("⚠️  Warning: llm_providers module not found.")
        print("   Make sure llm_providers.py is in the same directory or in the Python path")
        HAS_LLM = False
        LocalLLMProvider = None


class SpecGenerator:
    """
    Generate standardized ALU specification from user requirements.

    This is the ONLY component that uses LLM in the pipeline.
    After this step, all downstream processing is deterministic.

    ENHANCED: Now supports LLM-specific file organization
    """

    # Standard ALU operations mapping
    STANDARD_OPERATIONS = {
        "ADD": {"opcode": "0000", "description": "Addition (A + B)"},
        "SUB": {"opcode": "0001", "description": "Subtraction (A - B)"},
        "AND": {"opcode": "0010", "description": "Bitwise AND (A & B)"},
        "OR":  {"opcode": "0011", "description": "Bitwise OR (A | B)"},
        "XOR": {"opcode": "0100", "description": "Bitwise XOR (A ^ B)"},
        "NOT": {"opcode": "0101", "description": "Bitwise NOT (~A)"},
        "SHL": {"opcode": "0110", "description": "Shift Left (A << B)"},
        "SHR": {"opcode": "0111", "description": "Shift Right (A >> B)"},
    }

    # LLM provider name mapping
    LLM_NAMES = {
        'LocalLLMProvider': 'local',
        'GeminiProvider': 'gemini',
        'GroqProvider': 'groq',
        'DeepSeekProvider': 'deepseek',
        'OpenAIProvider': 'openai',
        'ClaudeProvider': 'claude',
    }

    # LLMs that require proxy (international APIs)
    PROXY_REQUIRED_LLMS = ['gemini', 'groq', 'openai', 'claude']
    # LLMs that don't need proxy (domestic or local)
    NO_PROXY_LLMS = ['deepseek', 'local']

    def __init__(
        self,
        output_dir: Optional[str] = None,
        project_root: Optional[str] = None,
        llm_provider: Optional['LLMProvider'] = None,
        debug: bool = True
    ):
        """
        Initialize SPEC generator

        Args:
            output_dir: Base directory to save generated SPEC files
            project_root: Project root directory
            llm_provider: LLM provider instance (optional)
            debug: Enable debug output
        """
        self.llm = llm_provider
        self.debug = debug

        # Detect LLM provider name
        self.llm_name = self._detect_llm_name()

        # Check proxy requirements
        self._check_proxy_requirements()

        # Setup output paths (now includes LLM-specific subdirectory)
        self._setup_paths(project_root, output_dir)

    def _check_proxy_requirements(self):
        """
        Check if current LLM requires proxy and warn if not set
        """
        # Skip check for local or no LLM
        if self.llm_name in self.NO_PROXY_LLMS:
            if self.debug:
                print(f"   ℹ️  {self.llm_name} 不需要代理")
            return

        # Check if proxy is set for international APIs
        if self.llm_name in self.PROXY_REQUIRED_LLMS:
            https_proxy = os.environ.get('HTTPS_PROXY') or os.environ.get('https_proxy')

            if not https_proxy:
                print(f"\n⚠️  警告: {self.llm_name} 需要代理访问国际 API")
                print(f"   请确保已设置代理，否则可能无法访问")
                print(f"   方法1: 配置 config/llm_config.json 中的 proxy 设置")
                print(f"   方法2: 设置环境变量 HTTPS_PROXY")
                print()
            else:
                if self.debug:
                    print(f"   ✅ 代理已设置: {https_proxy}")

    def _detect_llm_name(self) -> str:
        """
        Automatically detect LLM provider name from instance

        Returns:
            LLM provider name (e.g., 'gemini', 'groq', 'local')
        """
        if self.llm is None:
            return 'local'

        # Get class name
        class_name = type(self.llm).__name__

        # Map to standard name
        llm_name = self.LLM_NAMES.get(class_name, 'unknown')

        if self.debug:
            print(f"   🔍 Detected LLM provider: {class_name} → {llm_name}")

        return llm_name

    def _setup_paths(self, project_root: Optional[str], output_dir: Optional[str]):
        """
        Setup output directory paths with LLM-specific subdirectories

        Structure: <base_dir>/specs/<llm_name>/
        """
        # Determine base specs directory
        if output_dir:
            base_dir = Path(output_dir)
        elif project_root:
            base_dir = Path(project_root) / "specs"
        else:
            # Try to find specs directory dynamically
            current = Path.cwd()

            # Check common locations
            possible_paths = [
                current / "specs",
                current / "src" / "specs",
                current.parent / "specs",
                current.parent / "src" / "specs",
            ]

            for path in possible_paths:
                if path.exists():
                    base_dir = path
                    break
            else:
                # Default to current/specs
                base_dir = current / "specs"

        # Create LLM-specific subdirectory
        self.output_dir = base_dir / self.llm_name

        # Ensure directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"📁 Spec output directory: {self.output_dir.absolute()}")
        print(f"   LLM provider: {self.llm_name}")

    def _debug_print(self, message: str, level: str = "INFO"):
        """Unified debug output method"""
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

    def generate_spec(self, user_input: str) -> Dict:
        """
        Generate formal specification from user input

        This is the main entry point. It uses LLM to understand the user's
        natural language requirements and converts them to a structured spec.

        即使 LLM API 调用失败，也会生成一个标记为失败的 spec

        Args:
            user_input: Natural language requirements from user

        Returns:
            Dictionary containing the generated specification (may be marked as failed)
        """
        print(f"\n🔍 Processing requirements: {user_input}")
        print("=" * 70)

        # Step 1: Use LLM to understand and structure requirements
        print("\nStep 1: Understanding requirements with LLM...")
        spec_content = self._understand_requirements(user_input)

        # 检查是否失败
        is_failed = spec_content.get("api_call_failed", False)
        if is_failed:
            print(f"\n   ⚠️  API call FAILED: {spec_content.get('error', 'Unknown error')}")
            print(f"   ⚠️  Will save FAILED spec for debugging")

        # Step 2: Parse and validate the specification
        print("\nStep 2: Parsing specification...")
        spec_dict = self._parse_spec(spec_content, user_input)

        # Step 3: Validate completeness
        print("\nStep 3: Validating specification...")
        self._validate_spec(spec_dict)

        if is_failed:
            print(f"\n   ⚠️  Specification marked as FAILED")
            print(f"   ⚠️  LLM Provider: {self.llm_name}")
            print(f"   ⚠️  Files will be saved but marked as incomplete")
        else:
            print(f"\n   ✅ Specification generated successfully")
            print(f"   ✅ Bitwidth: {spec_dict['bitwidth']}-bit")
            print(f"   ✅ Operations: {len(spec_dict['operations'])} defined")
            print(f"   ✅ LLM Provider: {self.llm_name}")

        return spec_dict

    def save_spec(self, spec_dict: Dict, filename_prefix: str = "spec") -> str:
        """
        Save specification to both JSON and TXT formats

        Files are saved to: specs/<llm_name>/spec_<timestamp>.json|txt

        Args:
            spec_dict: Specification dictionary
            filename_prefix: Prefix for filename (default: "spec")

        Returns:
            Path to saved TXT file
        """
        # Generate timestamp-based filename to avoid overwrites
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{filename_prefix}_{timestamp}"

        # Save paths
        json_path = self.output_dir / f"{base_filename}.json"
        txt_path = self.output_dir / f"{base_filename}.txt"

        # Add metadata
        spec_dict['metadata'] = {
            'generated_by': f'spec_generator ({self.llm_name})',
            'timestamp': timestamp,
            'llm_provider': self.llm_name,
        }

        # Save JSON
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(spec_dict, f, indent=2, ensure_ascii=False)
        print(f"   ✅ JSON saved: {json_path}")

        # Save TXT
        txt_content = self._format_spec_txt(spec_dict)
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(txt_content)
        print(f"   ✅ TXT saved: {txt_path}")

        return str(txt_path)

    # ========================================================================
    # Copy the rest of the methods from the original file
    # (这里需要保留原文件中的所有其他方法)
    # ========================================================================

    def _understand_requirements(self, user_input: str) -> Dict:
        """
        Use LLM to understand user requirements and generate structured spec

        注意：即使 API 调用失败，也会返回一个包含错误信息的 spec，
        而不是 fallback 到 local 模式
        """
        prompt = self._create_spec_prompt(user_input)

        try:
            # Check LLM provider type
            if self.llm is None:
                self._debug_print("No LLM provider, using local template", "INFO")
                return self._create_basic_spec(user_input)

            llm_type = type(self.llm).__name__
            self._debug_print(f"LLM Provider: {llm_type}", "INFO")

            # Skip LLM for LocalLLMProvider
            if HAS_LLM and isinstance(self.llm, LocalLLMProvider):
                self._debug_print("Using local template mode", "INFO")
                return self._create_basic_spec(user_input)

            # Call LLM API
            self._debug_print("Calling LLM API...", "STEP")

            if hasattr(self.llm, "_call_api"):
                response = self.llm._call_api(
                    prompt,
                    max_tokens=2000,
                    system_prompt="You are a hardware design specification expert. Generate detailed, structured ALU specifications."
                )
            elif hasattr(self.llm, "call_api"):
                response = self.llm.call_api(
                    prompt,
                    max_tokens=2000
                )
            else:
                self._debug_print("LLM does not have call_api method", "WARN")
                # 不 fallback，而是返回错误信息
                return self._create_failed_spec(
                    user_input,
                    error_message="LLM provider does not have call_api method"
                )

            self._debug_print(f"Raw LLM Response (first 500 chars):", "RAW")
            self._debug_print(response[:500], "RAW")

            # 检查是否是 fallback response（API 实际失败了）
            if self._is_fallback_response(response):
                self._debug_print("Detected FALLBACK response from LLM provider", "ERROR")
                self._debug_print("This indicates API call actually FAILED", "ERROR")

                # 提取错误信息
                error_msg = self._extract_error_from_fallback(response)

                # 视为失败
                return self._create_failed_spec(user_input, error_message=error_msg)

            return {"llm_response": response, "user_input": user_input, "status": "success"}

        except Exception as e:
            self._debug_print(f"LLM call failed: {str(e)}", "ERROR")
            # 不 fallback 到 local，而是创建失败的 spec
            self._debug_print("Creating FAILED spec (not falling back to local)", "WARN")
            import traceback
            traceback.print_exc()

            # 返回包含错误信息的 spec
            return self._create_failed_spec(user_input, error_message=str(e))

    def _create_failed_spec(self, user_input: str, error_message: str) -> Dict:
        """
        创建一个标记为失败的 spec

        这个 spec 会被保存到对应 LLM 的文件夹，
        标记为 API 调用失败，但仍然包含基本信息

        Args:
            user_input: 用户输入
            error_message: 错误信息

        Returns:
            包含失败信息的 spec dict
        """
        self._debug_print(f"Creating FAILED spec: {error_message}", "ERROR")

        # 尝试从用户输入中提取基本信息
        bitwidth = 16  # default
        bitwidth_patterns = [
            r'(\d+)[\s-]*bit',
            r'(\d+)b\s',
            r'(\d+)\s*bits?'
        ]
        for pattern in bitwidth_patterns:
            match = re.search(pattern, user_input.lower())
            if match:
                bitwidth = int(match.group(1))
                break

        # 创建失败的响应
        failed_response = f"""
API_CALL_FAILED: {error_message}

User Input: {user_input}

This is a FAILED API call. The spec below is a fallback structure.

BITWIDTH: {bitwidth}
OPERATIONS:
  [API call failed - no operations generated]
"""

        return {
            "llm_response": failed_response,
            "user_input": user_input,
            "status": "failed",
            "error": error_message,
            "api_call_failed": True
        }

    def _is_fallback_response(self, response: str) -> bool:
        """
        检测 LLM 响应是否是 fallback response（实际 API 调用失败）

        llm_providers 中的某些 provider 在 API 失败后会返回 fallback response，
        我们需要检测这种情况并标记为失败

        检测标志：
        - 包含 "Fallback response due to API failure"
        - 包含 "error" 字段且值提到 fallback
        - 包含 "operation": "UNKNOWN"

        Args:
            response: LLM 返回的响应字符串

        Returns:
            bool: True 表示是 fallback response (实际失败)
        """
        if not response:
            return False

        response_lower = response.lower()

        # 检测明确的 fallback 标志
        fallback_indicators = [
            "fallback response due to api failure",
            "fallback response",
            '"error": "fallback',
            "operation\": \"unknown",
            "api failure",
        ]

        for indicator in fallback_indicators:
            if indicator in response_lower:
                return True

        # 尝试解析 JSON 检测
        try:
            import json
            data = json.loads(response)

            # 检查是否有 error 字段提到 fallback
            if isinstance(data, dict):
                error_field = data.get('error', '')
                if error_field and 'fallback' in str(error_field).lower():
                    return True

                # 检查是否有 operation: UNKNOWN
                if data.get('operation') == 'UNKNOWN':
                    return True
        except:
            pass

        return False

    def _extract_error_from_fallback(self, response: str) -> str:
        """
        从 fallback response 中提取错误信息

        Args:
            response: Fallback response 字符串

        Returns:
            str: 错误信息
        """
        try:
            import json
            data = json.loads(response)

            if isinstance(data, dict):
                # 尝试从 error 字段提取
                error_msg = data.get('error', '')
                if error_msg:
                    return error_msg
        except:
            pass

        # 如果无法提取，返回通用消息
        return "LLM API call failed (fallback response detected)"

    def _create_spec_prompt(self, user_input: str) -> str:
        """Create prompt for LLM to generate specification"""
        return f"""You are a hardware specification expert. Convert the following user requirements into a formal ALU specification.

    User Requirements: {user_input}

    Please return your response as a JSON-formatted specification with this structure:

    BITWIDTH: <8/16/32/64>
    OPERATIONS:
      ADD: Addition (A + B)
      SUB: Subtraction (A - B)
      AND: Bitwise AND (A & B)
      OR: Bitwise OR (A | B)
      XOR: Bitwise XOR (A ^ B)
      NOT: Bitwise NOT (~A)
      [Include only operations mentioned in requirements]

    IMPORTANT:
    1. Use ONLY standard operations: ADD, SUB, AND, OR, XOR, NOT, SHL, SHR
    2. Detect bitwidth from requirements (default: 16 if not specified)
    3. Include operation descriptions
    4. Return the specification in JSON format

    Return ONLY the specification in the format above."""

    def _create_basic_spec(self, user_input: str) -> Dict:
        """
        Create basic specification from user input without LLM
        """
        # Extract bitwidth
        bitwidth = 16  # default
        bitwidth_patterns = [
            r'(\d+)[\s-]*bit',
            r'(\d+)b\s',
            r'(\d+)\s*bits?'
        ]
        for pattern in bitwidth_patterns:
            match = re.search(pattern, user_input.lower())
            if match:
                bitwidth = int(match.group(1))
                break

        # Extract operations
        operations = []
        input_lower = user_input.lower()

        operation_keywords = {
            'ADD': ['add', 'addition', 'plus', 'sum'],
            'SUB': ['sub', 'subtract', 'minus', 'difference'],
            'AND': ['and', 'logical and'],
            'OR': ['or', 'logical or'],
            'XOR': ['xor', 'exclusive or'],
            'NOT': ['not', 'invert', 'complement'],
            'SHL': ['shift left', 'shl', 'left shift'],
            'SHR': ['shift right', 'shr', 'right shift'],
        }

        for op, keywords in operation_keywords.items():
            if any(kw in input_lower for kw in keywords):
                operations.append(op)

        # Default to basic operations if none found
        if not operations:
            operations = ['ADD', 'SUB', 'AND', 'OR']

        # Create spec text
        spec_lines = [
            f"BITWIDTH: {bitwidth}",
            "OPERATIONS:"
        ]

        for op in operations:
            if op in self.STANDARD_OPERATIONS:
                desc = self.STANDARD_OPERATIONS[op]['description']
                spec_lines.append(f"  {op}: {desc}")

        return {
            "llm_response": "\n".join(spec_lines),
            "user_input": user_input
        }

    def _parse_spec(self, spec_content: Dict, user_input: str) -> Dict:
        """
        Parse specification content into structured dictionary

        处理成功和失败的 API 响应
        """
        llm_response = spec_content.get("llm_response", "")

        # 检查是否是失败的 API 调用
        is_failed = spec_content.get("api_call_failed", False)
        error_message = spec_content.get("error", "")

        spec_dict = {
            "bitwidth": 16,
            "operations": {},
            "description": user_input,
        }

        # 如果 API 调用失败，添加失败标记
        if is_failed:
            spec_dict["api_call_failed"] = True
            spec_dict["error_message"] = error_message
            spec_dict["status"] = "failed"

            self._debug_print(f"Parsing FAILED spec: {error_message}", "WARN")

            # 仍然尝试从用户输入中提取基本信息
            for pattern in [r'(\d+)[\s-]*bit', r'(\d+)b\s']:
                match = re.search(pattern, user_input.lower())
                if match:
                    spec_dict["bitwidth"] = int(match.group(1))
                    break

            # 不添加任何操作，保持为空
            self._debug_print(f"FAILED spec will have NO operations", "WARN")
            return spec_dict

        # 正常解析（成功的 API 响应）
        # Extract bitwidth
        bitwidth_match = re.search(r'BITWIDTH:\s*(\d+)', llm_response, re.IGNORECASE)
        if bitwidth_match:
            spec_dict["bitwidth"] = int(bitwidth_match.group(1))
        else:
            # Try to extract from user input
            for pattern in [r'(\d+)[\s-]*bit', r'(\d+)b\s']:
                match = re.search(pattern, user_input.lower())
                if match:
                    spec_dict["bitwidth"] = int(match.group(1))
                    break

        # Extract operations
        operations_section = re.search(
            r'OPERATIONS:(.*?)(?=\n\n|\Z)',
            llm_response,
            re.DOTALL | re.IGNORECASE
        )

        if operations_section:
            ops_text = operations_section.group(1)
            for line in ops_text.split('\n'):
                line = line.strip()
                if not line:
                    continue

                # Parse: "ADD: Addition (A + B)"
                op_match = re.match(r'([A-Z]+):\s*(.+)', line)
                if op_match:
                    op_name = op_match.group(1).upper()
                    op_desc = op_match.group(2).strip()

                    if op_name in self.STANDARD_OPERATIONS:
                        spec_dict["operations"][op_name] = {
                            "opcode": self.STANDARD_OPERATIONS[op_name]["opcode"],
                            "description": op_desc
                        }

        # If no operations found, use defaults (only for successful API calls)
        if not spec_dict["operations"] and not is_failed:
            for op in ['ADD', 'SUB', 'AND', 'OR']:
                spec_dict["operations"][op] = {
                    "opcode": self.STANDARD_OPERATIONS[op]["opcode"],
                    "description": self.STANDARD_OPERATIONS[op]["description"]
                }

        return spec_dict

    def _validate_spec(self, spec_dict: Dict):
        """
        Validate specification completeness

        对于失败的 spec，只验证基本结构，不要求 operations
        """
        # 检查是否是失败的 spec
        is_failed = spec_dict.get("api_call_failed", False)

        if is_failed:
            self._debug_print("Validating FAILED spec (relaxed validation)", "WARN")

            if not spec_dict.get("bitwidth"):
                print(f"   ⚠️  Failed spec: No bitwidth")

            if not spec_dict.get("operations"):
                print(f"   ⚠️  Failed spec: No operations (expected)")

            print(f"   ⚠️  Specification marked as FAILED")
            print(f"   ⚠️  Error: {spec_dict.get('error_message', 'Unknown')}")
            return

        # 正常验证（成功的 spec）
        if not spec_dict.get("bitwidth"):
            raise ValueError("Bitwidth not specified")

        if spec_dict["bitwidth"] not in [8, 16, 32, 64]:
            print(f"   ⚠️  Unusual bitwidth: {spec_dict['bitwidth']}")

        if not spec_dict.get("operations"):
            raise ValueError("No operations defined")

        print(f"   ✅ Specification valid")

    def _format_spec_txt(self, spec_dict: Dict) -> str:
        """
        Format specification as human-readable text

        对于失败的 spec，会清晰标记为 API 调用失败
        """
        lines = []
        lines.append("=" * 80)

        # 检查是否是失败的 spec
        is_failed = spec_dict.get("api_call_failed", False)

        if is_failed:
            lines.append("⚠️  API CALL FAILED - INCOMPLETE SPECIFICATION")
            lines.append("=" * 80)
        else:
            lines.append("ALU SPECIFICATION")
            lines.append("=" * 80)

        lines.append("")

        # Metadata
        if 'metadata' in spec_dict:
            meta = spec_dict['metadata']
            lines.append(f"Generated by: {meta.get('generated_by', 'unknown')}")
            lines.append(f"Timestamp: {meta.get('timestamp', 'unknown')}")
            lines.append(f"LLM Provider: {meta.get('llm_provider', 'unknown')}")
            lines.append("")

        # 如果是失败的 spec，显示错误信息
        if is_failed:
            lines.append("⚠️  STATUS: FAILED")
            lines.append(f"Error: {spec_dict.get('error_message', 'Unknown error')}")
            lines.append("")
            lines.append("This specification is INCOMPLETE because the LLM API call failed.")
            lines.append("The file is saved for debugging purposes.")
            lines.append("")

        lines.append(f"BITWIDTH: {spec_dict['bitwidth']}")
        lines.append("")

        if is_failed:
            lines.append("OPERATIONS: NONE (API call failed)")
            lines.append("-" * 80)
            lines.append("  No operations were generated due to API failure.")
        else:
            lines.append("OPERATIONS:")
            lines.append("-" * 80)

            if spec_dict["operations"]:
                for op_name, op_info in spec_dict["operations"].items():
                    lines.append(f"  {op_name:6} (opcode: {op_info['opcode']})")
                    lines.append(f"         {op_info['description']}")
                    lines.append("")
            else:
                lines.append("  (No operations defined)")

        lines.append("")
        lines.append("=" * 80)
        lines.append(f"Description: {spec_dict.get('description', 'N/A')}")
        lines.append("=" * 80)

        return "\n".join(lines)


def interactive_mode(project_root: Optional[str] = None):
    """Interactive mode with LLM provider selection"""
    print("=" * 70)
    print("🎯 SPEC Generator - Interactive Mode (Multi-LLM Support)")
    print("=" * 70)
    print("\n📝 Examples of requirements:")
    print("   • 'I need a 16-bit ALU with ADD, SUB, AND, OR operations'")
    print("   • '8-bit ALU with full arithmetic and logic operations'")
    print("   • '32-bit ALU supporting basic math'")
    print("\n💡 Tip: The more detailed your requirements, the better the specification!")
    print("Type 'quit' or 'exit' to stop.\n")

    # Check proxy status
    https_proxy = os.environ.get('HTTPS_PROXY') or os.environ.get('https_proxy')
    if https_proxy:
        print(f"🌐 代理状态: 已设置 ({https_proxy})")
    else:
        print("⚠️  代理状态: 未设置")
        print("   注意: Gemini, Groq, OpenAI, Claude 需要代理访问")
        print("   DeepSeek 和 Local 不需要代理\n")

    # LLM Provider selection
    llm = None
    if HAS_LLM:
        use_llm = input("Use LLM provider for enhanced generation? (y/n, default: n): ").strip().lower()

        if use_llm == 'y':
            print("\nSelect LLM provider:")
            print("\n🆓 FREE Providers (Recommended):")
            print("  1. Local template (no external API, zero setup)")
            print("  2. Google Gemini (FREE, 60 req/min) ⭐ [需要代理]")
            print("  3. Groq (FREE, ultra-fast) ⚡ [需要代理]")
            print("  4. DeepSeek (FREE, Chinese LLM) [不需要代理] 🇨🇳")
            print("\n💰 PAID Providers (High Quality):")
            print("  5. OpenAI GPT-5 Series 🎯 [需要代理]")
            print("  6. Anthropic Claude (premium) [需要代理]")

            choice = input("\nYour choice (1-6): ").strip()

            if choice == "2":
                api_key = os.getenv("GEMINI_API_KEY")
                if not api_key:
                    print("\n💡 Get FREE key at: https://makersuite.google.com/app/apikey")
                    api_key = input("Enter Gemini API key: ").strip()
                llm = LLMFactory.create_provider("gemini", api_key=api_key)
            elif choice == "3":
                api_key = os.getenv("GROQ_API_KEY")
                if not api_key:
                    print("\n💡 Get FREE key at: https://console.groq.com/keys")
                    api_key = input("Enter Groq API key: ").strip()
                llm = LLMFactory.create_provider("groq", api_key=api_key)
            elif choice == "4":
                api_key = os.getenv("DEEPSEEK_API_KEY")
                if not api_key:
                    print("\n💡 Get key at: https://platform.deepseek.com/")
                    api_key = input("Enter DeepSeek API key: ").strip()
                llm = LLMFactory.create_provider("deepseek", api_key=api_key)
            elif choice == "5":
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    print("\n💡 Get API key at: https://platform.openai.com/api-keys")
                    api_key = input("Enter OpenAI API key: ").strip()

                print("\n🤖 Select GPT-5 model:")
                print("  1. gpt-5-mini (Recommended)")
                print("  2. gpt-5")
                print("  3. gpt-5.1")
                print("  4. gpt-5.1-codex")
                model_choice = input("Model choice (1-4, default: 1): ").strip() or "1"

                model_map = {"1": "gpt-5-mini", "2": "gpt-5", "3": "gpt-5.1", "4": "gpt-5.1-codex"}
                model = model_map.get(model_choice, "gpt-5-mini")

                llm = LLMFactory.create_provider("openai", api_key=api_key, model=model)
                print(f"✅ Using OpenAI GPT-5 model: {model}")

            elif choice == "6":
                api_key = os.getenv("ANTHROPIC_API_KEY")
                if not api_key:
                    print("\n💡 Get API key at: https://console.anthropic.com/")
                    api_key = input("Enter Anthropic API key: ").strip()
                llm = LLMFactory.create_provider("claude", api_key=api_key)
            else:
                llm = LocalLLMProvider() if LocalLLMProvider else None
        else:
            llm = LocalLLMProvider() if LocalLLMProvider else None
    else:
        print("⚠️  LLM providers not available, using local template mode")
        llm = None

    # Create generator
    generator = SpecGenerator(project_root=project_root, llm_provider=llm, debug=True)

    print("\n" + "=" * 70)
    print("🚀 Ready! Enter your ALU requirements.\n")

    while True:
        try:
            user_input = input("\n💬 Enter your ALU requirements: ").strip()

            if user_input.lower() in ["quit", "exit", "q"]:
                print("\n👋 Bye!")
                break

            if not user_input:
                continue

            # Generate specification
            spec_dict = generator.generate_spec(user_input)

            # Save specification
            spec_path = generator.save_spec(spec_dict)

            print("\n" + "=" * 70)
            print("📋 NEXT STEPS (Independent Generation Workflow):")
            print("=" * 70)
            print(f"1. Review specification: {spec_path}")
            print(f"2. Generate BDD:         python bdd_generator.py --spec-dir {generator.output_dir}")
            print(f"3. Generate Verilog:     python alu_generator.py --spec-dir {generator.output_dir}")
            print(f"   [Note: Steps 2 and 3 are INDEPENDENT - no LLM required]")
            print("=" * 70)

        except KeyboardInterrupt:
            print("\n\n⚠️ Interrupted by user")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Generate formal ALU specification from natural language requirements',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Interactive mode (recommended)
  python spec_generator.py

  # Command line mode
  python spec_generator.py -i "16-bit ALU with ADD, SUB, AND, OR"

  # With LLM
  python spec_generator.py -i "8-bit ALU" --llm openai --model gpt-5-mini

  # Specify output directory
  python spec_generator.py -i "16-bit ALU" -o ./my_specs
        '''
    )

    parser.add_argument('-i', '--input', help='User requirements (natural language)')
    parser.add_argument('-o', '--output-dir', help='Output directory for spec files')
    parser.add_argument('--project-root', help='Project root directory')
    parser.add_argument('--llm', default='local',
                        choices=['local', 'openai', 'gemini', 'groq', 'deepseek', 'claude'])
    parser.add_argument('--model', help='Model name for LLM provider')
    parser.add_argument('--api-key', help='API key for LLM provider')
    parser.add_argument('--debug', action='store_true', default=True, help='Enable debug output')

    args = parser.parse_args()

    # If no input, enter interactive mode
    if not args.input:
        interactive_mode(project_root=args.project_root)
        return

    # Create LLM provider
    llm = None
    if HAS_LLM and args.llm != 'local':
        llm_config = {}
        if args.api_key:
            llm_config['api_key'] = args.api_key
        if args.model:
            llm_config['model'] = args.model

        # Try environment variables
        env_keys = {
            'openai': 'OPENAI_API_KEY',
            'gemini': 'GEMINI_API_KEY',
            'groq': 'GROQ_API_KEY',
            'deepseek': 'DEEPSEEK_API_KEY',
            'claude': 'ANTHROPIC_API_KEY'
        }
        if 'api_key' not in llm_config and args.llm in env_keys:
            llm_config['api_key'] = os.getenv(env_keys[args.llm])

        try:
            llm = LLMFactory.create_provider(args.llm, **llm_config)
        except Exception as e:
            print(f"⚠️  Failed to create LLM provider: {e}")
            llm = None

    # Create generator and run
    generator = SpecGenerator(
        output_dir=args.output_dir,
        project_root=args.project_root,
        llm_provider=llm,
        debug=args.debug
    )

    spec_dict = generator.generate_spec(args.input)
    spec_path = generator.save_spec(spec_dict)

    print("\n" + "=" * 70)
    print("📋 NEXT STEPS:")
    print("=" * 70)
    print(f"1. Review: {spec_path}")
    print(f"2. BDD:    python bdd_generator.py --spec-dir {generator.output_dir}")
    print(f"3. Verilog: python alu_generator.py --spec-dir {generator.output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()