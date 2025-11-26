"""
SPEC Generator - Generate standardized ALU specification from natural language
===============================================================================

This is the ONLY module that uses LLM in the entire pipeline.
It converts user's natural language requirements into a formal specification.

PURPOSE:
This file creates a standardized SPEC document that serves as the single source
of truth for BOTH BDD scenario generation AND Verilog DUT generation.
This ensures independence between test generation and design generation.

ARCHITECTURE:
    Natural Language Input
           │
           ▼
    ┌──────────────────┐
    │  spec_generator  │  ◄── LLM used here ONLY
    │    (this file)   │
    └────────┬─────────┘
             │
             ▼
        spec.txt / spec.json
             │
    ┌────────┴────────┐
    ▼                 ▼
 bdd_gen.py      alu_gen.py    ◄── No LLM (deterministic)
    │                 │
    ▼                 ▼
 .feature           ALU.v

Fixed: Enhanced LLM integration with detailed debugging output
Fixed: Better JSON parsing and error handling from bdd_generator
"""

import os
import re
import json
import argparse
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime

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
            output_dir: Directory to save generated SPEC files
            project_root: Project root directory
            llm_provider: LLM provider instance (optional)
            debug: Enable debug output
        """
        self.llm = llm_provider
        self.debug = debug

        # Setup output paths
        self._setup_paths(project_root, output_dir)

    def _setup_paths(self, project_root: Optional[str], output_dir: Optional[str]):
        """Setup output directory paths"""
        if output_dir:
            self.output_dir = Path(output_dir)
        elif project_root:
            self.output_dir = Path(project_root) / "src" / "specs"
        else:
            # Try to find specs directory dynamically
            current = Path.cwd()

            # Check common locations
            possible_paths = [
                current / "src" / "specs",
                current / "specs",
                current.parent / "src" / "specs",
                current.parent / "specs",
            ]

            for path in possible_paths:
                if path.exists():
                    self.output_dir = path
                    break
            else:
                # Default to current/specs
                self.output_dir = current / "specs"

        # Ensure directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 Spec output directory: {self.output_dir.absolute()}")

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

        Args:
            user_input: Natural language requirements from user

        Returns:
            Dictionary containing the generated specification
        """
        print(f"\n🔍 Processing requirements: {user_input}")
        print("=" * 70)

        # Step 1: Use LLM to understand and structure requirements
        print("\nStep 1: Understanding requirements with LLM...")
        spec_content = self._understand_requirements(user_input)

        # Step 2: Parse and validate the specification
        print("\nStep 2: Parsing specification...")
        spec_dict = self._parse_spec(spec_content, user_input)

        # Step 3: Validate completeness
        print("\nStep 3: Validating specification...")
        self._validate_spec(spec_dict)

        print(f"\n   ✅ Specification generated successfully")
        print(f"   ✅ Bitwidth: {spec_dict['bitwidth']}-bit")
        print(f"   ✅ Operations: {len(spec_dict['operations'])} defined")

        return spec_dict

    def _understand_requirements(self, user_input: str) -> Dict:
        """
        Use LLM to understand user requirements and generate structured spec

        🔧 Enhanced version with better error handling and debug output
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
                    system_prompt="You are a hardware design specification expert. Generate detailed, structured ALU specifications. Output valid JSON only."
                )
            else:
                response = self._call_llm_generic(prompt)

            # Debug output
            print("\n" + "-" * 50)
            print("   📤 LLM API RESPONSE DETAILS:")
            print("-" * 50)

            self._debug_print(f"Response Type: {type(response).__name__}", "DEBUG")
            self._debug_print(f"Response Length: {len(response)} chars", "DEBUG")

            if len(response) < 500:
                print(f"   📝 [RAW] Full Response:\n   '''\n{response}\n   '''")
            else:
                print(f"   📝 [RAW] Response Preview:\n   '''\n{response[:500]}...\n   '''")

            # Clean and parse response
            cleaned = self._clean_llm_response(response)

            print(f"\n   🔧 [CLEANED] Response length: {len(cleaned)} chars")
            print("-" * 50)

            # Try to parse as JSON first
            try:
                spec_dict = json.loads(cleaned)
                self._debug_print("JSON parsing successful!", "SUCCESS")
                return spec_dict
            except json.JSONDecodeError:
                self._debug_print("Response is not JSON, treating as text spec", "INFO")
                return self._parse_text_spec(cleaned, user_input)

        except Exception as e:
            self._debug_print(f"LLM call failed: {e}", "ERROR")
            self._debug_print("Using fallback template", "INFO")
            if self.debug:
                import traceback
                traceback.print_exc()
            return self._create_basic_spec(user_input)

    def _call_llm_generic(self, prompt: str) -> str:
        """Generic LLM call for different provider types"""

        # OpenAI provider
        if hasattr(self.llm, 'client') and hasattr(self.llm.client, 'chat'):
            response = self.llm.client.chat.completions.create(
                model=self.llm.model,
                messages=[
                    {"role": "system", "content": "You are a hardware design specification expert."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2000
            )
            return response.choices[0].message.content

        # Gemini provider
        elif hasattr(self.llm, 'client') and hasattr(self.llm, 'models_to_try'):
            response = self.llm.client.models.generate_content(
                model=self.llm.models_to_try[0],
                contents=prompt,
            )
            return response.text

        # REST API providers (Groq, DeepSeek, Claude)
        elif hasattr(self.llm, 'api_url'):
            import requests
            headers = {
                "Authorization": f"Bearer {self.llm.api_key}",
                "Content-Type": "application/json"
            }

            # Claude uses different header
            if 'anthropic' in self.llm.api_url:
                headers = {
                    "x-api-key": self.llm.api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json"
                }
                payload = {
                    "model": self.llm.model,
                    "max_tokens": 2000,
                    "messages": [{"role": "user", "content": prompt}]
                }
            else:
                payload = {
                    "model": self.llm.model,
                    "messages": [
                        {"role": "system", "content": "You are a hardware design specification expert."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 2000
                }

            response = requests.post(self.llm.api_url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()

            # Different response structures
            if 'content' in result:  # Claude
                return result['content'][0]['text']
            else:  # OpenAI-compatible
                return result['choices'][0]['message']['content']

        raise Exception("Unable to determine provider interface")

    def _clean_llm_response(self, response: str) -> str:
        """Clean LLM response for parsing"""
        response = response.strip()

        # Remove markdown code blocks
        if "```" in response:
            json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
            if json_match:
                response = json_match.group(1)
            else:
                response = response.replace("```json", "").replace("```", "")

        # Try to extract JSON object
        if not response.startswith("{"):
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                response = json_match.group(0)

        return response.strip()

    def _create_spec_prompt(self, user_input: str) -> str:
        """Create prompt for LLM to generate specification"""
        return f"""You are a hardware design specification expert. Convert the following user requirements into a formal, detailed ALU specification.

User Requirements: "{user_input}"

Generate a JSON specification with this EXACT structure:
{{
    "module_name": "alu_Nbit",
    "bitwidth": N,
    "description": "N-bit Arithmetic Logic Unit",
    "operations": [
        {{"name": "ADD", "opcode": "0000", "description": "Addition (A + B)"}},
        {{"name": "SUB", "opcode": "0001", "description": "Subtraction (A - B)"}},
        ...
    ],
    "interface": {{
        "inputs": [
            {{"name": "a", "width": N, "description": "First operand"}},
            {{"name": "b", "width": N, "description": "Second operand"}},
            {{"name": "opcode", "width": 4, "description": "Operation select"}}
        ],
        "outputs": [
            {{"name": "result", "width": N, "description": "Operation result"}},
            {{"name": "zero", "width": 1, "description": "Zero flag"}},
            {{"name": "carry", "width": 1, "description": "Carry flag"}},
            {{"name": "overflow", "width": 1, "description": "Overflow flag"}},
            {{"name": "negative", "width": 1, "description": "Negative flag"}}
        ]
    }},
    "test_cases": [
        {{"operation": "ADD", "a": 10, "b": 5, "expected": 15}},
        ...
    ]
}}

CRITICAL INSTRUCTIONS:
- Respond with ONLY valid JSON
- Do NOT include markdown code blocks
- Infer bitwidth from user input (default: 16-bit)
- Include ALL operations mentioned by user
- If user mentions "basic operations", include ADD, SUB, AND, OR
- If user mentions "full operations", include ADD, SUB, AND, OR, XOR, NOT, SHL, SHR
"""

    def _create_basic_spec(self, user_input: str) -> Dict:
        """Create basic specification template when LLM is not available"""

        # Extract bitwidth from input
        bitwidth_match = re.search(r'(\d+)[-\s]?bit', user_input, re.IGNORECASE)
        bitwidth = int(bitwidth_match.group(1)) if bitwidth_match else 16

        # Detect requested operations
        text_lower = user_input.lower()
        operations = []

        for op_name, op_info in self.STANDARD_OPERATIONS.items():
            if op_name.lower() in text_lower:
                operations.append({
                    "name": op_name,
                    "opcode": op_info["opcode"],
                    "description": op_info["description"]
                })

        # Default operations if none specified
        if not operations:
            if "full" in text_lower or "all" in text_lower:
                operations = [
                    {"name": k, "opcode": v["opcode"], "description": v["description"]}
                    for k, v in self.STANDARD_OPERATIONS.items()
                ]
            else:
                # Basic operations
                for op in ["ADD", "SUB", "AND", "OR"]:
                    operations.append({
                        "name": op,
                        "opcode": self.STANDARD_OPERATIONS[op]["opcode"],
                        "description": self.STANDARD_OPERATIONS[op]["description"]
                    })

        return {
            "module_name": f"alu_{bitwidth}bit",
            "bitwidth": bitwidth,
            "description": f"{bitwidth}-bit Arithmetic Logic Unit",
            "operations": operations,
            "interface": {
                "inputs": [
                    {"name": "a", "width": bitwidth, "description": "First operand"},
                    {"name": "b", "width": bitwidth, "description": "Second operand"},
                    {"name": "opcode", "width": 4, "description": "Operation select"}
                ],
                "outputs": [
                    {"name": "result", "width": bitwidth, "description": "Operation result"},
                    {"name": "zero", "width": 1, "description": "Zero flag"},
                    {"name": "carry", "width": 1, "description": "Carry flag"},
                    {"name": "overflow", "width": 1, "description": "Overflow flag"},
                    {"name": "negative", "width": 1, "description": "Negative flag"}
                ]
            },
            "test_cases": self._generate_test_cases(operations, bitwidth),
            "_generated_by": "local_template"
        }

    def _parse_text_spec(self, text: str, user_input: str) -> Dict:
        """Parse text-based specification into structured format"""
        # This handles cases where LLM returns text instead of JSON
        spec = self._create_basic_spec(user_input)
        spec["raw_text"] = text
        spec["_generated_by"] = "llm_text_parsed"
        return spec

    def _parse_spec(self, spec_content: Dict, user_input: str) -> Dict:
        """Parse and structure the specification"""

        # Ensure required fields exist
        if isinstance(spec_content, dict):
            spec = spec_content
        else:
            spec = self._create_basic_spec(user_input)

        # Add metadata
        spec['original_input'] = user_input
        spec['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        spec['version'] = '1.0'

        # Ensure bitwidth exists
        if 'bitwidth' not in spec:
            bitwidth_match = re.search(r'(\d+)[-\s]?bit', user_input, re.IGNORECASE)
            spec['bitwidth'] = int(bitwidth_match.group(1)) if bitwidth_match else 16

        # 🔧 FIX: Ensure operations field exists (handles API fallback responses)
        if 'operations' not in spec or not spec['operations']:
            self._debug_print("Operations missing, reconstructing from user input...", "WARN")
            # Parse operations from user input
            text_lower = user_input.lower()
            operations = []

            for op_name, op_info in self.STANDARD_OPERATIONS.items():
                if op_name.lower() in text_lower:
                    operations.append({
                        "name": op_name,
                        "opcode": op_info["opcode"],
                        "description": op_info["description"]
                    })

            # Default operations if none detected
            if not operations:
                if "full" in text_lower or "all" in text_lower:
                    operations = [
                        {"name": k, "opcode": v["opcode"], "description": v["description"]}
                        for k, v in self.STANDARD_OPERATIONS.items()
                    ]
                else:
                    # Basic default operations
                    for op in ["ADD", "SUB", "AND", "OR"]:
                        operations.append({
                            "name": op,
                            "opcode": self.STANDARD_OPERATIONS[op]["opcode"],
                            "description": self.STANDARD_OPERATIONS[op]["description"]
                        })

            spec['operations'] = operations
            self._debug_print(f"Reconstructed {len(operations)} operations from user input", "SUCCESS")

        # 🔧 FIX: Ensure interface exists
        if 'interface' not in spec:
            bitwidth = spec.get('bitwidth', 16)
            spec['interface'] = {
                "inputs": [
                    {"name": "a", "width": bitwidth, "description": "First operand"},
                    {"name": "b", "width": bitwidth, "description": "Second operand"},
                    {"name": "opcode", "width": 4, "description": "Operation select"}
                ],
                "outputs": [
                    {"name": "result", "width": bitwidth, "description": "Operation result"},
                    {"name": "zero", "width": 1, "description": "Zero flag"},
                    {"name": "carry", "width": 1, "description": "Carry flag"},
                    {"name": "overflow", "width": 1, "description": "Overflow flag"},
                    {"name": "negative", "width": 1, "description": "Negative flag"}
                ]
            }

        # 🔧 FIX: Ensure module_name exists
        if 'module_name' not in spec:
            spec['module_name'] = f"alu_{spec['bitwidth']}bit"

        # 🔧 FIX: Ensure test_cases exist
        if 'test_cases' not in spec:
            spec['test_cases'] = self._generate_test_cases(spec['operations'], spec['bitwidth'])

        return spec

    def _validate_spec(self, spec: Dict) -> bool:
        """Validate specification completeness"""
        required_fields = ['bitwidth', 'operations']

        for field in required_fields:
            if field not in spec:
                self._debug_print(f"Missing required field: {field}", "WARN")
                return False

        if not spec.get('operations'):
            self._debug_print("No operations defined", "WARN")
            return False

        self._debug_print("Specification validation passed", "SUCCESS")
        return True

    def _generate_test_cases(self, operations: List[Dict], bitwidth: int) -> List[Dict]:
        """Generate sample test cases for operations"""
        import random
        test_cases = []
        max_val = (1 << bitwidth) - 1

        for op in operations:
            op_name = op['name']
            a = random.randint(0, max_val // 2)
            b = random.randint(0, max_val // 2)

            if op_name == "ADD":
                expected = (a + b) & max_val
            elif op_name == "SUB":
                expected = (a - b) & max_val
            elif op_name == "AND":
                expected = a & b
            elif op_name == "OR":
                expected = a | b
            elif op_name == "XOR":
                expected = a ^ b
            elif op_name == "NOT":
                expected = (~a) & max_val
                b = 0
            elif op_name == "SHL":
                b = random.randint(0, 4)
                expected = (a << b) & max_val
            elif op_name == "SHR":
                b = random.randint(0, 4)
                expected = a >> b
            else:
                expected = 0

            test_cases.append({
                "operation": op_name,
                "a": a,
                "b": b,
                "expected": expected
            })

        return test_cases

    def save_spec(self, spec_dict: Dict, filename: Optional[str] = None) -> Path:
        """
        Save specification to files (both .txt and .json)

        Args:
            spec_dict: Specification dictionary
            filename: Optional filename (without extension)

        Returns:
            Path to the saved specification text file
        """
        if filename is None:
            filename = f"alu_{spec_dict.get('bitwidth', 16)}bit_spec"

        # Save as JSON (primary format for programmatic access)
        json_path = self.output_dir / f"{filename}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(spec_dict, f, indent=2, ensure_ascii=False)

        # Save as text (for human readability and paper documentation)
        txt_path = self.output_dir / f"{filename}.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(self._format_spec_as_text(spec_dict))

        print(f"\n✅ Specification saved:")
        print(f"   📄 {txt_path}  (human readable)")
        print(f"   📊 {json_path}  (machine readable)")

        return txt_path

    def _format_spec_as_text(self, spec: Dict) -> str:
        """Format specification as readable text"""
        lines = [
            "=" * 80,
            "ALU SPECIFICATION DOCUMENT",
            "=" * 80,
            "",
            f"Generated: {spec.get('timestamp', 'N/A')}",
            f"Version: {spec.get('version', '1.0')}",
            "",
            "=" * 80,
            "MODULE INFORMATION",
            "=" * 80,
            f"Module Name: {spec.get('module_name', 'alu')}",
            f"Bit Width: {spec.get('bitwidth', 16)}-bit",
            f"Description: {spec.get('description', 'Arithmetic Logic Unit')}",
            "",
            "=" * 80,
            "ORIGINAL USER INPUT",
            "=" * 80,
            spec.get('original_input', 'N/A'),
            "",
            "=" * 80,
            "OPERATIONS AND OPCODES",
            "=" * 80,
        ]

        for op in spec.get('operations', []):
            lines.append(f"{op.get('opcode', '????')}: {op.get('name', '?')} - {op.get('description', '')}")

        lines.extend([
            "",
            "=" * 80,
            "INTERFACE DEFINITION",
            "=" * 80,
            "Inputs:",
        ])

        for inp in spec.get('interface', {}).get('inputs', []):
            lines.append(f"  - {inp['name']}[{inp['width']-1}:0]: {inp.get('description', '')}")

        lines.append("Outputs:")
        for out in spec.get('interface', {}).get('outputs', []):
            width = out.get('width', 1)
            if width == 1:
                lines.append(f"  - {out['name']}: {out.get('description', '')}")
            else:
                lines.append(f"  - {out['name']}[{width-1}:0]: {out.get('description', '')}")

        lines.extend([
            "",
            "=" * 80,
            "TEST EXAMPLES",
            "=" * 80,
        ])

        for i, tc in enumerate(spec.get('test_cases', [])[:5], 1):
            lines.append(f"Example {i}: {tc.get('a', 0)} {tc.get('operation', 'OP')} {tc.get('b', 0)} = {tc.get('expected', 0)}")

        lines.append("")
        lines.append("=" * 80)

        return "\n".join(lines)


def interactive_mode(project_root: Optional[str] = None):
    """Interactive mode with LLM provider selection"""
    print("=" * 70)
    print("🎯 SPEC Generator - Interactive Mode")
    print("=" * 70)
    print("\n📝 Examples of requirements:")
    print("   • 'I need a 16-bit ALU with ADD, SUB, AND, OR operations'")
    print("   • '8-bit ALU with full arithmetic and logic operations'")
    print("   • '32-bit ALU supporting basic math'")
    print("\n💡 Tip: The more detailed your requirements, the better the specification!")
    print("Type 'quit' or 'exit' to stop.\n")

    # LLM Provider selection
    llm = None
    if HAS_LLM:
        use_llm = input("Use LLM provider for enhanced generation? (y/n, default: n): ").strip().lower()

        if use_llm == 'y':
            print("\nSelect LLM provider:")
            print("\n🆓 FREE Providers (Recommended):")
            print("  1. Local template (no external API, zero setup)")
            print("  2. Google Gemini (FREE, 60 req/min) ⭐ Recommended")
            print("  3. Groq (FREE, ultra-fast) ⚡")
            print("  4. DeepSeek (FREE, Chinese LLM)")
            print("\n💰 PAID Providers (High Quality):")
            print("  5. OpenAI GPT-5 Series 🎯 Best Quality")
            print("  6. Anthropic Claude (premium)")

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