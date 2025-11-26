"""
BDD Generator - Generate BDD .feature files from specification
===============================================================

This module reads spec files (JSON/TXT) and generates BDD .feature files.
NO LLM is used - this is a deterministic transformation.

ARCHITECTURE:
    spec.txt / spec.json  (from spec_generator.py)
           │
           ▼
    ┌──────────────────┐
    │   bdd_generator  │  ◄── NO LLM (deterministic)
    │    (this file)   │
    └────────┬─────────┘
             │
             ▼
        .feature files
             │
             ▼
      verilog_generator.py → testbench.v

PURPOSE:
- Read specification from specs directory
- Generate BDD scenarios for each operation
- Create parameterized Examples tables with test data
- Output .feature files in Gherkin format

This ensures:
1. Deterministic output (same spec → same BDD)
2. Independence from LLM (no API calls needed)
3. Traceability (BDD directly maps to spec)
"""

import os
import re
import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Union
from datetime import datetime


class BDDGenerator:
    """
    Generate BDD .feature files from specification.

    This is a deterministic generator - NO LLM is used.
    Same input spec always produces the same output.
    """

    # Standard ALU operations (fallback if not in spec)
    DEFAULT_OPERATIONS = {
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
        spec_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
        project_root: Optional[str] = None,
        debug: bool = True
    ):
        """
        Initialize BDD generator.

        Args:
            spec_dir: Directory containing spec files
            output_dir: Directory to save .feature files
            project_root: Project root directory
            debug: Enable debug output
        """
        self.debug = debug
        self.project_root = Path(project_root) if project_root else None

        # Setup paths
        self.spec_dir = self._find_spec_dir(spec_dir)
        self.output_dir = self._setup_output_dir(output_dir)

        print(f"📁 Spec directory: {self.spec_dir}")
        print(f"📁 Output directory: {self.output_dir}")

    def _find_spec_dir(self, spec_dir: Optional[str]) -> Path:
        """
        Find specs directory dynamically.

        Priority:
        1. Explicitly specified spec_dir
        2. project_root/src/specs
        3. Search common locations
        4. Fallback to absolute path
        """
        # 1. Explicit specification
        if spec_dir:
            path = Path(spec_dir)
            if path.exists():
                return path
            print(f"⚠️  Specified spec_dir not found: {spec_dir}")

        # 2. Project root based
        if self.project_root:
            path = self.project_root / "src" / "specs"
            if path.exists():
                return path

        # 3. Search common locations
        current = Path.cwd()
        search_paths = [
            current / "src" / "specs",
            current / "specs",
            current.parent / "src" / "specs",
            current.parent / "specs",
            current / "AluBDDVerilog" / "src" / "specs",
            current.parent / "AluBDDVerilog" / "src" / "specs",
        ]

        for path in search_paths:
            if path.exists():
                self._debug_print(f"Found specs at: {path}", "SUCCESS")
                return path

        # 4. Fallback to absolute path (Windows specific)
        fallback_path = Path(r"D:\DE\HdlFormalVerifierLLM\HdlFormalVerifier\AluBDDVerilog\src\specs")
        if fallback_path.exists():
            self._debug_print(f"Using fallback path: {fallback_path}", "INFO")
            return fallback_path

        # 5. Create default location
        default = current / "specs"
        default.mkdir(parents=True, exist_ok=True)
        print(f"⚠️  No existing specs directory found, created: {default}")
        return default

    def _setup_output_dir(self, output_dir: Optional[str]) -> Path:
        """Setup output directory for .feature files"""
        if output_dir:
            path = Path(output_dir)
        elif self.project_root:
            path = self.project_root / "output" / "bdd"
        else:
            # Put output next to specs directory
            path = self.spec_dir.parent.parent / "output" / "bdd"

        path.mkdir(parents=True, exist_ok=True)
        return path

    def _debug_print(self, message: str, level: str = "INFO"):
        """Debug output"""
        if not self.debug and level == "DEBUG":
            return

        icons = {
            "INFO": "ℹ️ ",
            "DEBUG": "🔍",
            "WARN": "⚠️ ",
            "ERROR": "❌",
            "SUCCESS": "✅",
            "STEP": "📌",
        }
        icon = icons.get(level, "  ")
        print(f"   {icon} [{level}] {message}")

    def scan_specs(self) -> List[Path]:
        """
        Scan specs directory for specification files.

        Returns:
            List of spec file paths (JSON preferred over TXT)
        """
        print(f"\n🔍 Scanning for specs in: {self.spec_dir}")

        json_files = list(self.spec_dir.glob("*.json"))
        txt_files = list(self.spec_dir.glob("*.txt"))

        # Prefer JSON files
        spec_files = json_files if json_files else txt_files

        if not spec_files:
            print(f"   ⚠️  No spec files found in {self.spec_dir}")
            return []

        print(f"   ✅ Found {len(spec_files)} spec file(s):")
        for f in spec_files:
            print(f"      • {f.name}")

        return spec_files

    def load_spec(self, spec_path: Path) -> Dict:
        """
        Load specification from file.

        Args:
            spec_path: Path to spec file

        Returns:
            Specification dictionary
        """
        print(f"\n📖 Loading spec: {spec_path.name}")

        if spec_path.suffix == '.json':
            return self._load_json_spec(spec_path)
        else:
            return self._load_txt_spec(spec_path)

    def _load_json_spec(self, path: Path) -> Dict:
        """Load JSON specification"""
        with open(path, 'r', encoding='utf-8') as f:
            spec = json.load(f)

        self._debug_print(f"Loaded JSON spec: {spec.get('module_name', 'unknown')}", "SUCCESS")
        self._debug_print(f"Bitwidth: {spec.get('bitwidth', 16)}", "DEBUG")
        self._debug_print(f"Operations: {len(spec.get('operations', []))}", "DEBUG")

        return spec

    def _load_txt_spec(self, path: Path) -> Dict:
        """Load and parse TXT specification"""
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Extract bitwidth
        bitwidth_match = re.search(r'(\d+)[-\s]?bit', content, re.IGNORECASE)
        bitwidth = int(bitwidth_match.group(1)) if bitwidth_match else 16

        # Extract operations from OPERATIONS section
        operations = []
        op_match = re.search(r'OPERATIONS.*?(?=\n\n|\nFLAG|\nINTERFACE|\nTEST|\Z)', content, re.DOTALL | re.IGNORECASE)

        if op_match:
            op_section = op_match.group(0)
            for line in op_section.split('\n'):
                # Match pattern like "0000: ADD - Addition"
                match = re.match(r'(\d{4}):\s*(\w+)\s*[-–]\s*(.+)', line.strip())
                if match:
                    operations.append({
                        "opcode": match.group(1),
                        "name": match.group(2),
                        "description": match.group(3).strip()
                    })

        # If no operations found, use defaults
        if not operations:
            operations = [
                {"name": k, "opcode": v["opcode"], "description": v["description"]}
                for k, v in list(self.DEFAULT_OPERATIONS.items())[:4]
            ]

        spec = {
            "module_name": f"alu_{bitwidth}bit",
            "bitwidth": bitwidth,
            "operations": operations,
            "raw_text": content
        }

        self._debug_print(f"Parsed TXT spec: {spec['module_name']}", "SUCCESS")
        return spec

    def generate_feature(self, spec: Dict, num_examples: int = 5) -> str:
        """
        Generate BDD .feature content from specification.

        Args:
            spec: Specification dictionary
            num_examples: Number of test examples per scenario

        Returns:
            Complete .feature file content
        """
        print(f"\n🔧 Generating BDD feature...")

        bitwidth = spec.get('bitwidth', 16)
        module_name = spec.get('module_name', f'alu_{bitwidth}bit')
        operations = spec.get('operations', [])

        # Build feature header
        feature = f"""# Auto-generated BDD Feature File
# Generated from: {module_name}
# Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
# Generator: bdd_generator.py (deterministic, no LLM)

Feature: {bitwidth}-bit ALU Verification
  As a hardware verification engineer
  I want to verify the {bitwidth}-bit ALU implementation
  So that I can ensure it correctly performs all arithmetic and logical operations

  Background:
    Given the ALU is initialized with {bitwidth}-bit operands

"""

        # Generate scenario for each operation
        for op in operations:
            op_name = op.get('name', 'UNKNOWN')
            op_desc = op.get('description', f'{op_name} operation')
            opcode = op.get('opcode', '0000')

            self._debug_print(f"Generating scenario for: {op_name}", "DEBUG")

            scenario = self._generate_scenario(
                op_name=op_name,
                op_desc=op_desc,
                opcode=opcode,
                bitwidth=bitwidth,
                num_examples=num_examples
            )
            feature += scenario + "\n"

        self._debug_print(f"Generated {len(operations)} scenarios", "SUCCESS")
        return feature

    def _generate_scenario(
        self,
        op_name: str,
        op_desc: str,
        opcode: str,
        bitwidth: int,
        num_examples: int
    ) -> str:
        """Generate a single scenario with Examples table"""

        scenario = f"""  @{op_name.lower()} @arithmetic
  Scenario Outline: Verify {op_name} operation
    Given I have operand A = <A>
    And I have operand B = <B>
    When I perform the {op_name} operation with opcode {opcode}
    Then the result should be <Expected_Result>
    And the zero flag should be <Zero_Flag>
    And the overflow flag should be <Overflow>
    And the negative flag should be <Negative_Flag>

    # {op_desc}
    Examples:
      | A     | B     | Opcode | Expected_Result | Zero_Flag | Overflow | Negative_Flag |
"""

        # Generate test examples
        examples = self._generate_test_examples(op_name, bitwidth, num_examples)

        for ex in examples:
            scenario += f"      | {ex['a']:<5} | {ex['b']:<5} | {opcode:<6} | {ex['result']:<15} | {str(ex['zero']):<9} | {str(ex['overflow']):<8} | {str(ex['negative']):<13} |\n"

        return scenario

    def _generate_test_examples(
        self,
        op_name: str,
        bitwidth: int,
        num_examples: int
    ) -> List[Dict]:
        """Generate deterministic test examples for an operation"""

        examples = []
        max_val = (1 << bitwidth) - 1  # 8-bit: 255, 16-bit: 65535, etc.
        half_max = max_val // 2

        # Use deterministic seed based on operation name for reproducibility
        seed = sum(ord(c) for c in op_name)
        rng = random.Random(seed)

        # Calculate safe ranges based on bitwidth
        # For small bitwidths (8-bit), use smaller values
        small_max = min(100, half_max)  # 8-bit: 100, but never exceed half_max
        medium_max = min(1000, half_max)  # For equal values test

        for i in range(num_examples):
            # Generate varied test cases - ALL values must be within bitwidth range
            if i == 0:
                # Edge case: small values (always safe)
                a, b = 10, 5
            elif i == 1:
                # Edge case: equal values (within range)
                a = rng.randint(small_max // 2, min(medium_max, half_max))
                b = a
            elif i == 2:
                # Edge case: one operand is zero
                a = rng.randint(1, small_max)
                b = 0
            else:
                # Random values within safe range
                a = rng.randint(0, half_max)
                b = rng.randint(0, half_max)

            # Calculate result based on operation
            result, overflow = self._calculate_result(op_name, a, b, bitwidth)

            # Calculate flags
            zero = (result == 0)
            negative = (result & (1 << (bitwidth - 1))) != 0

            examples.append({
                'a': a,
                'b': b,
                'result': result,
                'zero': zero,
                'overflow': overflow,
                'negative': negative
            })

        return examples

    def _calculate_result(
        self,
        op_name: str,
        a: int,
        b: int,
        bitwidth: int
    ) -> tuple:
        """Calculate ALU operation result and overflow flag"""

        max_val = (1 << bitwidth) - 1
        overflow = False

        if op_name == "ADD":
            raw = a + b
            overflow = raw > max_val
            result = raw & max_val
        elif op_name == "SUB":
            raw = a - b
            overflow = raw < 0
            result = raw & max_val
        elif op_name == "AND":
            result = a & b
        elif op_name == "OR":
            result = a | b
        elif op_name == "XOR":
            result = a ^ b
        elif op_name == "NOT":
            result = (~a) & max_val
        elif op_name == "SHL":
            shift = b & 0xF  # Limit shift amount
            raw = a << shift
            overflow = raw > max_val
            result = raw & max_val
        elif op_name == "SHR":
            shift = b & 0xF
            result = a >> shift
        else:
            result = 0

        return result, overflow

    def generate_all(self, num_examples: int = 5) -> List[Path]:
        """
        Generate .feature files for all specs in the directory.

        Args:
            num_examples: Number of test examples per scenario

        Returns:
            List of generated .feature file paths
        """
        print("\n" + "=" * 70)
        print("🚀 BDD Generator - Starting generation")
        print("=" * 70)

        spec_files = self.scan_specs()

        if not spec_files:
            print("\n❌ No spec files found. Please run spec_generator.py first.")
            return []

        generated_files = []

        for spec_path in spec_files:
            try:
                # Load spec
                spec = self.load_spec(spec_path)

                # Generate feature content
                feature_content = self.generate_feature(spec, num_examples)

                # Save feature file
                feature_name = spec_path.stem.replace('_spec', '') + '.feature'
                feature_path = self.output_dir / feature_name

                with open(feature_path, 'w', encoding='utf-8') as f:
                    f.write(feature_content)

                print(f"\n✅ Generated: {feature_path}")
                generated_files.append(feature_path)

            except Exception as e:
                print(f"\n❌ Error processing {spec_path.name}: {e}")
                if self.debug:
                    import traceback
                    traceback.print_exc()

        print("\n" + "=" * 70)
        print(f"✨ Generation complete! Created {len(generated_files)} .feature file(s)")
        print("=" * 70)

        return generated_files

    def generate_from_spec_file(self, spec_path: Union[str, Path], num_examples: int = 5) -> Path:
        """
        Generate .feature file from a specific spec file.

        Args:
            spec_path: Path to the spec file
            num_examples: Number of test examples per scenario

        Returns:
            Path to generated .feature file
        """
        spec_path = Path(spec_path)

        if not spec_path.exists():
            raise FileNotFoundError(f"Spec file not found: {spec_path}")

        spec = self.load_spec(spec_path)
        feature_content = self.generate_feature(spec, num_examples)

        feature_name = spec_path.stem.replace('_spec', '') + '.feature'
        feature_path = self.output_dir / feature_name

        with open(feature_path, 'w', encoding='utf-8') as f:
            f.write(feature_content)

        print(f"\n✅ Generated: {feature_path}")
        return feature_path


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Generate BDD .feature files from ALU specification (No LLM)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Generate from all specs in default directory
  python bdd_generator.py

  # Specify spec directory
  python bdd_generator.py --spec-dir ./specs

  # Generate from specific spec file
  python bdd_generator.py --spec-file ./specs/alu_16bit_spec.json

  # Custom output directory
  python bdd_generator.py --output-dir ./output/bdd

  # More examples per scenario
  python bdd_generator.py --examples 10

  # With project root
  python bdd_generator.py --project-root D:/DE/HdlFormalVerifierLLM/HdlFormalVerifier/AluBDDVerilog

Note: This generator does NOT use LLM. It performs deterministic
transformation from spec to BDD scenarios.
        '''
    )

    parser.add_argument('--spec-dir', help='Directory containing spec files')
    parser.add_argument('--spec-file', help='Specific spec file to process')
    parser.add_argument('--output-dir', help='Output directory for .feature files')
    parser.add_argument('--project-root', help='Project root directory')
    parser.add_argument('--examples', type=int, default=5, help='Number of examples per scenario (default: 5)')
    parser.add_argument('--debug', action='store_true', default=True, help='Enable debug output')

    args = parser.parse_args()

    # Create generator
    generator = BDDGenerator(
        spec_dir=args.spec_dir,
        output_dir=args.output_dir,
        project_root=args.project_root,
        debug=args.debug
    )

    # Generate
    if args.spec_file:
        # Single file mode
        feature_path = generator.generate_from_spec_file(args.spec_file, args.examples)
        print(f"\n📄 Generated: {feature_path}")
    else:
        # Batch mode - process all specs
        generated = generator.generate_all(args.examples)

        if generated:
            print("\n📋 Generated files:")
            for path in generated:
                print(f"   • {path}")

            print("\n📋 NEXT STEPS:")
            print("=" * 70)
            print(f"1. Review .feature files in: {generator.output_dir}")
            print(f"2. Generate testbench: python verilog_generator.py --feature-dir {generator.output_dir}")
            print(f"3. Run simulation:     python simulation_controller.py")
            print("=" * 70)


if __name__ == "__main__":
    main()