"""
Multi-LLM BDD Generator - Generate BDD .feature files for multiple LLMs
========================================================================

This module reads spec files from different LLM providers and generates
corresponding BDD .feature files. NO LLM is used - deterministic transformation.

NEW FEATURES:
- ✅ Support multiple LLM directories (specs/gemini/, specs/groq/, etc.)
- ✅ Skip failed specs (api_call_failed: true)
- ✅ Generate to LLM-specific output directories (output/bdd/gemini/, etc.)
- ✅ Preserve relative paths (no hardcoded absolute paths)
- ✅ Batch processing for all LLMs
- ✅ Summary report of successful/failed/skipped

DIRECTORY STRUCTURE:
    specs/
    ├── gemini/
    │   ├── spec_success.json → output/bdd/gemini/xxx.feature
    │   └── spec_failed.json  → SKIPPED (api_call_failed)
    ├── groq/
    └── deepseek/

ARCHITECTURE:
    spec.json (from spec_generator.py)
           │
           ▼
    ┌──────────────────┐
    │ bdd_generator_v2 │  ◄── NO LLM (deterministic)
    │    (this file)   │      Skips failed specs
    └────────┬─────────┘
             │
             ▼
        .feature files
"""

import argparse
import json
import random
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple


class MultiLLMBDDGenerator:
    """
    Generate BDD .feature files for multiple LLM providers.

    This is a deterministic generator - NO LLM is used.
    Automatically skips failed specs.
    """

    # Standard ALU operations (fallback if not in spec)
    DEFAULT_OPERATIONS = {
        "ADD": {"opcode": "0000", "description": "Addition (A + B)"},
        "SUB": {"opcode": "0001", "description": "Subtraction (A - B)"},
        "AND": {"opcode": "0010", "description": "Bitwise AND (A & B)"},
        "OR": {"opcode": "0011", "description": "Bitwise OR (A | B)"},
        "XOR": {"opcode": "0100", "description": "Bitwise XOR (A ^ B)"},
        "NOT": {"opcode": "0101", "description": "Bitwise NOT (~A)"},
        "SHL": {"opcode": "0110", "description": "Shift Left (A << B)"},
        "SHR": {"opcode": "0111", "description": "Shift Right (A >> B)"},
    }

    def __init__(
        self,
        specs_base_dir: Optional[str] = None,
        output_base_dir: Optional[str] = None,
        project_root: Optional[str] = None,
        debug: bool = True
    ):
        """
        Initialize Multi-LLM BDD generator.

        Args:
            specs_base_dir: Base directory containing LLM subdirectories (e.g., "specs/")
            output_base_dir: Base output directory (e.g., "output/bdd/")
            project_root: Project root directory
            debug: Enable debug output
        """
        self.debug = debug

        # Setup paths
        self.specs_base_dir = self._find_specs_base_dir(specs_base_dir, project_root)
        self.output_base_dir = self._setup_output_base_dir(output_base_dir, project_root)

        print(f"📁 Specs base directory: {self.specs_base_dir}")
        print(f"📁 Output base directory: {self.output_base_dir}")

    def _find_specs_base_dir(self, specs_dir: Optional[str], project_root: Optional[str]) -> Path:
        """Find specs base directory"""
        if specs_dir:
            return Path(specs_dir)

        # Try common locations
        current = Path.cwd()
        possible_paths = [
            current / "specs",
            current / "src" / "specs",
            current.parent / "specs",
        ]

        if project_root:
            possible_paths.insert(0, Path(project_root) / "specs")

        for path in possible_paths:
            if path.exists() and path.is_dir():
                return path

        # Default
        return current / "specs"

    def _setup_output_base_dir(self, output_dir: Optional[str], project_root: Optional[str]) -> Path:
        """Setup output base directory"""
        if output_dir:
            base_dir = Path(output_dir)
        elif project_root:
            base_dir = Path(project_root) / "output" / "bdd"
        else:
            # Try to find existing output directory
            current = Path.cwd()
            possible_paths = [
                current / "output" / "bdd",
                current / "outputs" / "bdd",
                current.parent / "output" / "bdd",
            ]

            for path in possible_paths:
                if path.parent.exists():  # Check if parent (output/) exists
                    base_dir = path
                    break
            else:
                # Default
                base_dir = current / "output" / "bdd"

        # Ensure directory exists
        base_dir.mkdir(parents=True, exist_ok=True)
        return base_dir

    def process_all_llms(self, num_examples: int = 5) -> Dict:
        """
        Process all LLM directories and generate BDD files.

        Args:
            num_examples: Number of test examples per scenario

        Returns:
            Dictionary with results for each LLM
        """
        print("\n" + "=" * 80)
        print("🚀 Multi-LLM BDD Generation")
        print("=" * 80)

        # Find all LLM directories
        llm_dirs = [d for d in self.specs_base_dir.iterdir()
                   if d.is_dir() and not d.name.startswith('.')]

        if not llm_dirs:
            print(f"\n❌ No LLM directories found in {self.specs_base_dir}")
            return {}

        print(f"\n📂 Found {len(llm_dirs)} LLM directories:")
        for d in llm_dirs:
            print(f"   • {d.name}")

        results = {}

        # Process each LLM directory
        for llm_dir in sorted(llm_dirs):
            llm_name = llm_dir.name
            print(f"\n{'='*80}")
            print(f"🔄 Processing LLM: {llm_name.upper()}")
            print(f"{'='*80}")

            result = self.process_llm_directory(llm_dir, llm_name, num_examples)
            results[llm_name] = result

        # Print summary
        self._print_summary(results)

        return results

    def process_llm_directory(self, llm_dir: Path, llm_name: str, num_examples: int = 5) -> Dict:
        """
        Process a single LLM directory.

        Args:
            llm_dir: Path to LLM directory (e.g., specs/gemini/)
            llm_name: LLM name (e.g., "gemini")
            num_examples: Number of test examples per scenario

        Returns:
            Dictionary with processing results
        """
        result = {
            'llm': llm_name,
            'total': 0,
            'success': 0,
            'failed_specs': 0,
            'skipped': 0,
            'errors': 0,
            'generated_files': []
        }

        # Find all JSON spec files
        spec_files = list(llm_dir.glob('*.json'))
        result['total'] = len(spec_files)

        if not spec_files:
            print(f"   ⚠️  No spec files found in {llm_dir}")
            return result

        print(f"\n   📄 Found {len(spec_files)} spec files")

        # Create output directory for this LLM
        llm_output_dir = self.output_base_dir / llm_name
        llm_output_dir.mkdir(parents=True, exist_ok=True)

        # Process each spec file
        for spec_file in spec_files:
            try:
                # Load spec
                with open(spec_file, 'r', encoding='utf-8') as f:
                    spec = json.load(f)

                # Check if spec failed
                if spec.get('api_call_failed'):
                    print(f"   ⏭️  Skipping FAILED spec: {spec_file.name}")
                    print(f"      Error: {spec.get('error_message', 'Unknown')}")
                    result['failed_specs'] += 1
                    continue

                # Check if has operations
                operations = spec.get('operations', {})
                if not operations or len(operations) == 0:
                    print(f"   ⏭️  Skipping spec with NO operations: {spec_file.name}")
                    result['skipped'] += 1
                    continue

                # Generate BDD feature
                print(f"\n   ✅ Processing: {spec_file.name}")
                feature_content = self.generate_feature(spec, num_examples)

                # Generate output filename
                timestamp = spec.get('metadata', {}).get('timestamp',
                                                        datetime.now().strftime("%Y%m%d_%H%M%S"))
                bitwidth = spec.get('bitwidth', 16)
                feature_filename = f"alu_{bitwidth}bit_{timestamp}.feature"

                # Save feature file
                feature_path = llm_output_dir / feature_filename
                with open(feature_path, 'w', encoding='utf-8') as f:
                    f.write(feature_content)

                print(f"      💾 Saved: {feature_path.relative_to(self.output_base_dir.parent)}")

                result['success'] += 1
                result['generated_files'].append(str(feature_path))

            except Exception as e:
                print(f"   ❌ Error processing {spec_file.name}: {e}")
                result['errors'] += 1
                if self.debug:
                    import traceback
                    traceback.print_exc()

        return result

    def generate_feature(self, spec: Dict, num_examples: int = 5) -> str:
        """
        Generate BDD .feature content from specification.

        Args:
            spec: Specification dictionary
            num_examples: Number of test examples per scenario

        Returns:
            Complete .feature file content
        """
        bitwidth = spec.get('bitwidth', 16)
        operations = spec.get('operations', {})

        # Get metadata
        metadata = spec.get('metadata', {})
        llm_provider = metadata.get('llm_provider', 'unknown')
        timestamp = metadata.get('timestamp', datetime.now().strftime("%Y%m%d_%H%M%S"))

        # Build feature header
        feature = f"""# Auto-generated BDD Feature File
# Source: {llm_provider} LLM
# Generated: {timestamp}
# Generator: bdd_generator_v2.py (deterministic, no LLM)
# Bitwidth: {bitwidth}-bit

Feature: {bitwidth}-bit ALU Verification
  As a hardware verification engineer
  I want to verify the {bitwidth}-bit ALU implementation
  So that I can ensure it correctly performs all arithmetic and logical operations

  Background:
    Given the ALU is initialized with {bitwidth}-bit operands

"""

        # Generate scenario for each operation
        for op_name, op_info in operations.items():
            op_desc = op_info.get('description', f'{op_name} operation')
            opcode = op_info.get('opcode', '0000')

            if self.debug:
                print(f"      • Generating scenario for: {op_name}")

            scenario = self._generate_scenario(
                op_name=op_name,
                op_desc=op_desc,
                opcode=opcode,
                bitwidth=bitwidth,
                num_examples=num_examples
            )
            feature += scenario + "\n"

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

    def _generate_test_examples(self, op_name: str, bitwidth: int, num_examples: int) -> List[Dict]:
        """
        Generate test examples for an operation.

        Includes:
        - Basic cases
        - Edge cases (0, max value)
        - Random cases
        """
        examples = []
        max_val = (1 << bitwidth) - 1

        # Edge cases
        edge_cases = [
            (0, 0),
            (0, 1),
            (1, 0),
            (max_val, 0),
            (0, max_val),
            (max_val, max_val),
        ]

        # Add edge cases
        for a, b in edge_cases[:min(3, num_examples)]:
            result, zero, overflow, negative = self._compute_result(op_name, a, b, bitwidth)
            examples.append({
                'a': a,
                'b': b,
                'result': result,
                'zero': zero,
                'overflow': overflow,
                'negative': negative
            })

        # Add random cases
        remaining = num_examples - len(examples)
        for _ in range(remaining):
            a = random.randint(0, max_val)
            b = random.randint(0, max_val)
            result, zero, overflow, negative = self._compute_result(op_name, a, b, bitwidth)
            examples.append({
                'a': a,
                'b': b,
                'result': result,
                'zero': zero,
                'overflow': overflow,
                'negative': negative
            })

        return examples

    def _compute_result(self, op_name: str, a: int, b: int, bitwidth: int) -> Tuple[int, bool, bool, bool]:
        """
        Compute expected result and flags for an operation.

        Returns:
            (result, zero_flag, overflow_flag, negative_flag)
        """
        max_val = (1 << bitwidth) - 1

        # Compute result
        if op_name == "ADD":
            result = a + b
            overflow = result > max_val
            result = result & max_val
        elif op_name == "SUB":
            result = a - b
            overflow = result < 0
            result = result & max_val
        elif op_name == "AND":
            result = a & b
            overflow = False
        elif op_name == "OR":
            result = a | b
            overflow = False
        elif op_name == "XOR":
            result = a ^ b
            overflow = False
        elif op_name == "NOT":
            result = ~a & max_val
            overflow = False
        elif op_name == "SHL":
            result = (a << (b & 0xF)) & max_val
            overflow = False
        elif op_name == "SHR":
            result = (a >> (b & 0xF)) & max_val
            overflow = False
        else:
            result = 0
            overflow = False

        # Compute flags
        zero = (result == 0)
        negative = bool(result & (1 << (bitwidth - 1)))

        return result, zero, overflow, negative

    def _print_summary(self, results: Dict):
        """Print summary of all LLM processing"""
        print("\n" + "=" * 80)
        print("📊 BDD Generation Summary")
        print("=" * 80)

        total_success = 0
        total_failed = 0
        total_skipped = 0
        total_errors = 0

        for llm_name, result in sorted(results.items()):
            total_success += result['success']
            total_failed += result['failed_specs']
            total_skipped += result['skipped']
            total_errors += result['errors']

            print(f"\n{llm_name.upper()}:")
            print(f"  Total specs: {result['total']}")
            print(f"  ✅ Generated: {result['success']}")
            if result['failed_specs'] > 0:
                print(f"  ⏭️  Skipped (failed specs): {result['failed_specs']}")
            if result['skipped'] > 0:
                print(f"  ⏭️  Skipped (no operations): {result['skipped']}")
            if result['errors'] > 0:
                print(f"  ❌ Errors: {result['errors']}")

            if result['generated_files']:
                print(f"  📄 Files:")
                for f in result['generated_files']:
                    print(f"     • {Path(f).name}")

        print(f"\n{'='*80}")
        print(f"Overall:")
        print(f"  ✅ Total generated: {total_success}")
        print(f"  ⏭️  Total skipped: {total_failed + total_skipped}")
        print(f"  ❌ Total errors: {total_errors}")
        print(f"{'='*80}")

        if total_failed > 0:
            print(f"\n💡 Tip: {total_failed} spec(s) were skipped because API calls failed.")
            print(f"   Check proxy settings and re-run spec_generator.py for these LLMs.")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Generate BDD .feature files for multiple LLMs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Process all LLMs (auto-detect directories)
  python bdd_generator_v2.py

  # Specify directories
  python bdd_generator_v2.py --specs specs/ --output output/bdd/

  # With project root
  python bdd_generator_v2.py --project-root /path/to/project

  # Custom number of test examples
  python bdd_generator_v2.py --num-examples 10
        '''
    )

    parser.add_argument('--specs', help='Specs base directory (default: auto-detect)')
    parser.add_argument('--output', help='Output base directory (default: output/bdd/)')
    parser.add_argument('--project-root', help='Project root directory')
    parser.add_argument('--num-examples', type=int, default=5,
                       help='Number of test examples per scenario (default: 5)')
    parser.add_argument('--no-debug', action='store_true',
                       help='Disable debug output')

    args = parser.parse_args()

    # Create generator
    generator = MultiLLMBDDGenerator(
        specs_base_dir=args.specs,
        output_base_dir=args.output,
        project_root=args.project_root,
        debug=not args.no_debug
    )

    # Process all LLMs
    results = generator.process_all_llms(num_examples=args.num_examples)

    # Return exit code
    total_errors = sum(r['errors'] for r in results.values())
    return 0 if total_errors == 0 else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())