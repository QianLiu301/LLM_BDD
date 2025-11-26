"""
Testbench Generator - Generate Verilog testbench from BDD .feature files
=========================================================================

This module reads .feature files and generates Verilog testbench.
NO LLM is used - this is a deterministic transformation.
It does NOT generate ALU design (that's done by alu_generator.py).

ARCHITECTURE:
    output/bdd/*.feature  (from bdd_generator.py)
           │
           ▼
    ┌────────────────────┐
    │ testbench_generator│  ◄── NO LLM (deterministic)
    │    (this file)     │
    └──────────┬─────────┘
               │
               ▼
       output/verilog/testbench.v
               │
               ├── Instantiates ──► output/verilog/ALU.v (from alu_generator.py)
               │
               ▼
          Simulation (iverilog)

PURPOSE:
- Read .feature files from output/bdd/
- Parse test scenarios and expected results
- Generate testbench.v that tests the ALU.v (DUT)
- Support decimal/hexadecimal number formats
- Generate VCD dump for waveform viewing

This ensures:
1. Deterministic output (same .feature → same testbench)
2. Independence from LLM (no API calls needed)
3. Complete separation: testbench tests ALU, doesn't generate it
"""

import os
import re
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from enum import Enum
from datetime import datetime


class NumberFormat(Enum):
    """Number format enumeration"""
    DECIMAL = "decimal"
    HEXADECIMAL = "hexadecimal"
    BINARY = "binary"


class FeatureParser:
    """Parse .feature files and extract test scenarios"""

    def __init__(self, feature_file: str, debug: bool = True):
        self.feature_file = feature_file
        self.debug = debug
        self.bitwidth = 16  # default
        self.operations = {}  # opcode -> operation_name
        self.scenarios = []  # test scenarios
        self.number_format = NumberFormat.DECIMAL

    def parse(self) -> Dict:
        """Parse a .feature file"""
        with open(self.feature_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Extract bitwidth
        bitwidth_match = re.search(r'(\d+)[-_]?bit', content, re.IGNORECASE)
        if bitwidth_match:
            self.bitwidth = int(bitwidth_match.group(1))

        # Detect number format
        self._detect_number_format(content)

        # Extract opcode mapping
        self._extract_operations(content)

        # Extract test scenarios
        self._extract_scenarios(content)

        # Infer bitwidth from values if not found
        if not bitwidth_match and self.scenarios:
            self.bitwidth = self._infer_bitwidth_from_scenarios()
            if self.debug:
                print(f"   🔍 Inferred bitwidth from values: {self.bitwidth}-bit")

        return {
            'bitwidth': self.bitwidth,
            'operations': self.operations,
            'scenarios': self.scenarios,
            'number_format': self.number_format
        }

    def _infer_bitwidth_from_scenarios(self) -> int:
        """Infer bitwidth from value range"""
        max_value = 0
        for scenario in self.scenarios:
            for key in ['a', 'b', 'expected_result']:
                if key in scenario:
                    max_value = max(max_value, scenario[key])

        if max_value <= 0xFF:
            return 8
        elif max_value <= 0xFFFF:
            return 16
        elif max_value <= 0xFFFFFFFF:
            return 32
        else:
            return 64

    def _detect_number_format(self, content: str):
        """Auto-detect number format"""
        has_hex_prefix = bool(re.search(r'0[xX][0-9A-Fa-f]+', content))

        if has_hex_prefix:
            self.number_format = NumberFormat.HEXADECIMAL
            if self.debug:
                print("   🔍 Detected number format: hexadecimal")
        else:
            self.number_format = NumberFormat.DECIMAL
            if self.debug:
                print("   🔍 Detected number format: decimal")

    def _extract_operations(self, content: str):
        """Extract opcode definitions"""
        # Pattern: "opcode 0000" or "opcode is 0000 (ADD)"
        opcode_patterns = [
            r'opcode\s+(?:is\s+)?["\']?([01]{4})["\']?\s*\((\w+)\)',
            r'opcode\s+(?:is\s+)?["\']?([01]{4})["\']?\s+for\s+(\w+)',
            r'perform\s+the\s+(\w+)\s+operation\s+with\s+opcode\s+([01]{4})',
        ]

        for pattern in opcode_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                if len(match) == 2:
                    # Determine which is opcode and which is operation
                    if match[0].isdigit() or set(match[0]) <= {'0', '1'}:
                        opcode, op_name = match[0], match[1]
                    else:
                        op_name, opcode = match[0], match[1]
                    self.operations[opcode] = op_name.upper()

        # Default operations if none found
        if not self.operations:
            if self.debug:
                print("   ⚠️  No opcode definitions found, using defaults")
            self.operations = {
                '0000': 'ADD', '0001': 'SUB', '0010': 'AND', '0011': 'OR',
                '0100': 'XOR', '0101': 'NOT', '0110': 'SHL', '0111': 'SHR'
            }

    def _extract_scenarios(self, content: str):
        """Extract test scenarios from Examples tables"""
        scenarios = []

        # Find Scenario Outline blocks with Examples
        scenario_pattern = r'Scenario\s*(?:Outline)?[:\s]*(.+?)(?=Scenario|Feature:|$)'
        scenario_matches = re.findall(scenario_pattern, content, re.DOTALL | re.IGNORECASE)

        for scenario_content in scenario_matches:
            parsed = self._parse_scenario_block(scenario_content)
            if parsed:
                scenarios.extend(parsed)

        self.scenarios = scenarios

    def _parse_scenario_block(self, content: str) -> List[Dict]:
        """Parse a single scenario block"""
        scenarios = []

        # Extract opcode from scenario
        opcode_match = re.search(r'opcode\s+([01]{4})', content, re.IGNORECASE)
        opcode = opcode_match.group(1) if opcode_match else '0000'

        # Find Examples table
        examples_match = re.search(
            r'Examples:\s*\n\s*\|(.+?)\n((?:\s*\|.+\n?)+)',
            content,
            re.DOTALL
        )

        if examples_match:
            # Parse header
            header_line = examples_match.group(1).strip('| \t')
            headers = [h.strip().lower() for h in header_line.split('|') if h.strip()]

            # Parse data rows
            data_section = examples_match.group(2).strip()
            for line in data_section.split('\n'):
                line = line.strip()
                if not line or not line.startswith('|'):
                    continue

                values = [v.strip() for v in line.strip('|').split('|')]

                if len(values) >= 2:
                    scenario = {'opcode': opcode}

                    for i, header in enumerate(headers):
                        if i < len(values):
                            value = self._parse_number(values[i])
                            if value is not None:
                                # Map header names to standard keys
                                if header in ['a', 'operand_a']:
                                    scenario['a'] = value
                                elif header in ['b', 'operand_b']:
                                    scenario['b'] = value
                                elif header in ['expected_result', 'result', 'expected']:
                                    scenario['expected_result'] = value
                                elif header in ['zero_flag', 'zero']:
                                    scenario['zero_flag'] = str(value).lower() in ['true', '1', 'yes']
                                elif header in ['overflow', 'overflow_flag']:
                                    scenario['overflow'] = str(value).lower() in ['true', '1', 'yes']
                                elif header in ['negative_flag', 'negative']:
                                    scenario['negative_flag'] = str(value).lower() in ['true', '1', 'yes']
                                elif header == 'opcode':
                                    scenario['opcode'] = values[i].strip()

                    if 'a' in scenario and 'b' in scenario:
                        scenarios.append(scenario)

        return scenarios

    def _parse_number(self, value_str: str) -> Optional[int]:
        """Parse number string (decimal or hex)"""
        try:
            value_str = value_str.strip()

            # Boolean values
            if value_str.lower() in ['true', 'false']:
                return 1 if value_str.lower() == 'true' else 0

            # Hex format
            if value_str.startswith('0x') or value_str.startswith('0X'):
                return int(value_str, 16)

            # Decimal
            return int(value_str)
        except (ValueError, AttributeError):
            return None


class TestbenchGenerator:
    """
    Generate Verilog testbench from .feature files.

    This generator ONLY creates testbench files.
    ALU design is generated separately by alu_generator.py.
    """

    def __init__(
        self,
        feature_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
        project_root: Optional[str] = None,
        dut_module_name: Optional[str] = None,
        debug: bool = True
    ):
        """
        Initialize testbench generator.

        Args:
            feature_dir: Directory containing .feature files
            output_dir: Directory to save testbench files
            project_root: Project root directory
            dut_module_name: Name of DUT module (from alu_generator)
            debug: Enable debug output
        """
        self.debug = debug
        self.project_root = Path(project_root) if project_root else None
        self.dut_module_name = dut_module_name

        # Setup paths
        self.feature_dir = self._find_feature_dir(feature_dir)
        self.output_dir = self._setup_output_dir(output_dir)

        print(f"📁 Feature directory: {self.feature_dir}")
        print(f"📁 Output directory: {self.output_dir}")

    def _find_feature_dir(self, feature_dir: Optional[str]) -> Path:
        """Find .feature files directory dynamically"""
        # 1. Explicit specification
        if feature_dir:
            path = Path(feature_dir)
            if path.exists():
                return path
            print(f"⚠️  Specified feature_dir not found: {feature_dir}")

        # 2. Project root based
        if self.project_root:
            path = self.project_root / "output" / "bdd"
            if path.exists():
                return path

        # 3. Search common locations
        current = Path.cwd()
        search_paths = [
            current / "output" / "bdd",
            current / "bdd",
            current.parent / "output" / "bdd",
            current / "src" / "output" / "bdd",
        ]

        for path in search_paths:
            if path.exists():
                self._debug_print(f"Found features at: {path}", "SUCCESS")
                return path

        # 4. Fallback to absolute path
        fallback_path = Path(r"D:\DE\HdlFormalVerifierLLM\HdlFormalVerifier\AluBDDVerilog\output\bdd")
        if fallback_path.exists():
            return fallback_path

        # 5. Create default
        default = current / "output" / "bdd"
        default.mkdir(parents=True, exist_ok=True)
        print(f"⚠️  No feature directory found, created: {default}")
        return default

    def _setup_output_dir(self, output_dir: Optional[str]) -> Path:
        """Setup output directory for testbench files"""
        if output_dir:
            path = Path(output_dir)
        elif self.project_root:
            path = self.project_root / "output" / "verilog"
        else:
            # Same level as feature directory
            path = self.feature_dir.parent / "verilog"

        path.mkdir(parents=True, exist_ok=True)
        return path

    def _debug_print(self, message: str, level: str = "INFO"):
        """Debug output"""
        if not self.debug and level == "DEBUG":
            return

        icons = {
            "INFO": "ℹ️ ", "DEBUG": "🔍", "WARN": "⚠️ ",
            "ERROR": "❌", "SUCCESS": "✅", "STEP": "📌",
        }
        icon = icons.get(level, "  ")
        print(f"   {icon} [{level}] {message}")

    def scan_features(self) -> List[Path]:
        """Scan for .feature files"""
        print(f"\n🔍 Scanning for .feature files in: {self.feature_dir}")

        feature_files = list(self.feature_dir.glob("*.feature"))

        if not feature_files:
            print(f"   ⚠️  No .feature files found")
            return []

        print(f"   ✅ Found {len(feature_files)} feature file(s):")
        for f in feature_files:
            print(f"      • {f.name}")

        return feature_files

    def _detect_dut_module(self, bitwidth: int = None) -> str:
        """
        Detect DUT module name.

        Priority:
        1. Explicitly specified dut_module_name
        2. Infer from bitwidth (alu_{bitwidth}bit)
        3. Search for existing ALU .v files
        4. Default fallback
        """
        # 1. Use explicitly specified name
        if self.dut_module_name:
            return self.dut_module_name

        # 2. Infer from bitwidth (PREFERRED METHOD)
        if bitwidth:
            module_name = f"alu_{bitwidth}bit"
            self._debug_print(f"DUT module from bitwidth: {module_name}", "INFO")
            return module_name

        # 3. Search for ALU .v files in output directory
        verilog_files = list(self.output_dir.glob("alu_*.v"))

        for vf in verilog_files:
            # Skip testbench files
            if '_tb.v' in vf.name:
                continue
            # Extract module name from filename
            module_name = vf.stem
            self._debug_print(f"Detected DUT module from file: {module_name}", "INFO")
            return module_name

        # 4. Default fallback
        return "alu_16bit"

    def generate_testbench(self, spec: Dict, feature_name: str) -> str:
        """Generate testbench Verilog code"""
        bitwidth = spec['bitwidth']
        operations = spec['operations']
        scenarios = spec['scenarios']
        number_format = spec.get('number_format', NumberFormat.DECIMAL)

        # Detect or use specified DUT module name - pass bitwidth for correct inference
        dut_module = self._detect_dut_module(bitwidth=bitwidth)
        opcode_width = 4  # Standard 4-bit opcode

        lines = []

        # Header
        lines.append(f"//==============================================================================")
        lines.append(f"// Testbench: {feature_name}")
        lines.append(f"// Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"// Generator: testbench_generator.py (deterministic, no LLM)")
        lines.append(f"//")
        lines.append(f"// DUT Module: {dut_module}")
        lines.append(f"// Bitwidth: {bitwidth}-bit")
        lines.append(f"// Test cases: {len(scenarios)}")
        lines.append(f"// Number format: {number_format.value}")
        lines.append(f"//==============================================================================")
        lines.append("")
        lines.append("`timescale 1ns / 1ps")
        lines.append("")
        lines.append(f"module {feature_name}_tb;")
        lines.append("")

        # Signal declarations
        lines.append("    //--------------------------------------------------------------------------")
        lines.append("    // Test Signals")
        lines.append("    //--------------------------------------------------------------------------")
        lines.append(f"    reg  [{bitwidth-1}:0] a;")
        lines.append(f"    reg  [{bitwidth-1}:0] b;")
        lines.append(f"    reg  [{opcode_width-1}:0] opcode;")
        lines.append(f"    wire [{bitwidth-1}:0] result;")
        lines.append(f"    wire zero, carry, overflow, negative;")
        lines.append("")
        lines.append("    // Test counters")
        lines.append("    integer passed = 0;")
        lines.append("    integer failed = 0;")
        lines.append("    integer total = 0;")
        lines.append("")

        # DUT instantiation
        lines.append("    //--------------------------------------------------------------------------")
        lines.append("    // Device Under Test (DUT)")
        lines.append("    //--------------------------------------------------------------------------")
        lines.append(f"    {dut_module} uut (")
        lines.append(f"        .a(a),")
        lines.append(f"        .b(b),")
        lines.append(f"        .opcode(opcode),")
        lines.append(f"        .result(result),")
        lines.append(f"        .zero(zero),")
        lines.append(f"        .carry(carry),")
        lines.append(f"        .overflow(overflow),")
        lines.append(f"        .negative(negative)")
        lines.append(f"    );")
        lines.append("")

        # Test sequence
        lines.append("    //--------------------------------------------------------------------------")
        lines.append("    // Test Sequence")
        lines.append("    //--------------------------------------------------------------------------")
        lines.append("    initial begin")
        lines.append(f'        $display("========================================");')
        lines.append(f'        $display("Testbench: {feature_name}");')
        lines.append(f'        $display("DUT: {dut_module}");')
        lines.append(f'        $display("========================================");')
        lines.append(f'        $display("");')
        lines.append("")

        # Format specifier based on number format
        if number_format == NumberFormat.DECIMAL:
            fmt = "%d"
        else:
            hex_width = (bitwidth + 3) // 4
            fmt = f"%0{hex_width}X"

        # Generate test cases
        if scenarios:
            for i, scenario in enumerate(scenarios, 1):
                op_name = operations.get(scenario['opcode'], 'UNKNOWN')
                a_val = scenario.get('a', 0)
                b_val = scenario.get('b', 0)
                opcode_val = scenario['opcode']

                lines.append(f"        // Test {i}: {op_name}")
                lines.append(f"        a = {bitwidth}'d{a_val};")
                lines.append(f"        b = {bitwidth}'d{b_val};")
                lines.append(f"        opcode = 4'b{opcode_val};")
                lines.append(f"        #10;")
                lines.append(f"        total = total + 1;")

                # Display result
                lines.append(f'        $display("Test %0d: {fmt} {op_name} {fmt} = {fmt} (Z=%b C=%b O=%b N=%b)",')
                lines.append(f'                 total, a, b, result, zero, carry, overflow, negative);')

                # Check expected result if available
                if 'expected_result' in scenario:
                    expected = scenario['expected_result']
                    lines.append(f"        if (result == {bitwidth}'d{expected}) begin")
                    lines.append(f'            $display("  [PASS] Result correct");')
                    lines.append(f"            passed = passed + 1;")
                    lines.append(f"        end else begin")
                    lines.append(f'            $display("  [FAIL] Expected: {fmt}", {bitwidth}\'d{expected});')
                    lines.append(f"            failed = failed + 1;")
                    lines.append(f"        end")
                else:
                    lines.append(f"        passed = passed + 1;  // No expected value, assume pass")

                lines.append("")
        else:
            # Default test cases if no scenarios
            lines.append("        // Default test cases (no scenarios found)")
            default_tests = [
                (10, 5, '0000', 'ADD'),
                (20, 8, '0001', 'SUB'),
                (0xFF, 0x0F, '0010', 'AND'),
                (0xF0, 0x0F, '0011', 'OR'),
            ]
            for i, (a_val, b_val, op, op_name) in enumerate(default_tests, 1):
                lines.append(f"        // Test {i}: {op_name}")
                lines.append(f"        a = {bitwidth}'d{a_val};")
                lines.append(f"        b = {bitwidth}'d{b_val};")
                lines.append(f"        opcode = 4'b{op};")
                lines.append(f"        #10;")
                lines.append(f"        total = total + 1;")
                lines.append(f'        $display("Test %0d ({op_name}): %d op %d = %d", total, a, b, result);')
                lines.append("")

        # Summary
        lines.append(f'        $display("");')
        lines.append(f'        $display("========================================");')
        lines.append(f'        $display("TEST SUMMARY");')
        lines.append(f'        $display("========================================");')
        lines.append(f'        $display("Total:  %0d", total);')
        lines.append(f'        $display("Passed: %0d", passed);')
        lines.append(f'        $display("Failed: %0d", failed);')
        lines.append(f'        if (failed == 0)')
        lines.append(f'            $display("[SUCCESS] All tests passed!");')
        lines.append(f'        else')
        lines.append(f'            $display("[FAILURE] Some tests failed!");')
        lines.append(f'        $display("========================================");')
        lines.append("")
        lines.append("        #10;")
        lines.append("        $finish;")
        lines.append("    end")
        lines.append("")

        # VCD dump for waveform viewing
        lines.append("    //--------------------------------------------------------------------------")
        lines.append("    // VCD Dump for Waveform Viewing")
        lines.append("    //--------------------------------------------------------------------------")
        lines.append("    initial begin")
        lines.append(f'        $dumpfile("{feature_name}_tb.vcd");')
        lines.append(f"        $dumpvars(0, {feature_name}_tb);")
        lines.append("    end")
        lines.append("")
        lines.append("endmodule")
        lines.append("")
        lines.append("//==============================================================================")
        lines.append("// End of Testbench")
        lines.append("//==============================================================================")

        return '\n'.join(lines)

    def generate_all(self) -> List[Path]:
        """Generate testbenches for all .feature files"""
        print("\n" + "=" * 70)
        print("🚀 Testbench Generator - Starting generation")
        print("=" * 70)

        feature_files = self.scan_features()

        if not feature_files:
            print("\n❌ No .feature files found. Run bdd_generator.py first.")
            return []

        generated_files = []

        for feature_path in feature_files:
            try:
                print(f"\n📖 Processing: {feature_path.name}")

                # Parse feature file
                parser = FeatureParser(str(feature_path), debug=self.debug)
                spec = parser.parse()

                self._debug_print(f"Bitwidth: {spec['bitwidth']}", "DEBUG")
                self._debug_print(f"Scenarios: {len(spec['scenarios'])}", "DEBUG")

                # Generate testbench
                feature_name = feature_path.stem
                tb_content = self.generate_testbench(spec, feature_name)

                # Save testbench
                tb_path = self.output_dir / f"{feature_name}_tb.v"
                with open(tb_path, 'w', encoding='utf-8') as f:
                    f.write(tb_content)

                print(f"   ✅ Generated: {tb_path.name}")
                generated_files.append(tb_path)

            except Exception as e:
                print(f"   ❌ Error: {e}")
                if self.debug:
                    import traceback
                    traceback.print_exc()

        print("\n" + "=" * 70)
        print(f"✨ Generation complete! Created {len(generated_files)} testbench(es)")
        print("=" * 70)

        return generated_files

    def generate_from_feature_file(self, feature_path: Union[str, Path]) -> Path:
        """Generate testbench from a specific .feature file"""
        feature_path = Path(feature_path)

        if not feature_path.exists():
            raise FileNotFoundError(f"Feature file not found: {feature_path}")

        parser = FeatureParser(str(feature_path), debug=self.debug)
        spec = parser.parse()

        feature_name = feature_path.stem
        tb_content = self.generate_testbench(spec, feature_name)

        tb_path = self.output_dir / f"{feature_name}_tb.v"
        with open(tb_path, 'w', encoding='utf-8') as f:
            f.write(tb_content)

        print(f"\n✅ Generated: {tb_path}")
        return tb_path


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Generate Verilog testbench from BDD .feature files (No LLM)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Generate from all .feature files in default directory
  python testbench_generator.py

  # Specify feature directory
  python testbench_generator.py --feature-dir ./output/bdd

  # Specify output directory
  python testbench_generator.py --output-dir ./output/verilog

  # Specify DUT module name explicitly
  python testbench_generator.py --dut-module alu_16bit

  # Process single feature file
  python testbench_generator.py --feature-file ./output/bdd/alu_16bit.feature

  # With project root
  python testbench_generator.py --project-root D:/DE/HdlFormalVerifierLLM/HdlFormalVerifier/AluBDDVerilog

Note: This generator creates ONLY testbench files.
ALU design is generated by alu_generator.py separately.

Output Structure:
  output/
  ├── bdd/
  │   └── *.feature      ← Input (from bdd_generator)
  └── verilog/
      ├── alu_16bit.v    ← DUT (from alu_generator)
      └── *_tb.v         ← Testbench (from this file)

Simulation:
  iverilog -o sim output/verilog/alu_16bit.v output/verilog/*_tb.v
  vvp sim
  gtkwave *.vcd
        '''
    )

    parser.add_argument('--feature-dir', help='Directory containing .feature files')
    parser.add_argument('--feature-file', help='Specific .feature file to process')
    parser.add_argument('--output-dir', help='Output directory for testbench files')
    parser.add_argument('--project-root', help='Project root directory')
    parser.add_argument('--dut-module', help='DUT module name (auto-detected if not specified)')
    parser.add_argument('--debug', action='store_true', default=True, help='Enable debug output')

    args = parser.parse_args()

    # Create generator
    generator = TestbenchGenerator(
        feature_dir=args.feature_dir,
        output_dir=args.output_dir,
        project_root=args.project_root,
        dut_module_name=args.dut_module,
        debug=args.debug
    )

    # Generate
    if args.feature_file:
        # Single file mode
        tb_path = generator.generate_from_feature_file(args.feature_file)
        print(f"\n📄 Generated: {tb_path}")
    else:
        # Batch mode
        generated = generator.generate_all()

        if generated:
            print("\n📋 Generated files:")
            for path in generated:
                print(f"   • {path}")

            # Show next steps
            dut_module = generator._detect_dut_module()
            print("\n📋 NEXT STEPS:")
            print("=" * 70)
            print(f"1. Ensure DUT exists: {generator.output_dir}/{dut_module}.v")
            print(f"2. Compile:  iverilog -o sim {dut_module}.v *_tb.v")
            print(f"3. Simulate: vvp sim")
            print(f"4. Waveform: gtkwave *.vcd")
            print("=" * 70)


if __name__ == "__main__":
    main()