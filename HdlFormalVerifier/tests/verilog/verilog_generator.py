"""
Verilog Generator - Generate Verilog code from BDD feature files
Supports dynamic path configuration and automatic number format detection (decimal/hexadecimal)
"""

import os
import re
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from enum import Enum


class NumberFormat(Enum):
    """Number format enum"""
    DECIMAL = "decimal"
    HEXADECIMAL = "hexadecimal"
    BINARY = "binary"


class FeatureParser:
    """Parse .feature files and extract ALU specification information"""

    def __init__(self, feature_file: str):
        self.feature_file = feature_file
        self.bitwidth = 16  # Default bitwidth
        self.operations = {}  # opcode -> operation_name
        self.scenarios = []  # Test scenarios
        self.number_format = NumberFormat.DECIMAL  # Default: decimal

    def parse(self) -> Dict:
        """Parse feature file"""
        with open(self.feature_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Extract bitwidth
        bitwidth_match = re.search(r'(\d+)[-_]bit', content, re.IGNORECASE)
        if bitwidth_match:
            self.bitwidth = int(bitwidth_match.group(1))

        # Detect number format
        self._detect_number_format(content)

        # Extract opcode mapping
        self._extract_operations(content)

        # Extract test scenarios
        self._extract_scenarios(content)

        return {
            'bitwidth': self.bitwidth,
            'operations': self.operations,
            'scenarios': self.scenarios,
            'module_name': f"alu_{self.bitwidth}bit",
            'number_format': self.number_format
        }

    def _detect_number_format(self, content: str):
        """Automatically detect number format"""
        # Check for hexadecimal markers (0x, 0X, or uppercase hex numbers in Examples table)
        hex_patterns = [
            r'0x[0-9A-Fa-f]+',  # 0x prefix
            r'0X[0-9A-Fa-f]+',  # 0X prefix
            r'\b[0-9A-F]{4,}\b'  # 4 or more uppercase hex digits (likely hexadecimal)
        ]

        has_hex_prefix = bool(re.search(r'0[xX][0-9A-Fa-f]+', content))

        # Check inside Examples section
        examples_match = re.search(r'Examples:(.*?)(?=Scenario:|$)', content, re.DOTALL)
        if examples_match:
            examples_section = examples_match.group(1)
            # Extract numbers
            numbers = re.findall(r'\b\d+\b', examples_section)

            if has_hex_prefix:
                self.number_format = NumberFormat.HEXADECIMAL
                print("  Detected number format: hexadecimal (0x prefix)")
            elif numbers:
                # If there is no 0x prefix and numbers are small (<1000), assume decimal
                max_num = max([int(n) for n in numbers]) if numbers else 0
                if max_num < 1000 or any(int(n) < 10 for n in numbers):
                    self.number_format = NumberFormat.DECIMAL
                    print("  Detected number format: decimal")
                else:
                    # If numbers are large, they might be hexadecimal without prefix
                    self.number_format = NumberFormat.HEXADECIMAL
                    print("  Detected number format: hexadecimal (no prefix)")
            else:
                self.number_format = NumberFormat.DECIMAL
                print("  Using default number format: decimal")
        else:
            # If there is no Examples section, check whole file
            if has_hex_prefix:
                self.number_format = NumberFormat.HEXADECIMAL
                print("  Detected number format: hexadecimal")
            else:
                self.number_format = NumberFormat.DECIMAL
                print("  Using default number format: decimal")

    def _extract_operations(self, content: str):
        """Extract opcode definitions from feature file"""
        # Find opcode definition lines, e.g.:
        # "Given the opcode is 0000 (ADD)"
        # "When I set opcode to "0001" for SUB operation"
        opcode_patterns = [
            r'opcode\s+is\s+["\']?([01]+)["\']?\s*\((\w+)\)',
            r'opcode\s+to\s+["\']([01]+)["\']?\s+for\s+(\w+)',
            r'opcode\s+["\']([01]+)["\'].*?(\w+)\s+operation'
        ]

        for pattern in opcode_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for opcode, op_name in matches:
                self.operations[opcode] = op_name.upper()

        # If no opcode mapping is found, use default mapping
        if not self.operations:
            print("  Warning: No opcode definitions found, using default mapping")
            self.operations = {
                '0000': 'ADD',
                '0001': 'SUB',
                '0010': 'AND',
                '0011': 'OR',
                '0100': 'XOR',
                '0101': 'SHL',
                '0110': 'SHR',
                '0111': 'NOT'
            }

    def _extract_scenarios(self, content: str):
        """Extract test scenarios"""
        scenarios = []

        # Find all Scenarios and their Examples
        scenario_pattern = r'Scenario:\s*(.+?)(?=Scenario:|$)'
        scenario_matches = re.findall(scenario_pattern, content, re.DOTALL)

        for scenario_content in scenario_matches:
            scenario_list = self._parse_scenario(scenario_content)
            if scenario_list:
                scenarios.extend(scenario_list)

        self.scenarios = scenarios

    def _parse_scenario(self, scenario_content: str) -> List[Dict]:
        """Parse a single test scenario, supporting Examples tables"""
        scenarios = []

        # Extract opcode
        opcode_patterns = [
            r'opcode\s+(?:is|to)\s+["\']?([01]+)["\']?',
            r'set\s+opcode.*?["\']([01]+)["\']'
        ]

        opcode = None
        for pattern in opcode_patterns:
            opcode_match = re.search(pattern, scenario_content, re.IGNORECASE)
            if opcode_match:
                opcode = opcode_match.group(1)
                break

        if not opcode:
            return scenarios

        # Find Examples table
        examples_match = re.search(
            r'Examples:\s*\n\s*\|(.+?)\|(.*?)(?=\n\s*\n|\n\s*Scenario:|\Z)',
            scenario_content,
            re.DOTALL
        )

        if examples_match:
            # Parse header
            header_line = examples_match.group(1).strip()
            headers = [h.strip() for h in header_line.split('|') if h.strip()]

            # Parse data rows
            data_lines = examples_match.group(2).strip().split('\n')

            for line in data_lines:
                line = line.strip()
                if not line or not line.startswith('|'):
                    continue

                # Remove leading and trailing |
                line = line.strip('|')
                values = [v.strip() for v in line.split('|')]

                if len(values) < 2:
                    continue

                # Create scenario dict
                scenario = {'opcode': opcode}

                # Parse A and B values
                for i, header in enumerate(headers):
                    if i < len(values):
                        value_str = values[i]
                        # Parse number (supports decimal and hexadecimal)
                        value = self._parse_number(value_str)
                        if value is not None:
                            if header.upper() == 'A':
                                scenario['a'] = value
                                scenario['a_str'] = value_str  # Keep original string
                            elif header.upper() == 'B':
                                scenario['b'] = value
                                scenario['b_str'] = value_str

                # Extract expected result (if present)
                result_patterns = [
                    r'result\s+should\s+be\s+(\d+)',
                    r'result\s+should\s+be\s+0x([0-9A-Fa-f]+)'
                ]

                for pattern in result_patterns:
                    result_match = re.search(pattern, scenario_content, re.IGNORECASE)
                    if result_match:
                        result_str = result_match.group(1)
                        if 'x' in pattern or 'X' in pattern:
                            scenario['expected_result'] = int(result_str, 16)
                        else:
                            scenario['expected_result'] = int(result_str)
                        break

                if 'a' in scenario and 'b' in scenario:
                    scenarios.append(scenario)
        else:
            # Legacy format: extract directly from Given/When/Then
            a_patterns = [
                r'input\s+A\s+is\s+0x([0-9A-Fa-f]+)',
                r'input\s+A\s+is\s+(\d+)',
                r'operand\s+A\s+is\s+(\d+)',
                r'A\s*=\s*(\d+)'
            ]

            b_patterns = [
                r'input\s+B\s+is\s+0x([0-9A-Fa-f]+)',
                r'input\s+B\s+is\s+(\d+)',
                r'operand\s+B\s+is\s+(\d+)',
                r'B\s*=\s*(\d+)'
            ]

            a_value = None
            a_str = None
            for pattern in a_patterns:
                a_match = re.search(pattern, scenario_content, re.IGNORECASE)
                if a_match:
                    a_str = a_match.group(1)
                    a_value = self._parse_number(a_str, '0x' in pattern)
                    break

            b_value = None
            b_str = None
            for pattern in b_patterns:
                b_match = re.search(pattern, scenario_content, re.IGNORECASE)
                if b_match:
                    b_str = b_match.group(1)
                    b_value = self._parse_number(b_str, '0x' in pattern)
                    break

            if a_value is not None and b_value is not None:
                scenario = {
                    'a': a_value,
                    'b': b_value,
                    'a_str': a_str,
                    'b_str': b_str,
                    'opcode': opcode
                }

                # Extract expected result
                result_patterns = [
                    r'result\s+should\s+be\s+0x([0-9A-Fa-f]+)',
                    r'result\s+should\s+be\s+(\d+)'
                ]

                for pattern in result_patterns:
                    result_match = re.search(pattern, scenario_content, re.IGNORECASE)
                    if result_match:
                        result_str = result_match.group(1)
                        if 'x' in pattern:
                            scenario['expected_result'] = int(result_str, 16)
                        else:
                            scenario['expected_result'] = int(result_str)
                        break

                scenarios.append(scenario)

        return scenarios

    def _parse_number(self, value_str: str, force_hex: bool = False) -> Optional[int]:
        """Parse number and automatically recognize format"""
        try:
            value_str = value_str.strip()

            # Check for hexadecimal prefix
            if value_str.startswith('0x') or value_str.startswith('0X') or force_hex:
                # Remove 0x prefix if present
                hex_str = value_str[2:] if value_str.startswith('0') else value_str
                return int(hex_str, 16)
            else:
                # Decimal
                return int(value_str)
        except (ValueError, AttributeError):
            return None


class VerilogGenerator:
    """Generate Verilog code from feature files"""

    def __init__(self, bdd_dir: str = None, verilog_dir: str = None):
        """
        Initialize generator
        :param bdd_dir: Directory containing BDD feature files
        :param verilog_dir: Verilog output directory
        """
        # Set default paths or use provided paths
        if bdd_dir:
            self.bdd_dir = Path(bdd_dir)
        else:
            # Default path
            default_bdd = Path(r"D:\DE\HdlFormalVerifierLLM\HdlFormalVerifier\AluBDDVerilog\src\output\bdd")
            self.bdd_dir = default_bdd if default_bdd.exists() else Path("./bdd")

        if verilog_dir:
            self.verilog_dir = Path(verilog_dir)
        else:
            # Default path
            default_verilog = Path(r"D:\DE\HdlFormalVerifierLLM\HdlFormalVerifier\AluBDDVerilog\src\output\verilog")
            self.verilog_dir = default_verilog if default_verilog.parent.exists() else Path("./verilog")

        # Ensure output directory exists
        self.verilog_dir.mkdir(parents=True, exist_ok=True)
        print(f"BDD input directory: {self.bdd_dir}")
        print(f"Verilog output directory: {self.verilog_dir}")

    def process_all_features(self):
        """Process all feature files"""
        if not self.bdd_dir.exists():
            print(f"Error: BDD directory does not exist: {self.bdd_dir}")
            print(f"Please create the directory or use -b to specify the correct path")
            return

        feature_files = list(self.bdd_dir.glob("*.feature"))

        if not feature_files:
            print(f"Warning: No .feature files found in {self.bdd_dir}")
            return

        print(f"\nFound {len(feature_files)} feature file(s)")
        print("=" * 60)

        success_count = 0
        fail_count = 0

        for feature_file in feature_files:
            print(f"\nProcessing file: {feature_file.name}")
            try:
                self.process_single_feature(str(feature_file))
                success_count += 1
                print(f"✓ Successfully processed: {feature_file.name}")
            except Exception as e:
                fail_count += 1
                print(f"✗ Error while processing {feature_file.name}: {e}")
                import traceback
                traceback.print_exc()

        print("\n" + "=" * 60)
        print(f"Processing finished: {success_count} succeeded, {fail_count} failed")

    def process_single_feature(self, feature_file: str):
        """Process a single feature file"""
        # Parse feature file
        parser = FeatureParser(feature_file)
        spec = parser.parse()

        print(f"  Bitwidth: {spec['bitwidth']} bit")
        print(f"  Operations: {len(spec['operations'])}")
        print(f"  Test scenarios: {len(spec['scenarios'])}")

        # Generate Verilog module
        module_code = self.generate_module(spec)
        module_file = self.verilog_dir / f"{spec['module_name']}.v"
        with open(module_file, 'w', encoding='utf-8') as f:
            f.write(module_code)
        print(f"  Generated module: {module_file.name}")

        # Generate testbench
        tb_code = self.generate_testbench(spec)
        tb_file = self.verilog_dir / f"{spec['module_name']}_tb.v"
        with open(tb_file, 'w', encoding='utf-8') as f:
            f.write(tb_code)
        print(f"  Generated testbench: {tb_file.name}")

    def generate_module(self, spec: Dict) -> str:
        """Generate Verilog module code"""
        bitwidth = spec['bitwidth']
        operations = spec['operations']
        opcode_width = len(list(operations.keys())[0]) if operations else 4
        module_name = spec['module_name']

        lines = []

        # File header comment
        lines.append(f"/*")
        lines.append(f" * {bitwidth}-bit ALU Module")
        lines.append(f" * Generated from BDD Feature File")
        lines.append(f" * Operations: {len(operations)}")
        lines.append(f" * Supported operations:")
        for opcode, op_name in sorted(operations.items()):
            lines.append(f" *   - {op_name} ({opcode})")
        lines.append(f" */")
        lines.append("")
        lines.append(f"`timescale 1ns/1ps")
        lines.append("")

        # Module declaration
        lines.append(f"module {module_name} (")
        lines.append(f"    input  wire [{bitwidth - 1}:0] a,      // Operand A")
        lines.append(f"    input  wire [{bitwidth - 1}:0] b,      // Operand B")
        lines.append(f"    input  wire [{opcode_width - 1}:0] opcode, // Operation code")
        lines.append(f"    output reg  [{bitwidth - 1}:0] result, // Result")
        lines.append(f"    output reg  carry,           // Carry flag")
        lines.append(f"    output reg  zero,            // Zero flag")
        lines.append(f"    output reg  negative,        // Negative flag")
        lines.append(f"    output reg  overflow         // Overflow flag")
        lines.append(");")
        lines.append("")

        # Internal signals
        lines.append("    // Internal signals")
        lines.append(f"    reg [{bitwidth}:0] temp_result;  // Extended for carry detection")
        lines.append("")

        # ALU main logic
        lines.append("    // ALU operation logic")
        lines.append("    always @(*) begin")
        lines.append("        // Default values")
        lines.append(f"        result = {bitwidth}'b0;")
        lines.append("        carry = 1'b0;")
        lines.append("        overflow = 1'b0;")
        lines.append(f"        temp_result = {bitwidth + 1}'b0;")
        lines.append("")
        lines.append("        case (opcode)")

        # Case branches for each operation
        for opcode, op_name in sorted(operations.items()):
            case_lines = self._generate_operation_case(opcode, op_name, bitwidth)
            lines.extend(case_lines)

        lines.append("            default: begin")
        lines.append(f"                result = {bitwidth}'b0;")
        lines.append("            end")
        lines.append("        endcase")
        lines.append("    end")
        lines.append("")

        # Flag generation
        lines.append("    // Flag generation")
        lines.append("    always @(*) begin")
        lines.append(f"        zero = (result == {bitwidth}'b0);")
        lines.append(f"        negative = result[{bitwidth - 1}];")
        lines.append("    end")
        lines.append("")
        lines.append("endmodule")

        return '\n'.join(lines)

    def _generate_operation_case(self, opcode: str, op_name: str, bitwidth: int) -> List[str]:
        """Generate case branch for a single operation"""
        lines = []
        opcode_width = len(opcode)
        opcode_str = f"{opcode_width}'b{opcode}"

        lines.append(f"            {opcode_str}: begin // {op_name}")

        if op_name == 'ADD':
            lines.append(f"                temp_result = a + b;")
            lines.append(f"                result = temp_result[{bitwidth - 1}:0];")
            lines.append(f"                carry = temp_result[{bitwidth}];")
            lines.append(
                f"                overflow = (a[{bitwidth - 1}] == b[{bitwidth - 1}]) && (result[{bitwidth - 1}] != a[{bitwidth - 1}]);"
            )

        elif op_name == 'SUB':
            lines.append(f"                temp_result = a - b;")
            lines.append(f"                result = temp_result[{bitwidth - 1}:0];")
            lines.append(f"                carry = ~temp_result[{bitwidth}];")
            lines.append(
                f"                overflow = (a[{bitwidth - 1}] != b[{bitwidth - 1}]) && (result[{bitwidth - 1}] != a[{bitwidth - 1}]);"
            )

        elif op_name == 'AND':
            lines.append(f"                result = a & b;")

        elif op_name == 'OR':
            lines.append(f"                result = a | b;")

        elif op_name == 'XOR':
            lines.append(f"                result = a ^ b;")

        elif op_name == 'NOT':
            lines.append(f"                result = ~a;")

        elif op_name == 'SHL':
            lines.append(f"                result = a << b[4:0];  // Shift by lower 5 bits")

        elif op_name == 'SHR':
            lines.append(f"                result = a >> b[4:0];  // Shift by lower 5 bits")

        elif op_name == 'ROL':
            lines.append(f"                result = (a << b[4:0]) | (a >> ({bitwidth} - b[4:0]));")

        elif op_name == 'ROR':
            lines.append(f"                result = (a >> b[4:0]) | (a << ({bitwidth} - b[4:0]));")

        elif op_name == 'INC':
            lines.append(f"                temp_result = a + 1;")
            lines.append(f"                result = temp_result[{bitwidth - 1}:0];")
            lines.append(f"                carry = temp_result[{bitwidth}];")

        elif op_name == 'DEC':
            lines.append(f"                temp_result = a - 1;")
            lines.append(f"                result = temp_result[{bitwidth - 1}:0];")

        else:
            lines.append(f"                result = {bitwidth}'b0;  // Undefined operation")

        lines.append(f"            end")
        lines.append("")

        return lines

    def generate_testbench(self, spec: Dict) -> str:
        """Generate testbench code"""
        bitwidth = spec['bitwidth']
        operations = spec['operations']
        scenarios = spec['scenarios']
        opcode_width = len(list(operations.keys())[0]) if operations else 4
        module_name = spec['module_name']
        number_format = spec.get('number_format', NumberFormat.DECIMAL)

        lines = []

        # File header
        lines.append(f"`timescale 1ns/1ps")
        lines.append("")
        lines.append(f"/*")
        lines.append(f" * Testbench for {bitwidth}-bit ALU")
        lines.append(f" * Generated from BDD Feature File")
        lines.append(f" * Number format: {number_format.value}")
        lines.append(f" * Test cases: {len(scenarios)}")
        lines.append(f" */")
        lines.append(f"module {module_name}_tb;")
        lines.append("")

        # Signal declarations
        lines.append("    // Test signals")
        lines.append(f"    reg [{bitwidth - 1}:0] a, b;")
        lines.append(f"    reg [{opcode_width - 1}:0] opcode;")
        lines.append(f"    wire [{bitwidth - 1}:0] result;")
        lines.append(f"    wire carry, zero, negative, overflow;")
        lines.append("")
        lines.append(f"    // Test statistics")
        lines.append(f"    integer passed = 0;")
        lines.append(f"    integer failed = 0;")
        lines.append("")

        # Instantiate DUT
        lines.append(f"    // Instantiate ALU")
        lines.append(f"    {module_name} uut (")
        lines.append(f"        .a(a),")
        lines.append(f"        .b(b),")
        lines.append(f"        .opcode(opcode),")
        lines.append(f"        .result(result),")
        lines.append(f"        .carry(carry),")
        lines.append(f"        .zero(zero),")
        lines.append(f"        .negative(negative),")
        lines.append(f"        .overflow(overflow)")
        lines.append(f"    );")
        lines.append("")

        # Test sequence
        lines.append("    // Test sequence")
        lines.append("    initial begin")
        lines.append(f'        $display("========================================");')
        lines.append(f'        $display("{bitwidth}-bit ALU Testbench");')
        lines.append(f'        $display("Number format: {number_format.value}");')
        lines.append(f'        $display("========================================");')
        lines.append(f'        $display("");')
        lines.append("")

        # Decide display format according to number format
        if number_format == NumberFormat.DECIMAL:
            format_spec = "%d"
            format_comment = "decimal"
        else:
            # Calculate hex width
            hex_width = (bitwidth + 3) // 4
            format_spec = f"%0{hex_width}X"
            format_comment = "hexadecimal"

        # Generate test cases
        if scenarios:
            for i, scenario in enumerate(scenarios, 1):
                op_name = operations.get(scenario['opcode'], 'UNKNOWN')

                lines.append(f"        // Test case {i}: {op_name}")

                # Assign inputs according to number format
                if number_format == NumberFormat.DECIMAL:
                    lines.append(f"        a = {bitwidth}'d{scenario['a']};")
                    lines.append(f"        b = {bitwidth}'d{scenario['b']};")
                else:
                    lines.append(f"        a = {bitwidth}'h{scenario['a']:X};")
                    lines.append(f"        b = {bitwidth}'h{scenario['b']:X};")

                lines.append(f"        opcode = {opcode_width}'b{scenario['opcode']};")
                lines.append(f"        #10;")

                # Display result using consistent format
                lines.append(
                    f'        $display("Test {i}: {format_spec} {op_name} {format_spec} = {format_spec} (C=%b Z=%b N=%b V=%b)",'
                )
                lines.append(f'                 a, b, result, carry, zero, negative, overflow);')

                # Check expected result if available
                if 'expected_result' in scenario:
                    expected = scenario['expected_result']
                    lines.append(f"        if (result == {bitwidth}'d{expected}) begin")
                    lines.append(f'            $display("  ✓ PASS");')
                    lines.append(f"            passed = passed + 1;")
                    lines.append(f"        end else begin")
                    lines.append(
                        f'            $display("  ✗ FAIL - Expected: {format_spec}", {bitwidth}\'d{expected});'
                    )
                    lines.append(f"            failed = failed + 1;")
                    lines.append(f"        end")

                lines.append("")
        else:
            # If no scenarios, generate some default tests
            lines.append(f"        // Default test cases")
            test_cases = [
                (15, 10, '0000', 'ADD'),
                (32, 16, '0001', 'SUB'),
            ]

            for i, (a_val, b_val, op_code, op_name) in enumerate(test_cases, 1):
                lines.append(f"        // Test {i}: {op_name}")
                if number_format == NumberFormat.DECIMAL:
                    lines.append(f"        a = {bitwidth}'d{a_val};")
                    lines.append(f"        b = {bitwidth}'d{b_val};")
                else:
                    lines.append(f"        a = {bitwidth}'h{a_val:X};")
                    lines.append(f"        b = {bitwidth}'h{b_val:X};")
                lines.append(f"        opcode = {opcode_width}'b{op_code};")
                lines.append(f"        #10;")
                lines.append(
                    f'        $display("Test {i} ({op_name}): {format_spec} op {format_spec} = {format_spec}",'
                )
                lines.append(f'                 a, b, result);')
                lines.append("")

        lines.append(f'        $display("");')
        lines.append(f'        $display("========================================");')
        lines.append(f'        $display("Test Summary");')
        lines.append(f'        $display("========================================");')
        lines.append(f'        $display("Total tests: %0d", passed + failed);')
        lines.append(f'        $display("Passed: %0d", passed);')
        lines.append(f'        $display("Failed: %0d", failed);')
        if scenarios and any('expected_result' in s for s in scenarios):
            lines.append(f'        if (failed == 0)')
            lines.append(f'            $display("✓ All tests PASSED!");')
            lines.append(f'        else')
            lines.append(f'            $display("✗ Some tests FAILED!");')
        lines.append(f'        $display("========================================");')
        lines.append("        #10;")
        lines.append("        $finish;")
        lines.append("    end")
        lines.append("")

        # VCD dump
        lines.append("    // VCD dump for waveform viewing")
        lines.append("    initial begin")
        lines.append(f'        $dumpfile("{module_name}_tb.vcd");')
        lines.append(f"        $dumpvars(0, {module_name}_tb);")
        lines.append("    end")
        lines.append("")
        lines.append("endmodule")

        return '\n'.join(lines)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Generate Verilog code from BDD feature files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Use default paths
  python verilog_generator_enhanced.py

  # Specify input and output directories
  python verilog_generator_enhanced.py -b ./bdd -v ./verilog

  # Use absolute paths
  python verilog_generator_enhanced.py -b D:\\path\\to\\bdd -v D:\\path\\to\\verilog
        '''
    )

    parser.add_argument('-b', '--bdd-dir',
                        help='Directory containing BDD feature files',
                        default=None)

    parser.add_argument('-v', '--verilog-dir',
                        help='Directory for Verilog output files',
                        default=None)

    args = parser.parse_args()

    print("-" * 60)
    print("Verilog Generator - Generate Verilog code from BDD feature files")
    print("Supports dynamic paths and automatic number format detection")
    print("-" * 60)
    print()

    # Create generator instance
    generator = VerilogGenerator(
        bdd_dir=args.bdd_dir,
        verilog_dir=args.verilog_dir
    )

    # Process all feature files
    generator.process_all_features()

    print()
    print("-" * 60)
    print("Generation finished!")
    print("-" * 60)


if __name__ == "__main__":
    main()
