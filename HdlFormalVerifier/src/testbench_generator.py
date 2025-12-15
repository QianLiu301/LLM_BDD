"""
Testbench Generator - Generate Verilog testbench from BDD .feature files
=========================================================================

ENHANCED VERSION: Multi-LLM Support with organized directory structure

This module reads .feature files and generates Verilog testbench.
NO LLM is used - this is a deterministic transformation.

ARCHITECTURE:
    output/bdd/{llm}/*.feature  (from bdd_generator.py)
           │
           ▼
    ┌────────────────────┐
    │ testbench_generator│  ◄── NO LLM (deterministic)
    │    (this file)     │
    └──────────┬─────────┘
               │
               ▼
       output/verilog/{llm}/testbench.v
               │
               ├── Tests ──► output/dut/alu.v (fixed DUT)
               │
               ▼
          Simulation (iverilog)

DIRECTORY MAPPING:
    output/bdd/groq/alu_16bit.feature     → output/verilog/groq/alu_16bit_tb.v
    output/bdd/deepseek/alu_16bit.feature → output/verilog/deepseek/alu_16bit_tb.v
    output/bdd/openai/alu_16bit.feature   → output/verilog/openai/alu_16bit_tb.v

All testbenches test the SAME fixed DUT: output/dut/alu_16bit.v

NEW FEATURES:
- ✅ Recursive scanning of LLM subdirectories
- ✅ Maintains LLM-specific directory structure
- ✅ Batch processing for all LLMs
- ✅ No hardcoded paths
- ✅ Automatic DUT detection

PURPOSE:
- Read .feature files from output/bdd/{llm}/
- Parse test scenarios and expected results
- Generate testbench.v for each LLM in separate directories
- Support decimal/hexadecimal number formats
- Generate VCD dump for waveform viewing

This ensures:
1. Fair comparison (all test the same DUT)
2. Deterministic output (same .feature → same testbench)
3. Independence from LLM (no API calls needed)
4. Clear organization by LLM provider
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
            for key in ['a', 'b', 'result']:
                if key in scenario:
                    val = scenario[key]
                    if isinstance(val, int) and val > max_value:
                        max_value = val

        # Determine bitwidth from max value
        if max_value <= 255:
            return 8
        elif max_value <= 65535:
            return 16
        elif max_value <= 4294967295:
            return 32
        else:
            return 64

    def _detect_number_format(self, content: str):
        """Detect number format from content"""
        # Check for hex patterns (0x...)
        if re.search(r'\b0x[0-9a-fA-F]+\b', content):
            self.number_format = NumberFormat.HEXADECIMAL
        # Check for binary patterns (0b...)
        elif re.search(r'\b0b[01]+\b', content):
            self.number_format = NumberFormat.BINARY
        else:
            self.number_format = NumberFormat.DECIMAL

    def _extract_operations(self, content: str):
        """Extract operation-to-opcode mapping"""
        # Look for opcode definitions in comments or background
        opcode_pattern = r'(\w+)\s*(?:operation|opcode|code)?\s*(?:is|=|:)?\s*["\']?([0-9a-fA-Fx]+)["\']?'

        for match in re.finditer(opcode_pattern, content, re.IGNORECASE):
            op_name = match.group(1).upper()
            opcode = match.group(2)

            # Normalize opcode
            if opcode.startswith('0x') or opcode.startswith('0X'):
                opcode_int = int(opcode, 16)
                opcode = format(opcode_int, '04b')
            elif all(c in '01' for c in opcode):
                opcode = opcode.zfill(4)

            self.operations[opcode] = op_name

    def _extract_scenarios(self, content: str):
        """Extract test scenarios from Examples tables"""
        # Find all Examples sections
        examples_pattern = r'Examples?:\s*\n((?:\s*\|.*\n)+)'

        for match in re.finditer(examples_pattern, content, re.MULTILINE):
            table_text = match.group(1)
            rows = [row.strip() for row in table_text.strip().split('\n')]

            if len(rows) < 2:
                continue

            # Parse header
            header = [col.strip() for col in rows[0].split('|') if col.strip()]

            # Parse data rows
            for row in rows[1:]:
                cols = [col.strip() for col in row.split('|') if col.strip()]

                if len(cols) != len(header):
                    continue

                scenario = {}
                for col_name, col_value in zip(header, cols):
                    parsed_value = self._parse_value(col_value)
                    if parsed_value is not None:
                        scenario[col_name.lower()] = parsed_value
                    else:
                        scenario[col_name.lower()] = col_value

                if scenario:
                    self.scenarios.append(scenario)

    def _parse_value(self, value_str: str) -> Optional[int]:
        """Parse value (hex/decimal/binary)"""
        value_str = str(value_str).strip()

        try:
            # Binary (0b...)
            if value_str.startswith('0b') or value_str.startswith('0B'):
                return int(value_str, 2)

            # Hexadecimal (0x...)
            if value_str.startswith('0x') or value_str.startswith('0X'):
                return int(value_str, 16)

            # Decimal
            return int(value_str)
        except (ValueError, AttributeError):
            return None


class TestbenchGenerator:
    """
    Generate Verilog testbench from .feature files.

    ENHANCED: Multi-LLM support with directory structure preservation

    This generator ONLY creates testbench files.
    ALU design is generated separately by alu_generator.py.
    """

    def __init__(
        self,
        feature_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
        dut_dir: Optional[str] = None,
        project_root: Optional[str] = None,
        dut_module_name: Optional[str] = None,
        debug: bool = True
    ):
        """
        Initialize testbench generator.

        Args:
            feature_dir: Directory containing .feature files (supports LLM subdirs)
            output_dir: Base directory to save testbench files
            dut_dir: Directory containing DUT .v files (default: output/dut)
            project_root: Project root directory
            dut_module_name: Name of DUT module (auto-detected if not specified)
            debug: Enable debug output
        """
        self.debug = debug
        self.project_root = Path(project_root) if project_root else None
        self.dut_module_name = dut_module_name

        # Setup paths
        self.feature_dir = self._find_feature_dir(feature_dir)
        self.output_base_dir = self._setup_output_base_dir(output_dir)
        self.dut_dir = self._find_dut_dir(dut_dir)

        print(f"📁 Feature directory: {self.feature_dir}")
        print(f"📁 Output base directory: {self.output_base_dir}")
        print(f"📁 DUT directory: {self.dut_dir}")

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

        # 4. Create default
        default = current / "output" / "bdd"
        default.mkdir(parents=True, exist_ok=True)
        print(f"⚠️  No feature directory found, created: {default}")
        return default

    def _setup_output_base_dir(self, output_dir: Optional[str]) -> Path:
        """Setup base output directory for testbench files"""
        if output_dir:
            path = Path(output_dir)
        elif self.project_root:
            path = self.project_root / "output" / "verilog"
        else:
            # Same level as feature directory
            path = self.feature_dir.parent / "verilog"

        path.mkdir(parents=True, exist_ok=True)
        return path

    def _find_dut_dir(self, dut_dir: Optional[str]) -> Path:
        """Find DUT directory"""
        if dut_dir:
            path = Path(dut_dir)
            if path.exists():
                return path

        # Default: output/dut
        if self.project_root:
            path = self.project_root / "output" / "dut"
        else:
            path = self.feature_dir.parent / "dut"

        if not path.exists():
            print(f"⚠️  DUT directory not found: {path}")
            print(f"   Creating directory...")
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

    def scan_features(self) -> List[Tuple[Path, str]]:
        """
        Scan for .feature files, including LLM subdirectories.

        Returns:
            List of (feature_path, llm_name) tuples
        """
        print(f"\n🔍 Scanning for .feature files in: {self.feature_dir}")

        feature_files = []

        # Check root directory
        for f in self.feature_dir.glob("*.feature"):
            feature_files.append((f, "default"))

        # Check LLM subdirectories
        for subdir in self.feature_dir.iterdir():
            if subdir.is_dir():
                llm_name = subdir.name
                for f in subdir.glob("*.feature"):
                    feature_files.append((f, llm_name))

        if not feature_files:
            print(f"   ⚠️  No .feature files found")
            return []

        print(f"   ✅ Found {len(feature_files)} feature file(s):")

        # Group by LLM for display
        by_llm = {}
        for path, llm in feature_files:
            if llm not in by_llm:
                by_llm[llm] = []
            by_llm[llm].append(path)

        for llm, paths in sorted(by_llm.items()):
            print(f"      📂 {llm}:")
            for p in paths:
                print(f"         • {p.name}")

        return feature_files

    def _detect_dut_module(self, bitwidth: int = None) -> str:
        """
        Detect DUT module name.

        Priority:
        1. Explicitly specified dut_module_name
        2. Infer from bitwidth (alu_{bitwidth}bit)
        3. Search for existing ALU .v files in DUT directory
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

        # 3. Search for ALU .v files in DUT directory
        verilog_files = list(self.dut_dir.glob("alu_*.v"))

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

    def generate_testbench(self, spec: Dict, feature_name: str, llm_name: str = "default") -> str:
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
        lines.append(f"// LLM Provider: {llm_name}")
        lines.append(f"// Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"// Generator: testbench_generator.py (deterministic, no LLM)")
        lines.append(f"//")
        lines.append(f"// DUT Module: {dut_module} (from {self.dut_dir})")
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
        lines.append(f"    reg clk;")
        lines.append(f"    reg rst;")
        lines.append(f"    reg  [{bitwidth-1}:0] a;")
        lines.append(f"    reg  [{bitwidth-1}:0] b;")
        lines.append(f"    reg  [{opcode_width-1}:0] opcode;")
        lines.append(f"    wire [{bitwidth-1}:0] result;")
        lines.append(f"    wire zero, overflow, negative;")
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
        lines.append(f"    {dut_module} dut (")
        lines.append(f"        .clk(clk),")
        lines.append(f"        .rst(rst),")
        lines.append(f"        .a(a),")
        lines.append(f"        .b(b),")
        lines.append(f"        .opcode(opcode),")
        lines.append(f"        .result(result),")
        lines.append(f"        .zero(zero),")
        lines.append(f"        .overflow(overflow),")
        lines.append(f"        .negative(negative)")
        lines.append(f"    );")
        lines.append("")

        # Clock generation
        lines.append("    //--------------------------------------------------------------------------")
        lines.append("    // Clock Generation")
        lines.append("    //--------------------------------------------------------------------------")
        lines.append("    initial begin")
        lines.append("        clk = 0;")
        lines.append("        forever #5 clk = ~clk;  // 100MHz clock")
        lines.append("    end")
        lines.append("")

        # Test stimulus
        lines.append("    //--------------------------------------------------------------------------")
        lines.append("    // Test Stimulus")
        lines.append("    //--------------------------------------------------------------------------")
        lines.append("    initial begin")
        lines.append("        // VCD dump for waveform")
        lines.append(f"        $dumpfile(\"{feature_name}_{llm_name}.vcd\");")
        lines.append("        $dumpvars(0, dut);")
        lines.append("")
        lines.append(f"        $display(\"\\n{'='*70}\");")
        lines.append(f"        $display(\"Testbench: {feature_name}\");")
        lines.append(f"        $display(\"LLM Provider: {llm_name}\");")
        lines.append(f"        $display(\"DUT: {dut_module} ({bitwidth}-bit)\");")
        lines.append(f"        $display(\"Test cases: {len(scenarios)}\");")
        lines.append(f"        $display(\"{'='*70}\\n\");")
        lines.append("")
        lines.append("        // Reset")
        lines.append("        rst = 1;")
        lines.append("        #20 rst = 0;")
        lines.append("        #10;")
        lines.append("")

        # Generate test cases
        lines.append("        //----------------------------------------------------------------------")
        lines.append("        // Test Cases")
        lines.append("        //----------------------------------------------------------------------")

        for i, scenario in enumerate(scenarios, 1):
            a_val = scenario.get('a', 0)
            b_val = scenario.get('b', 0)
            op_val = scenario.get('opcode', scenario.get('operation', '0000'))
            expected = scenario.get('result', scenario.get('expected', 0))

            # Convert opcode to binary if needed
            if isinstance(op_val, str):
                if op_val.startswith('0x'):
                    op_int = int(op_val, 16)
                    op_val = format(op_int, '04b')
                elif not all(c in '01' for c in op_val):
                    # Try to find in operations mapping
                    op_val = '0000'

            # Format numbers based on number format
            if number_format == NumberFormat.HEXADECIMAL:
                a_str = f"{bitwidth}'h{a_val:X}"
                b_str = f"{bitwidth}'h{b_val:X}"
                exp_str = f"{bitwidth}'h{expected:X}"
            else:
                a_str = f"{bitwidth}'d{a_val}"
                b_str = f"{bitwidth}'d{b_val}"
                exp_str = f"{bitwidth}'d{expected}"

            lines.append(f"        // Test case {i}")
            lines.append(f"        a = {a_str}; b = {b_str}; opcode = 4'b{op_val};")
            lines.append(f"        #10;  // Wait for one clock cycle")
            lines.append(f"        #10;  // Wait for result")
            lines.append(f"        total = total + 1;")
            lines.append(f"        if (result == {exp_str}) begin")
            lines.append(f"            $display(\"✅ Test {i} PASSED: %d op %d = %d\", a, b, result);")
            lines.append(f"            passed = passed + 1;")
            lines.append(f"        end else begin")
            lines.append(f"            $display(\"❌ Test {i} FAILED: %d op %d = %d (expected %d)\", a, b, result, {expected});")
            lines.append(f"            failed = failed + 1;")
            lines.append(f"        end")
            lines.append("")

        # Summary
        lines.append("        //----------------------------------------------------------------------")
        lines.append("        // Test Summary")
        lines.append("        //----------------------------------------------------------------------")
        lines.append(f"        $display(\"\\n{'='*70}\");")
        lines.append(f"        $display(\"Test Summary for {llm_name}\");")
        lines.append(f"        $display(\"{'='*70}\");")
        lines.append("        $display(\"Total:  %0d\", total);")
        lines.append("        $display(\"Passed: %0d\", passed);")
        lines.append("        $display(\"Failed: %0d\", failed);")
        lines.append("")
        lines.append("        if (failed == 0) begin")
        lines.append(f"            $display(\"\\n🎉 ALL TESTS PASSED!\");")
        lines.append("        end else begin")
        lines.append(f"            $display(\"\\n⚠️  SOME TESTS FAILED\");")
        lines.append("        end")
        lines.append(f"        $display(\"{'='*70}\\n\");")
        lines.append("")
        lines.append("        $finish;")
        lines.append("    end")
        lines.append("")
        lines.append("endmodule")

        return '\n'.join(lines)

    def generate_all(self) -> Dict[str, List[Path]]:
        """
        Generate testbenches for all .feature files.

        Returns:
            Dictionary mapping LLM names to list of generated testbench paths
        """
        print("\n" + "=" * 70)
        print("🚀 Testbench Generator - Multi-LLM Mode")
        print("=" * 70)

        # Scan for features
        feature_files = self.scan_features()

        if not feature_files:
            print("\n❌ No .feature files found. Run bdd_generator.py first.")
            return {}

        generated_by_llm = {}

        for feature_path, llm_name in feature_files:
            try:
                print(f"\n📖 Processing: {llm_name}/{feature_path.name}")

                # Parse feature file
                parser = FeatureParser(str(feature_path), debug=self.debug)
                spec = parser.parse()

                self._debug_print(f"Bitwidth: {spec['bitwidth']}", "DEBUG")
                self._debug_print(f"Scenarios: {len(spec['scenarios'])}", "DEBUG")

                # Generate testbench
                feature_name = feature_path.stem
                tb_content = self.generate_testbench(spec, feature_name, llm_name)

                # Create LLM-specific output directory
                llm_output_dir = self.output_base_dir / llm_name
                llm_output_dir.mkdir(parents=True, exist_ok=True)

                # Save testbench
                tb_path = llm_output_dir / f"{feature_name}_tb.v"
                with open(tb_path, 'w', encoding='utf-8') as f:
                    f.write(tb_content)

                print(f"   ✅ Generated: {tb_path.relative_to(self.output_base_dir.parent)}")

                # Track by LLM
                if llm_name not in generated_by_llm:
                    generated_by_llm[llm_name] = []
                generated_by_llm[llm_name].append(tb_path)

            except Exception as e:
                print(f"   ❌ Error: {e}")
                if self.debug:
                    import traceback
                    traceback.print_exc()

        # Summary
        print("\n" + "=" * 70)
        print(f"✨ Generation Complete!")
        print("=" * 70)

        total = sum(len(files) for files in generated_by_llm.values())
        print(f"\n📊 Summary:")
        print(f"   Total testbenches: {total}")
        print(f"   LLM providers: {len(generated_by_llm)}")
        print()

        for llm_name, files in sorted(generated_by_llm.items()):
            print(f"   📂 {llm_name}: {len(files)} testbench(es)")
            for f in files:
                print(f"      • {f.name}")

        return generated_by_llm

    def generate_from_feature_file(self, feature_path: Union[str, Path], llm_name: str = "default") -> Path:
        """Generate testbench from a specific .feature file"""
        feature_path = Path(feature_path)

        if not feature_path.exists():
            raise FileNotFoundError(f"Feature file not found: {feature_path}")

        parser = FeatureParser(str(feature_path), debug=self.debug)
        spec = parser.parse()

        feature_name = feature_path.stem
        tb_content = self.generate_testbench(spec, feature_name, llm_name)

        # Create LLM-specific output directory
        llm_output_dir = self.output_base_dir / llm_name
        llm_output_dir.mkdir(parents=True, exist_ok=True)

        tb_path = llm_output_dir / f"{feature_name}_tb.v"
        with open(tb_path, 'w', encoding='utf-8') as f:
            f.write(tb_content)

        print(f"\n✅ Generated: {tb_path}")
        return tb_path


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Generate Verilog testbench from BDD .feature files (No LLM) - Multi-LLM Support',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Generate from all .feature files (including LLM subdirs)
  python testbench_generator.py

  # Specify feature directory
  python testbench_generator.py --feature-dir ./output/bdd

  # Specify output directory
  python testbench_generator.py --output-dir ./output/verilog

  # Specify DUT directory
  python testbench_generator.py --dut-dir ./output/dut

  # Process single feature file
  python testbench_generator.py --feature-file ./output/bdd/groq/alu_16bit.feature --llm groq

  # With project root
  python testbench_generator.py --project-root D:/DE/RQ/MultiLLM_BDD_Comparison/HdlFormalVerifier

DIRECTORY STRUCTURE:
  This generator supports multi-LLM organization:
  
  Input:
    output/bdd/
    ├── groq/
    │   └── alu_16bit.feature
    ├── deepseek/
    │   └── alu_16bit.feature
    └── openai/
        └── alu_16bit.feature

  Output:
    output/verilog/
    ├── groq/
    │   └── alu_16bit_tb.v       → tests output/dut/alu_16bit.v
    ├── deepseek/
    │   └── alu_16bit_tb.v       → tests output/dut/alu_16bit.v
    └── openai/
        └── alu_16bit_tb.v       → tests output/dut/alu_16bit.v

  All testbenches test the SAME fixed DUT in output/dut/

SIMULATION:
  cd output/verilog/groq
  iverilog -o sim ../../dut/alu_16bit.v alu_16bit_tb.v
  vvp sim
  gtkwave *.vcd
        '''
    )

    parser.add_argument('--feature-dir', help='Directory containing .feature files (supports LLM subdirs)')
    parser.add_argument('--feature-file', help='Specific .feature file to process')
    parser.add_argument('--output-dir', help='Base output directory for testbench files')
    parser.add_argument('--dut-dir', help='Directory containing DUT .v files (default: output/dut)')
    parser.add_argument('--project-root', help='Project root directory')
    parser.add_argument('--dut-module', help='DUT module name (auto-detected if not specified)')
    parser.add_argument('--llm', help='LLM provider name (for single file mode)')
    parser.add_argument('--debug', action='store_true', default=True, help='Enable debug output')

    args = parser.parse_args()

    # Create generator
    generator = TestbenchGenerator(
        feature_dir=args.feature_dir,
        output_dir=args.output_dir,
        dut_dir=args.dut_dir,
        project_root=args.project_root,
        dut_module_name=args.dut_module,
        debug=args.debug
    )

    # Generate
    if args.feature_file:
        # Single file mode
        llm_name = args.llm or "default"
        tb_path = generator.generate_from_feature_file(args.feature_file, llm_name)
        print(f"\n📄 Generated: {tb_path}")
    else:
        # Batch mode (multi-LLM)
        generated_by_llm = generator.generate_all()

        if generated_by_llm:
            # Show next steps
            print("\n📋 NEXT STEPS:")
            print("=" * 70)

            dut_module = generator._detect_dut_module()
            dut_path = generator.dut_dir / f"{dut_module}.v"

            print(f"1. Ensure DUT exists: {dut_path}")
            print(f"\n2. Run simulations for each LLM:")

            for llm_name in sorted(generated_by_llm.keys()):
                llm_dir = generator.output_base_dir / llm_name
                print(f"\n   📂 {llm_name}:")
                print(f"      cd {llm_dir}")
                print(f"      iverilog -o sim ../../dut/{dut_module}.v *_tb.v")
                print(f"      vvp sim")
                print(f"      gtkwave *.vcd")

            print("\n" + "=" * 70)


if __name__ == "__main__":
    main()