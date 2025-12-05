"""
ALU Generator - Generate Verilog ALU module from specification
==============================================================

This module reads spec files (JSON/TXT) and generates Verilog ALU design.
NO LLM is used - this is a deterministic transformation.

ARCHITECTURE:
    spec.txt / spec.json  (from spec_generator.py)
           │
           ▼
    ┌──────────────────┐
    │   alu_generator  │  ◄── NO LLM (deterministic)
    │    (this file)   │
    └────────┬─────────┘
             │
             ▼
         ALU.v (DUT)
             │
             ▼
      verilog_generator.py → testbench.v

PURPOSE:
- Read specification from specs directory
- Generate synthesizable Verilog ALU module
- Support all standard ALU operations
- Generate proper flag logic (zero, carry, overflow, negative)

This ensures:
1. Deterministic output (same spec → same ALU.v)
2. Independence from LLM (no API calls needed)
3. Traceability (Verilog directly maps to spec)
4. Independence from BDD generation
"""

import os
import re
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Union
from datetime import datetime


class ALUGenerator:
    """
    Generate Verilog ALU module from specification.

    This is a deterministic generator - NO LLM is used.
    Same input spec always produces the same Verilog output.
    """

    # Standard ALU operations with Verilog implementation
    OPERATION_TEMPLATES = {
        "ADD": {
            "opcode": "0000",
            "verilog": "{result_full} = a + b;",
            "description": "Addition (A + B)"
        },
        "SUB": {
            "opcode": "0001",
            "verilog": "{result_full} = a - b;",
            "description": "Subtraction (A - B)"
        },
        "AND": {
            "opcode": "0010",
            "verilog": "{result_full} = {{1'b0, a & b}};",
            "description": "Bitwise AND (A & B)"
        },
        "OR": {
            "opcode": "0011",
            "verilog": "{result_full} = {{1'b0, a | b}};",
            "description": "Bitwise OR (A | B)"
        },
        "XOR": {
            "opcode": "0100",
            "verilog": "{result_full} = {{1'b0, a ^ b}};",
            "description": "Bitwise XOR (A ^ B)"
        },
        "NOT": {
            "opcode": "0101",
            "verilog": "{result_full} = {{1'b0, ~a}};",
            "description": "Bitwise NOT (~A)"
        },
        "SHL": {
            "opcode": "0110",
            "verilog": "{result_full} = {{1'b0, a << b[3:0]}};",
            "description": "Shift Left (A << B)"
        },
        "SHR": {
            "opcode": "0111",
            "verilog": "{result_full} = {{1'b0, a >> b[3:0]}};",
            "description": "Shift Right (A >> B)"
        },
    }

    def __init__(
            self,
            spec_dir: Optional[str] = None,
            output_dir: Optional[str] = None,
            project_root: Optional[str] = None,
            debug: bool = True
    ):
        """
        Initialize ALU generator.

        Args:
            spec_dir: Directory containing spec files
            output_dir: Directory to save Verilog files
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
        ]

        for path in search_paths:
            if path.exists():
                self._debug_print(f"Found specs at: {path}", "SUCCESS")
                return path

        # 4. Fallback to absolute path (Windows specific)
        fallback_path = Path(r"D:\DE\HdlFormalVerifierLLM\HdlFormalVerifier\specs")
        if fallback_path.exists():
            self._debug_print(f"Using fallback path: {fallback_path}", "INFO")
            return fallback_path

        # 5. Create default location
        default = current / "specs"
        default.mkdir(parents=True, exist_ok=True)
        print(f"⚠️  No existing specs directory found, created: {default}")
        return default

    def _setup_output_dir(self, output_dir: Optional[str]) -> Path:
        """Setup output directory for Verilog files"""
        if output_dir:
            path = Path(output_dir)
        elif self.project_root:
            path = self.project_root / "output" / "verilog"
        else:
            # Put output next to specs directory (output/verilog, same level as output/bdd)
            path = self.spec_dir.parent / "output" / "verilog"

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
                for k, v in list(self.OPERATION_TEMPLATES.items())[:4]
            ]

        spec = {
            "module_name": f"alu_{bitwidth}bit",
            "bitwidth": bitwidth,
            "operations": operations,
            "raw_text": content
        }

        self._debug_print(f"Parsed TXT spec: {spec['module_name']}", "SUCCESS")
        return spec

    def generate_verilog(self, spec: Dict) -> str:
        """
        Generate Verilog ALU module from specification.

        Args:
            spec: Specification dictionary

        Returns:
            Complete Verilog module code
        """
        print(f"\n🔧 Generating Verilog ALU module...")

        bitwidth = spec.get('bitwidth', 16)
        module_name = spec.get('module_name', f'alu_{bitwidth}bit')
        operations = spec.get('operations', [])

        self._debug_print(f"Module: {module_name}", "DEBUG")
        self._debug_print(f"Bitwidth: {bitwidth}", "DEBUG")
        self._debug_print(f"Operations: {len(operations)}", "DEBUG")

        # Generate Verilog code
        verilog = self._generate_module_header(module_name, bitwidth)
        verilog += self._generate_port_declarations(bitwidth)
        verilog += self._generate_internal_signals(bitwidth)
        verilog += self._generate_alu_logic(operations, bitwidth)
        verilog += self._generate_flag_logic(bitwidth)
        verilog += self._generate_module_footer()

        self._debug_print(f"Generated {len(verilog)} characters of Verilog", "SUCCESS")
        return verilog

    def _generate_module_header(self, module_name: str, bitwidth: int) -> str:
        """Generate module header with comments"""
        return f"""//==============================================================================
// ALU Module: {module_name}
// Bitwidth: {bitwidth}-bit
// Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
// Generator: alu_generator.py (deterministic, no LLM)
//
// This is a synthesizable {bitwidth}-bit ALU module generated from specification.
// All operations are purely combinational.
//==============================================================================

`timescale 1ns / 1ps

module {module_name} (
"""

    def _generate_port_declarations(self, bitwidth: int) -> str:
        """Generate port declarations"""
        return f"""    // Inputs
    input  wire [{bitwidth - 1}:0] a,           // First operand
    input  wire [{bitwidth - 1}:0] b,           // Second operand
    input  wire [3:0]  opcode,      // Operation select

    // Outputs
    output reg  [{bitwidth - 1}:0] result,      // Operation result
    output reg         zero,        // Zero flag (result == 0)
    output reg         carry,       // Carry/borrow flag
    output reg         overflow,    // Overflow flag (signed)
    output reg         negative     // Negative flag (MSB of result)
);

"""

    def _generate_internal_signals(self, bitwidth: int) -> str:
        """Generate internal signal declarations"""
        return f"""    //--------------------------------------------------------------------------
    // Internal Signals
    //--------------------------------------------------------------------------
    reg [{bitwidth}:0] result_full;  // Extended result for carry detection

    // Operation opcodes (from specification)
    localparam OP_ADD = 4'b0000;
    localparam OP_SUB = 4'b0001;
    localparam OP_AND = 4'b0010;
    localparam OP_OR  = 4'b0011;
    localparam OP_XOR = 4'b0100;
    localparam OP_NOT = 4'b0101;
    localparam OP_SHL = 4'b0110;
    localparam OP_SHR = 4'b0111;

"""

    def _generate_alu_logic(self, operations: List[Dict], bitwidth: int) -> str:
        """Generate ALU operation logic"""
        code = f"""    //--------------------------------------------------------------------------
    // ALU Operation Logic (Combinational)
    //--------------------------------------------------------------------------
    always @(*) begin
        // Default values
        result_full = {{{bitwidth + 1}'b0}};

        case (opcode)
"""

        # Generate case for each operation
        for op in operations:
            op_name = op.get('name', 'UNKNOWN')
            opcode = op.get('opcode', '0000')

            # Get Verilog template for this operation
            if op_name in self.OPERATION_TEMPLATES:
                template = self.OPERATION_TEMPLATES[op_name]
                verilog_code = template['verilog'].format(result_full='result_full')
                description = template['description']
            else:
                # Fallback for unknown operations
                verilog_code = f"result_full = {{{bitwidth + 1}'b0}};"
                description = f"Unknown operation: {op_name}"

            code += f"""            // {description}
            4'b{opcode}: begin
                {verilog_code}
            end

"""

        # Default case
        code += f"""            // Default: No operation
            default: begin
                result_full = {{{bitwidth + 1}'b0}};
            end
        endcase
    end

"""
        return code

    def _generate_flag_logic(self, bitwidth: int) -> str:
        """Generate flag calculation logic"""
        return f"""    //--------------------------------------------------------------------------
    // Flag Logic
    //--------------------------------------------------------------------------
    always @(*) begin
        // Extract result (lower {bitwidth} bits)
        result = result_full[{bitwidth - 1}:0];

        // Zero flag: set when result is zero
        zero = (result == {bitwidth}'b0);

        // Carry flag: set from extended bit (for ADD/SUB)
        carry = result_full[{bitwidth}];

        // Negative flag: set when MSB is 1 (signed interpretation)
        negative = result[{bitwidth - 1}];

        // Overflow flag: signed overflow detection
        // For ADD: overflow when both operands have same sign but result has different sign
        // For SUB: overflow when operands have different signs and result sign != a's sign
        case (opcode)
            OP_ADD: overflow = (a[{bitwidth - 1}] == b[{bitwidth - 1}]) && (result[{bitwidth - 1}] != a[{bitwidth - 1}]);
            OP_SUB: overflow = (a[{bitwidth - 1}] != b[{bitwidth - 1}]) && (result[{bitwidth - 1}] != a[{bitwidth - 1}]);
            default: overflow = 1'b0;
        endcase
    end

"""

    def _generate_module_footer(self) -> str:
        """Generate module footer"""
        return """endmodule

//==============================================================================
// End of ALU Module
//==============================================================================
"""

    def generate_all(self) -> List[Path]:
        """
        Generate Verilog files for all specs in the directory.

        Returns:
            List of generated Verilog file paths
        """
        print("\n" + "=" * 70)
        print("🚀 ALU Generator - Starting Verilog generation")
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

                # Generate Verilog content
                verilog_content = self.generate_verilog(spec)

                # Save Verilog file
                module_name = spec.get('module_name', 'alu')
                verilog_name = f"{module_name}.v"
                verilog_path = self.output_dir / verilog_name

                with open(verilog_path, 'w', encoding='utf-8') as f:
                    f.write(verilog_content)

                print(f"\n✅ Generated: {verilog_path}")
                generated_files.append(verilog_path)

            except Exception as e:
                print(f"\n❌ Error processing {spec_path.name}: {e}")
                if self.debug:
                    import traceback
                    traceback.print_exc()

        print("\n" + "=" * 70)
        print(f"✨ Generation complete! Created {len(generated_files)} Verilog file(s)")
        print("=" * 70)

        return generated_files

    def generate_from_spec_file(self, spec_path: Union[str, Path]) -> Path:
        """
        Generate Verilog file from a specific spec file.

        Args:
            spec_path: Path to the spec file

        Returns:
            Path to generated Verilog file
        """
        spec_path = Path(spec_path)

        if not spec_path.exists():
            raise FileNotFoundError(f"Spec file not found: {spec_path}")

        spec = self.load_spec(spec_path)
        verilog_content = self.generate_verilog(spec)

        module_name = spec.get('module_name', 'alu')
        verilog_name = f"{module_name}.v"
        verilog_path = self.output_dir / verilog_name

        with open(verilog_path, 'w', encoding='utf-8') as f:
            f.write(verilog_content)

        print(f"\n✅ Generated: {verilog_path}")
        return verilog_path


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Generate Verilog ALU module from specification (No LLM)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Generate from all specs in default directory
  python alu_generator.py

  # Specify spec directory
  python alu_generator.py --spec-dir ./specs

  # Generate from specific spec file
  python alu_generator.py --spec-file ./specs/alu_16bit_spec.json

  # Custom output directory
  python alu_generator.py --output-dir ./output/verilog

  # With project root
  python alu_generator.py --project-root D:/DE/HdlFormalVerifierLLM/HdlFormalVerifier

Note: This generator does NOT use LLM. It performs deterministic
transformation from spec to synthesizable Verilog code.

Output Structure:
  output/
  ├── bdd/           ← from bdd_generator.py
  │   └── *.feature
  └── verilog/       ← from this file
      └── *.v
        '''
    )

    parser.add_argument('--spec-dir', help='Directory containing spec files')
    parser.add_argument('--spec-file', help='Specific spec file to process')
    parser.add_argument('--output-dir', help='Output directory for Verilog files')
    parser.add_argument('--project-root', help='Project root directory')
    parser.add_argument('--debug', action='store_true', default=True, help='Enable debug output')

    args = parser.parse_args()

    # Create generator
    generator = ALUGenerator(
        spec_dir=args.spec_dir,
        output_dir=args.output_dir,
        project_root=args.project_root,
        debug=args.debug
    )

    # Generate
    if args.spec_file:
        # Single file mode
        verilog_path = generator.generate_from_spec_file(args.spec_file)
        print(f"\n📄 Generated: {verilog_path}")
    else:
        # Batch mode - process all specs
        generated = generator.generate_all()

        if generated:
            print("\n📋 Generated files:")
            for path in generated:
                print(f"   • {path}")

            print("\n📋 NEXT STEPS:")
            print("=" * 70)
            print(f"1. Review Verilog files in: {generator.output_dir}")
            print(f"2. Generate testbench: python verilog_generator.py")
            print(f"3. Run simulation:     iverilog -o sim ALU.v testbench.v && vvp sim")
            print(f"4. View waveform:      gtkwave waveform.vcd")
            print("=" * 70)


if __name__ == "__main__":
    main()