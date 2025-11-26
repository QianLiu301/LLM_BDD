"""
SPEC Generator - Generate standardized ALU specification from user input
This module uses LLM to convert user requirements into a formal specification

PURPOSE:
This file creates a standardized SPEC document that serves as the single source
of truth for BOTH BDD scenario generation AND Verilog DUT generation.
This ensures independence between test generation and design generation.
"""

import argparse
from pathlib import Path
from typing import Dict, Optional
import json

# Import your existing LLM provider
try:
    from HdlFormalVerifier.tests.bdd_test.llm_providers import get_llm_response
    HAS_LLM = True
except ImportError:
    print("Warning: llm_providers module not found.")
    HAS_LLM = False


class SpecGenerator:
    """Generate standardized specification from user requirements"""

    def __init__(self, output_dir: str = "./specs"):
        """
        Initialize SPEC generator

        Args:
            output_dir: Directory to save generated SPEC files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_spec(self, user_input: str,
                     llm_provider: str = "openai",
                     model: str = "gpt-4") -> Dict[str, any]:
        """
        Generate formal specification from user input

        Args:
            user_input: Natural language requirements from user
            llm_provider: LLM provider to use
            model: Model name

        Returns:
            Dictionary containing the generated specification
        """

        print(f"Generating specification from user input...")

        # Create prompt for LLM
        prompt = self._create_spec_prompt(user_input)

        # Get LLM response
        if HAS_LLM:
            print(f"✓ Using LLM: {llm_provider} / {model}")
            try:
                spec_text = get_llm_response(prompt, provider=llm_provider, model=model)
            except Exception as e:
                print(f"⚠️  LLM call failed: {e}")
                print("⚠️  Falling back to template mode")
                spec_text = self._create_basic_spec(user_input)
        else:
            print(f"⚠️  Using template mode (LLM not available)")
            # Fallback: create a basic spec template
            spec_text = self._create_basic_spec(user_input)

        # Parse and structure the specification
        spec_dict = self._parse_spec_response(spec_text, user_input)

        return spec_dict

    def _create_spec_prompt(self, user_input: str) -> str:
        """Create prompt for LLM to generate specification"""

        prompt = f"""You are a hardware design specification expert. Convert the following user requirements into a formal, detailed ALU (Arithmetic Logic Unit) specification.

User Requirements:
{user_input}

Generate a complete specification document that includes:

1. MODULE INFORMATION
   - Module name (e.g., alu_8bit, alu_16bit)
   - Bit width (e.g., 8-bit, 16-bit, 32-bit)
   - Brief description

2. INTERFACE DEFINITION
   Inputs:
   - a[N-1:0]: First operand (N-bit)
   - b[N-1:0]: Second operand (N-bit)
   - opcode[M-1:0]: Operation select (M-bit)
   
   Outputs:
   - result[N-1:0]: Operation result (N-bit)
   - carry: Carry/borrow flag
   - zero: Zero flag
   - negative: Negative flag
   - overflow: Overflow flag

3. OPERATIONS AND OPCODES
   List all supported operations with opcodes:
   - 0000: ADD - Addition (A + B)
   - 0001: SUB - Subtraction (A - B)
   - 0010: AND - Bitwise AND (A & B)
   - 0011: OR - Bitwise OR (A | B)
   [... continue for all operations]

4. FLAG DEFINITIONS
   - Carry: Set when operation produces a carry
   - Zero: Set when result is zero
   - Negative: Set when result MSB is 1
   - Overflow: Set for signed arithmetic overflow

5. FUNCTIONAL DESCRIPTION
   - Detailed description of ALU behavior
   - How each operation works
   - Edge cases and special conditions

6. TEST EXAMPLES
   Provide at least 5 example operations:
   Example 1: 15 + 10 = 25 (ADD, no overflow)
   Example 2: 255 + 1 = 0 (ADD, with overflow)
   [... more examples]

Format your response as a clear, structured document.
"""
        return prompt

    def _create_basic_spec(self, user_input: str) -> str:
        """Create a basic specification template when LLM is not available"""

        import re
        bitwidth_match = re.search(r'(\d+)[-\s]bit', user_input, re.IGNORECASE)
        bitwidth = int(bitwidth_match.group(1)) if bitwidth_match else 8

        spec = f"""
MODULE INFORMATION
==================
Module Name: alu_{bitwidth}bit
Bit Width: {bitwidth}-bit
Description: {bitwidth}-bit Arithmetic Logic Unit

INTERFACE DEFINITION
====================
Inputs:
  - a[{bitwidth-1}:0]: First operand ({bitwidth}-bit)
  - b[{bitwidth-1}:0]: Second operand ({bitwidth}-bit)
  - opcode[3:0]: Operation select (4-bit)

Outputs:
  - result[{bitwidth-1}:0]: Operation result ({bitwidth}-bit)
  - carry: Carry/borrow flag
  - zero: Zero flag (result == 0)
  - negative: Negative flag (result[{bitwidth-1}] == 1)
  - overflow: Overflow flag

OPERATIONS AND OPCODES
=======================
0000: ADD - Addition (A + B)
0001: SUB - Subtraction (A - B)
0010: AND - Bitwise AND (A & B)
0011: OR  - Bitwise OR (A | B)
0100: XOR - Bitwise XOR (A ^ B)
0101: SHL - Shift Left (A << 1)
0110: SHR - Shift Right (A >> 1)
0111: NOT - Bitwise NOT (~A)

FLAG DEFINITIONS
================
- Carry: Set when addition produces carry or subtraction produces borrow
- Zero: Set when result equals zero
- Negative: Set when result MSB is 1 (for signed interpretation)
- Overflow: Set when signed arithmetic overflow occurs

FUNCTIONAL DESCRIPTION
======================
The ALU performs arithmetic and logic operations based on the opcode input.
Results are computed combinationally and flags are set accordingly.

TEST EXAMPLES
=============
Example 1: 15 ADD 10 = 25 (Carry=0, Zero=0)
Example 2: 255 ADD 1 = 0 (Carry=1, Zero=1, Overflow detected)
Example 3: 20 SUB 10 = 10 (Carry=0, Zero=0)
Example 4: 0xFF AND 0x0F = 0x0F
Example 5: 0xF0 OR 0x0F = 0xFF
"""
        return spec

    def _parse_spec_response(self, spec_text: str, original_input: str) -> Dict:
        """Parse LLM response and structure it"""

        # Extract bit width
        import re
        bitwidth_match = re.search(r'(\d+)[-\s]bit', spec_text, re.IGNORECASE)
        bitwidth = int(bitwidth_match.group(1)) if bitwidth_match else 8

        # Create structured specification
        spec_dict = {
            'original_input': original_input,
            'bitwidth': bitwidth,
            'spec_text': spec_text,
            'timestamp': self._get_timestamp(),
            'version': '1.0'
        }

        return spec_dict

    def save_spec(self, spec_dict: Dict, filename: Optional[str] = None) -> Path:
        """
        Save specification to files

        Args:
            spec_dict: Specification dictionary
            filename: Optional filename (without extension)

        Returns:
            Path to the saved specification file
        """
        if filename is None:
            filename = f"alu_{spec_dict['bitwidth']}bit_spec"

        # Save as text file (main format for paper traceability)
        txt_path = self.output_dir / f"{filename}.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("ALU SPECIFICATION DOCUMENT\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Generated: {spec_dict['timestamp']}\n")
            f.write(f"Version: {spec_dict['version']}\n")
            f.write(f"Bit Width: {spec_dict['bitwidth']}-bit\n")
            f.write("\n" + "=" * 80 + "\n\n")
            f.write("ORIGINAL USER INPUT:\n")
            f.write("-" * 80 + "\n")
            f.write(spec_dict['original_input'] + "\n")
            f.write("\n" + "=" * 80 + "\n\n")
            f.write("FORMAL SPECIFICATION:\n")
            f.write("-" * 80 + "\n")
            f.write(spec_dict['spec_text'] + "\n")
            f.write("\n" + "=" * 80 + "\n")

        # Also save as JSON for easy parsing by other tools
        json_path = self.output_dir / f"{filename}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(spec_dict, f, indent=2, ensure_ascii=False)

        print(f"\n✓ Specification saved to:")
        print(f"  - {txt_path}  (for paper/documentation)")
        print(f"  - {json_path}  (for automated processing)")

        return txt_path

    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Generate formal ALU specification from user requirements',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Interactive mode
  python spec_generator.py
  
  # From command line
  python spec_generator.py -i "8-bit ALU with ADD, SUB, AND, OR operations"
  
  # Specify output directory
  python spec_generator.py -i "16-bit ALU" -o ./my_specs
  
  # Use different LLM provider
  python spec_generator.py -i "32-bit ALU" --llm gemini --model gemini-pro
        '''
    )

    parser.add_argument('-i', '--input',
                       help='User requirements (natural language)',
                       default=None)

    parser.add_argument('-o', '--output-dir',
                       help='Output directory for specification files',
                       default='./specs')

    parser.add_argument('--llm',
                       help='LLM provider to use',
                       default='openai',
                       choices=['openai', 'claude', 'gemini', 'groq', 'deepseek'])

    parser.add_argument('--model',
                       help='Model name to use',
                       default='gpt-4')

    args = parser.parse_args()

    print("=" * 80)
    print("SPEC Generator - Generate Formal ALU Specification")
    print("This creates the single source of truth for independent generation")
    print("=" * 80)
    print()

    # Get user input
    if args.input:
        user_input = args.input
    else:
        print("Please enter your ALU requirements (natural language):")
        print("Example: 'I need an 8-bit ALU that supports ADD, SUB, AND, OR operations'")
        print()
        user_input = input("Your requirements: ").strip()

        if not user_input:
            print("Error: No input provided.")
            return

    print(f"\nUser Input: {user_input}")
    print()

    # Create generator
    generator = SpecGenerator(output_dir=args.output_dir)

    # Generate specification
    spec_dict = generator.generate_spec(
        user_input=user_input,
        llm_provider=args.llm,
        model=args.model
    )

    # Save specification
    spec_path = generator.save_spec(spec_dict)

    print()
    print("=" * 80)
    print("NEXT STEPS (Independent Generation Workflow):")
    print("=" * 80)
    print(f"1. Review the generated specification: {spec_path}")
    print(f"2. Generate BDD scenarios:  python bdd_generator.py -s {spec_path}")
    print(f"3. Generate Verilog DUT:    python alu_generator.py -s {spec_path}")
    print(f"   [Note: Steps 2 and 3 are INDEPENDENT - order doesn't matter]")
    print(f"4. Generate testbench:      python verilog_generator_enhanced.py")
    print(f"5. Run simulation:          python simulation_controller.py")
    print("=" * 80)


if __name__ == "__main__":
    main()