"""
ALU to BDD to Verilog Generator Package

This package provides tools for converting ALU specifications to BDD representations
and generating Verilog hardware description language code.

Modules:
    - alu_parser: Parse ALU specification files
    - bdd: Generate Binary Decision Diagrams
    - verilog_generator: Generate Verilog RTL code
    - visualize_bdd: Visualize BDD structures
"""

__version__ = '1.0.0'
__author__ = 'ALU BDD Verilog Generator'

# Import main classes for convenience
from .alu_parser import ALUParser, ALUSpec, create_default_alu_spec

__all__ = [
    'ALUParser',
    'ALUSpec',
    'create_default_alu_spec',
    'BDDGenerator',
    'BDDGraph',
    'BDDNode',
    'VerilogGenerator',
    'VerilogTestbenchGenerator',
]