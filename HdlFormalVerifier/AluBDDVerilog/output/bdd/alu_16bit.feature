# Auto-generated BDD Feature File
# Generated from: alu_16bit
# Timestamp: 2025-11-27 00:28:15
# Generator: bdd_generator.py (deterministic, no LLM)

Feature: 16-bit ALU Verification
  As a hardware verification engineer
  I want to verify the 16-bit ALU implementation
  So that I can ensure it correctly performs all arithmetic and logical operations

  Background:
    Given the ALU is initialized with 16-bit operands

  @and @arithmetic
  Scenario Outline: Verify AND operation
    Given I have operand A = <A>
    And I have operand B = <B>
    When I perform the AND operation with opcode 0010
    Then the result should be <Expected_Result>
    And the zero flag should be <Zero_Flag>
    And the overflow flag should be <Overflow>
    And the negative flag should be <Negative_Flag>

    # Bitwise AND (A & B)
    Examples:
      | A     | B     | Opcode | Expected_Result | Zero_Flag | Overflow | Negative_Flag |
      | 10    | 5     | 0010   | 0               | True      | False    | False         |
      | 628   | 628   | 0010   | 628             | False     | False    | False         |
      | 46    | 0     | 0010   | 0               | True      | False    | False         |
      | 15865 | 1376  | 0010   | 1376            | False     | False    | False         |
      | 20554 | 22473 | 0010   | 20552           | False     | False    | False         |

