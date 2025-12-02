# Auto-generated BDD Feature File
# Generated from: alu_8bit
# Timestamp: 2025-12-01 21:46:26
# Generator: bdd_generator.py (deterministic, no LLM)

Feature: 8-bit ALU Verification
  As a hardware verification engineer
  I want to verify the 8-bit ALU implementation
  So that I can ensure it correctly performs all arithmetic and logical operations

  Background:
    Given the ALU is initialized with 8-bit operands

  @add @arithmetic
  Scenario Outline: Verify ADD operation
    Given I have operand A = <A>
    And I have operand B = <B>
    When I perform the ADD operation with opcode 0000
    Then the result should be <Expected_Result>
    And the zero flag should be <Zero_Flag>
    And the overflow flag should be <Overflow>
    And the negative flag should be <Negative_Flag>

    # Addition (A + B)
    Examples:
      | A     | B     | Opcode | Expected_Result | Zero_Flag | Overflow | Negative_Flag |
      | 10    | 5     | 0000   | 15              | False     | False    | False         |
      | 58    | 58    | 0000   | 116             | False     | False    | False         |
      | 92    | 0     | 0000   | 92              | False     | False    | False         |
      | 83    | 107   | 0000   | 190             | False     | False    | True          |
      | 0     | 72    | 0000   | 72              | False     | False    | False         |

  @sub @arithmetic
  Scenario Outline: Verify SUB operation
    Given I have operand A = <A>
    And I have operand B = <B>
    When I perform the SUB operation with opcode 0001
    Then the result should be <Expected_Result>
    And the zero flag should be <Zero_Flag>
    And the overflow flag should be <Overflow>
    And the negative flag should be <Negative_Flag>

    # Subtraction (A - B)
    Examples:
      | A     | B     | Opcode | Expected_Result | Zero_Flag | Overflow | Negative_Flag |
      | 10    | 5     | 0001   | 5               | False     | False    | False         |
      | 93    | 93    | 0001   | 0               | True      | False    | False         |
      | 34    | 0     | 0001   | 34              | False     | False    | False         |
      | 14    | 105   | 0001   | 165             | False     | True     | True          |
      | 123   | 58    | 0001   | 65              | False     | False    | False         |

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
      | 122   | 122   | 0010   | 122             | False     | False    | False         |
      | 46    | 0     | 0010   | 0               | True      | False    | False         |
      | 61    | 5     | 0010   | 5               | False     | False    | False         |
      | 80    | 87    | 0010   | 80              | False     | False    | False         |

  @or @arithmetic
  Scenario Outline: Verify OR operation
    Given I have operand A = <A>
    And I have operand B = <B>
    When I perform the OR operation with opcode 0011
    Then the result should be <Expected_Result>
    And the zero flag should be <Zero_Flag>
    And the overflow flag should be <Overflow>
    And the negative flag should be <Negative_Flag>

    # Bitwise OR (A | B)
    Examples:
      | A     | B     | Opcode | Expected_Result | Zero_Flag | Overflow | Negative_Flag |
      | 10    | 5     | 0011   | 15              | False     | False    | False         |
      | 116   | 116   | 0011   | 116             | False     | False    | False         |
      | 27    | 0     | 0011   | 27              | False     | False    | False         |
      | 43    | 58    | 0011   | 59              | False     | False    | False         |
      | 122   | 8     | 0011   | 122             | False     | False    | False         |

