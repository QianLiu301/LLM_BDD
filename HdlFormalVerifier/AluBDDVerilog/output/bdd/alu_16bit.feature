# Auto-generated BDD Feature File
# Generated from: alu_16bit
# Timestamp: 2025-12-01 21:46:26
# Generator: bdd_generator.py (deterministic, no LLM)

Feature: 16-bit ALU Verification
  As a hardware verification engineer
  I want to verify the 16-bit ALU implementation
  So that I can ensure it correctly performs all arithmetic and logical operations

  Background:
    Given the ALU is initialized with 16-bit operands

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
      | 118   | 118   | 0000   | 236             | False     | False    | False         |
      | 92    | 0     | 0000   | 92              | False     | False    | False         |
      | 21338 | 27547 | 0000   | 48885           | False     | False    | True          |
      | 240   | 18533 | 0000   | 18773           | False     | False    | False         |

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
      | 400   | 400   | 0001   | 0               | True      | False    | False         |
      | 34    | 0     | 0001   | 34              | False     | False    | False         |
      | 3773  | 26952 | 0001   | 42357           | False     | True     | True          |
      | 31490 | 15021 | 0001   | 16469           | False     | False    | False         |

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
      | 580   | 580   | 0011   | 580             | False     | False    | False         |
      | 27    | 0     | 0011   | 27              | False     | False    | False         |
      | 11011 | 15044 | 0011   | 15303           | False     | False    | False         |
      | 31452 | 2244  | 0011   | 31452           | False     | False    | False         |

