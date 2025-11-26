# Auto-generated BDD Feature File
# Generated from: alu_64bit
# Timestamp: 2025-11-27 00:28:15
# Generator: bdd_generator.py (deterministic, no LLM)

Feature: 64-bit ALU Verification
  As a hardware verification engineer
  I want to verify the 64-bit ALU implementation
  So that I can ensure it correctly performs all arithmetic and logical operations

  Background:
    Given the ALU is initialized with 64-bit operands

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
      | 7753987954478579207 | 6017043200574059716 | 0000   | 13771031155052638923 | False     | False    | True          |
      | 4993603137919118814 | 8894918318955573194 | 0000   | 13888521456874692008 | False     | False    | True          |

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
      | 1062015355818845848 | 7586402456324509308 | 0001   | 11922356973203888156 | False     | True     | True          |
      | 4228065651077726551 | 4817880938540546958 | 0001   | 17856928786246731209 | False     | True     | True          |

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
      | 4465834584413815543 | 5785536705484297034 | 0010   | 1173276626018894402 | False     | False    | False         |
      | 7676810840197284270 | 7848942153956742598 | 0010   | 7532554364863529350 | False     | False    | False         |

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
      | 3099515833590915264 | 631655993265934062 | 0011   | 3154692374118985454 | False     | False    | False         |
      | 8873789065998484944 | 8495185410343105479 | 0011   | 9216333237999499223 | False     | False    | False         |

