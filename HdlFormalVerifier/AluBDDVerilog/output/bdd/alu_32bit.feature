# Auto-generated BDD Feature File
# Generated from: alu_32bit
# Timestamp: 2025-11-27 00:28:15
# Generator: bdd_generator.py (deterministic, no LLM)

Feature: 32-bit ALU Verification
  As a hardware verification engineer
  I want to verify the 32-bit ALU implementation
  So that I can ensure it correctly performs all arithmetic and logical operations

  Background:
    Given the ALU is initialized with 32-bit operands

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
      | 1398439431 | 1805366006 | 0000   | 3203805437      | False     | False    | True          |
      | 15785585 | 1214608580 | 0000   | 1230394165      | False     | False    | False         |

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
      | 247269718 | 1766346966 | 0001   | 2775890048      | False     | True     | True          |
      | 2063751945 | 984423246 | 0001   | 1079328699      | False     | False    | False         |

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
      | 1039783140 | 90182474 | 0010   | 90181696        | False     | False    | False         |
      | 1347050235 | 1472843118 | 0010   | 1346914410      | False     | False    | False         |

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
      | 721662266 | 985956352 | 0011   | 1002942778      | False     | False    | False         |
      | 2061291246 | 147068871 | 0011   | 2061295599      | False     | False    | False         |

