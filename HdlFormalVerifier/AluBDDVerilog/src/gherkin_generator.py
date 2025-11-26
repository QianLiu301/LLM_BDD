"""
Enhanced Gherkin Generator with LLM Integration
Gherkin BDD generator enhanced with LLM-generated descriptions.
"""

from typing import Optional
from HdlFormalVerifier.tests.llm_providers_v2 import LLMProvider, LocalLLMProvider


class GherkinGenerator:
    """Gherkin BDD feature file generator (with LLM support)."""

    def __init__(self, spec, llm_provider: Optional[LLMProvider] = None):
        """
        Initialize the generator.

        Args:
            spec: ALU specification object
            llm_provider: LLM provider instance; if None, a local template-based provider is used
        """
        self.spec = spec
        self.llm_provider = llm_provider or LocalLLMProvider()
        self.content = []

    def generate(self) -> str:
        """Generate a complete Gherkin Feature file as a string."""
        self.content = []

        # Generate Feature header
        self._generate_feature_header()

        # Generate Background (if needed)
        self._generate_background()

        # Generate a scenario for each operation
        for opcode, op_name in self.spec.operations.items():
            op_details = self.spec.operation_details[op_name]
            self._generate_scenario(opcode, op_name, op_details)

        return "\n".join(self.content)

    def _generate_feature_header(self):
        """Generate the Feature header (using LLM)."""
        operations_list = list(self.spec.operations.values())

        # Use LLM to generate the Feature description
        print(f"   🤖 Using {type(self.llm_provider).__name__} to generate Feature description...")
        feature_description = self.llm_provider.generate_feature_description(
            bitwidth=self.spec.bitwidth,
            operations_count=len(self.spec.operations),
            operations_list=operations_list,
        )

        self.content.append(f"Feature: {self.spec.bitwidth}-bit ALU Verification")
        self.content.append(f"  {feature_description}")
        self.content.append("")

    def _generate_background(self):
        """Generate the Background section."""
        self.content.append("  Background:")
        self.content.append(f"    Given a {self.spec.bitwidth}-bit ALU")
        self.content.append("")

    def _generate_scenario(self, opcode: str, op_name: str, op_details: dict):
        """Generate a single scenario (using LLM)."""
        description = op_details.get("description", "")

        # Use LLM to generate scenario description
        print(f"   🤖 Generating scenario: {op_name} ({opcode})...")
        scenario_desc = self.llm_provider.generate_scenario_description(
            operation_name=op_name,
            operation_code=opcode,
            operation_description=description,
            bitwidth=self.spec.bitwidth,
        )

        # Generate the scenario
        self.content.append(f"  Scenario: Test {op_name} operation")
        self.content.append(f"    # {scenario_desc}")
        self.content.append(f"    When the operation code is '{opcode}'")
        self.content.append(f"    And the operation name is '{op_name}'")

        # Generate concrete steps based on operation type
        self._generate_operation_steps(op_name, opcode)

        self.content.append("")

    def _generate_operation_steps(self, op_name: str, opcode: str):
        """Generate concrete test steps based on the operation type."""

        # Two-operand operations
        dual_operand_ops = [
            "ADD",
            "SUB",
            "AND",
            "OR",
            "XOR",
            "NAND",
            "NOR",
            "SHL",
            "SHR",
            "ROL",
            "ROR",
            "CMP",
        ]

        # Single-operand operations
        single_operand_ops = ["NOT", "INC", "DEC", "PASS"]

        if op_name in dual_operand_ops:
            self.content.append("    And operand A is set to a test value")
            self.content.append("    And operand B is set to a test value")
            self.content.append(f"    Then the ALU output should match the expected {op_name} result")
            self.content.append("    And the flags should be updated correctly")

        elif op_name in single_operand_ops:
            self.content.append("    And operand A is set to a test value")
            self.content.append(f"    Then the ALU output should match the expected {op_name} result")
            self.content.append("    And the flags should be updated correctly")

        else:
            # Generic steps for other operations
            self.content.append("    And input operands are set")
            self.content.append("    Then the operation should complete successfully")
            self.content.append("    And the result should be correct")

    def save(self, filename: str):
        """Save the generated Feature content to a file."""
        with open(filename, "w", encoding="utf-8") as f:
            f.write(self.generate())


def generate_step_definitions(
    spec, language: str = "python", llm_provider: Optional[LLMProvider] = None
) -> str:
    """
    Generate a step-definitions file as a string.

    Args:
        spec: ALU specification object
        language: 'python' or 'javascript'
        llm_provider: optional LLM provider (reserved for enhanced comments or documentation)
    """
    if language == "python":
        return _generate_python_steps(spec, llm_provider)
    elif language == "javascript":
        return _generate_javascript_steps(spec, llm_provider)
    else:
        raise ValueError(f"Unsupported language: {language}")


def _generate_python_steps(spec, llm_provider: Optional[LLMProvider] = None) -> str:
    """Generate Python step definitions (behave)."""

    # If an LLM is provided, we could use it to enhance comments or documentation
    header_comment = "# BDD Step Definitions for ALU Verification"
    if llm_provider:
        print("   🤖 Using LLM to enhance Python step-definition comments...")

    template = f'''"""
{header_comment}
Auto-generated step definitions for {spec.bitwidth}-bit ALU
"""

from behave import given, when, then
from alu_model import ALU{spec.bitwidth}


@given('a {spec.bitwidth}-bit ALU')
def step_given_alu(context):
    """Initialize the ALU."""
    context.alu = ALU{spec.bitwidth}()
    context.bitwidth = {spec.bitwidth}


@when("the operation code is '{{opcode}}'")
def step_when_opcode(context, opcode):
    """Set the operation code."""
    context.opcode = opcode


@when("the operation name is '{{op_name}}'")
def step_when_operation_name(context, op_name):
    """Set the operation name."""
    context.op_name = op_name


@when('operand A is set to a test value')
def step_when_operand_a(context):
    """Set operand A to a fixed test pattern."""
    context.operand_a = 0x5A5A & ((1 << {spec.bitwidth}) - 1)


@when('operand B is set to a test value')
def step_when_operand_b(context):
    """Set operand B to a fixed test pattern."""
    context.operand_b = 0x3C3C & ((1 << {spec.bitwidth}) - 1)


@when('input operands are set')
def step_when_inputs_set(context):
    """Set generic input operands."""
    context.operand_a = 0x1234 & ((1 << {spec.bitwidth}) - 1)
    context.operand_b = 0x5678 & ((1 << {spec.bitwidth}) - 1)


@then('the ALU output should match the expected {{operation}} result')
def step_then_output_matches(context, operation):
    """Verify that the ALU output matches the expected result for the given operation."""
    result = context.alu.execute(
        context.opcode,
        context.operand_a,
        context.operand_b
    )
    expected = context.alu.calculate_expected(
        operation,
        context.operand_a,
        context.operand_b
    )
    assert result == expected, f"Expected {{expected}}, got {{result}}"


@then('the flags should be updated correctly')
def step_then_flags_correct(context):
    """Verify that the ALU flags are set correctly (zero, carry, overflow, etc.)."""
    assert context.alu.verify_flags(), "Flags not set correctly"


@then('the operation should complete successfully')
def step_then_operation_completes(context):
    """Verify that the operation completes without failure."""
    assert context.alu.execute(context.opcode, context.operand_a, context.operand_b) is not None


@then('the result should be correct')
def step_then_result_correct(context):
    """Verify that the ALU result passes internal verification."""
    assert context.alu.verify_result(), "Result verification failed"
'''

    return template


def _generate_javascript_steps(spec, llm_provider: Optional[LLMProvider] = None) -> str:
    """Generate JavaScript step definitions (Cucumber.js)."""

    template = f'''/**
 * BDD Step Definitions for ALU Verification
 * Auto-generated for {spec.bitwidth}-bit ALU
 */

const {{ Given, When, Then }} = require('@cucumber/cucumber');
const {{ ALU{spec.bitwidth} }} = require('./alu_model');
const assert = require('assert');

Given('a {spec.bitwidth}-bit ALU', function() {{
    this.alu = new ALU{spec.bitwidth}();
    this.bitwidth = {spec.bitwidth};
}});

When("the operation code is '{{string}}'", function(opcode) {{
    this.opcode = opcode;
}});

When("the operation name is '{{string}}'", function(opName) {{
    this.opName = opName;
}});

When('operand A is set to a test value', function() {{
    this.operandA = 0x5A5A & ((1 << {spec.bitwidth}) - 1);
}});

When('operand B is set to a test value', function() {{
    this.operandB = 0x3C3C & ((1 << {spec.bitwidth}) - 1);
}});

When('input operands are set', function() {{
    this.operandA = 0x1234 & ((1 << {spec.bitwidth}) - 1);
    this.operandB = 0x5678 & ((1 << {spec.bitwidth}) - 1);
}});

Then('the ALU output should match the expected {{word}} result', function(operation) {{
    const result = this.alu.execute(this.opcode, this.operandA, this.operandB);
    const expected = this.alu.calculateExpected(operation, this.operandA, this.operandB);
    assert.strictEqual(result, expected, `Expected ${{expected}}, got ${{result}}`);
}});

Then('the flags should be updated correctly', function() {{
    assert(this.alu.verifyFlags(), 'Flags not set correctly');
}});

Then('the operation should complete successfully', function() {{
    const result = this.alu.execute(this.opcode, this.operandA, this.operandB);
    assert(result !== null && result !== undefined, 'Operation failed');
}});

Then('the result should be correct', function() {{
    assert(this.alu.verifyResult(), 'Result verification failed');
}});
'''

    return template


if __name__ == "__main__":
    # Simple test runner
    from HdlFormalVerifier.tests.alu_parser import ALUParser

    print("🧪 Testing the LLM-enhanced Gherkin generator\n")

    # Create a test spec
    test_spec_text = """
    BITWIDTH: 16

    OPERATIONS:
    0000 | ADD | Addition (A + B)
    0001 | SUB | Subtraction (A - B)
    0010 | AND | Bitwise AND (A & B)
    """

    parser = ALUParser()
    spec = parser.parse_text(test_spec_text)

    # Use local provider for testing
    print("1️⃣  Using local template-based provider:")
    local_provider = LocalLLMProvider()
    generator = GherkinGenerator(spec, local_provider)
    feature_content = generator.generate()

    print("Generated Feature file preview:")
    print("=" * 60)
    print(feature_content[:500] + "...")
    print("=" * 60)

    print("\n✅ Test finished!")
