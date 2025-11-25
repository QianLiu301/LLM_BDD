"""
Step Definitions for 16-bit ALU BDD Tests
Using behave framework
"""

from behave import given, when, then
from hamcrest import assert_that, equal_to, is_


class ALUContext:
    """ALU 测试上下文"""
    def __init__(self):
        self.operand_a = 0
        self.operand_b = 0
        self.opcode = "0000"
        self.result = 0
        self.zero_flag = False
        self.carry_flag = False
        self.overflow_flag = False
        self.negative_flag = False


@given('a 16-bit ALU module is instantiated')
def step_init_alu(context):
    """初始化 ALU"""
    context.alu = ALUContext()


@given('the ALU has {bitwidth}-bit operands A and B')
def step_setup_operands(context, bitwidth):
    """设置操作数位宽"""
    context.bitwidth = int(bitwidth)


@given('operand A is {value:d}')
def step_set_operand_a(context, value):
    """设置操作数 A"""
    context.alu.operand_a = value


@given('operand B is {value:d}')
def step_set_operand_b(context, value):
    """设置操作数 B"""
    context.alu.operand_b = value


@given('operand A is {binary_value} ({decimal_value:d})')
def step_set_operand_a_binary(context, binary_value, decimal_value):
    """设置操作数 A（二进制表示）"""
    context.alu.operand_a = decimal_value


@given('operand B is {binary_value} ({decimal_value:d})')
def step_set_operand_b_binary(context, binary_value, decimal_value):
    """设置操作数 B（二进制表示）"""
    context.alu.operand_b = decimal_value


@when('I set opcode to "{opcode}" for {operation} operation')
def step_set_opcode(context, opcode, operation):
    """设置操作码并执行运算"""
    context.alu.opcode = opcode

    # 这里应该调用实际的 ALU 模块进行计算
    # 示例：使用生成的 Verilog 进行仿真
    # 或者使用 Python 实现的 ALU 模型

    a = context.alu.operand_a
    b = context.alu.operand_b

    # 简化的 ALU 逻辑（实际应该调用 Verilog 仿真）
    if operation == "ADD":
        result = a + b
        context.alu.carry_flag = result > 65535
        context.alu.result = result & 65535
    elif operation == "SUB":
        result = a - b
        context.alu.result = result & 65535
        context.alu.negative_flag = result < 0
    elif operation == "AND":
        context.alu.result = a & b
    elif operation == "OR":
        context.alu.result = a | b
    elif operation == "XOR":
        context.alu.result = a ^ b
    elif operation == "NOT":
        context.alu.result = (~a) & 65535
    elif operation == "SHL":
        context.alu.result = (a << (b & 0x1F)) & 65535
    elif operation == "SHR":
        context.alu.result = (a >> (b & 0x1F))
    elif operation == "INC":
        context.alu.result = (a + 1) & 65535
    elif operation == "DEC":
        context.alu.result = (a - 1) & 65535

    # 更新 zero flag
    context.alu.zero_flag = (context.alu.result == 0)


@then('the result should be {expected:d}')
def step_check_result(context, expected):
    """验证结果"""
    assert_that(context.alu.result, equal_to(expected))


@then('the result should be {binary_value} ({expected:d})')
def step_check_result_binary(context, binary_value, expected):
    """验证结果（二进制表示）"""
    assert_that(context.alu.result, equal_to(expected))


@then('the zero flag should be {state}')
def step_check_zero_flag(context, state):
    """验证 zero flag"""
    expected = (state.lower() == 'true')
    assert_that(context.alu.zero_flag, is_(expected))


@then('the carry flag should be {state}')
def step_check_carry_flag(context, state):
    """验证 carry flag"""
    expected = (state.lower() == 'true')
    assert_that(context.alu.carry_flag, is_(expected))


@then('the overflow flag should be {state}')
def step_check_overflow_flag(context, state):
    """验证 overflow flag"""
    expected = (state.lower() == 'true')
    assert_that(context.alu.overflow_flag, is_(expected))


@then('the negative flag should be {state}')
def step_check_negative_flag(context, state):
    """验证 negative flag"""
    expected = (state.lower() == 'true')
    assert_that(context.alu.negative_flag, is_(expected))


@then('all bits in result should be {value:d}')
def step_check_all_bits(context, value):
    """验证所有位"""
    if value == 1:
        expected = 65535
    else:
        expected = 0
    assert_that(context.alu.result, equal_to(expected))
