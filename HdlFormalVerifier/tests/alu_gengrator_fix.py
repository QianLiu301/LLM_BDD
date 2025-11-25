#!/usr/bin/env python3
"""
修正的ALU生成器 - 根据scenario动态生成Verilog代码
"""
import re
from pathlib import Path
from typing import Dict, List, Tuple


class ALUGenerator:
    """根据Feature文件生成对应的ALU Verilog代码"""

    # 操作码映射
    OPCODE_MAP = {
        'ADD': '0000',
        'SUB': '0001',
        'AND': '0010',
        'OR': '0011',
        'XOR': '0100',
        'NOT': '0101',
        'SHL': '0110',  # Shift Left
        'SHR': '0111',  # Shift Right
        'MUL': '1000',
        'DIV': '1001'
    }

    def __init__(self):
        self.operations = {}  # {opcode: operation_name}

    def parse_feature_file(self, feature_file: Path) -> Dict:
        """解析feature文件，提取操作信息"""
        with open(feature_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # 提取操作类型
        operations = set()

        # 查找opcode设置语句
        opcode_pattern = r'opcode to "(\d{4})" for (\w+) operation'
        matches = re.findall(opcode_pattern, content, re.IGNORECASE)

        for opcode, op_name in matches:
            op_name = op_name.upper()
            operations.add((opcode, op_name))

        # 也可以从Scenario标题中提取
        scenario_pattern = r'Scenario:\s*(\w+)'
        scenario_matches = re.findall(scenario_pattern, content, re.IGNORECASE)

        print(f"\n解析feature文件: {feature_file.name}")
        print(f"  找到的操作: {operations}")
        print(f"  Scenario: {scenario_matches}")

        return {
            'operations': dict(operations),
            'scenarios': scenario_matches,
            'content': content
        }

    def generate_operation_logic(self, opcode: str, op_name: str) -> str:
        """生成单个操作的Verilog逻辑"""
        logic_templates = {
            'ADD': f"""
            4'b{opcode}: begin // ADD
                temp_result = a + b;
                result = temp_result[15:0];
                carry = temp_result[16];
                overflow = (a[15] == b[15]) && (result[15] != a[15]);
            end""",

            'SUB': f"""
            4'b{opcode}: begin // SUB
                temp_result = a - b;
                result = temp_result[15:0];
                carry = ~temp_result[16];
                overflow = (a[15] != b[15]) && (result[15] != a[15]);
            end""",

            'AND': f"""
            4'b{opcode}: begin // AND
                result = a & b;
                carry = 1'b0;
                overflow = 1'b0;
            end""",

            'OR': f"""
            4'b{opcode}: begin // OR
                result = a | b;
                carry = 1'b0;
                overflow = 1'b0;
            end""",

            'XOR': f"""
            4'b{opcode}: begin // XOR
                result = a ^ b;
                carry = 1'b0;
                overflow = 1'b0;
            end""",

            'NOT': f"""
            4'b{opcode}: begin // NOT (操作A)
                result = ~a;
                carry = 1'b0;
                overflow = 1'b0;
            end""",

            'SHL': f"""
            4'b{opcode}: begin // SHL (左移)
                result = a << b[3:0];  // 使用b的低4位作为移位量
                carry = 1'b0;
                overflow = 1'b0;
            end""",

            'SHR': f"""
            4'b{opcode}: begin // SHR (右移)
                result = a >> b[3:0];  // 使用b的低4位作为移位量
                carry = 1'b0;
                overflow = 1'b0;
            end""",

            'MUL': f"""
            4'b{opcode}: begin // MUL
                temp_result = a * b;
                result = temp_result[15:0];
                carry = |temp_result[31:16];  // 如果高16位有任何1
                overflow = |temp_result[31:16];
            end""",

            'DIV': f"""
            4'b{opcode}: begin // DIV
                if (b != 0) begin
                    result = a / b;
                end else begin
                    result = 16'hFFFF;  // 除零错误
                end
                carry = 1'b0;
                overflow = 1'b0;
            end"""
        }

        return logic_templates.get(op_name, f"""
            4'b{opcode}: begin // {op_name} (未实现)
                result = 16'b0;
                carry = 1'b0;
                overflow = 1'b0;
            end""")

    def generate_verilog(self, module_name: str, operations: Dict[str, str]) -> str:
        """生成完整的Verilog模块"""

        # 生成操作列表
        op_list = ', '.join([f"{name} ({code})" for code, name in operations.items()])

        # 生成所有操作的case语句
        case_statements = '\n'.join([
            self.generate_operation_logic(opcode, op_name)
            for opcode, op_name in operations.items()
        ])

        verilog_code = f"""/*
 * 16-bit ALU Module
 * Generated from BDD Feature File
 * Operations: {len(operations)}
 * Supported operations:
{chr(10).join([f' *   - {name} ({code})' for code, name in operations.items()])}
 */

`timescale 1ns/1ps

module {module_name} (
    input  wire [15:0] a,      // Operand A
    input  wire [15:0] b,      // Operand B
    input  wire [3:0] opcode, // Operation code
    output reg  [15:0] result, // Result
    output reg  carry,           // Carry flag
    output reg  zero,            // Zero flag
    output reg  negative,        // Negative flag
    output reg  overflow         // Overflow flag
);

    // Internal signals
    reg [16:0] temp_result;  // Extended for carry detection

    // ALU operation logic
    always @(*) begin
        // Default values
        result = 16'b0;
        carry = 1'b0;
        overflow = 1'b0;
        temp_result = 17'b0;

        case (opcode)
{case_statements}

            default: begin
                result = 16'b0;
            end
        endcase
    end

    // Flag generation
    always @(*) begin
        zero = (result == 16'b0);
        negative = result[15];
    end

endmodule"""

        return verilog_code

    def process_scenario(self, feature_file: Path, output_dir: Path):
        """处理单个scenario文件，生成对应的Verilog"""

        # 解析feature文件
        feature_info = self.parse_feature_file(feature_file)
        operations = feature_info['operations']

        if not operations:
            print(f"  ⚠️  警告: 未找到任何操作定义")
            return

        # 生成模块名（基于文件名）
        module_name = feature_file.stem.replace('.feature', '')
        if not module_name.endswith('_16bit'):
            module_name = f"{module_name}_16bit"

        # 生成Verilog代码
        verilog_code = self.generate_verilog(module_name, operations)

        # 保存文件
        output_file = output_dir / f"{module_name}.v"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(verilog_code)

        print(f"  ✓ 生成Verilog文件: {output_file.name}")
        print(f"    - 模块名: {module_name}")
        print(f"    - 支持操作: {', '.join(operations.values())}")

        return output_file


def main():
    """主函数 - 批量处理feature文件"""
    import argparse

    parser = argparse.ArgumentParser(description='从Feature文件生成对应的ALU Verilog代码')
    parser.add_argument('feature_file', help='Feature文件路径')
    parser.add_argument('-o', '--output-dir', default='.', help='输出目录')

    args = parser.parse_args()

    feature_file = Path(args.feature_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generator = ALUGenerator()
    generator.process_scenario(feature_file, output_dir)


if __name__ == '__main__':
    main()