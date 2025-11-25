"""
ALU Specification Parser
解析ALU文本规格文件，提取位宽、操作类型等信息
"""

import re
from typing import Dict, List, Tuple


class ALUSpec:
    """ALU规格类"""

    def __init__(self, bitwidth: int):
        self.bitwidth = bitwidth
        self.operations = {}  # 操作码 -> 操作名称
        self.operation_details = {}  # 操作名称 -> 详细信息

    def add_operation(self, opcode: str, name: str, description: str = ""):
        """添加ALU操作"""
        self.operations[opcode] = name
        self.operation_details[name] = {
            'opcode': opcode,
            'description': description
        }

    def __str__(self):
        return f"ALU Spec: {self.bitwidth}-bit, {len(self.operations)} operations"


class ALUParser:
    """ALU规格解析器"""

    def __init__(self):
        self.spec = None

    def parse_file(self, filename: str) -> ALUSpec:
        """从文件解析ALU规格"""
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()

        return self.parse_text(content)

    def parse_text(self, text: str) -> ALUSpec:
        """从文本解析ALU规格"""
        lines = text.strip().split('\n')

        # 解析位宽
        bitwidth = self._parse_bitwidth(lines)
        self.spec = ALUSpec(bitwidth)

        # 解析操作
        self._parse_operations(lines)

        return self.spec

    def _parse_bitwidth(self, lines: List[str]) -> int:
        """解析位宽信息"""
        for line in lines:
            line = line.strip()
            if line.startswith('BITWIDTH:') or line.startswith('WIDTH:'):
                match = re.search(r'(\d+)', line)
                if match:
                    return int(match.group(1))

        # 默认返回16位
        return 16

    def _parse_operations(self, lines: List[str]):
        """解析操作定义"""
        in_operations = False

        for line in lines:
            line = line.strip()

            # 跳过空行和注释
            if not line or line.startswith('#') or line.startswith('//'):
                continue

            # 检查是否进入操作定义部分
            if 'OPERATIONS:' in line.upper():
                in_operations = True
                continue

            if in_operations:
                # 解析操作定义格式: OPCODE | NAME | DESCRIPTION
                parts = [p.strip() for p in line.split('|')]
                if len(parts) >= 2:
                    opcode = parts[0]
                    name = parts[1]
                    description = parts[2] if len(parts) > 2 else ""

                    self.spec.add_operation(opcode, name, description)


def create_default_alu_spec(bitwidth: int) -> ALUSpec:
    """创建默认的ALU规格"""
    spec = ALUSpec(bitwidth)

    # 添加标准ALU操作
    spec.add_operation('0000', 'ADD', 'Addition')
    spec.add_operation('0001', 'SUB', 'Subtraction')
    spec.add_operation('0010', 'AND', 'Bitwise AND')
    spec.add_operation('0011', 'OR', 'Bitwise OR')
    spec.add_operation('0100', 'XOR', 'Bitwise XOR')
    spec.add_operation('0101', 'NOT', 'Bitwise NOT')
    spec.add_operation('0110', 'SHL', 'Shift Left')
    spec.add_operation('0111', 'SHR', 'Shift Right')
    spec.add_operation('1000', 'ROL', 'Rotate Left')
    spec.add_operation('1001', 'ROR', 'Rotate Right')
    spec.add_operation('1010', 'INC', 'Increment')
    spec.add_operation('1011', 'DEC', 'Decrement')

    return spec


if __name__ == '__main__':
    # 测试解析器
    parser = ALUParser()

    # 测试文本
    test_spec = """
    BITWIDTH: 16
    
    OPERATIONS:
    0000 | ADD | Addition
    0001 | SUB | Subtraction
    0010 | AND | Bitwise AND
    0011 | OR  | Bitwise OR
    """

    spec = parser.parse_text(test_spec)
    print(spec)
    print(f"Operations: {spec.operations}")
    print("\n✅ alu_parser.py 测试成功！")