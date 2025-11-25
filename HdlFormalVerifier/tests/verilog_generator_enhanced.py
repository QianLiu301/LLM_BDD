"""
Verilog Generator - 从BDD Feature文件生成Verilog代码
支持动态路径配置和自动数字格式识别（十进制/十六进制）
"""

import os
import re
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from enum import Enum


class NumberFormat(Enum):
    """数字格式枚举"""
    DECIMAL = "decimal"
    HEXADECIMAL = "hexadecimal"
    BINARY = "binary"


class FeatureParser:
    """解析.feature文件,提取ALU规格信息"""

    def __init__(self, feature_file: str):
        self.feature_file = feature_file
        self.bitwidth = 16  # 默认位宽
        self.operations = {}  # opcode -> operation_name
        self.scenarios = []  # 测试场景
        self.number_format = NumberFormat.DECIMAL  # 默认十进制

    def parse(self) -> Dict:
        """解析feature文件"""
        with open(self.feature_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # 提取位宽
        bitwidth_match = re.search(r'(\d+)[-_]bit', content, re.IGNORECASE)
        if bitwidth_match:
            self.bitwidth = int(bitwidth_match.group(1))

        # 检测数字格式
        self._detect_number_format(content)

        # 提取操作码映射
        self._extract_operations(content)

        # 提取测试场景
        self._extract_scenarios(content)
        
        # 如果没有在文件名中找到位宽，从测试场景的数值范围推断
        if not bitwidth_match and self.scenarios:
            self.bitwidth = self._infer_bitwidth_from_scenarios()
            print(f"  从数值范围推断位宽: {self.bitwidth} bit")

        return {
            'bitwidth': self.bitwidth,
            'operations': self.operations,
            'scenarios': self.scenarios,
            'module_name': f"alu_{self.bitwidth}bit",
            'number_format': self.number_format
        }
    
    def _infer_bitwidth_from_scenarios(self) -> int:
        """从测试场景的数值范围推断位宽"""
        max_value = 0
        for scenario in self.scenarios:
            if 'a' in scenario:
                max_value = max(max_value, scenario['a'])
            if 'b' in scenario:
                max_value = max(max_value, scenario['b'])
            if 'expected_result' in scenario:
                max_value = max(max_value, scenario['expected_result'])
        
        # 根据最大值推断位宽
        if max_value <= 0xFF:  # 255
            return 8
        elif max_value <= 0xFFFF:  # 65535
            return 16
        elif max_value <= 0xFFFFFFFF:  # 4294967295
            return 32
        else:
            return 64  # 超大值使用 64-bit

    def _detect_number_format(self, content: str):
        """自动检测数字格式"""
        # 检查是否有明确的十六进制标记 (0x 或 0X 前缀)
        has_hex_prefix = bool(re.search(r'0[xX][0-9A-Fa-f]+', content))
        
        if has_hex_prefix:
            # 如果有 0x 前缀，明确是十六进制
            self.number_format = NumberFormat.HEXADECIMAL
            print("  检测到数字格式: 十六进制 (0x前缀)")
        else:
            # 没有 0x 前缀，默认为十进制
            # 只有明确的 0x 前缀才认为是十六进制
            self.number_format = NumberFormat.DECIMAL
            print("  检测到数字格式: 十进制")

    def _extract_operations(self, content: str):
        """从feature文件中提取操作码定义"""
        # 查找操作码定义行,例如: "Given the opcode is 0000 (ADD)" 或 "When I set opcode to "0001" for SUB operation"
        opcode_patterns = [
            r'opcode\s+is\s+["\']?([01]+)["\']?\s*\((\w+)\)',
            r'opcode\s+to\s+["\']([01]+)["\']?\s+for\s+(\w+)',
            r'opcode\s+["\']([01]+)["\'].*?(\w+)\s+operation'
        ]
        
        for pattern in opcode_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for opcode, op_name in matches:
                self.operations[opcode] = op_name.upper()

        # 如果没有找到操作码,使用默认映射
        if not self.operations:
            print("  警告: 未找到操作码定义,使用默认映射")
            self.operations = {
                '0000': 'ADD',
                '0001': 'SUB',
                '0010': 'AND',
                '0011': 'OR',
                '0100': 'XOR',
                '0101': 'SHL',
                '0110': 'SHR',
                '0111': 'NOT'
            }

    def _extract_scenarios(self, content: str):
        """提取测试场景"""
        scenarios = []

        # 查找所有Scenario及其Examples
        scenario_pattern = r'Scenario:\s*(.+?)(?=Scenario:|$)'
        scenario_matches = re.findall(scenario_pattern, content, re.DOTALL)

        for scenario_content in scenario_matches:
            scenario_list = self._parse_scenario(scenario_content)
            if scenario_list:
                scenarios.extend(scenario_list)

        self.scenarios = scenarios

    def _parse_scenario(self, scenario_content: str) -> List[Dict]:
        """解析单个测试场景，支持Examples表格"""
        scenarios = []
        
        # 提取opcode
        opcode_patterns = [
            r'opcode\s+(?:is|to)\s+["\']?([01]+)["\']?',
            r'set\s+opcode.*?["\']([01]+)["\']'
        ]
        
        opcode = None
        for pattern in opcode_patterns:
            opcode_match = re.search(pattern, scenario_content, re.IGNORECASE)
            if opcode_match:
                opcode = opcode_match.group(1)
                break
        
        if not opcode:
            return scenarios

        # 查找Examples表格 - 改进的正则表达式
        # 匹配 Examples: 后的表格，直到遇到空行、新Scenario或文件结束
        examples_match = re.search(r'Examples:\s*\n\s*\|(.+?)\n((?:\s*\|.+\n?)+)', 
                                   scenario_content, re.DOTALL)
        
        if examples_match:
            # 解析表头 - 第一组捕获的内容
            header_line = examples_match.group(1).strip()
            # 移除首尾的 | 然后分割
            header_line = header_line.strip('|').strip()
            headers = [h.strip() for h in header_line.split('|') if h.strip()]
            
            # 解析数据行 - 第二组捕获的内容
            data_section = examples_match.group(2).strip()
            data_lines = data_section.split('\n')
            
            for line in data_lines:
                line = line.strip()
                if not line or line.startswith('|') is False:
                    continue
                    
                # 移除首尾的 |
                line = line.strip('|')
                values = [v.strip() for v in line.split('|')]
                
                if len(values) < 2:
                    continue
                
                # 创建scenario字典
                scenario = {'opcode': opcode}
                
                # 解析A和B的值
                for i, header in enumerate(headers):
                    if i < len(values):
                        value_str = values[i]
                        # 解析数字（支持十进制和十六进制）
                        value = self._parse_number(value_str)
                        if value is not None:
                            if header.upper() == 'A':
                                scenario['a'] = value
                                scenario['a_str'] = value_str  # 保存原始字符串
                            elif header.upper() == 'B':
                                scenario['b'] = value
                                scenario['b_str'] = value_str
                
                # 提取期望结果（如果有）
                result_patterns = [
                    r'result\s+should\s+be\s+(\d+)',
                    r'result\s+should\s+be\s+0x([0-9A-Fa-f]+)'
                ]
                
                for pattern in result_patterns:
                    result_match = re.search(pattern, scenario_content, re.IGNORECASE)
                    if result_match:
                        result_str = result_match.group(1)
                        if 'x' in pattern or 'X' in pattern:
                            scenario['expected_result'] = int(result_str, 16)
                        else:
                            scenario['expected_result'] = int(result_str)
                        break
                
                if 'a' in scenario and 'b' in scenario:
                    scenarios.append(scenario)
        else:
            # 旧格式：直接从Given/When/Then提取
            a_patterns = [
                r'input\s+A\s+is\s+0x([0-9A-Fa-f]+)',
                r'input\s+A\s+is\s+(\d+)',
                r'operand\s+A\s+is\s+(\d+)',
                r'A\s*=\s*(\d+)'
            ]
            
            b_patterns = [
                r'input\s+B\s+is\s+0x([0-9A-Fa-f]+)',
                r'input\s+B\s+is\s+(\d+)',
                r'operand\s+B\s+is\s+(\d+)',
                r'B\s*=\s*(\d+)'
            ]
            
            a_value = None
            a_str = None
            for pattern in a_patterns:
                a_match = re.search(pattern, scenario_content, re.IGNORECASE)
                if a_match:
                    a_str = a_match.group(1)
                    a_value = self._parse_number(a_str, '0x' in pattern)
                    break
            
            b_value = None
            b_str = None
            for pattern in b_patterns:
                b_match = re.search(pattern, scenario_content, re.IGNORECASE)
                if b_match:
                    b_str = b_match.group(1)
                    b_value = self._parse_number(b_str, '0x' in pattern)
                    break
            
            if a_value is not None and b_value is not None:
                scenario = {
                    'a': a_value,
                    'b': b_value,
                    'a_str': a_str,
                    'b_str': b_str,
                    'opcode': opcode
                }
                
                # 提取期望结果
                result_patterns = [
                    r'result\s+should\s+be\s+0x([0-9A-Fa-f]+)',
                    r'result\s+should\s+be\s+(\d+)'
                ]
                
                for pattern in result_patterns:
                    result_match = re.search(pattern, scenario_content, re.IGNORECASE)
                    if result_match:
                        result_str = result_match.group(1)
                        if 'x' in pattern:
                            scenario['expected_result'] = int(result_str, 16)
                        else:
                            scenario['expected_result'] = int(result_str)
                        break
                
                scenarios.append(scenario)

        return scenarios

    def _parse_number(self, value_str: str, force_hex: bool = False) -> Optional[int]:
        """解析数字，自动识别格式"""
        try:
            value_str = value_str.strip()
            
            # 检查十六进制前缀
            if value_str.startswith('0x') or value_str.startswith('0X') or force_hex:
                # 移除0x前缀
                hex_str = value_str[2:] if value_str.startswith('0') else value_str
                return int(hex_str, 16)
            else:
                # 十进制
                return int(value_str)
        except (ValueError, AttributeError):
            return None


class VerilogGenerator:
    """从Feature文件生成Verilog代码"""

    def __init__(self, bdd_dir: str = None, verilog_dir: str = None):
        """
        初始化生成器
        :param bdd_dir: BDD feature文件所在目录
        :param verilog_dir: Verilog输出目录
        """
        # 设置默认路径或使用提供的路径
        if bdd_dir:
            self.bdd_dir = Path(bdd_dir)
        else:
            # 默认路径
            default_bdd = Path(r"/HdlFormalVerifier/AluBDDVerilog/src/output/bdd")
            self.bdd_dir = default_bdd if default_bdd.exists() else Path("./bdd")

        if verilog_dir:
            self.verilog_dir = Path(verilog_dir)
        else:
            # 默认路径
            default_verilog = Path(r"/HdlFormalVerifier/AluBDDVerilog/src/output/verilog")
            self.verilog_dir = default_verilog if default_verilog.parent.exists() else Path("./verilog")

        # 确保输出目录存在
        self.verilog_dir.mkdir(parents=True, exist_ok=True)
        print(f"BDD输入目录: {self.bdd_dir}")
        print(f"Verilog输出目录: {self.verilog_dir}")

    def process_all_features(self):
        """处理所有feature文件"""
        if not self.bdd_dir.exists():
            print(f"错误: BDD目录不存在: {self.bdd_dir}")
            print(f"请创建目录或使用 -b 参数指定正确的路径")
            return

        feature_files = list(self.bdd_dir.glob("*.feature"))

        if not feature_files:
            print(f"警告: 在 {self.bdd_dir} 中没有找到.feature文件")
            return

        print(f"\n找到 {len(feature_files)} 个feature文件")
        print("=" * 60)

        success_count = 0
        fail_count = 0

        for feature_file in feature_files:
            print(f"\n处理文件: {feature_file.name}")
            try:
                self.process_single_feature(str(feature_file))
                success_count += 1
                print(f"✓ 成功处理: {feature_file.name}")
            except Exception as e:
                fail_count += 1
                print(f"✗ 错误: 处理 {feature_file.name} 时出错: {e}")
                import traceback
                traceback.print_exc()

        print("\n" + "=" * 60)
        print(f"处理完成: 成功 {success_count} 个, 失败 {fail_count} 个")

    def process_single_feature(self, feature_file: str):
        """处理单个feature文件"""
        # 解析feature文件
        parser = FeatureParser(feature_file)
        spec = parser.parse()

        # 从 feature 文件名提取基础名称
        feature_path = Path(feature_file)
        base_name = feature_path.stem  # 不带扩展名的文件名
        
        # 生成唯一的模块名称
        # 格式: 文件名_位宽bit (例如: scenario_20251117_081137_16bit)
        module_name = f"{base_name}_{spec['bitwidth']}bit"
        spec['module_name'] = module_name

        print(f"  位宽: {spec['bitwidth']} bit")
        print(f"  操作数: {len(spec['operations'])}")
        print(f"  测试场景: {len(spec['scenarios'])}")
        print(f"  模块名: {module_name}")

        # 生成Verilog模块
        module_code = self.generate_module(spec)
        module_file = self.verilog_dir / f"{module_name}.v"
        with open(module_file, 'w', encoding='utf-8') as f:
            f.write(module_code)
        print(f"  已生成模块: {module_file.name}")

        # 生成测试台
        tb_code = self.generate_testbench(spec)
        tb_file = self.verilog_dir / f"{module_name}_tb.v"
        with open(tb_file, 'w', encoding='utf-8') as f:
            f.write(tb_code)
        print(f"  已生成测试台: {tb_file.name}")

    def generate_module(self, spec: Dict) -> str:
        """生成Verilog模块代码"""
        bitwidth = spec['bitwidth']
        operations = spec['operations']
        opcode_width = len(list(operations.keys())[0]) if operations else 4
        module_name = spec['module_name']

        lines = []

        # 文件头注释
        lines.append(f"/*")
        lines.append(f" * {bitwidth}-bit ALU Module")
        lines.append(f" * Generated from BDD Feature File")
        lines.append(f" * Operations: {len(operations)}")
        lines.append(f" * Supported operations:")
        for opcode, op_name in sorted(operations.items()):
            lines.append(f" *   - {op_name} ({opcode})")
        lines.append(f" */")
        lines.append("")
        lines.append(f"`timescale 1ns/1ps")
        lines.append("")

        # 模块声明
        lines.append(f"module {module_name} (")
        lines.append(f"    input  wire [{bitwidth - 1}:0] a,      // Operand A")
        lines.append(f"    input  wire [{bitwidth - 1}:0] b,      // Operand B")
        lines.append(f"    input  wire [{opcode_width - 1}:0] opcode, // Operation code")
        lines.append(f"    output reg  [{bitwidth - 1}:0] result, // Result")
        lines.append(f"    output reg  carry,           // Carry flag")
        lines.append(f"    output reg  zero,            // Zero flag")
        lines.append(f"    output reg  negative,        // Negative flag")
        lines.append(f"    output reg  overflow         // Overflow flag")
        lines.append(");")
        lines.append("")

        # 内部信号
        lines.append("    // Internal signals")
        lines.append(f"    reg [{bitwidth}:0] temp_result;  // Extended for carry detection")
        lines.append("")

        # ALU主逻辑
        lines.append("    // ALU operation logic")
        lines.append("    always @(*) begin")
        lines.append("        // Default values")
        lines.append(f"        result = {bitwidth}'b0;")
        lines.append("        carry = 1'b0;")
        lines.append("        overflow = 1'b0;")
        lines.append(f"        temp_result = {bitwidth + 1}'b0;")
        lines.append("")
        lines.append("        case (opcode)")

        # 为每个操作生成case分支
        for opcode, op_name in sorted(operations.items()):
            case_lines = self._generate_operation_case(opcode, op_name, bitwidth)
            lines.extend(case_lines)

        lines.append("            default: begin")
        lines.append(f"                result = {bitwidth}'b0;")
        lines.append("            end")
        lines.append("        endcase")
        lines.append("    end")
        lines.append("")

        # 标志位生成
        lines.append("    // Flag generation")
        lines.append("    always @(*) begin")
        lines.append(f"        zero = (result == {bitwidth}'b0);")
        lines.append(f"        negative = result[{bitwidth - 1}];")
        lines.append("    end")
        lines.append("")

        lines.append("endmodule")

        return '\n'.join(lines)

    def _generate_operation_case(self, opcode: str, op_name: str, bitwidth: int) -> List[str]:
        """生成单个操作的case分支"""
        lines = []
        opcode_width = len(opcode)
        opcode_str = f"{opcode_width}'b{opcode}"

        lines.append(f"            {opcode_str}: begin // {op_name}")

        if op_name == 'ADD':
            lines.append(f"                temp_result = a + b;")
            lines.append(f"                result = temp_result[{bitwidth - 1}:0];")
            lines.append(f"                carry = temp_result[{bitwidth}];")
            lines.append(
                f"                overflow = (a[{bitwidth - 1}] == b[{bitwidth - 1}]) && (result[{bitwidth - 1}] != a[{bitwidth - 1}]);")

        elif op_name == 'SUB':
            lines.append(f"                temp_result = a - b;")
            lines.append(f"                result = temp_result[{bitwidth - 1}:0];")
            lines.append(f"                carry = ~temp_result[{bitwidth}];")
            lines.append(
                f"                overflow = (a[{bitwidth - 1}] != b[{bitwidth - 1}]) && (result[{bitwidth - 1}] != a[{bitwidth - 1}]);")

        elif op_name == 'AND':
            lines.append(f"                result = a & b;")

        elif op_name == 'OR':
            lines.append(f"                result = a | b;")

        elif op_name == 'XOR':
            lines.append(f"                result = a ^ b;")

        elif op_name == 'NOT':
            lines.append(f"                result = ~a;")

        elif op_name == 'SHL':
            # 根据位宽决定移位位数
            shift_bits = min(5, (bitwidth - 1).bit_length())  # 最多 5 位，支持到 32-bit
            lines.append(f"                result = a << b[{shift_bits-1}:0];  // Shift by lower {shift_bits} bits")

        elif op_name == 'SHR':
            shift_bits = min(5, (bitwidth - 1).bit_length())
            lines.append(f"                result = a >> b[{shift_bits-1}:0];  // Shift by lower {shift_bits} bits")

        elif op_name == 'ROL':
            shift_bits = min(5, (bitwidth - 1).bit_length())
            lines.append(f"                result = (a << b[{shift_bits-1}:0]) | (a >> ({bitwidth} - b[{shift_bits-1}:0]));")

        elif op_name == 'ROR':
            shift_bits = min(5, (bitwidth - 1).bit_length())
            lines.append(f"                result = (a >> b[{shift_bits-1}:0]) | (a << ({bitwidth} - b[{shift_bits-1}:0]));")

        elif op_name == 'INC':
            lines.append(f"                temp_result = a + 1;")
            lines.append(f"                result = temp_result[{bitwidth - 1}:0];")
            lines.append(f"                carry = temp_result[{bitwidth}];")

        elif op_name == 'DEC':
            lines.append(f"                temp_result = a - 1;")
            lines.append(f"                result = temp_result[{bitwidth - 1}:0];")

        else:
            lines.append(f"                result = {bitwidth}'b0;  // Undefined operation")

        lines.append(f"            end")
        lines.append("")

        return lines

    def generate_testbench(self, spec: Dict) -> str:
        """生成测试台代码"""
        bitwidth = spec['bitwidth']
        operations = spec['operations']
        scenarios = spec['scenarios']
        opcode_width = len(list(operations.keys())[0]) if operations else 4
        module_name = spec['module_name']
        number_format = spec.get('number_format', NumberFormat.DECIMAL)

        lines = []

        # 文件头
        lines.append(f"`timescale 1ns/1ps")
        lines.append("")
        lines.append(f"/*")
        lines.append(f" * Testbench for {bitwidth}-bit ALU")
        lines.append(f" * Generated from BDD Feature File")
        lines.append(f" * Number format: {number_format.value}")
        lines.append(f" * Test cases: {len(scenarios)}")
        lines.append(f" */")
        lines.append(f"module {module_name}_tb;")
        lines.append("")

        # 信号声明
        lines.append("    // Test signals")
        lines.append(f"    reg [{bitwidth - 1}:0] a, b;")
        lines.append(f"    reg [{opcode_width - 1}:0] opcode;")
        lines.append(f"    wire [{bitwidth - 1}:0] result;")
        lines.append(f"    wire carry, zero, negative, overflow;")
        lines.append("")
        lines.append(f"    // Test statistics")
        lines.append(f"    integer passed = 0;")
        lines.append(f"    integer failed = 0;")
        lines.append("")

        # 实例化被测模块
        lines.append(f"    // Instantiate ALU")
        lines.append(f"    {module_name} uut (")
        lines.append(f"        .a(a),")
        lines.append(f"        .b(b),")
        lines.append(f"        .opcode(opcode),")
        lines.append(f"        .result(result),")
        lines.append(f"        .carry(carry),")
        lines.append(f"        .zero(zero),")
        lines.append(f"        .negative(negative),")
        lines.append(f"        .overflow(overflow)")
        lines.append(f"    );")
        lines.append("")

        # 测试序列
        lines.append("    // Test sequence")
        lines.append("    initial begin")
        lines.append(f'        $display("========================================");')
        lines.append(f'        $display("{bitwidth}-bit ALU Testbench");')
        lines.append(f'        $display("Number format: {number_format.value}");')
        lines.append(f'        $display("========================================");')
        lines.append(f'        $display("");')
        lines.append("")

        # 根据数字格式决定显示格式
        if number_format == NumberFormat.DECIMAL:
            format_spec = "%d"
            format_comment = "decimal"
        else:
            # 计算十六进制显示的位数
            hex_width = (bitwidth + 3) // 4
            format_spec = f"%0{hex_width}X"
            format_comment = "hexadecimal"

        # 生成测试用例
        if scenarios:
            for i, scenario in enumerate(scenarios, 1):
                op_name = operations.get(scenario['opcode'], 'UNKNOWN')
                
                lines.append(f"        // Test case {i}: {op_name}")
                
                # 根据数字格式设置输入值
                if number_format == NumberFormat.DECIMAL:
                    lines.append(f"        a = {bitwidth}'d{scenario['a']};")
                    lines.append(f"        b = {bitwidth}'d{scenario['b']};")
                else:
                    lines.append(f"        a = {bitwidth}'h{scenario['a']:X};")
                    lines.append(f"        b = {bitwidth}'h{scenario['b']:X};")
                
                lines.append(f"        opcode = {opcode_width}'b{scenario['opcode']};")
                lines.append(f"        #10;")
                
                # 显示结果 - 使用一致的格式
                lines.append(f'        $display("Test {i}: {format_spec} {op_name} {format_spec} = {format_spec} (C=%b Z=%b N=%b V=%b)",')
                lines.append(f'                 a, b, result, carry, zero, negative, overflow);')
                
                # 如果有期望结果，进行验证
                if 'expected_result' in scenario:
                    expected = scenario['expected_result']
                    lines.append(f"        if (result == {bitwidth}'d{expected}) begin")
                    lines.append(f'            $display("  ✓ PASS");')
                    lines.append(f"            passed = passed + 1;")
                    lines.append(f"        end else begin")
                    lines.append(f'            $display("  ✗ FAIL - Expected: {format_spec}", {bitwidth}\'d{expected});')
                    lines.append(f"            failed = failed + 1;")
                    lines.append(f"        end")
                
                lines.append("")
        else:
            # 如果没有scenarios，生成默认测试用例
            lines.append(f"        // Default test cases")
            test_cases = [
                (15, 10, '0000', 'ADD'),
                (32, 16, '0001', 'SUB'),
            ]
            
            for i, (a_val, b_val, op_code, op_name) in enumerate(test_cases, 1):
                lines.append(f"        // Test {i}: {op_name}")
                if number_format == NumberFormat.DECIMAL:
                    lines.append(f"        a = {bitwidth}'d{a_val};")
                    lines.append(f"        b = {bitwidth}'d{b_val};")
                else:
                    lines.append(f"        a = {bitwidth}'h{a_val:X};")
                    lines.append(f"        b = {bitwidth}'h{b_val:X};")
                lines.append(f"        opcode = {opcode_width}'b{op_code};")
                lines.append(f"        #10;")
                lines.append(f'        $display("Test {i} ({op_name}): {format_spec} op {format_spec} = {format_spec}",')
                lines.append(f'                 a, b, result);')
                lines.append("")

        lines.append(f'        $display("");')
        lines.append(f'        $display("========================================");')
        lines.append(f'        $display("Test Summary");')
        lines.append(f'        $display("========================================");')
        lines.append(f'        $display("Total tests: %0d", passed + failed);')
        lines.append(f'        $display("Passed: %0d", passed);')
        lines.append(f'        $display("Failed: %0d", failed);')
        if scenarios and any('expected_result' in s for s in scenarios):
            lines.append(f'        if (failed == 0)')
            lines.append(f'            $display("✓ All tests PASSED!");')
            lines.append(f'        else')
            lines.append(f'            $display("✗ Some tests FAILED!");')
        lines.append(f'        $display("========================================");')
        lines.append("        #10;")
        lines.append("        $finish;")
        lines.append("    end")
        lines.append("")

        # VCD文件生成
        lines.append("    // VCD dump for waveform viewing")
        lines.append("    initial begin")
        lines.append(f'        $dumpfile("{module_name}_tb.vcd");')
        lines.append(f"        $dumpvars(0, {module_name}_tb);")
        lines.append("    end")
        lines.append("")

        lines.append("endmodule")

        return '\n'.join(lines)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='从BDD Feature文件生成Verilog代码',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  # 使用默认路径
  python verilog_generator_enhanced.py
  
  # 指定输入和输出目录
  python verilog_generator_enhanced.py -b ./bdd -v ./verilog
  
  # 使用完整路径
  python verilog_generator_enhanced.py -b D:\\path\\to\\bdd -v D:\\path\\to\\verilog
        '''
    )
    
    parser.add_argument('-b', '--bdd-dir',
                        help='BDD feature文件所在目录',
                        default=None)
    
    parser.add_argument('-v', '--verilog-dir',
                        help='Verilog输出目录',
                        default=None)
    
    args = parser.parse_args()

    print("=" * 60)
    print("Verilog Generator - 从BDD Feature文件生成Verilog代码")
    print("支持动态路径和自动数字格式识别")
    print("=" * 60)
    print()

    # 创建生成器实例
    generator = VerilogGenerator(
        bdd_dir=args.bdd_dir,
        verilog_dir=args.verilog_dir
    )

    # 处理所有feature文件
    generator.process_all_features()

    print()
    print("=" * 60)
    print("生成完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
