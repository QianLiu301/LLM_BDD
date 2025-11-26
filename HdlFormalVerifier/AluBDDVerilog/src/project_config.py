#!/usr/bin/env python3
"""
项目配置文件
包含ALU验证系统的所有配置项
"""

import os
from pathlib import Path


class ProjectConfig:
    """项目配置类"""

    # 项目根目录
    PROJECT_ROOT = Path.cwd()

    # 目录结构
    DIRECTORIES = {
        'verilog': PROJECT_ROOT / 'verilog',
        'src': PROJECT_ROOT / 'src',
        'output': PROJECT_ROOT / 'output',
        'bdd_spec': PROJECT_ROOT / 'output' / 'bdd_spec',
        'bdd': PROJECT_ROOT / 'output' / 'bdd',
        'specs1': PROJECT_ROOT / 'specs1',
        'reports': PROJECT_ROOT / 'output' / 'reports'
    }

    # Verilog相关配置
    VERILOG = {
        'alu_module': 'verilog/alu_16bit.v',
        'testbench': 'verilog/alu_16bit_tb.v',
        'sim_executable': 'output/alu_sim',
        'vcd_file': 'output/alu_16bit.vcd',
        'compiler': 'iverilog',
        'simulator': 'vvp'
    }

    # ALU操作码定义
    ALU_OPCODES = {
        'ADD': {'code': 0b0000, 'name': '加法', 'symbol': '+'},
        'SUB': {'code': 0b0001, 'name': '减法', 'symbol': '-'},
        'AND': {'code': 0b0010, 'name': '按位与', 'symbol': '&'},
        'OR': {'code': 0b0011, 'name': '按位或', 'symbol': '|'},
        'XOR': {'code': 0b0100, 'name': '按位异或', 'symbol': '^'},
        'SHL': {'code': 0b0101, 'name': '左移', 'symbol': '<<'},
        'SHR': {'code': 0b0110, 'name': '右移', 'symbol': '>>'},
        'NOT': {'code': 0b0111, 'name': '按位非', 'symbol': '~'}
    }

    # 测试向量配置
    TEST_VECTORS = {
        'basic': [
            {'a': 0x000F, 'b': 0x000A, 'op': 0, 'desc': '基本加法'},
            {'a': 0x0020, 'b': 0x0010, 'op': 1, 'desc': '基本减法'},
            {'a': 0xF0F0, 'b': 0x0F0F, 'op': 2, 'desc': '按位与'},
            {'a': 0xF0F0, 'b': 0x0F0F, 'op': 3, 'desc': '按位或'},
            {'a': 0xFFFF, 'b': 0x0000, 'op': 4, 'desc': '异或操作'},
            {'a': 0x8000, 'b': 0x0000, 'op': 5, 'desc': '左移操作'},
            {'a': 0x0001, 'b': 0x0000, 'op': 6, 'desc': '右移操作'},
            {'a': 0x0000, 'b': 0x0000, 'op': 7, 'desc': '非操作'}
        ],
        'boundary': [
            {'a': 0x7FFF, 'b': 0x0001, 'op': 0, 'desc': '正溢出测试'},
            {'a': 0x8000, 'b': 0x0001, 'op': 1, 'desc': '负溢出测试'},
            {'a': 0xFFFF, 'b': 0x0001, 'op': 0, 'desc': '无符号溢出'},
            {'a': 0x0000, 'b': 0x0000, 'op': 0, 'desc': '零值测试'},
            {'a': 0xFFFF, 'b': 0xFFFF, 'op': 2, 'desc': '全1测试'}
        ]
    }

    # GTKWave配置
    GTKWAVE = {
        'executable': 'gtkwave',
        'save_file': 'output/alu_display.gtkw',
        'signals': [
            'alu_16bit_tb.a[15:0]',
            'alu_16bit_tb.b[15:0]',
            'alu_16bit_tb.opcode[3:0]',
            'alu_16bit_tb.result[15:0]',
            'alu_16bit_tb.carry',
            'alu_16bit_tb.zero',
            'alu_16bit_tb.negative',
            'alu_16bit_tb.overflow'
        ],
        'colors': {
            'input': 2,  # 绿色
            'output': 5,  # 蓝色
            'control': 4,  # 红色
            'status': 1  # 白色
        }
    }

    # BDD配置
    BDD = {
        'feature_dir': 'specs1',
        'steps_dir': 'src',
        'language': 'zh-CN',
        'framework': 'behave'
    }

    # 仿真配置
    SIMULATION = {
        'timescale': '1ns/1ps',
        'simulation_time': 100,  # 仿真时间（时间单位）
        'clock_period': 10,  # 时钟周期（ns）
        'setup_time': 5,  # 建立时间（ns）
        'hold_time': 2  # 保持时间（ns）
    }

    # 文件模板
    TEMPLATES = {
        'verilog_header': '''`timescale {timescale}
/*
 * 16位ALU测试台
 * 生成时间: {timestamp}
 * 测试用例: A=0x{a:04X}, B=0x{b:04X}, OpCode={opcode:04b}
 */''',

        'bdd_header': '''# Language: {language}
# 自动生成于: {timestamp}

功能: 16位ALU {operation_name}操作测试
  作为一个数字电路设计师
  我想要测试ALU的{operation_name}功能
  以确保电路设计的正确性'''
    }

    # 验证规则
    VALIDATION_RULES = {
        'input_range': {'min': 0x0000, 'max': 0xFFFF},
        'opcode_range': {'min': 0, 'max': 7},
        'file_size_limits': {
            'vcd_max': 50 * 1024 * 1024,  # 50MB
            'log_max': 10 * 1024 * 1024  # 10MB
        }
    }

    @classmethod
    def create_directories(cls):
        """创建所有必要的目录"""
        for name, path in cls.DIRECTORIES.items():
            path.mkdir(parents=True, exist_ok=True)
            print(f"确认目录存在: {path}")

    @classmethod
    def get_operation_info(cls, opcode):
        """根据操作码获取操作信息"""
        for op_name, info in cls.ALU_OPCODES.items():
            if info['code'] == opcode:
                return {
                    'name': op_name,
                    'chinese_name': info['name'],
                    'symbol': info['symbol'],
                    'code': info['code']
                }
        return None

    @classmethod
    def validate_inputs(cls, a, b, opcode):
        """验证输入参数"""
        errors = []

        # 验证A值
        if not (cls.VALIDATION_RULES['input_range']['min'] <= a <=
                cls.VALIDATION_RULES['input_range']['max']):
            errors.append(f"A值超出范围: {a}")

        # 验证B值
        if not (cls.VALIDATION_RULES['input_range']['min'] <= b <=
                cls.VALIDATION_RULES['input_range']['max']):
            errors.append(f"B值超出范围: {b}")

        # 验证操作码
        if not (cls.VALIDATION_RULES['opcode_range']['min'] <= opcode <=
                cls.VALIDATION_RULES['opcode_range']['max']):
            errors.append(f"操作码超出范围: {opcode}")

        return errors

    @classmethod
    def get_test_vector_set(cls, set_name='basic'):
        """获取测试向量集合"""
        return cls.TEST_VECTORS.get(set_name, [])

    @classmethod
    def get_all_test_vectors(cls):
        """获取所有测试向量"""
        all_vectors = []
        for set_name, vectors in cls.TEST_VECTORS.items():
            for vector in vectors:
                vector['set'] = set_name
                all_vectors.append(vector)
        return all_vectors


# 环境检查函数
def check_environment():
    """检查环境配置"""
    issues = []

    # 检查必要的工具
    required_tools = ['iverilog', 'vvp', 'gtkwave']
    for tool in required_tools:
        if not shutil.which(tool):
            issues.append(f"未找到必要工具: {tool}")

    # 检查目录权限
    for name, path in ProjectConfig.DIRECTORIES.items():
        if not os.access(path.parent, os.W_OK):
            issues.append(f"目录无写权限: {path}")

    return issues


# 配置初始化
def initialize_project():
    """初始化项目配置"""
    print("初始化ALU验证系统...")

    # 创建目录结构
    ProjectConfig.create_directories()

    # 检查环境
    issues = check_environment()
    if issues:
        print("环境检查发现问题:")
        for issue in issues:
            print(f"  ⚠ {issue}")
    else:
        print("✓ 环境检查通过")

    # 创建配置文件
    config_file = ProjectConfig.PROJECT_ROOT / 'project_config.json'
    if not config_file.exists():
        import json
        config_data = {
            'project_name': 'ALU动态验证系统',
            'version': '1.0.0',
            'created': str(ProjectConfig.PROJECT_ROOT),
            'directories': {k: str(v) for k, v in ProjectConfig.DIRECTORIES.items()}
        }

        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
        print(f"✓ 配置文件已创建: {config_file}")

    print("项目初始化完成!")


if __name__ == "__main__":
    # 运行项目初始化
    initialize_project()

    # 显示配置信息
    print("\n=== 项目配置信息 ===")
    print(f"项目根目录: {ProjectConfig.PROJECT_ROOT}")
    print(f"支持的ALU操作: {list(ProjectConfig.ALU_OPCODES.keys())}")
    print(f"测试向量集合: {list(ProjectConfig.TEST_VECTORS.keys())}")
    print(f"基本测试用例数量: {len(ProjectConfig.get_test_vector_set('basic'))}")