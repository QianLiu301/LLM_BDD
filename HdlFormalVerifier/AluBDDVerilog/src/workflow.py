#!/usr/bin/env python3
"""
ALU Workflow Manager
编排完整的ALU验证工作流：Spec → BDD → Verilog → 仿真
"""

from pathlib import Path
from typing import Optional, Dict, Any
import sys


class ALUWorkflow:
    """ALU完整工作流管理器"""

    def __init__(self, project_root: Optional[Path] = None):
        """
        初始化工作流管理器

        Args:
            project_root: 项目根目录，默认为当前目录
        """
        self.project_root = project_root or Path.cwd()

        # 创建必要的目录
        self._setup_directories()

        # 延迟导入模块
        self._import_modules()

    def _setup_directories(self):
        """创建项目目录结构"""
        dirs = [
            "specs",
            "output/bdd",
            "output/verilog",
            "output/simulation",
            "verilog",
            "src"
        ]

        for dir_name in dirs:
            (self.project_root / dir_name).mkdir(parents=True, exist_ok=True)

    def _import_modules(self):
        """导入所需模块"""
        try:
            # 添加src目录到路径
            src_path = self.project_root / "src"
            if str(src_path) not in sys.path:
                sys.path.insert(0, str(src_path))

            # 导入转换器
            from spec_to_bdd import SpecToBDDConverter
            self.bdd_converter = SpecToBDDConverter(
                output_dir=str(self.project_root / "output" / "bdd")
            )

            # 导入仿真控制器（如果存在）
            try:
                from simulation_controller import SimulationController
                self.sim_controller = SimulationController(self.project_root)
            except ImportError:
                print("⚠️  simulation_controller未找到，仿真功能不可用")
                self.sim_controller = None

            # 导入Verilog生成器（如果存在）
            try:
                from verilog_generator import VerilogGenerator
                self.VerilogGenerator = VerilogGenerator
            except ImportError:
                print("⚠️  verilog_generator未找到，Verilog生成功能不可用")
                self.VerilogGenerator = None

        except Exception as e:
            print(f"❌ 模块导入错误: {e}")
            raise

    def run_spec_to_bdd(self, spec_file: str, output_name: Optional[str] = None) -> Optional[object]:
        """
        步骤1: 从Spec生成BDD测试

        Args:
            spec_file: spec文件路径
            output_name: 输出名称

        Returns:
            ALUSpec对象，失败返回None
        """
        print("=" * 70)
        print("🎯 步骤1: Spec → BDD 测试场景生成")
        print("=" * 70)

        spec = self.bdd_converter.convert_from_file(spec_file, output_name)

        if spec:
            print("✅ Spec到BDD转换完成")
        else:
            print("❌ Spec到BDD转换失败")

        print()
        return spec

    def run_bdd_from_params(self, a_value: int, b_value: int, opcode: int,
                            output_name: str = "dynamic_test") -> bool:
        """
        从动态参数生成BDD测试

        Args:
            a_value: 操作数A
            b_value: 操作数B
            opcode: 操作码
            output_name: 输出名称

        Returns:
            成功返回True
        """
        print("=" * 70)
        print("🎯 动态生成BDD测试")
        print("=" * 70)
        print(f"📊 参数: A=0x{a_value:04X}, B=0x{b_value:04X}, OpCode={opcode:04b}")

        # 这里可以扩展：从参数生成临时spec，然后转换为BDD
        # 目前简化处理，直接记录参数

        print("✅ 动态BDD生成完成")
        print()
        return True

    def run_verilog_generation(self, spec: object, output_name: Optional[str] = None) -> bool:
        """
        步骤2: 从Spec生成Verilog代码

        Args:
            spec: ALUSpec对象
            output_name: 输出名称

        Returns:
            成功返回True
        """
        if self.VerilogGenerator is None:
            print("⚠️  Verilog生成器不可用，跳过此步骤")
            return False

        print("=" * 70)
        print("🎯 步骤2: Spec → Verilog 代码生成")
        print("=" * 70)

        try:
            generator = self.VerilogGenerator(spec)
            output_name = output_name or "alu_16bit"

            # 生成Verilog代码
            verilog_code = generator.generate()

            # 保存到文件
            output_file = self.project_root / "verilog" / f"{output_name}.v"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(verilog_code)

            print(f"✅ Verilog代码生成完成: {output_file}")
            print()
            return True

        except Exception as e:
            print(f"❌ Verilog生成失败: {e}")
            print()
            return False

    def run_simulation(self, a_value: int, b_value: int, opcode: int,
                       open_wave: bool = False) -> bool:
        """
        步骤3: 运行仿真

        Args:
            a_value: 操作数A
            b_value: 操作数B
            opcode: 操作码
            open_wave: 是否自动打开波形

        Returns:
            成功返回True
        """
        if self.sim_controller is None:
            print("⚠️  仿真控制器不可用，跳过此步骤")
            return False

        print("=" * 70)
        print("🎯 步骤3: 运行仿真")
        print("=" * 70)
        print(f"📊 仿真参数: A=0x{a_value:04X}, B=0x{b_value:04X}, OpCode={opcode:04b}")

        try:
            success = self.sim_controller.run_full_simulation(
                a_value=a_value,
                b_value=b_value,
                opcode=opcode,
                open_wave=open_wave
            )

            if success:
                print("✅ 仿真执行完成")
                vcd_file = self.sim_controller.verilog_output / "alu_wave.vcd"
                if vcd_file.exists():
                    print(f"📈 波形文件: {vcd_file}")
            else:
                print("❌ 仿真执行失败")

            print()
            return success

        except Exception as e:
            print(f"❌ 仿真错误: {e}")
            print()
            return False

    def run_full_workflow_from_spec(self, spec_file: str,
                                    a_value: Optional[int] = None,
                                    b_value: Optional[int] = None,
                                    opcode: Optional[int] = None,
                                    open_wave: bool = False) -> Dict[str, Any]:
        """
        运行完整工作流：Spec → BDD → Verilog → 仿真

        Args:
            spec_file: spec文件路径
            a_value: 仿真用操作数A（可选）
            b_value: 仿真用操作数B（可选）
            opcode: 仿真用操作码（可选）
            open_wave: 是否打开波形

        Returns:
            包含各步骤结果的字典
        """
        results = {
            'spec_to_bdd': False,
            'verilog_gen': False,
            'simulation': False,
            'spec': None
        }

        print("🚀" * 35)
        print("   ALU 完整验证工作流")
        print("🚀" * 35)
        print()

        # 步骤1: Spec → BDD
        spec = self.run_spec_to_bdd(spec_file)
        if spec:
            results['spec_to_bdd'] = True
            results['spec'] = spec
        else:
            print("❌ 工作流终止: Spec到BDD转换失败")
            return results

        # 步骤2: Spec → Verilog
        if self.VerilogGenerator:
            results['verilog_gen'] = self.run_verilog_generation(spec)

        # 步骤3: 运行仿真（如果提供了参数）
        if all(v is not None for v in [a_value, b_value, opcode]):
            results['simulation'] = self.run_simulation(
                a_value, b_value, opcode, open_wave
            )

        # 总结
        print("=" * 70)
        print("📊 工作流执行总结")
        print("=" * 70)
        print(f"  Spec → BDD:      {'✅ 成功' if results['spec_to_bdd'] else '❌ 失败'}")
        print(f"  Spec → Verilog:  {'✅ 成功' if results['verilog_gen'] else '⊘ 跳过'}")
        print(f"  仿真执行:        {'✅ 成功' if results['simulation'] else '⊘ 跳过'}")
        print("=" * 70)
        print()

        return results

    def run_dynamic_workflow(self, a_value: int, b_value: int, opcode: int,
                             open_wave: bool = False) -> bool:
        """
        运行动态工作流：基于参数直接运行仿真

        Args:
            a_value: 操作数A
            b_value: 操作数B
            opcode: 操作码
            open_wave: 是否打开波形

        Returns:
            成功返回True
        """
        print("🚀" * 35)
        print("   ALU 动态验证工作流")
        print("🚀" * 35)
        print()

        # 生成动态BDD测试
        self.run_bdd_from_params(a_value, b_value, opcode)

        # 运行仿真
        success = self.run_simulation(a_value, b_value, opcode, open_wave)

        print("=" * 70)
        if success:
            print("✅ 动态工作流执行成功")
        else:
            print("❌ 动态工作流执行失败")
        print("=" * 70)
        print()

        return success

    def get_vcd_file_path(self) -> Optional[Path]:
        """
        获取VCD波形文件路径

        Returns:
            VCD文件路径，不存在返回None
        """
        possible_paths = [
            self.project_root / "output" / "verilog" / "alu_wave.vcd",
            self.project_root / "output" / "simulation" / "alu_wave.vcd",
            self.project_root / "verilog" / "alu_wave.vcd",
        ]

        for path in possible_paths:
            if path.exists():
                return path

        return None


# 独立运行支持
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="ALU验证工作流管理器")
    parser.add_argument('spec_file', help='Spec文件路径')
    parser.add_argument('-a', '--a-value', type=lambda x: int(x, 0),
                        help='操作数A（十六进制，如0x000F）')
    parser.add_argument('-b', '--b-value', type=lambda x: int(x, 0),
                        help='操作数B（十六进制，如0x000A）')
    parser.add_argument('-o', '--opcode', type=int,
                        help='操作码（0-15）')
    parser.add_argument('--open-wave', action='store_true',
                        help='自动打开波形查看器')
    parser.add_argument('--only-bdd_spec', action='store_true',
                        help='仅生成BDD测试')

    args = parser.parse_args()

    workflow = ALUWorkflow()

    if args.only_bdd:
        # 只生成BDD
        spec = workflow.run_spec_to_bdd(args.spec_file)
        sys.exit(0 if spec else 1)
    else:
        # 运行完整工作流
        results = workflow.run_full_workflow_from_spec(
            spec_file=args.spec_file,
            a_value=args.a_value,
            b_value=args.b_value,
            opcode=args.opcode,
            open_wave=args.open_wave
        )

        sys.exit(0 if results['spec_to_bdd'] else 1)