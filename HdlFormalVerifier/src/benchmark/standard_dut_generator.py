"""
standard_dut_generator.py
标准DUT生成器

在实验开始前，生成固定的Verilog DUT
这个DUT将被所有LLM实验共用
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime
# 添加父目录到路径以便导入alu_generator
from alu_generator import ALUGenerator

current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir.parent))


class StandardDUTGenerator:
    """
    生成标准DUT（固定，确定性）

    关键：这个生成器只运行一次！
    """

    def __init__(self, output_dir: str = "standard_dut"):
        """
        初始化标准DUT生成器

        Args:
            output_dir: DUT输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 初始化ALU生成器，传入输出目录
        self.alu_generator = ALUGenerator(output_dir=str(self.output_dir))

    def generate_standard_dut(self, spec_json_file: str, spec_name: str) -> str:
        """
        从规格JSON生成标准DUT

        Args:
            spec_json_file: 规格JSON文件路径
            spec_name: 规格名称

        Returns:
            生成的DUT文件路径
        """
        print(f"\n{'=' * 60}")
        print(f"Generating Standard DUT: {spec_name}")
        print(f"{'=' * 60}")

        # 1. 加载规格
        with open(spec_json_file, 'r', encoding='utf-8') as f:
            spec = json.load(f)

        print(f"✓ Loaded spec from: {spec_json_file}")
        print(f"  Bit-width: {spec.get('bit_width', 16)}")
        print(f"  Operations: {spec.get('operations', [])}")

        # 2. 生成DUT（确定性）
        dut_code = self.alu_generator.generate_verilog(spec)

        # 3. 保存DUT
        dut_filename = f"{spec_name}_dut.v"
        dut_path = self.output_dir / dut_filename

        with open(dut_path, 'w', encoding='utf-8') as f:
            f.write(dut_code)

        print(f"✓ DUT saved to: {dut_path}")

        # 4. 保存元数据
        metadata = {
            'spec_name': spec_name,
            'spec_file': spec_json_file,
            'dut_file': str(dut_path),
            'bit_width': spec.get('bit_width', 16),
            'operations': spec.get('operations', []),
            'generated_at': datetime.now().isoformat(),
            'generator_version': '1.0'
        }

        metadata_path = self.output_dir / f"{spec_name}_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)

        print(f"✓ Metadata saved to: {metadata_path}")
        print(f"{'=' * 60}\n")

        return str(dut_path)

    def generate_all_standard_duts(self, specs: list) -> dict:
        """
        为所有规格生成标准DUT

        Args:
            specs: 规格列表

        Returns:
            {spec_name: dut_path} 映射
        """
        dut_map = {}

        print("\n" + "=" * 60)
        print("🔧 Generating All Standard DUTs")
        print("=" * 60)
        print(f"Total specs: {len(specs)}")
        print("=" * 60 + "\n")

        for spec in specs:
            spec_name = spec['name']
            json_file = spec['json_file']

            # 检查文件是否存在
            if not Path(json_file).exists():
                print(f"⚠️  Warning: {json_file} not found, skipping...")
                continue

            # 生成DUT
            dut_path = self.generate_standard_dut(json_file, spec_name)
            dut_map[spec_name] = dut_path

        print("=" * 60)
        print(f"✅ Generated {len(dut_map)} standard DUTs")
        print("=" * 60)

        # 保存总映射
        mapping_file = self.output_dir / "dut_mapping.json"
        with open(mapping_file, 'w', encoding='utf-8') as f:
            json.dump(dut_map, f, indent=2)

        print(f"\n📄 DUT mapping saved to: {mapping_file}\n")

        return dut_map

    def verify_dut_exists(self, spec_name: str) -> bool:
        """
        验证标准DUT是否已存在

        Args:
            spec_name: 规格名称

        Returns:
            是否存在
        """
        dut_file = self.output_dir / f"{spec_name}_dut.v"
        return dut_file.exists()


# ========== 使用示例 ==========
if __name__ == "__main__":
    from experiment_config import SPECS

    # 创建生成器
    generator = StandardDUTGenerator(output_dir="standard_dut")

    # 生成所有标准DUT
    dut_map = generator.generate_all_standard_duts(SPECS)

    print("\n" + "=" * 60)
    print("📊 Generated DUTs:")
    print("=" * 60)
    for spec_name, dut_path in dut_map.items():
        print(f"  {spec_name}: {dut_path}")
    print("=" * 60)
