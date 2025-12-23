#!/usr/bin/env python3
"""
run_benchmark.py
LLM BDD Benchmark 主入口脚本

运行完整的LLM对比实验
"""

import sys
from pathlib import Path

# ============================================================================
# 代理设置 - 自动从配置文件读取
# ============================================================================
def setup_proxy():
    """从配置文件读取并设置代理环境变量"""
    import os
    import json
    from pathlib import Path

    # 查找配置文件
    config_paths = [
        Path('config/llm_config.json'),
        Path('llm_config.json'),
        Path('../config/llm_config.json'),
    ]

    config_file = None
    for path in config_paths:
        if path.exists():
            config_file = path
            break

    if not config_file:
        print("⚠️  未找到 llm_config.json，代理未设置")
        return

    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)

        proxy_config = config.get('proxy', {})

        if proxy_config.get('enabled'):
            os.environ['HTTP_PROXY'] = proxy_config.get('http_proxy', '')
            os.environ['HTTPS_PROXY'] = proxy_config.get('https_proxy', '')
            print(f"🌐 代理已启用: {os.environ['HTTPS_PROXY']}")
        else:
            print("ℹ️  代理未启用（配置中 enabled=false）")

    except Exception as e:
        print(f"⚠️  读取代理配置失败: {e}")

# 调用代理设置
setup_proxy()
# ============================================================================



# 修改这两行
sys.path.insert(0, str(Path(__file__).parent / 'src'))  # 添加src到路径
sys.path.insert(0, str(Path(__file__).parent))           # 添加根目录

# 然后修改导入
from benchmark.standard_dut_generator import StandardDUTGenerator
from benchmark.benchmark_runner import BenchmarkRunner
from benchmark.experiment_config import SPECS


def main():
    """主函数"""

    print("\n" + "=" * 80)
    print("🎯 LLM BDD Quality Benchmark Experiment")
    print("=" * 80)
    print("\nThis experiment will:")
    print("  1. Generate standard DUTs (once)")
    print("  2. Test multiple LLMs on BDD generation")
    print("  3. Compare quality metrics")
    print("=" * 80 + "\n")

    # ========== Phase 1: 生成标准DUT（只做一次）==========
    print("📋 Phase 1: Generating Standard DUTs...")
    print("-" * 80)

    dut_generator = StandardDUTGenerator(output_dir="standard_dut")
    dut_map = dut_generator.generate_all_standard_duts(SPECS)

    if not dut_map:
        print("\n❌ Error: No DUTs generated. Please check your spec files.")
        return 1

    print(f"\n✅ Phase 1 Complete: Generated {len(dut_map)} standard DUTs")
    print("=" * 80 + "\n")

    # ========== Phase 2: LLM Benchmark ==========
    print("📋 Phase 2: Running LLM Benchmark...")
    print("-" * 80 + "\n")

    runner = BenchmarkRunner(standard_dut_map=dut_map)
    results = runner.run_benchmark()

    print(f"\n✅ Phase 2 Complete: {len(results)} experiments finished")
    print("=" * 80)

    # ========== 结果位置 ==========
    print("\n📁 Results saved to:")
    print("  - Raw outputs: benchmark_results/raw_outputs/")
    print("  - Metrics: benchmark_results/metrics/intermediate_results.json")
    print("  - Invalid JSON: benchmark_results/invalid/json_errors/")
    print("  - Standard DUTs: standard_dut/")
    print("=" * 80 + "\n")

    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️  Experiment interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)