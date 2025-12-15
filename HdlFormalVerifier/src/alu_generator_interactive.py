"""
交互式 ALU Generator - 使用自然语言指定参数
===========================================

支持自然语言输入，自动解析：
- Bitwidth: 8-bit, 16-bit, 32-bit, 64-bit
- LLM Provider: groq, deepseek, etc.

Example inputs:
  "Generate a 32-bit ALU using groq"
  "I need 8-bit ALU"
  "Create 64-bit ALU with deepseek"
"""

import re
import sys
import argparse
from pathlib import Path

# 添加当前目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent))

try:
    from alu_generator import ALUGenerator
except ImportError:
    print("❌ Cannot import alu_generator.py")
    print("   Make sure alu_generator.py is in the same directory")
    sys.exit(1)


def parse_natural_language(input_text: str) -> dict:
    """
    从自然语言中提取参数

    Args:
        input_text: 用户输入的自然语言

    Returns:
        dict: 包含 bitwidth 和 llm_provider 的字典
    """
    input_lower = input_text.lower()

    # 提取 bitwidth
    bitwidth = 16  # 默认

    # 匹配模式：8-bit, 8bit, 8 bit, eight bit
    bitwidth_patterns = [
        (r'(\d+)\s*-?\s*bit', lambda m: int(m.group(1))),
        (r'(\d+)\s*位', lambda m: int(m.group(1))),
        (r'eight', lambda: 8),
        (r'sixteen', lambda: 16),
        (r'thirty[-\s]?two', lambda: 32),
        (r'sixty[-\s]?four', lambda: 64),
    ]

    for pattern, extract_func in bitwidth_patterns:
        match = re.search(pattern, input_lower)
        if match:
            if callable(extract_func):
                if match.groups():
                    bitwidth = extract_func(match)
                else:
                    bitwidth = extract_func()
            break

    # 验证 bitwidth
    valid_bitwidths = [8, 16, 32, 64]
    if bitwidth not in valid_bitwidths:
        print(f"⚠️  Invalid bitwidth: {bitwidth}")
        print(f"   Valid options: {valid_bitwidths}")
        print(f"   Using default: 16")
        bitwidth = 16

    # 提取 LLM provider
    llm_provider = 'groq'  # 默认

    llm_keywords = {
        'groq': ['groq'],
        'deepseek': ['deepseek', 'deep seek'],
        'openai': ['openai', 'gpt', 'chatgpt'],
        'claude': ['claude', 'anthropic'],
        'gemini': ['gemini', 'google'],
    }

    for provider, keywords in llm_keywords.items():
        for keyword in keywords:
            if keyword in input_lower:
                llm_provider = provider
                break
        if llm_provider != 'groq':  # 找到了就跳出
            break

    return {
        'bitwidth': bitwidth,
        'llm_provider': llm_provider
    }


def interactive_mode():
    """交互模式"""
    print("=" * 80)
    print("🤖 Interactive ALU Generator")
    print("=" * 80)
    print()
    print("📝 Examples:")
    print("   • 'Generate a 32-bit ALU using groq'")
    print("   • 'I need 8-bit ALU'")
    print("   • 'Create 64-bit ALU with deepseek'")
    print("   • '16-bit ALU'")
    print()
    print("💡 Just describe what you need in natural language!")
    print("   Type 'quit' or 'exit' to stop.")
    print()

    while True:
        print("-" * 80)
        user_input = input("💬 What ALU do you need? ").strip()

        if not user_input:
            continue

        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!")
            break

        # 解析输入
        params = parse_natural_language(user_input)

        print(f"\n✅ Understood:")
        print(f"   Bitwidth: {params['bitwidth']}-bit")
        print(f"   LLM: {params['llm_provider']}")
        print()

        # 确认
        confirm = input("👉 Generate this ALU? (y/n, default: y): ").strip().lower()
        if confirm and confirm not in ['y', 'yes', '']:
            print("⏭️  Skipped")
            continue

        # 生成
        print()
        try:
            generator = ALUGenerator(
                llm_provider=params['llm_provider'],
                debug=False
            )

            alu_path = generator.generate_alu(
                bitwidth=params['bitwidth'],
                module_name='alu'
            )

            if alu_path:
                print(f"\n🎉 Success! ALU saved to:")
                print(f"   {alu_path}")
            else:
                print(f"\n❌ Generation failed")

        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()

        print()


def command_line_mode(args):
    """命令行模式"""
    if args.input:
        # 从命令行参数解析
        params = parse_natural_language(args.input)

        print("=" * 80)
        print("🤖 ALU Generator - Command Line Mode")
        print("=" * 80)
        print(f"\n✅ Parsed:")
        print(f"   Bitwidth: {params['bitwidth']}-bit")
        print(f"   LLM: {params['llm_provider']}")
        print()

        generator = ALUGenerator(
            llm_provider=params['llm_provider'],
            output_dir=args.output,
            project_root=args.project_root,
            debug=not args.no_debug
        )

        alu_path = generator.generate_alu(
            bitwidth=params['bitwidth'],
            module_name=args.module_name
        )

        if alu_path:
            print(f"\n🎉 Success!")
            return 0
        else:
            print(f"\n❌ Failed")
            return 1
    else:
        # 手动指定参数
        generator = ALUGenerator(
            llm_provider=args.llm,
            output_dir=args.output,
            project_root=args.project_root,
            debug=not args.no_debug
        )

        alu_path = generator.generate_alu(
            bitwidth=args.bitwidth,
            module_name=args.module_name
        )

        if alu_path:
            return 0
        else:
            return 1


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='Interactive ALU Generator with Natural Language Support',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Interactive mode (recommended)
  python alu_generator_interactive.py
  
  # Natural language input from command line
  python alu_generator_interactive.py -i "Generate a 32-bit ALU using groq"
  python alu_generator_interactive.py -i "I need 8-bit ALU"
  python alu_generator_interactive.py -i "Create 64-bit ALU with deepseek"
  
  # Traditional parameter mode
  python alu_generator_interactive.py --llm groq --bitwidth 32
        '''
    )

    parser.add_argument('-i', '--input',
                       help='Natural language input describing the ALU')
    parser.add_argument('--llm', default='groq',
                       help='LLM provider (groq, deepseek, openai, claude)')
    parser.add_argument('--bitwidth', type=int, default=16,
                       help='ALU bitwidth (8, 16, 32, 64)')
    parser.add_argument('--output', help='Output directory')
    parser.add_argument('--project-root', help='Project root directory')
    parser.add_argument('--module-name', default='alu', help='Module name')
    parser.add_argument('--no-debug', action='store_true', help='Disable debug')

    args = parser.parse_args()

    # 如果没有任何参数，进入交互模式
    if len(sys.argv) == 1:
        interactive_mode()
        return 0
    else:
        return command_line_mode(args)


if __name__ == "__main__":
    sys.exit(main())