#!/usr/bin/env python3
"""GPT-5.1-Codex 基础功能测试"""

import os
from llm_providers import OpenAIProvider

# 方式 1：使用环境变量
print("=" * 70)
print("测试 1: 使用环境变量中的 API Key")
print("=" * 70)

try:
    provider = OpenAIProvider(model="gpt-5.1-codex")

    # 测试场景描述生成
    result = provider.generate_scenario_description(
        operation_name="ADD",
        operation_code="0000",
        operation_description="Addition (A + B)",
        bitwidth=16
    )

    print(f"\n✅ 测试成功!")
    print(f"生成的描述:\n{result}\n")

except Exception as e:
    print(f"\n❌ 测试失败: {e}\n")

# 方式 2：直接传入 API Key
print("=" * 70)
print("测试 2: 直接传入 API Key")
print("=" * 70)

api_key = os.getenv("OPENAI_API_KEY")
if api_key:
    try:
        provider = OpenAIProvider(api_key=api_key, model="gpt-5.1-codex")

        result = provider.generate_feature_description(
            bitwidth=32,
            operations_count=12,
            operations_list=["ADD", "SUB", "MUL", "DIV", "AND", "OR", "XOR"]
        )

        print(f"\n✅ 测试成功!")
        print(f"生成的功能描述:\n{result}\n")

    except Exception as e:
        print(f"\n❌ 测试失败: {e}\n")
else:
    print("⚠️  未找到 API Key，跳过此测试\n")

print("=" * 70)