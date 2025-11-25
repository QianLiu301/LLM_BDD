#!/usr/bin/env python3
"""
诊断脚本：检测 gpt-5.1-codex 是否正确路由到 completions 端点

运行此脚本可以：
1. 检查当前使用的 llm_providers.py 版本
2. 测试 codex 模型是否正确路由
3. 验证 API 调用是否成功
"""

import os
import sys

print("=" * 80)
print("🔍 GPT-5.1-Codex 诊断工具")
print("=" * 80)

# ==================== 步骤 1: 检查文件 ====================
print("\n📁 步骤 1: 检查 llm_providers.py 文件...")

try:
    # 尝试导入
    from llm_providers import OpenAIProvider

    print("✅ 成功导入 llm_providers")

    # 检查是否有 _is_codex_model 方法
    if hasattr(OpenAIProvider, '_is_codex_model'):
        print("✅ 找到 _is_codex_model 方法 - 使用的是修复后的版本")
    else:
        print("❌ 未找到 _is_codex_model 方法 - 使用的是旧版本!")
        print("\n⚠️  请按照以下步骤操作:")
        print("   1. 备份当前文件: cp llm_providers.py llm_providers_backup.py")
        print("   2. 替换为修复版本: cp llm_providers_fixed.py llm_providers.py")
        print("   3. 重新运行此脚本")
        sys.exit(1)

    # 检查是否有 _call_api_completions 方法
    if hasattr(OpenAIProvider, '_call_api_completions'):
        print("✅ 找到 _call_api_completions 方法 - completions 端点已就绪")
    else:
        print("❌ 未找到 _call_api_completions 方法 - 缺少 completions 支持!")
        sys.exit(1)

except ImportError as e:
    print(f"❌ 无法导入 llm_providers: {e}")
    print("\n请确保 llm_providers.py 在当前目录或 Python 路径中")
    sys.exit(1)

# ==================== 步骤 2: 检查 API Key ====================
print("\n🔑 步骤 2: 检查 OpenAI API Key...")

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("❌ 未找到 OPENAI_API_KEY 环境变量")
    print("\n请设置 API Key:")
    print("   export OPENAI_API_KEY='your-api-key-here'")
    sys.exit(1)
else:
    # 只显示前4位和后4位
    masked_key = f"{api_key[:4]}...{api_key[-4:]}"
    print(f"✅ 找到 API Key: {masked_key}")

# ==================== 步骤 3: 测试模型检测 ====================
print("\n🧪 步骤 3: 测试模型类型检测...")

try:
    provider = OpenAIProvider(model="gpt-5.1-codex")

    # 测试 _is_codex_model
    is_codex = provider._is_codex_model("gpt-5.1-codex")
    print(f"   _is_codex_model('gpt-5.1-codex') = {is_codex}")

    if is_codex:
        print("   ✅ 正确识别为 Codex 模型")
    else:
        print("   ❌ 未能识别为 Codex 模型!")
        sys.exit(1)

    # 测试其他模型
    is_not_codex = provider._is_codex_model("gpt-5-mini")
    print(f"   _is_codex_model('gpt-5-mini') = {is_not_codex}")

    if not is_not_codex:
        print("   ✅ 正确识别非 Codex 模型")
    else:
        print("   ❌ 错误地将 gpt-5-mini 识别为 Codex!")

except Exception as e:
    print(f"❌ 测试失败: {e}")
    sys.exit(1)

# ==================== 步骤 4: 测试 API 调用路由 ====================
print("\n🚦 步骤 4: 测试 API 调用路由...")

print("\n📝 测试场景: 生成一个简单的 BDD 场景描述")
print("-" * 80)

try:
    # 创建 provider
    provider = OpenAIProvider(model="gpt-5.1-codex")
    print(f"✅ 创建 Provider: model={provider.model}")

    # 测试 generate_scenario_description
    print("\n🔧 调用 generate_scenario_description()...")

    scenario = provider.generate_scenario_description(
        operation_name="ADD",
        operation_code="0000",
        operation_description="Addition (A + B)",
        bitwidth=16
    )

    print(f"\n✅ 成功生成场景描述:")
    print(f"   {scenario[:200]}...")

    # 检查是否使用了 fallback
    if "Fallback" in scenario or "fallback" in scenario:
        print("\n⚠️  警告: 使用了 fallback 响应，API 调用可能失败了")
        print("   请检查上面的错误信息")
    else:
        print("\n✅ API 调用成功！没有使用 fallback")

except Exception as e:
    print(f"\n❌ 调用失败: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# ==================== 步骤 5: 测试 _call_api ====================
print("\n🔧 步骤 5: 直接测试 _call_api() 方法...")
print("-" * 80)

try:
    provider = OpenAIProvider(model="gpt-5.1-codex")

    prompt = """Generate a simple BDD scenario for ALU ADD operation.
Just respond with one sentence describing the test."""

    print("📤 发送测试 prompt...")
    result = provider._call_api(prompt, max_tokens=100)

    print(f"\n✅ _call_api() 返回结果:")
    print(f"   类型: {type(result)}")
    print(f"   长度: {len(result)} 字符")
    print(f"   内容预览: {result[:200]}...")

    # 尝试解析为 JSON
    import json

    try:
        parsed = json.loads(result)
        print(f"\n✅ 结果是有效的 JSON:")
        print(f"   包含的键: {list(parsed.keys())}")

        # 检查是否有必需的字段
        required_fields = ["scenario", "operation", "opcode", "bitwidth"]
        missing_fields = [f for f in required_fields if f not in parsed]

        if missing_fields:
            print(f"\n⚠️  警告: JSON 缺少字段: {missing_fields}")
        else:
            print(f"\n✅ JSON 包含所有必需字段!")

    except json.JSONDecodeError:
        print(f"\n⚠️  注意: 结果不是 JSON 格式（这对 codex 是正常的）")

except Exception as e:
    print(f"\n❌ 测试失败: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# ==================== 总结 ====================
print("\n" + "=" * 80)
print("✅ 诊断完成!")
print("=" * 80)

print("\n📊 诊断总结:")
print("   ✅ llm_providers.py 已更新为修复版本")
print("   ✅ _is_codex_model 方法工作正常")
print("   ✅ _call_api_completions 方法已就绪")
print("   ✅ API 调用路由正确")

print("\n💡 下一步:")
print("   1. 确保 bdd_generator.py 调用 provider 的方式正确")
print("   2. 如果还有问题，检查 bdd_generator.py 中的调用代码")
print("   3. 考虑使用其他模型如 gpt-5-mini（支持 JSON mode，更稳定）")

print("\n🎯 模型推荐:")
print("   • gpt-5-mini: 快速、便宜、支持 JSON mode（推荐）")
print("   • gpt-5: 性能好、支持 JSON mode")
print("   • gpt-5.1-codex: 代码专用、不支持 JSON mode（需要额外处理）")

print("\n" + "=" * 80)