#!/usr/bin/env python3
"""
LLM API Configuration Tool
一次性配置所有免费 LLM API keys
"""

import json
import os
from pathlib import Path


class LLMConfigTool:
    """LLM API 配置工具"""

    def __init__(self, config_file: str = "llm_config.json"):
        self.config_file = config_file
        self.config = self._load_or_create_config()

    def _load_or_create_config(self) -> dict:
        """加载或创建配置文件"""
        default_config = {
            "default_provider": "gemini",
            "providers": {
                "gemini": {
                    "enabled": False,
                    "api_key": "",
                    "model": "gemini-pro",
                    "description": "Google Gemini - FREE (60 req/min)"
                },
                "groq": {
                    "enabled": False,
                    "api_key": "",
                    "model": "mixtral-8x7b-32768",
                    "description": "Groq - FREE (ultra-fast)"
                },
                "deepseek": {
                    "enabled": False,
                    "api_key": "",
                    "model": "deepseek-chat",
                    "description": "DeepSeek - FREE (Chinese LLM)"
                },
                "openai": {
                    "enabled": False,
                    "api_key": "",
                    "model": "gpt-5.1",
                    "description": "OpenAI GPT - PAID"
                },
                "claude": {
                    "enabled": False,
                    "api_key": "",
                    "model": "claude-sonnet-4-20250514",
                    "description": "Anthropic Claude - PAID"
                },
                "local": {
                    "enabled": True,
                    "api_key": "",
                    "model": "",
                    "description": "Local Templates - FREE (no API needed)"
                }
            }
        }

        # 尝试加载现有配置
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)

                # 验证配置文件结构
                if isinstance(loaded_config, dict) and "providers" in loaded_config:
                    # 合并配置，确保所有必需的键都存在
                    for key in default_config:
                        if key not in loaded_config:
                            loaded_config[key] = default_config[key]

                    # 确保所有提供者都存在
                    for provider_key in default_config["providers"]:
                        if provider_key not in loaded_config["providers"]:
                            loaded_config["providers"][provider_key] = default_config["providers"][provider_key]

                    return loaded_config
            except Exception as e:
                print(f"⚠️  Warning: Failed to load config file: {e}")
                print("   Using default configuration...")

        # 返回默认配置
        return default_config

    def save_config(self):
        """保存配置到文件"""
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
        print(f"\n✅ Configuration saved to: {self.config_file}")

    def interactive_setup(self):
        """交互式设置所有 API keys"""
        print("\n" + "=" * 70)
        print("🔧 LLM API Configuration Tool")
        print("=" * 70)
        print("\n💡 This tool will help you configure all your API keys at once.")
        print("   You can skip any provider by pressing Enter.\n")

        # 配置每个提供者
        providers_info = {
            "gemini": {
                "name": "Google Gemini",
                "url": "https://makersuite.google.com/app/apikey",
                "key_format": "AIzaSyD-...",
                "free": True
            },
            "groq": {
                "name": "Groq",
                "url": "https://console.groq.com/keys",
                "key_format": "gsk_...",
                "free": True
            },
            "deepseek": {
                "name": "DeepSeek",
                "url": "https://platform.deepseek.com/",
                "key_format": "sk-...",
                "free": True
            },
            "openai": {
                "name": "OpenAI GPT",
                "url": "https://platform.openai.com/api-keys",
                "key_format": "sk-proj-... or sk-...",
                "free": False
            },
            "claude": {
                "name": "Anthropic Claude",
                "url": "https://console.anthropic.com/",
                "key_format": "sk-ant-...",
                "free": False
            }
        }

        for provider_key, info in providers_info.items():
            print("\n" + "-" * 70)
            status = "🆓 FREE" if info["free"] else "💰 PAID"
            print(f"{status} {info['name']}")
            print(f"   Get API key: {info['url']}")
            print(f"   Key format: {info['key_format']}")

            # 检查当前配置（安全检查）
            current_key = ""
            if "providers" in self.config and provider_key in self.config["providers"]:
                current_key = self.config["providers"][provider_key].get("api_key", "")

            if current_key:
                print(f"   Current: {current_key[:10]}... (configured)")
                update = input(f"   Update API key? (y/n) [default: n]: ").strip().lower()
                if update != 'y':
                    continue

            # 输入新 key
            api_key = input(f"   Enter {info['name']} API key (or press Enter to skip): ").strip()

            if api_key:
                # 确保结构存在
                if "providers" not in self.config:
                    self.config["providers"] = {}
                if provider_key not in self.config["providers"]:
                    self.config["providers"][provider_key] = {}

                self.config["providers"][provider_key]["api_key"] = api_key
                self.config["providers"][provider_key]["enabled"] = True
                print(f"   ✅ {info['name']} configured!")
            else:
                print(f"   ⏭️  Skipped")

        # 选择默认提供者
        print("\n" + "=" * 70)
        print("🎯 Select Default Provider")
        print("=" * 70)

        enabled_providers = []
        for key, provider in self.config["providers"].items():
            if provider["enabled"] and provider["api_key"]:
                enabled_providers.append((key, provider["description"]))

        if enabled_providers:
            print("\nConfigured providers:")
            for i, (key, desc) in enumerate(enabled_providers, 1):
                default_marker = " (current default)" if key == self.config.get("default_provider") else ""
                print(f"  {i}. {desc}{default_marker}")

            choice = input(f"\nSelect default provider (1-{len(enabled_providers)}) [default: 1]: ").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(enabled_providers):
                self.config["default_provider"] = enabled_providers[int(choice) - 1][0]
            elif not choice:
                self.config["default_provider"] = enabled_providers[0][0]

        # 保存配置
        self.save_config()

        # 显示摘要
        self.show_summary()

    def show_summary(self):
        """显示配置摘要"""
        print("\n" + "=" * 70)
        print("📊 Configuration Summary")
        print("=" * 70)

        print(f"\n✅ Default Provider: {self.config['default_provider']}")
        print("\n📋 Configured Providers:")

        for key, provider in self.config["providers"].items():
            if provider["enabled"]:
                status = "✅" if provider["api_key"] else "⚠️"
                key_display = f"{provider['api_key'][:10]}..." if provider["api_key"] else "Not configured"
                print(f"   {status} {key:10} - {key_display}")

        print("\n💡 Usage:")
        print("   # Use configured default")
        print("   python bdd_generator.py")
        print()
        print("   # Specify a provider")
        default = self.config['default_provider']
        print(f"   python bdd_generator.py --llm-provider {default}")

        # 生成环境变量设置脚本
        self.generate_env_script()

    def generate_env_script(self):
        """生成环境变量设置脚本"""
        print("\n" + "=" * 70)
        print("🔐 Environment Variables (Optional)")
        print("=" * 70)
        print("\nYou can also set API keys as environment variables:")

        # Windows PowerShell
        print("\n# Windows (PowerShell):")
        for key, provider in self.config["providers"].items():
            if provider["api_key"]:
                env_name = f"{key.upper()}_API_KEY"
                print(f'$env:{env_name}="{provider["api_key"]}"')

        # Linux/Mac Bash
        print("\n# Linux/Mac (Bash/Zsh):")
        for key, provider in self.config["providers"].items():
            if provider["api_key"]:
                env_name = f"{key.upper()}_API_KEY"
                print(f'export {env_name}="{provider["api_key"]}"')

        # Save to file
        self._save_env_files()

    def _save_env_files(self):
        """保存环境变量到文件"""
        # Windows batch file
        with open('set_env.bat', 'w', encoding='utf-8') as f:
            f.write('@echo off\n')
            f.write('REM Set LLM API Keys\n')
            for key, provider in self.config["providers"].items():
                if provider["api_key"]:
                    env_name = f"{key.upper()}_API_KEY"
                    f.write(f'set {env_name}={provider["api_key"]}\n')
            f.write('echo Environment variables set!\n')

        # Linux/Mac shell script
        with open('set_env.sh', 'w', encoding='utf-8') as f:
            f.write('#!/bin/bash\n')
            f.write('# Set LLM API Keys\n')
            for key, provider in self.config["providers"].items():
                if provider["api_key"]:
                    env_name = f"{key.upper()}_API_KEY"
                    f.write(f'export {env_name}="{provider["api_key"]}"\n')
            f.write('echo "Environment variables set!"\n')

        # Make shell script executable
        try:
            os.chmod('set_env.sh', 0o755)
        except:
            pass

        print(f"\n📝 Environment variable scripts saved:")
        print(f"   - set_env.bat (Windows)")
        print(f"   - set_env.sh (Linux/Mac)")

    def quick_test(self):
        """快速测试配置的 API keys"""
        print("\n" + "=" * 70)
        print("🧪 Quick API Key Test")
        print("=" * 70)

        test = input("\nDo you want to test your API keys? (y/n): ").strip().lower()
        if test == 'y':
            print("\n💡 Run: python test_free_api.py")
            print("   This will test each configured API key.")


def main():
    """主函数"""
    tool = LLMConfigTool()

    print("\n" + "-" * 70)
    print("🚀 Welcome to LLM API Configuration Tool")
    print("-" * 70)
    print("\nThis tool helps you:")
    print("  1. Configure all your FREE and PAID LLM API keys")
    print("  2. Save them to a config file")
    print("  3. Generate environment variable scripts")
    print("  4. Set a default provider")

    print("\n💡 Recommended FREE providers:")
    print("   • Google Gemini - 60 req/min (https://makersuite.google.com/app/apikey)")
    print("   • Groq - Ultra-fast (https://console.groq.com/keys)")
    print("   • DeepSeek - Chinese LLM (https://platform.deepseek.com/)")

    ready = input("\n📝 Ready to configure? (y/n): ").strip().lower()
    if ready != 'y':
        print("\n👋 Bye!")
        return

    # 交互式设置
    tool.interactive_setup()

    # 快速测试
    tool.quick_test()

    print("\n" + "=" * 70)
    print("✅ Configuration Complete!")
    print("=" * 70)
    print("\n🎉 You can now use your configured API keys:")
    print("   python bdd_generator.py")
    print()


if __name__ == "__main__":
    main()