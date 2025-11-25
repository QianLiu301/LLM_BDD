"""
LLM Provider Module
Supports multiple LLM APIs including FREE options for generating BDD scenario descriptions
"""

import os
import json
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import requests

# 尝试导入 google-genai，如果失败则标记
try:
    from google import genai
    from google.genai import errors
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    print("⚠️  google-genai not installed. Install with: pip install -U google-genai")

# 尝试导入 openai SDK，如果失败则标记
try:
    from openai import OpenAI
    OPENAI_SDK_AVAILABLE = True
except ImportError:
    OPENAI_SDK_AVAILABLE = False
    print("⚠️  openai SDK not installed. Install with: pip install openai")


class LLMProvider(ABC):
    """Abstract base class for LLM providers"""

    @abstractmethod
    def generate_scenario_description(
        self,
        operation_name: str,
        operation_code: str,
        operation_description: str,
        bitwidth: int
    ) -> str:
        """Generate a BDD scenario description"""
        pass

    @abstractmethod
    def generate_feature_description(
        self,
        bitwidth: int,
        operations_count: int,
        operations_list: List[str]
    ) -> str:
        """Generate a Feature-level description"""
        pass


# ========== FREE PROVIDERS ==========


class GeminiProvider(LLMProvider):
    """
    Google Gemini API Provider - FREE!

    Free tier: 60 requests per minute
    How to get API key:
    1. Visit: https://makersuite.google.com/app/apikey
    2. Click "Create API Key"
    3. Copy the key

    Note: Requires Google account
    Now uses google-genai SDK with retry and model fallback
    """

    def __init__(self, api_key: Optional[str] = None, model: str = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")

        if not self.api_key:
            raise ValueError("Gemini API key not provided. Get free key at: https://makersuite.google.com/app/apikey")

        # 如果 google-genai 可用，使用新的 SDK
        if GENAI_AVAILABLE:
            # 使用新的模型名称和 SDK
            self.models_to_try = ["gemini-2.5-flash", "gemini-1.5-flash"]
            self.client = genai.Client(api_key=self.api_key)
            self.use_sdk = True
            self.max_retries = 3
            self.sleep_seconds = 2.0
        else:
            # 降级到旧的 REST API 方式
            self.model = model or "gemini-1.5-flash"
            self.use_sdk = False
            print("⚠️  Using REST API fallback. For better reliability, install: pip install -U google-genai")

    def _call_api_sdk(self, prompt: str) -> str:
        """使用新的 google-genai SDK 调用 API，包含重试和模型降级"""
        last_error = None

        for model_name in self.models_to_try:
            for attempt in range(1, self.max_retries + 1):
                try:
                    response = self.client.models.generate_content(
                        model=model_name,
                        contents=prompt,
                    )
                    return response.text

                except errors.ServerError as e:
                    # 处理 5xx 错误（包括 503 overloaded）
                    last_error = e
                    if attempt < self.max_retries:
                        time.sleep(self.sleep_seconds)
                    else:
                        break  # 这个模型的重试次数用完了

                except Exception as e:
                    # 其他错误，不需要重试
                    last_error = e
                    break

            # 如果不是最后一个模型，继续尝试下一个
            if model_name != self.models_to_try[-1]:
                continue

        # 所有尝试失败，返回 fallback
        print(f"⚠️  Gemini API failed after all attempts: {last_error}")
        return self._fallback_description(prompt)

    def _call_api_rest(self, prompt: str, max_tokens: int = 200) -> str:
        """旧的 REST API 调用方式（作为备用）"""
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"

        payload = {
            "contents": [{
                "parts": [{
                    "text": prompt
                }]
            }],
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": 0.7,
            }
        }

        try:
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            return result['candidates'][0]['content']['parts'][0]['text'].strip()
        except Exception as e:
            print(f"⚠️  Gemini REST API request failed: {e}")
            return self._fallback_description(prompt)

    def _call_api(self, prompt: str, max_tokens: int = 200) -> str:
        """统一的 API 调用接口"""
        if self.use_sdk:
            return self._call_api_sdk(prompt)
        else:
            return self._call_api_rest(prompt, max_tokens)

    def generate_scenario_description(
        self,
        operation_name: str,
        operation_code: str,
        operation_description: str,
        bitwidth: int
    ) -> str:
        """Generate a BDD scenario description"""
        prompt = f"""Generate a clear BDD scenario description for testing a {bitwidth}-bit ALU operation.

Operation Details:
- Name: {operation_name}
- Opcode: {operation_code}
- Description: {operation_description}
- Bitwidth: {bitwidth} bits

Requirements:
1. Write in Gherkin/BDD style
2. Be specific about the operation being tested
3. Keep it concise (1-2 sentences)
4. Focus on functional behavior

Generate the scenario description:
"""
        return self._call_api(prompt, max_tokens=150)

    def generate_feature_description(
        self,
        bitwidth: int,
        operations_count: int,
        operations_list: List[str]
    ) -> str:
        """Generate a Feature-level description"""
        prompt = f"""Generate a BDD Feature description for a {bitwidth}-bit ALU verification suite.

ALU Details:
- Bitwidth: {bitwidth} bits
- Total Operations: {operations_count}
- Operations: {', '.join(operations_list[:10])}

Generate a concise Feature description (2-4 sentences):
"""
        return self._call_api(prompt, max_tokens=200)

    def _fallback_description(self, prompt: str) -> str:
        """Fallback description when API fails"""
        return "Test ALU operation with various input values and verify correct output"


class GroqProvider(LLMProvider):
    """
    Groq API Provider - FREE and FAST!

    Free tier: 30 requests per minute, 14,400 per day
    How to get API key:
    1. Visit: https://console.groq.com/keys
    2. Sign up (free, no credit card needed)
    3. Create API key

    Models: llama3-70b, mixtral-8x7b, gemma-7b
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "llama-3.3-70b-versatile"):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model = model
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"

        if not self.api_key:
            raise ValueError("Groq API key not provided. Get free key at: https://console.groq.com/keys")

    def _call_api(self, prompt: str, max_tokens: int = 200) -> str:
        """Call Groq API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that generates clear, concise BDD scenario descriptions for hardware verification."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": max_tokens,
            "temperature": 0.7
        }

        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            return result['choices'][0]['message']['content'].strip()
        except Exception as e:
            print(f"⚠️  Groq API request failed: {e}")
            return self._fallback_description(prompt)

    def generate_scenario_description(
        self,
        operation_name: str,
        operation_code: str,
        operation_description: str,
        bitwidth: int
    ) -> str:
        """Generate a BDD scenario description"""
        prompt = f"""Generate a clear BDD scenario description for testing a {bitwidth}-bit ALU operation.

Operation: {operation_name} (Opcode: {operation_code})
Description: {operation_description}
Bitwidth: {bitwidth} bits

Generate a concise BDD scenario description (1-2 sentences):
"""
        return self._call_api(prompt, max_tokens=150)

    def generate_feature_description(
        self,
        bitwidth: int,
        operations_count: int,
        operations_list: List[str]
    ) -> str:
        """Generate a Feature-level description"""
        prompt = f"""Generate a BDD Feature description for a {bitwidth}-bit ALU with {operations_count} operations including {', '.join(operations_list[:5])}.

Generate a concise Feature description (2-3 sentences):
"""
        return self._call_api(prompt, max_tokens=200)

    def _fallback_description(self, prompt: str) -> str:
        """Fallback description when API fails"""
        return "Verify ALU operation performs correctly with expected results"


class DeepSeekProvider(LLMProvider):
    """
    DeepSeek API Provider - Chinese LLM with free tier

    Free tier: Available with registration
    How to get API key:
    1. Visit: https://platform.deepseek.com/
    2. Sign up (requires Chinese phone number)
    3. Get API key
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "deepseek-chat"):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.model = model
        self.api_url = "https://api.deepseek.com/v1/chat/completions"

        if not self.api_key:
            raise ValueError("DeepSeek API key not provided. Get key at: https://platform.deepseek.com/")

    def _call_api(self, prompt: str, max_tokens: int = 200) -> str:
        """Call DeepSeek API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant for hardware verification."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": 0.7
        }

        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            return result['choices'][0]['message']['content'].strip()
        except Exception as e:
            print(f"⚠️  DeepSeek API request failed: {e}")
            return self._fallback_description(prompt)

    def generate_scenario_description(self, operation_name: str, operation_code: str,
                                     operation_description: str, bitwidth: int) -> str:
        prompt = f"Generate a BDD scenario description for {bitwidth}-bit ALU {operation_name} operation (opcode: {operation_code}). Keep it concise."
        return self._call_api(prompt, max_tokens=150)

    def generate_feature_description(self, bitwidth: int, operations_count: int,
                                    operations_list: List[str]) -> str:
        prompt = f"Generate a Feature description for {bitwidth}-bit ALU with {operations_count} operations. Keep it concise."
        return self._call_api(prompt, max_tokens=200)

    def _fallback_description(self, prompt: str) -> str:
        return "Test ALU operation with various inputs"


# ========== PAID PROVIDERS (Original) ==========
class OpenAIProvider(LLMProvider):
    """
    OpenAI GPT provider (PAID - requires credits)

    Now supports two modes:
    1. Official SDK (recommended) - more stable, better error handling
    2. REST API (fallback) - when SDK is unavailable
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model

        if not self.api_key:
            raise ValueError("OpenAI API key not provided")

        # 如果 SDK 可用，使用 SDK；否则使用 REST API
        if OPENAI_SDK_AVAILABLE:
            self.client = OpenAI(api_key=self.api_key)
            self.use_sdk = True
            print("🎯 Using OpenAI SDK (recommended)")
            # 🔍 调试：打印 base_url 和默认模型
            try:
                print(f"🔍 [DEBUG] OpenAI base_url: {self.client.base_url}")
            except Exception:
                print("⚠️ [DEBUG] Cannot read client.base_url (SDK version difference)")
        else:
            self.api_url = "https://api.openai.com/v1/chat/completions"
            self.use_sdk = False
            print("⚠️  Using REST API fallback. For better reliability, install: pip install openai")

    def _call_api_sdk(self, prompt: str, max_tokens: int = 200) -> str:
        """使用官方 OpenAI SDK 调用 API"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that generates clear, concise BDD scenario descriptions for hardware verification."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=max_tokens,
                temperature=0.7
            )

            # === 🔍 DEBUG: 确认真的在用 OpenAI，并查看费用情况 ===
            try:
                print("🔗 [DEBUG][OpenAI SDK] Response ID:", response.id)
                print("🤖 [DEBUG][OpenAI SDK] Model:", response.model)

                if hasattr(response, "usage") and response.usage is not None:
                    print(
                        "💰 [DEBUG][OpenAI SDK] Tokens - "
                        f"Prompt: {response.usage.prompt_tokens}, "
                        f"Completion: {response.usage.completion_tokens}, "
                        f"Total: {response.usage.total_tokens}"
                    )
                else:
                    print("⚠️ [DEBUG][OpenAI SDK] No usage info in response (maybe older API / special model)")
            except Exception as dbg_e:
                print(f"⚠️ [DEBUG][OpenAI SDK] Failed to read debug info: {dbg_e}")
            # ========================================================

            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"⚠️  OpenAI SDK request failed: {e}")
            return self._fallback_description(prompt)

    def _call_api_rest(self, prompt: str, max_tokens: int = 200) -> str:
        """使用 REST API 调用（备用方案）"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that generates clear, concise BDD scenario descriptions for hardware verification."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": max_tokens,
            "temperature": 0.7
        }

        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            result = response.json()

            # === 🔍 DEBUG: REST 模式下的调试输出 ===
            try:
                print("🔗 [DEBUG][OpenAI REST] HTTP Status:", response.status_code)
                print("🤖 [DEBUG][OpenAI REST] Model:", result.get("model"))
                usage = result.get("usage", {})
                if usage:
                    print(
                        "💰 [DEBUG][OpenAI REST] Tokens - "
                        f"Prompt: {usage.get('prompt_tokens')}, "
                        f"Completion: {usage.get('completion_tokens')}, "
                        f"Total: {usage.get('total_tokens')}"
                    )
                else:
                    print("⚠️ [DEBUG][OpenAI REST] No usage info in JSON response")
            except Exception as dbg_e:
                print(f"⚠️ [DEBUG][OpenAI REST] Failed to read debug info: {dbg_e}")
            # =====================================================

            return result['choices'][0]['message']['content'].strip()
        except Exception as e:
            print(f"⚠️  OpenAI REST API request failed: {e}")
            return self._fallback_description(prompt)

    def _call_api(self, prompt: str, max_tokens: int = 200) -> str:
        """统一的 API 调用接口"""
        if self.use_sdk:
            return self._call_api_sdk(prompt, max_tokens)
        else:
            return self._call_api_rest(prompt, max_tokens)

    def generate_scenario_description(
        self,
        operation_name: str,
        operation_code: str,
        operation_description: str,
        bitwidth: int
    ) -> str:
        """Generate a BDD scenario description"""
        prompt = f"""Generate a clear BDD scenario description for testing a {bitwidth}-bit ALU operation.

Operation Details:
- Name: {operation_name}
- Opcode: {operation_code}
- Description: {operation_description}
- Bitwidth: {bitwidth} bits

Requirements:
1. Write in Gherkin/BDD style
2. Be specific about the operation being tested
3. Keep it concise (1-2 sentences)
4. Focus on functional behavior

Generate the scenario description:
"""
        return self._call_api(prompt, max_tokens=150)

    def generate_feature_description(
        self,
        bitwidth: int,
        operations_count: int,
        operations_list: List[str]
    ) -> str:
        """Generate a Feature-level description"""
        prompt = f"""Generate a BDD Feature description for a {bitwidth}-bit ALU verification suite.

ALU Details:
- Bitwidth: {bitwidth} bits
- Total Operations: {operations_count}
- Operations: {', '.join(operations_list[:10])}

Generate the Feature description:
"""
        return self._call_api(prompt, max_tokens=200)

    def _fallback_description(self, prompt: str) -> str:
        """Fallback description when API fails"""
        return "Test ALU operation with various input values and verify correct output"



# class OpenAIProvider(LLMProvider):
#     """
#     OpenAI GPT provider (PAID - requires credits)
#
#     Now supports two modes:
#     1. Official SDK (recommended) - more stable, better error handling
#     2. REST API (fallback) - when SDK is unavailable
#     """
#
#     def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
#         self.api_key = api_key or os.getenv("OPENAI_API_KEY")
#         self.model = model
#
#         if not self.api_key:
#             raise ValueError("OpenAI API key not provided")
#
#         # 如果 SDK 可用，使用 SDK；否则使用 REST API
#         if OPENAI_SDK_AVAILABLE:
#             self.client = OpenAI(api_key=self.api_key)
#             self.use_sdk = True
#             print("🎯 Using OpenAI SDK (recommended)")
#         else:
#             self.api_url = "https://api.openai.com/v1/chat/completions"
#             self.use_sdk = False
#             print("⚠️  Using REST API fallback. For better reliability, install: pip install openai")
#
#     def _call_api_sdk(self, prompt: str, max_tokens: int = 200) -> str:
#         """使用官方 OpenAI SDK 调用 API"""
#         try:
#             response = self.client.chat.completions.create(
#                 model=self.model,
#                 messages=[
#                     {
#                         "role": "system",
#                         "content": "You are a helpful assistant that generates clear, concise BDD scenario descriptions for hardware verification."
#                     },
#                     {
#                         "role": "user",
#                         "content": prompt
#                     }
#                 ],
#                 max_tokens=max_tokens,
#                 temperature=0.7
#             )
#             return response.choices[0].message.content.strip()
#
#         except Exception as e:
#             print(f"⚠️  OpenAI SDK request failed: {e}")
#             return self._fallback_description(prompt)
#
#     def _call_api_rest(self, prompt: str, max_tokens: int = 200) -> str:
#         """使用 REST API 调用（备用方案）"""
#         headers = {
#             "Authorization": f"Bearer {self.api_key}",
#             "Content-Type": "application/json"
#         }
#
#         payload = {
#             "model": self.model,
#             "messages": [
#                 {
#                     "role": "system",
#                     "content": "You are a helpful assistant that generates clear, concise BDD scenario descriptions for hardware verification."
#                 },
#                 {
#                     "role": "user",
#                     "content": prompt
#                 }
#             ],
#             "max_tokens": max_tokens,
#             "temperature": 0.7
#         }
#
#         try:
#             response = requests.post(
#                 self.api_url,
#                 headers=headers,
#                 json=payload,
#                 timeout=30
#             )
#             response.raise_for_status()
#             result = response.json()
#             return result['choices'][0]['message']['content'].strip()
#         except Exception as e:
#             print(f"⚠️  OpenAI REST API request failed: {e}")
#             return self._fallback_description(prompt)
#
#     def _call_api(self, prompt: str, max_tokens: int = 200) -> str:
#         """统一的 API 调用接口"""
#         if self.use_sdk:
#             return self._call_api_sdk(prompt, max_tokens)
#         else:
#             return self._call_api_rest(prompt, max_tokens)
#
#     def generate_scenario_description(
#         self,
#         operation_name: str,
#         operation_code: str,
#         operation_description: str,
#         bitwidth: int
#     ) -> str:
#         """Generate a BDD scenario description"""
#         prompt = f"""Generate a clear BDD scenario description for testing a {bitwidth}-bit ALU operation.
#
# Operation Details:
# - Name: {operation_name}
# - Opcode: {operation_code}
# - Description: {operation_description}
# - Bitwidth: {bitwidth} bits
#
# Requirements:
# 1. Write in Gherkin/BDD style
# 2. Be specific about the operation being tested
# 3. Keep it concise (1-2 sentences)
# 4. Focus on functional behavior
#
# Generate the scenario description:
# """
#         return self._call_api(prompt, max_tokens=150)
#
#     def generate_feature_description(
#         self,
#         bitwidth: int,
#         operations_count: int,
#         operations_list: List[str]
#     ) -> str:
#         """Generate a Feature-level description"""
#         prompt = f"""Generate a BDD Feature description for a {bitwidth}-bit ALU verification suite.
#
# ALU Details:
# - Bitwidth: {bitwidth} bits
# - Total Operations: {operations_count}
# - Operations: {', '.join(operations_list[:10])}
#
# Generate the Feature description:
# """
#         return self._call_api(prompt, max_tokens=200)
#
#     def _fallback_description(self, prompt: str) -> str:
#         """Fallback description when API fails"""
#         return "Test ALU operation with various input values and verify correct output"


class ClaudeProvider(LLMProvider):
    """Anthropic Claude provider (PAID - requires credits)"""

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-sonnet-4-20250514"):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model = model
        self.api_url = "https://api.anthropic.com/v1/messages"

        if not self.api_key:
            raise ValueError("Anthropic API key not provided")

    def _call_api(self, prompt: str, max_tokens: int = 200) -> str:
        """Call Claude API"""
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }

        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }

        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            return result['content'][0]['text'].strip()
        except Exception as e:
            print(f"⚠️  Claude API request failed: {e}")
            return self._fallback_description(prompt)

    def generate_scenario_description(
        self,
        operation_name: str,
        operation_code: str,
        operation_description: str,
        bitwidth: int
    ) -> str:
        """Generate a BDD scenario description"""
        prompt = f"""Generate a clear BDD scenario description for testing a {bitwidth}-bit ALU operation.

Operation: {operation_name} (Opcode: {operation_code})
Description: {operation_description}

Generate a concise BDD scenario description (1-2 sentences):
"""
        return self._call_api(prompt, max_tokens=150)

    def generate_feature_description(
        self,
        bitwidth: int,
        operations_count: int,
        operations_list: List[str]
    ) -> str:
        """Generate a Feature-level description"""
        prompt = f"""Generate a professional BDD Feature description for a {bitwidth}-bit ALU with {operations_count} operations.

Main operations: {', '.join(operations_list[:10])}

Generate a concise Feature description (2-3 sentences):
"""
        return self._call_api(prompt, max_tokens=200)

    def _fallback_description(self, prompt: str) -> str:
        """Fallback description when API fails"""
        return "Verify ALU operation performs correctly with expected results"


# ========== LOCAL PROVIDER ==========


class LocalLLMProvider(LLMProvider):
    """Local LLM provider using templates (FREE - no API required)"""

    def generate_scenario_description(
        self,
        operation_name: str,
        operation_code: str,
        operation_description: str,
        bitwidth: int
    ) -> str:
        """Generate template-based scenario description"""
        templates = {
            'ADD': f"Verify that the {bitwidth}-bit ALU correctly performs addition of two operands",
            'SUB': f"Verify that the {bitwidth}-bit ALU correctly performs subtraction operation",
            'AND': f"Verify that the {bitwidth}-bit ALU correctly performs bitwise AND operation",
            'OR': f"Verify that the {bitwidth}-bit ALU correctly performs bitwise OR operation",
            'XOR': f"Verify that the {bitwidth}-bit ALU correctly performs bitwise XOR operation",
            'NOT': f"Verify that the {bitwidth}-bit ALU correctly performs bitwise NOT operation",
            'SHL': f"Verify that the {bitwidth}-bit ALU correctly performs left shift operation",
            'SHR': f"Verify that the {bitwidth}-bit ALU correctly performs right shift operation",
            'ROL': f"Verify that the {bitwidth}-bit ALU correctly performs rotate left operation",
            'ROR': f"Verify that the {bitwidth}-bit ALU correctly performs rotate right operation",
            'INC': f"Verify that the {bitwidth}-bit ALU correctly increments the operand by 1",
            'DEC': f"Verify that the {bitwidth}-bit ALU correctly decrements the operand by 1",
            'CMP': f"Verify that the {bitwidth}-bit ALU correctly compares two operands and sets flags",
            'NAND': f"Verify that the {bitwidth}-bit ALU correctly performs bitwise NAND operation",
            'NOR': f"Verify that the {bitwidth}-bit ALU correctly performs bitwise NOR operation",
            'PASS': f"Verify that the {bitwidth}-bit ALU correctly passes through the input operand"
        }

        if operation_name in templates:
            return templates[operation_name]
        else:
            return f"Verify that the {bitwidth}-bit ALU correctly performs {operation_name} operation as specified by {operation_description}"

    def generate_feature_description(
        self,
        bitwidth: int,
        operations_count: int,
        operations_list: List[str]
    ) -> str:
        """Generate Feature-level description"""
        return f"""As a hardware verification engineer
I want to verify the {bitwidth}-bit ALU implementation
So that I can ensure it correctly performs all {operations_count} supported arithmetic and logical operations,
including {', '.join(operations_list[:5])}{' and more' if len(operations_list) > 5 else ''}"""


# ========== FACTORY ==========


class LLMFactory:
    """Factory class for creating LLM providers"""

    @staticmethod
    def create_provider(provider_type: str, **kwargs) -> LLMProvider:
        """Create an LLM provider instance"""
        providers = {
            # FREE providers
            'gemini': GeminiProvider,
            'google': GeminiProvider,
            'groq': GroqProvider,
            'deepseek': DeepSeekProvider,
            'local': LocalLLMProvider,

            # PAID providers
            'openai': OpenAIProvider,
            'gpt': OpenAIProvider,
            'chatgpt': OpenAIProvider,
            'claude': ClaudeProvider,
            'anthropic': ClaudeProvider,
        }

        provider_type = provider_type.lower()
        if provider_type not in providers:
            print(f"⚠️  Unknown provider type: {provider_type}, using local template provider")
            provider_type = 'local'

        try:
            provider = providers[provider_type](**kwargs)

            # Print provider info
            if provider_type in ['gemini', 'google']:
                if GENAI_AVAILABLE:
                    print("🆓 Using Google Gemini (FREE) with SDK - retry & fallback enabled")
                else:
                    print("🆓 Using Google Gemini (FREE) with REST API")
            elif provider_type == 'groq':
                print("🆓 Using Groq (FREE and FAST)")
            elif provider_type == 'deepseek':
                print("🆓 Using DeepSeek (FREE)")
            elif provider_type == 'local':
                print("📝 Using Local Templates (FREE)")
            elif provider_type in ['openai', 'gpt', 'chatgpt']:
                print("💰 Using OpenAI (PAID)")
            elif provider_type in ['claude', 'anthropic']:
                print("💰 Using Claude (PAID)")

            return provider
        except Exception as e:
            print(f"⚠️  Failed to create provider {provider_type}: {e}")
            print("🔄 Falling back to local provider")
            return LocalLLMProvider()

    @staticmethod
    def list_providers() -> Dict[str, Dict[str, str]]:
        """List all available providers with their details"""
        return {
            "FREE": {
                "gemini": "Google Gemini - 60 req/min (Get key: https://makersuite.google.com/app/apikey)",
                "groq": "Groq - Fast and free (Get key: https://console.groq.com/keys)",
                "deepseek": "DeepSeek - Chinese LLM (Get key: https://platform.deepseek.com/)",
                "local": "Local Templates - No API needed"
            },
            "PAID": {
                "openai": "OpenAI GPT - Requires credits",
                "claude": "Anthropic Claude - Requires credits"
            }
        }


class LLMConfig:
    """Configuration manager for LLM providers"""

    def __init__(self, config_file: str = "llm_config.json"):
        self.config_file = config_file
        self.config = self._load_config()

    def _load_config(self) -> Dict:
        """Load configuration file"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"⚠️  Failed to load config: {e}")

        return {
            "provider": "local",
            "gemini": {
                "model": "gemini-2.5-flash",
                "api_key": ""
            },
            "groq": {
                "model": "mixtral-8x7b-32768",
                "api_key": ""
            },
            "deepseek": {
                "model": "deepseek-chat",
                "api_key": ""
            },
            "openai": {
                "model": "gpt-4o-mini",
                "api_key": ""
            },
            "claude": {
                "model": "claude-sonnet-4-20250514",
                "api_key": ""
            }
        }

    def save_config(self):
        """Save configuration file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            print(f"✅ Configuration saved to {self.config_file}")
        except Exception as e:
            print(f"⚠️  Failed to save config: {e}")

    def get_provider(self) -> LLMProvider:
        """Get configured LLM provider"""
        provider_type = self.config.get("provider", "local")

        kwargs = {}
        if provider_type in self.config:
            provider_config = self.config[provider_type]
            if "api_key" in provider_config:
                kwargs["api_key"] = provider_config["api_key"]
            if "model" in provider_config:
                kwargs["model"] = provider_config["model"]

        return LLMFactory.create_provider(provider_type, **kwargs)


if __name__ == '__main__':
    print("🧪 Testing LLM Provider Module\n")
    print("=" * 70)

    print("\n📋 Available Providers:")
    providers = LLMFactory.list_providers()

    print("\n🆓 FREE Providers:")
    for name, desc in providers["FREE"].items():
        print(f"   • {name}: {desc}")

    print("\n💰 PAID Providers:")
    for name, desc in providers["PAID"].items():
        print(f"   • {name}: {desc}")

    print("\n" + "=" * 70)
    print("\n1️⃣  Testing Local Provider:")
    local_provider = LocalLLMProvider()
    desc = local_provider.generate_scenario_description(
        "ADD", "0000", "Addition (A + B)", 16
    )
    print(f"   Scenario description: {desc}\n")

    feature_desc = local_provider.generate_feature_description(
        16, 12, ["ADD", "SUB", "AND", "OR", "XOR"]
    )
    print(f"   Feature description:\n{feature_desc}\n")

    print("2️⃣  Testing Configuration Manager:")
    config = LLMConfig()
    print(f"   Current provider: {config.config.get('provider')}")
    provider = config.get_provider()
    print(f"   Provider type: {type(provider).__name__}\n")

    # 如果有 Gemini API key，测试 Gemini
    if os.getenv("GEMINI_API_KEY"):
        print("3️⃣  Testing Gemini Provider:")
        try:
            gemini_provider = LLMFactory.create_provider("gemini")
            desc = gemini_provider.generate_scenario_description(
                "MUL", "1001", "Multiplication (A * B)", 16
            )
            print(f"   Scenario: {desc}\n")
        except Exception as e:
            print(f"   Failed: {e}\n")
    else:
        print("3️⃣  Gemini API Key not found. Set GEMINI_API_KEY to test.")

    # 如果有 OpenAI API key，测试 OpenAI
    if os.getenv("OPENAI_API_KEY"):
        print("4️⃣  Testing OpenAI Provider:")
        try:
            openai_provider = LLMFactory.create_provider("openai", model="gpt-4o-mini")
            desc = openai_provider.generate_scenario_description(
                "DIV", "1010", "Division (A / B)", 16
            )
            print(f"   Scenario: {desc}\n")
        except Exception as e:
            print(f"   Failed: {e}\n")
    else:
        print("4️⃣  OpenAI API Key not found. Set OPENAI_API_KEY to test.")

    print("=" * 70)
    print("✅ Testing completed!")
    print("\n💡 To use FREE providers:")
    print("   • Gemini: python script.py --llm-provider gemini --api-key YOUR_KEY")
    print("   • Groq:   python script.py --llm-provider groq --api-key YOUR_KEY")
    print("   • Local:  python script.py --llm-provider local (no key needed)")
    print("\n💡 To use OpenAI with SDK:")
    print("   • First install: pip install openai")
    print("   • Then run: python script.py --llm-provider openai --api-key YOUR_KEY")