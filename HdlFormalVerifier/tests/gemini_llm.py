"""
Simple Gemini LLM tester with retry & fallback.

前提：
1. 已安装： pip install -U google-genai
2. 已设置环境变量：GEMINI_API_KEY
"""

import time
from google import genai
from google.genai import errors


def generate_with_gemini(
    prompt: str,
    max_retries: int = 3,
    sleep_seconds: float = 2.0,
) -> str:
    """
    调用 Gemini，带重试 + 模型降级：
    - 先尝试 gemini-2.5-flash
    - 如果 503（模型过载），自动降级到 gemini-1.5-flash
    - 多次失败后抛出异常
    """
    # client 会自动从环境变量 GEMINI_API_KEY 读取 key
    client = genai.Client()

    # 优先尝试的模型列表：先 2.5，再 1.5
    models_to_try = ["gemini-2.5-flash", "gemini-1.5-flash"]

    last_error = None

    for model_name in models_to_try:
        print(f"\n🔎 Trying model: {model_name}")
        for attempt in range(1, max_retries + 1):
            try:
                print(f"   ▶ Attempt {attempt}/{max_retries} ...")
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                )
                # 如果成功，直接返回文本
                print("   ✅ Request succeeded.")
                return response.text

            except errors.ServerError as e:
                # 处理 5xx 错误（包括你遇到的 503 overloaded）
                last_error = e
                print(f"   ⚠️ ServerError (Gemini overloaded?): {e}")
                if attempt < max_retries:
                    print(f"   ⏳ Waiting {sleep_seconds} seconds before retry ...")
                    time.sleep(sleep_seconds)
                else:
                    print("   ❌ Max retries reached for this model.")

            except Exception as e:
                # 其他错误（认证问题、参数错误等）
                last_error = e
                print(f"   ❌ Non-server error with model {model_name}: {e}")
                # 这类错误通常没必要重试，直接换下一个模型
                break

        print(f"   ⚠️ Model {model_name} failed, switching to next candidate...")

    # 如果所有模型都失败了，抛出最后一个错误
    raise RuntimeError(f"All Gemini model attempts failed. Last error: {last_error}")


if __name__ == "__main__":
    test_prompt = "Explain how AI works in a few words"

    try:
        print("🚀 Calling Gemini with retry & fallback...")
        text = generate_with_gemini(test_prompt)
        print("\n✅ Final Gemini response:\n")
        print(text)

    except Exception as e:
        print("\n💥 Gemini call failed completely:")
        print(e)
