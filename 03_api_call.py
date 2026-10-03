"""
阶段 0 · 项目 0.2：调用大模型 API
================================================

这个脚本会让你第一次亲手调用大模型 —— 也是你从「写 Python」跨到
「做 AI Agent」的那一步。

【运行方式】在项目目录里：
    source .venv/bin/activate
    python 03_api_call.py

运行前请先把 API Key 填进 .env 文件（同目录下）。

【这个脚本想让你明白的一件事】
很多人以为「调用大模型」是件神秘的事。其实它就是一个最普通的 HTTP 请求：
    把一段 JSON POST 出去 → 拿回一段 JSON
所有框架（LangChain、LlamaIndex、Agent SDK）底层做的都是这一件事。
这段代码不依赖任何框架，就是为了让你看清这层「底」。
"""

import json
import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

# ============================================================
# 配置
# ============================================================

# load_dotenv() 会读取同目录下的 .env 文件，把里面的键值对
# 加载成环境变量，之后用 os.getenv() 就能取到。
load_dotenv()

API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
API_URL = "https://api.deepseek.com/chat/completions"

# 价格用于估算成本。注意：各家价格会变，用之前去官网核对一下。
# 单位：元 / 每百万 token
PRICE_INPUT = 2.0     # 输入（cache miss）
PRICE_OUTPUT = 8.0    # 输出

PLACEHOLDER_HINT = "在这里粘贴你的Key"


# ============================================================
# 第 0 步：检查密钥
# ============================================================
def check_api_key():
    """确认 API Key 已经填好。没填就给出明确的指引，而不是让程序崩掉。"""
    if not API_KEY or PLACEHOLDER_HINT in API_KEY:
        print("=" * 60)
        print("还差一步：你还没有填入 API Key")
        print("=" * 60)
        print()
        print("请打开项目里的 .env 文件，找到这一行：")
        print()
        print("    DEEPSEEK_API_KEY=在这里粘贴你的Key")
        print()
        print("把等号后面替换成你自己的 Key（sk- 开头），保存后重新运行。")
        print()
        print("Key 从哪来：https://platform.deepseek.com/api_keys")
        print()
        return False

    if not API_KEY.startswith("sk-"):
        print(f"⚠️  警告：Key 看起来不太对（不是 sk- 开头），但还是先试一下。\n")

    # 只显示头尾，避免完整密钥出现在屏幕或日志里
    masked = API_KEY[:6] + "..." + API_KEY[-4:] if len(API_KEY) > 12 else "***"
    print(f"已读取 API Key：{masked}")
    print(f"使用模型：{MODEL}\n")
    return True


# ============================================================
# 第 1 步：发起一次最简单的调用
# ============================================================
def ask(question, system_prompt="你是一个简洁、准确的中文助手。", temperature=1.0, verbose=True):
    """
    调用模型回答一个问题，返回 (回答文本, 用量字典)。

    注意最后返回的是「两个值」—— 这是 Python 的多返回值写法（本质是元组），
    调用时要写成 content, usage = ask(...)
    """
    # ---- 1. 请求头：告诉服务器「我是谁」「我发的是什么格式」 ----
    headers = {
        "Authorization": f"Bearer {API_KEY}",     # 密钥就放在这里
        "Content-Type": "application/json",       # 我发的是 JSON
    }

    # ---- 2. 请求体：一个普通的字典 ----
    payload = {
        "model": MODEL,
        "messages": [
            # system 消息设定模型的身份和规则，它会影响后面所有的回复
            {"role": "system", "content": system_prompt},
            # user 消息是用户说的话
            {"role": "user", "content": question},
        ],
        # temperature 控制随机性：0 最稳定、接近照抄；1-2 更有创造性也更飘
        "temperature": temperature,
        "stream": False,          # 先不用流式，一次性拿完整回答
    }

    if verbose:
        print(f"→ 正在请求 {MODEL} ...")

    # ---- 3. 发出去 ----
    # timeout 一定要设。不设的话网络卡住时程序会永远挂着，
    # 这是新手写 API 调用最常见的疏忽。
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
    except requests.Timeout:
        print("✘ 请求超时。可能是网络问题，稍后重试。")
        return None, None
    except requests.ConnectionError as error:
        print(f"✘ 连接失败：{error}")
        return None, None

    # ---- 4. 先看状态码，再解析内容 ----
    # 200 = 成功。其他都不是好事，而且不同状态码含义不同。
    if response.status_code != 200:
        print(f"✘ 请求失败，HTTP {response.status_code}")
        if response.status_code == 401:
            print("   原因：密钥无效或已过期。去 platform.deepseek.com 重新生成一个。")
        elif response.status_code == 402:
            print("   原因：账户余额不足。去 platform.deepseek.com 充值。")
        elif response.status_code == 429:
            print("   原因：请求太频繁，触发了限流。等几秒再试。")
        print(f"   原始返回：{response.text[:300]}")
        return None, None

    # ---- 5. 解析返回的 JSON ----
    data = response.json()

    # 这个嵌套结构很重要，记住它：
    # data["choices"] 是候选回答的列表（通常是 1 个）
    #   → [0] 取第一个
    #     → ["message"] 是这条消息
    #       → ["content"] 才是真正的文本
    content = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})

    return content, usage


# ============================================================
# 第 2 步：看看「请求」和「响应」到底长什么样
# ============================================================
def show_raw_structures():
    """
    把发出去的 JSON 和收回来的 JSON 都打印出来。
    这是整个脚本里最值得细看的一段 —— 看懂了这个，就没有黑箱了。
    """
    print("=" * 60)
    print("原始请求 / 响应结构")
    print("=" * 60)

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "你是一个简洁的中文助手。"},
            {"role": "user", "content": "用一句话解释什么是 token。"},
        ],
        "temperature": 1.0,
        "stream": False,
    }

    print("\n【我们发出去的 JSON】")
    print(json.dumps(payload, ensure_ascii=False, indent=2))

    response = requests.post(
        API_URL,
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        json=payload,
        timeout=60,
    )
    data = response.json()

    print("\n【服务器返回的 JSON】")
    print(json.dumps(data, ensure_ascii=False, indent=2)[:1200])

    print("\n【关键路径】")
    print(f"  data['choices'][0]['message']['content']  →  回答文本")
    print(f"  data['usage']                             →  {data.get('usage')}")
    print()


# ============================================================
# 第 3 步：多轮对话 —— 理解 messages 数组
# ============================================================
def multi_turn_demo():
    """
    模型本身没有记忆。所谓「多轮对话」，是你在每次请求时
    把之前的所有消息一起发过去 —— 它只是把整段历史重新读一遍。

    这一点非常关键：后面学 Agent 时会遇到「上下文越来越长、
    越来越贵」的问题，根源就在这里。
    """
    print("=" * 60)
    print("多轮对话：模型没有记忆，历史是你自己带上的")
    print("=" * 60)

    history = [
        {"role": "system", "content": "你是一个简洁的中文助手，回答不超过两句话。"}
    ]

    questions = [
        "我叫 sherry，正在学 AI Agent 开发。",
        "我刚才说我在学什么？",          # 测试它能否记住上一轮
    ]

    for question in questions:
        history.append({"role": "user", "content": question})
        print(f"\n👤 我：{question}")

        # 这里手动发一次请求，因为我们要用自己维护的 history
        response = requests.post(
            API_URL,
            headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
            json={"model": MODEL, "messages": history, "temperature": 0.7},
            timeout=60,
        )
        if response.status_code != 200:
            print(f"✘ HTTP {response.status_code}: {response.text[:200]}")
            return
        data = response.json()
        answer = data["choices"][0]["message"]["content"]
        print(f"🤖 AI：{answer}")

        # 关键：把模型的回答也追加进历史，否则下一轮它就"不记得"自己说过什么
        history.append({"role": "assistant", "content": answer})

    print(f"\n当前对话历史共 {len(history)} 条消息")
    print("→ 每一轮都要把全部历史重新发一遍，所以轮次越多，token 消耗越大\n")


# ============================================================
# 第 4 步：流式输出 —— 让文字一个字一个字蹦出来
# ============================================================
def stream_demo(question="用三句话介绍你自己。"):
    """
    非流式：等模型全部生成完，一次性返回（感觉卡顿、慢）
    流式：  模型生成一个字就立刻推给你（体验好得多）

    原理：服务器持续推来一行行 data: {...}，直到最后收到 data: [DONE]
    """
    print("=" * 60)
    print("流式输出")
    print("=" * 60)
    print(f"\n👤 我：{question}")
    print("🤖 AI：", end="", flush=True)

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "你是一个简洁、准确的中文助手。"},
            {"role": "user", "content": question},
        ],
        "stream": True,          # ← 只有这一处不同
    }

    response = requests.post(
        API_URL,
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        json=payload,
        stream=True,             # ← requests 这边也要开
        timeout=120,
    )

    if response.status_code != 200:
        print(f"\n✘ HTTP {response.status_code}: {response.text[:200]}")
        return

    for raw_line in response.iter_lines():
        if not raw_line:
            continue
        line = raw_line.decode("utf-8")
        if not line.startswith("data: "):
            continue

        payload_str = line[len("data: "):]
        if payload_str == "[DONE]":
            break

        chunk = json.loads(payload_str)
        delta = chunk["choices"][0]["delta"].get("content", "")
        print(delta, end="", flush=True)

    print("\n")


# ============================================================
# 主流程
# ============================================================
def main():
    if not check_api_key():
        return

    # 第 1 步：一次最简单的调用
    print("=" * 60)
    print("第 1 步：最简单的调用")
    print("=" * 60)

    question = "用两句话说明什么是大模型的 API。"
    print(f"\n👤 我：{question}\n")

    answer, usage = ask(question)

    if answer is None:
        print("\n调用失败，后面的步骤跳过。")
        return

    print(f"🤖 AI：{answer}\n")

    # 用量与成本 —— 从第一天就养成看数字的习惯
    if usage:
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total = usage.get("total_tokens", 0)
        cost = prompt_tokens / 1_000_000 * PRICE_INPUT + completion_tokens / 1_000_000 * PRICE_OUTPUT
        print("【本次调用消耗】")
        print(f"  输入 token：{prompt_tokens}")
        print(f"  输出 token：{completion_tokens}")
        print(f"  合计      ：{total}")
        print(f"  估算成本  ：约 {cost:.6f} 元")
        print("  （价格会变，记得去官网核对）")
        print()

    input("按回车继续看「原始请求/响应结构」...")

    # 第 2 步
    show_raw_structures()
    input("按回车继续看「多轮对话」...")

    # 第 3 步
    multi_turn_demo()
    input("按回车继续看「流式输出」...")

    # 第 4 步
    stream_demo()

    print("=" * 60)
    print("全部完成")
    print("=" * 60)
    print("\n你已经亲手调用过大模型了。回头看一遍这个脚本，")
    print("你会发现所谓「AI 应用开发」，起点就是这么一个 HTTP 请求。")


if __name__ == "__main__":
    main()
