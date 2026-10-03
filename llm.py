"""
llm.py —— 大模型调用的可复用模块
================================================

把「怎么发请求」这件事收进一个文件。之后所有脚本只要 import 它，
不用再复制粘贴那一堆 headers / payload / 错误处理。

用法：
    from llm import ask, chat, chat_stream, Conversation

    text, usage = ask("你好")
    print(text)

【为什么值得单独抽一个模块】
上一课的 03_api_call.py 里，请求逻辑散在四个函数里，每个都重复了一遍
headers 和错误处理。接下来你要写 Prompt 实验、知识库、Agent ——
如果每次都复制粘贴，改一处（比如换个模型、加个重试）就要改十处。

「重复三次就抽出来」是工程上的老规矩。
"""

import json
import os
from pathlib import Path

import requests
from dotenv import load_dotenv

# 用 __file__ 定位 .env，而不是依赖「当前工作目录」。
# 这样无论你在哪个目录下运行脚本，都能正确找到配置。
ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(ENV_PATH)

API_URL = "https://api.deepseek.com/chat/completions"
DEFAULT_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

# 价格（元 / 每百万 token）。会变，用之前去官网核对。
PRICE_INPUT = 2.0
PRICE_OUTPUT = 8.0

# 各状态码对应的原因，报错时直接查表
HTTP_ERROR_TIPS = {
    400: "请求格式不对",
    401: "密钥无效或已过期，去 platform.deepseek.com 重新生成",
    402: "账户余额不足，需要充值",
    422: "请求参数有误",
    429: "请求过于频繁，被限流，等几秒再试",
    500: "服务端内部错误，稍后重试",
    503: "服务暂时不可用，稍后重试",
}


class LLMError(Exception):
    """调用大模型失败时抛出。

    为什么要自定义异常类型？因为调用方可以精确地捕获「我知道怎么处理的错误」，
    而不是把整个程序包在一层宽泛的 except Exception 里，那样会把真正的 bug 也吞掉。
    """


def _get_api_key():
    key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not key or "在这里粘贴" in key:
        raise LLMError(
            "没有读取到 API Key。请在同目录的 .env 文件里设置：\n"
            "    DEEPSEEK_API_KEY=sk-你的key"
        )
    return key


def estimate_cost(usage):
    """根据 usage 估算本次花费（元）。"""
    if not usage:
        return 0.0
    return (
        usage.get("prompt_tokens", 0) / 1_000_000 * PRICE_INPUT
        + usage.get("completion_tokens", 0) / 1_000_000 * PRICE_OUTPUT
    )


def chat(messages, model=None, temperature=1.0, timeout=60, extra=None):
    """
    最底层函数：发一次请求，返回 (回答文本, usage 字典)。

    messages 是消息列表，每一项形如 {"role": "user", "content": "..."}
    role 有三种：
        system    —— 设定模型的角色和行为规则，影响后面所有回复
        user      —— 用户说的话
        assistant —— 模型之前说过的话（多轮对话时要把历史一起带上）

    extra 是一个「逃生舱口」：需要传一些本模块没封装的参数时用它，例如
        chat(msgs, extra={"response_format": {"type": "json_object"}})
    这样可以保证模型输出的一定是合法 JSON。
    """
    payload = {
        "model": model or DEFAULT_MODEL,
        "messages": messages,
        "temperature": temperature,
        "stream": False,
    }
    if extra:
        payload.update(extra)

    try:
        response = requests.post(
            API_URL,
            headers={
                "Authorization": f"Bearer {_get_api_key()}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=timeout,
        )
    except requests.Timeout as error:
        raise LLMError(f"请求超时（超过 {timeout} 秒）") from error
    except requests.ConnectionError as error:
        raise LLMError(f"网络连接失败：{error}") from error

    if response.status_code != 200:
        reason = HTTP_ERROR_TIPS.get(response.status_code, response.text[:200])
        raise LLMError(f"HTTP {response.status_code}：{reason}")

    data = response.json()
    content = data["choices"][0]["message"]["content"]
    return content, data.get("usage", {})


def ask(question, system=None, model=None, temperature=1.0, timeout=60):
    """
    单轮问答的快捷方式。等价于 chat()，只是省去自己拼 messages。

    system 参数传 None 时，不会发送 system 消息 —— 这正好可以用来
    对比「有角色设定」和「没有角色设定」的输出差异。
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": question})
    return chat(messages, model=model, temperature=temperature, timeout=timeout)


def chat_stream(messages, model=None, temperature=1.0, timeout=120):
    """
    流式版本：逐段产出文本。这是一个「生成器」——用 for 循环取，
    每拿到一小块就打印，所以文字会一个字一个字冒出来。

        for piece in chat_stream([{"role": "user", "content": "你好"}]):
            print(piece, end="", flush=True)
    """
    payload = {
        "model": model or DEFAULT_MODEL,
        "messages": messages,
        "temperature": temperature,
        "stream": True,
    }

    response = requests.post(
        API_URL,
        headers={
            "Authorization": f"Bearer {_get_api_key()}",
            "Content-Type": "application/json",
        },
        json=payload,
        stream=True,
        timeout=timeout,
    )

    if response.status_code != 200:
        reason = HTTP_ERROR_TIPS.get(response.status_code, response.text[:200])
        raise LLMError(f"HTTP {response.status_code}：{reason}")

    for raw_line in response.iter_lines():
        if not raw_line:
            continue
        line = raw_line.decode("utf-8")
        if not line.startswith("data: "):
            continue
        body = line[len("data: "):]
        if body == "[DONE]":
            break
        delta = json.loads(body)["choices"][0]["delta"].get("content", "")
        if delta:
            yield delta


class Conversation:
    """
    带记忆的对话。内部自己维护历史，每次自动把全部历史发出去。

        conv = Conversation(system="你是一个简洁的中文助手。")
        print(conv.say("我叫 sherry"))
        print(conv.say("我叫什么？"))     # 它会答对

    total_tokens 会累计所有轮次的消耗 —— 用来观察「越聊越贵」。
    """

    def __init__(self, system=None, model=None, temperature=1.0):
        self.messages = []
        self.model = model
        self.temperature = temperature
        self.total_tokens = 0
        self.total_cost = 0.0
        if system:
            self.messages.append({"role": "system", "content": system})

    def say(self, text):
        """说一句话，返回模型的回复。"""
        self.messages.append({"role": "user", "content": text})
        reply, usage = chat(
            self.messages,
            model=self.model,
            temperature=self.temperature,
        )
        self.messages.append({"role": "assistant", "content": reply})
        self.total_tokens += usage.get("total_tokens", 0)
        self.total_cost += estimate_cost(usage)
        return reply

    @property
    def message_count(self):
        return len(self.messages)

    def reset(self):
        """清空历史，但保留 system 消息。"""
        system = [m for m in self.messages if m["role"] == "system"]
        self.messages = system
        self.total_tokens = 0
        self.total_cost = 0.0


# 直接运行这个文件时，做一次自检，确认配置和网络都正常
if __name__ == "__main__":
    print("llm 模块自检")
    print("-" * 40)
    try:
        _get_api_key()
        print("✔ API Key 已读取")
    except LLMError as error:
        print(f"✘ {error}")
        raise SystemExit(1)

    print(f"✔ 使用模型：{DEFAULT_MODEL}")
    text, usage = ask("只回答两个字：正常", temperature=0)
    print(f"✔ 调用成功，模型回复：{text.strip()}")
    print(f"✔ 本次消耗 {usage.get('total_tokens')} token，约 {estimate_cost(usage):.6f} 元")
    print("\n模块可用。其他脚本可以 from llm import ask 了。")
