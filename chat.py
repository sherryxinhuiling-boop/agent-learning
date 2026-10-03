#!/usr/bin/env python3
"""
阶段 1 · 里程碑项目 1：个人命令行 AI 助手
================================================

这是你的第一个「产品」—— 不是一个演示脚本，而是一个你会真的想用的工具。

【运行方式】
    source .venv/bin/activate
    python chat.py

【它做了什么】
    · 连续对话 —— 它记得你之前说过什么
    · 流式输出 —— 文字一个字一个字冒出来，不用干等
    · 会话管理 —— 随时清空、保存成文件
    · 成本可见 —— 每一轮都告诉你花了多少 token、多少钱

【它用的东西，全是你前面几课学的】
    llm.Conversation    → 第 1 课抽出的模块
    say_stream()        → 第 2 课学的流式输出
    .env                → 阶段 0 学的密钥管理

没有新框架、没有魔法。你现在拥有的东西，已经足够做出一个能用的工具了。

【命令】
    /help    显示帮助
    /clear   清空对话历史（开始新话题）
    /save    把这次对话保存成 markdown 文件
    /stats   查看累计用量和花费
    /exit    退出（或者按 Ctrl+C / Ctrl+D）
"""

import sys
from datetime import datetime
from pathlib import Path

import llm

# ============================================================
# 配置
# ============================================================

# 助手的「人设」。改这里就能改变它的风格 —— 这是你第一个可调的产品参数。
DEFAULT_SYSTEM = """你是一个简洁、准确的中文助手。
回答时直接给结论，不要客套话，不要重复我的问题。
如果不确定，就说不确定，不要编造。"""

TEMPERATURE = 0.7
SAVE_DIR = Path(__file__).resolve().parent / "chats"

HELP_TEXT = """
可用命令：
  /help    显示这份帮助
  /clear   清空对话历史，开始新话题
  /save    把本次对话保存成 markdown 文件
  /stats   查看累计 token 与花费
  /exit    退出

其他输入都会作为问题发给模型。
按 Ctrl+C 随时中断当前回答，按 Ctrl+D 退出。
"""


# ============================================================
# 助手本体
# ============================================================
class ChatApp:
    def __init__(self, system=DEFAULT_SYSTEM, model=None):
        self.conv = llm.Conversation(
            system=system,
            model=model,
            temperature=TEMPERATURE,
        )
        self.started_at = datetime.now()
        self.rounds = 0
        self.model = model or llm.DEFAULT_MODEL

    # ---------- 界面 ----------
    def banner(self):
        print()
        print("=" * 60)
        print("  个人 AI 助手")
        print("=" * 60)
        print(f"  模型：{self.model}")
        print(f"  输入 /help 查看命令，/exit 退出")
        print()

    def print_result(self):
        """每轮结束后打印本轮的用量。养成看数字的习惯。"""
        usage = getattr(self.conv, "last_usage", None)
        if not usage:
            return
        cost = llm.estimate_cost(usage)
        print(f"  · {usage.get('total_tokens', 0)} token"
              f"（输入 {usage.get('prompt_tokens', 0)} / "
              f"输出 {usage.get('completion_tokens', 0)}）"
              f"  约 {cost:.6f} 元")

    # ---------- 命令 ----------
    def cmd_help(self):
        print(HELP_TEXT)

    def cmd_clear(self):
        if self.rounds == 0:
            print("  当前没有对话记录。")
            return
        self.conv.reset()
        self.rounds = 0
        print("  ✔ 对话历史已清空，可以开始新话题了。")

    def cmd_stats(self):
        elapsed = (datetime.now() - self.started_at).seconds
        minutes, seconds = divmod(elapsed, 60)
        print()
        print(f"  对话轮数   ：{self.rounds}")
        print(f"  历史消息数 ：{self.conv.message_count} 条（含 system）")
        print(f"  累计 token ：{self.conv.total_tokens}")
        print(f"  累计花费   ：约 {self.conv.total_cost:.6f} 元")
        print(f"  已用时间   ：{minutes} 分 {seconds} 秒")
        if self.rounds:
            avg = self.conv.total_cost / self.rounds
            print(f"  平均每轮   ：约 {avg:.6f} 元")
        print()
        print("  提示：轮数越多，历史越长，每轮消耗会持续上升。")
        print()

    def cmd_save(self):
        history = self.conv.history_without_system()
        if not history:
            print("  没有可保存的内容。")
            return

        SAVE_DIR.mkdir(exist_ok=True)
        stamp = self.started_at.strftime("%Y-%m-%d_%H%M")
        path = SAVE_DIR / f"chat_{stamp}.md"

        lines = [
            f"# 对话记录 · {self.started_at.strftime('%Y-%m-%d %H:%M')}",
            "",
            f"- 模型：`{self.model}`",
            f"- 轮数：{self.rounds}",
            f"- 累计消耗：{self.conv.total_tokens} token"
            f"（约 {self.conv.total_cost:.6f} 元）",
            "",
            "---",
            "",
        ]
        for message in history:
            who = "我" if message["role"] == "user" else "助手"
            lines.append(f"**{who}**")
            lines.append("")
            lines.append(message["content"])
            lines.append("")

        path.write_text("\n".join(lines), encoding="utf-8")
        print(f"  ✔ 已保存到 {path}")
        print(f"    （相对项目目录的路径：{path.relative_to(Path(__file__).resolve().parent)}）")

    # ---------- 主循环 ----------
    def handle(self, line):
        """处理一行输入。返回 False 表示要退出。"""
        if line in ("/exit", "/quit"):
            return False
        if line == "/help":
            self.cmd_help()
            return True
        if line == "/clear":
            self.cmd_clear()
            return True
        if line == "/stats":
            self.cmd_stats()
            return True
        if line == "/save":
            self.cmd_save()
            return True
        if line.startswith("/"):
            print(f"  未知命令：{line}。输入 /help 查看可用命令。")
            return True

        # ---- 正常提问 ----
        print("助手 > ", end="", flush=True)
        try:
            for piece in self.conv.say_stream(line):
                print(piece, end="", flush=True)
            print()
            self.rounds += 1
            self.print_result()
        except llm.LLMError as error:
            print(f"\n  ✘ 调用失败：{error}")
            if self.conv.drop_last_user_message():
                print("  （已撤回这条提问，历史保持完整）")
        except KeyboardInterrupt:
            # 用户按了 Ctrl+C：中断本次回答。
            # 这里的处理很关键 —— 不能把半截回答留在历史里，
            # 否则下一轮模型会看到一条断掉的回复，行为会变奇怪。
            print("\n  （已中断本次回答）")
            if self.conv.drop_last_user_message():
                print("  （这条提问也已撤回）")

        return True

    def run(self):
        self.banner()
        while True:
            try:
                line = input("你   > ").strip()
            except KeyboardInterrupt:
                print("\n  （按 Ctrl+C 退出）")
                break
            except EOFError:
                print()
                break

            if not line:
                continue

            if not self.handle(line):
                break

        # 退出时给个小结
        if self.rounds:
            print()
            print(f"本次会话：{self.rounds} 轮，"
                  f"{self.conv.total_tokens} token，"
                  f"约 {self.conv.total_cost:.6f} 元")
            print("想保存的话，下次会话结束前输入 /save。")
        print("再见。")


def main():
    # 允许从命令行覆盖模型，例如：python chat.py deepseek-reasoner
    model = sys.argv[1] if len(sys.argv) > 1 else None

    try:
        llm._get_api_key()
    except llm.LLMError as error:
        print(f"无法启动：{error}")
        return 1

    app = ChatApp(model=model)
    try:
        app.run()
    except KeyboardInterrupt:
        print("\n再见。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
