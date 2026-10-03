"""
阶段 1 · 第 2 课：让模型输出可靠的数据
================================================

上一课我们看到：模型的输出「格式对了」不等于「内容对了」。
这一课要把它做成一个能真正用的函数 —— 也就是工程上的三层防护。

    ┌─────────────────────────────────────────────┐
    │  第 1 层  Prompt     把要求说清楚           │  提高成功率
    │  第 2 层  机制       response_format        │  保证格式合法
    │  第 3 层  代码       校验 + 反馈重试         │  保证内容正确
    └─────────────────────────────────────────────┘

只有第 3 层是「确定性」的。前两层都是概率，唯独代码能给你保证。

【这一课真正的重点不是写函数】
而是最后那部分：**失败了该怎么办**。
这是区分「能跑的 demo」和「敢上线的系统」的地方。

【运行方式】
    source .venv/bin/activate
    python 05_reliable_extract.py
"""

import json

import llm

# ============================================================
# 第 1 层：Prompt —— 把要求说清楚
# ============================================================

SYSTEM_PROMPT = "你是一个信息抽取助手。你只输出 JSON，不输出任何其他文字。"

# 字段定义单独抽出来，因为「校验」和「Prompt」需要共用同一份约定。
# 如果这两处各写一份，改了一处忘了另一处，就会出现
# 「Prompt 要 A 字段、校验查 B 字段」的诡异 bug。
FIELD_SPEC = {
    "product":  ("商品名称", "字符串"),
    "rating":   ("用户满意度评分，1-5 的整数", "整数(1-5)"),
    "pros":     ("优点列表，没有则为空数组", "字符串数组"),
    "issues":   ("问题列表，没有则为空数组", "字符串数组"),
    "recommend": ("是否愿意推荐给朋友", "布尔值 true/false"),
    "sentiment": ("整体情绪，只能是 positive / neutral / negative", "枚举字符串"),
}

ALLOWED_SENTIMENTS = {"positive", "neutral", "negative"}


def build_extract_prompt(text):
    lines = ["从下面的用户评价中抽取信息。", "", "必须输出以下字段："]
    for name, (desc, typ) in FIELD_SPEC.items():
        lines.append(f"- {name}: {desc}（类型：{typ}）")
    lines += [
        "",
        "要求：",
        "1. 只输出 JSON 对象本身，不要解释文字",
        "2. 评价中没有提到的信息用 null，不要编造",
        "3. 严格使用上面的字段名，不要新增、删除或改名",
        "4. rating 必须是 JSON 数字，recommend 必须是 JSON 布尔值，不要写成字符串",
        "",
        "评价：",
        text,
    ]
    return "\n".join(lines)


# ============================================================
# 第 3 层之一：校验 —— 判断「出来的是不是我们想要的」
# ============================================================

def validate(data):
    """
    检查数据是否符合约定。返回 (是否通过, 问题列表)。

    问题列表要写清楚「哪里错了、应该是什么」，
    因为这个列表会原样发给模型让它自己修 ——
    反馈越具体，模型修对的概率越高。
    """
    problems = []

    if not isinstance(data, dict):
        return False, [f"顶层应该是 JSON 对象，实际是 {type(data).__name__}"]

    # 字段是否齐全、类型是否正确
    expected_types = {
        "product": str,
        "rating": int,
        "pros": list,
        "issues": list,
        "recommend": bool,
        "sentiment": str,
    }
    for key, expected in expected_types.items():
        if key not in data:
            problems.append(f"缺少字段 `{key}`")
            continue
        value = data[key]
        if value is None:
            continue            # 允许 null，表示「评价里没提到」
        if expected is int and isinstance(value, bool):
            problems.append(f"`{key}` 应该是整数，实际是布尔值 {value}")
        elif expected is bool and not isinstance(value, bool):
            problems.append(f"`{key}` 应该是布尔值 true/false，实际是 {type(value).__name__}：{value!r}")
        elif not isinstance(value, expected):
            problems.append(f"`{key}` 应该是 {expected.__name__}，实际是 {type(value).__name__}")

    # 有没有多出不该有的字段
    for key in data:
        if key not in FIELD_SPEC:
            problems.append(f"多出未定义的字段 `{key}`，请删除")

    # 取值范围
    rating = data.get("rating")
    if isinstance(rating, int) and not isinstance(rating, bool):
        if not 1 <= rating <= 5:
            problems.append(f"`rating` 必须在 1-5 之间，实际是 {rating}")

    sentiment = data.get("sentiment")
    if isinstance(sentiment, str) and sentiment not in ALLOWED_SENTIMENTS:
        problems.append(
            f"`sentiment` 只能是 positive / neutral / negative 之一，实际是 {sentiment!r}"
        )

    # 数组元素类型
    for key in ("pros", "issues"):
        value = data.get(key)
        if isinstance(value, list) and not all(isinstance(x, str) for x in value):
            problems.append(f"`{key}` 数组里只能放字符串")

    return len(problems) == 0, problems


def build_repair_prompt(problems):
    """把校验发现的问题反馈给模型，让它自己改。"""
    bullets = "\n".join(f"- {p}" for p in problems)
    return f"""你上一次的输出不符合要求，存在以下问题：

{bullets}

请重新输出修正后的 JSON 对象。只输出 JSON 本身，不要任何解释。"""


# ============================================================
# 第 3 层之二：带反馈重试的抽取函数
# ============================================================

def extract(text, max_retries=2, verbose=False, prompt_builder=None):
    """
    从一个文本里抽取结构化信息，带自动校验和反馈重试。

    prompt_builder 用于指定怎么构造提问。默认用上面那个规范版本；
    传一个别的进去，就能对比「弱 Prompt」会多花多少次重试。

    返回一个字典：
        ok        是否成功
        data      成功时的数据（失败时为 None）
        attempts  实际尝试了几次
        tokens    总 token 消耗
        cost      总花费
        problems  失败时的问题列表
        log       每次尝试的简要记录（便于排查）

    注意 max_retries 的含义：它是「重试次数」，所以最坏情况会调用
    1 + max_retries 次。设太大可能烧钱，设 0 就没有容错。
    生产环境里通常 1-2 次就够了 —— 如果两次都修不对，
    再试十次也不会对，那说明是 Prompt 或任务本身有问题。
    """
    builder = prompt_builder or build_extract_prompt
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": builder(text)},
    ]

    tokens = 0
    cost = 0.0
    log = []
    problems = []

    for attempt in range(1, max_retries + 2):
        raw, usage = llm.chat(
            messages,
            temperature=0,                                        # 抽取任务用 0
            extra={"response_format": {"type": "json_object"}},    # 第 2 层防护
        )
        tokens += usage.get("total_tokens", 0)
        cost += llm.estimate_cost(usage)

        # --- 先检查格式 ---
        try:
            data = json.loads(raw)
            problems = []
        except json.JSONDecodeError as error:
            data = None
            problems = [f"输出不是合法 JSON：{error.msg}"]

        # --- 再检查内容 ---
        if data is not None:
            ok, problems = validate(data)
            if ok:
                log.append(f"第 {attempt} 次尝试：通过")
                return {
                    "ok": True,
                    "data": data,
                    "attempts": attempt,
                    "tokens": tokens,
                    "cost": cost,
                    "problems": [],
                    "log": log,
                }
            log.append(f"第 {attempt} 次尝试：JSON 合法但校验未通过（{len(problems)} 项问题）")
        else:
            log.append(f"第 {attempt} 次尝试：输出不是合法 JSON")

        if verbose:
            print(f"    ↳ 第 {attempt} 次未通过，问题共 {len(problems)} 项")
            for problem in problems[:2]:
                print(f"       · {problem}")

        # --- 把失败反馈回去，准备下一次 ---
        messages.append({"role": "assistant", "content": raw})
        messages.append({"role": "user", "content": build_repair_prompt(problems)})

    # 重试用完了还是不行 —— 明确失败，不返回半成品
    log.append(f"重试 {max_retries} 次后仍失败，放弃")
    return {
        "ok": False,
        "data": None,
        "attempts": max_retries + 1,
        "tokens": tokens,
        "cost": cost,
        "problems": problems,
        "log": log,
    }


# ============================================================
# 测试样本 —— 这就是最原始的「评测集」
# ============================================================
# 注意：不要只用「典型」样本。真正会暴露问题的是边界情况。
# 这一组里我特意放了几条刁钻的：极短的、纯情绪的、什么都没说的。
#
# 这正是阶段 2 要系统做的事 —— 现在先手工攒几条，感受一下。

TEST_CASES = [
    {
        "note": "标准：有赞有踩",
        "text": "上周买了降噪耳机，等了四天才到货，有点慢。不过音质是真的好，"
                "戴一整天耳朵也不疼。包装盒被压扁了一角，客服二话不说就补发了。"
                "整体挺满意，会推荐给朋友，但物流希望改进。",
    },
    {
        "note": "纯赞美",
        "text": "这个保温杯太好用了，保温效果一流，早上装的开水到晚上还是烫的，"
                "做工也扎实，已经推荐给三个同事了。",
    },
    {
        "note": "纯吐槽",
        "text": "垃圾东西，用了三天就坏了，联系客服推来推去，最后说超出保修期。"
                "再也不会买了。",
    },
    {
        "note": "极短，信息极少",
        "text": "还行吧。",
    },
    {
        "note": "没有商品名，只有情绪",
        "text": "太差了，非常失望，浪费时间。",
    },
    {
        "note": "一句话里提到两个商品",
        "text": "买了个键盘手感不错，但是配套的鼠标太轻了，用着不习惯。",
    },
]


# ============================================================
# 批量运行 + 统计
# ============================================================

def run_batch(cases):
    print("=" * 68)
    print("批量抽取")
    print("=" * 68)
    print()

    results = []
    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] {case['note']}")
        print(f"    原文：{case['text'][:52]}{'...' if len(case['text']) > 52 else ''}")

        result = extract(case["text"], verbose=True)
        results.append({**case, **result})

        if result["ok"]:
            data = result["data"]
            print(f"    ✔ 通过（第 {result['attempts']} 次尝试）")
            print(f"      product={data.get('product')!r}  rating={data.get('rating')!r}")
            print(f"      recommend={data.get('recommend')!r}  sentiment={data.get('sentiment')!r}")
        else:
            print(f"    ✘ 失败（尝试 {result['attempts']} 次）")
            for problem in result["problems"][:3]:
                print(f"      · {problem}")
        print()

    return results


def print_summary(results):
    print("=" * 68)
    print("统计")
    print("=" * 68)
    print()

    total = len(results)
    passed = sum(1 for r in results if r["ok"])
    first_try = sum(1 for r in results if r["ok"] and r["attempts"] == 1)
    retried = sum(1 for r in results if r["ok"] and r["attempts"] > 1)
    failed = total - passed
    total_cost = sum(r["cost"] for r in results)
    total_tokens = sum(r["tokens"] for r in results)

    print(f"  样本总数   ：{total}")
    print(f"  成功       ：{passed}   （成功率 {passed / total:.0%}）")
    print(f"    ├ 一次通过：{first_try}")
    print(f"    └ 重试通过：{retried}")
    if retried == 0:
        print("                （本次规范 Prompt 表现很好，没触发重试 ——")
        print("                  但这不代表重试机制多余，见下面的演示 1）")
    else:
        print("                ← 这几个如果没有重试机制，就会变成失败样本")
    print(f"  失败       ：{failed}")
    print()
    print(f"  总 token   ：{total_tokens}")
    print(f"  总花费     ：约 {total_cost:.6f} 元")
    print(f"  单条均摊   ：约 {total_cost / total:.6f} 元")
    print()

    if failed:
        print("  失败的样本（这些必须人工看一眼，不能就这么算了）：")
        for item in results:
            if not item["ok"]:
                print(f"    · {item['note']}")
                print(f"      原文：{item['text'][:40]}...")
                for problem in item["problems"][:2]:
                    print(f"      问题：{problem}")
        print()


# ============================================================
# 演示 1：重试机制到底救回了什么
# ============================================================

def weak_prompt(text):
    """
    故意用「弱 Prompt」：只说要用 JSON，不说字段和约束。
    这正是上一课 V2 的写法 —— 它会让字段名变成中文、类型也乱掉。
    """
    return f"提取这段评价里的信息，用 JSON 格式输出。\n\n{text}"


def demo_retry_mechanism():
    """
    用弱 Prompt + 严格校验，强制触发重试，让你看见反馈重试怎么工作。

    这个场景在真实项目里很常见：你手上有校验规则，
    但 Prompt 没写清楚 —— 于是第一批输出全都不合格。
    没有重试机制，这些就全变成失败样本。
    """
    print("=" * 68)
    print("演示 1：反馈重试机制在做什么")
    print("=" * 68)
    print()
    print("故意用一个「弱 Prompt」（只说输出 JSON，不说字段名和类型），")
    print("但校验规则仍然是严格的。看会发生什么：")
    print()

    text = "上周买了降噪耳机，音质很好，就是物流有点慢。"

    result = extract(text, max_retries=2, verbose=True, prompt_builder=weak_prompt)

    print()
    print("  尝试过程：")
    for line in result["log"]:
        print(f"    · {line}")
    print()

    if result["ok"]:
        data = result["data"]
        print(f"  ✔ 最终通过（用了 {result['attempts']} 次尝试）")
        print(f"    product={data.get('product')!r}  rating={data.get('rating')!r}")
        print(f"    recommend={data.get('recommend')!r}  sentiment={data.get('sentiment')!r}")
        print()
        print(f"  → 第 1 次输出了中文字段名（商品名称/优点/是否推荐），被校验拦下。")
        print(f"    代码把「缺哪些字段、类型错在哪」反馈给模型，它据此修正 ——")
        print(f"    如此反复，第 {result['attempts']} 次才全部通过。")
        print()
        print("  ⚠️ 特别注意第 2 次的失败原因里，可能出现 sentiment = 'mixed'。")
        print("     这条评价确实又夸又贬（'音质很好，就是物流慢'），")
        print("     模型想表达「褒贬都有」，但你的枚举里没有这个选项。")
        print("     ——当模型反复尝试输出一个你没允许的值时，")
        print("       通常不是它笨，而是你的 schema 少了一个类别。")
    else:
        print(f"  ✘ 重试 {result['attempts'] - 1} 次后仍失败")

    print()
    print("  关键点：代码没有「教」模型怎么写，只是把不合格的地方告诉它。")
    print("  这是最省事也最有效的一种自修复 —— 不需要改 Prompt。")
    print()


# ============================================================
# 演示 2：为什么不能「兜底」
# ============================================================

DEFAULT_FALLBACK = {
    "product": "未知",
    "rating": 3,
    "pros": [],
    "issues": [],
    "recommend": False,
    "sentiment": "neutral",
}


def show_silent_failure_demo():
    """
    ❌ 反面教材：校验失败时，悄悄换成一个默认值。

    这里用一个真实存在过的坏输出（上一课 V2 的输出就是长这样）来演示。
    """
    print("=" * 68)
    print("演示 2：失败时给默认值会怎样（反例）")
    print("=" * 68)
    print()

    # 这是上一课 V2 的真实输出形态：JSON 合法，但字段名和类型全不对
    bad_output = '{"商品名称": "降噪耳机", "优点": ["音质好"], "是否推荐": "会推荐给朋友"}'

    print("  模型实际返回：")
    print(f"    {bad_output}")
    print()

    # ---- ❌ 错误做法：兜底 ----
    try:
        parsed = json.loads(bad_output)
        ok, problems = validate(parsed)
        if not ok:
            result_bad = DEFAULT_FALLBACK          # 悄悄替换成默认值
    except json.JSONDecodeError:
        result_bad = DEFAULT_FALLBACK

    print("  ❌ 兜底做法的返回值：")
    print(f"    {json.dumps(result_bad, ensure_ascii=False)}")
    print()
    print("  程序没崩、没报错、看起来一切正常。但注意 rating=3 ——")
    print("  这个数字不是用户说的，是程序编的。")
    print()

    # ---- ✅ 正确做法：明确失败 ----
    print("  ✅ 正确做法的返回值：")
    print('    {"ok": False, "data": None, "problems": ["缺少字段 product", ...]}')
    print()
    print("  区别在这里：")
    print("    · 兜底版：一千条里混进几条假数据，你看不出来")
    print("    · 正确版：这 N 条会出现在失败统计里，进入人工复核队列")
    print()
    print("  半年后的差别：")
    print("    · 兜底版：报表里 30% 的商品叫「未知」，平均分永远是 3.0，")
    print("      而且系统从头到尾没报过一次错 —— 你根本不知道数据已经烂了")
    print("    · 正确版：你知道有 30% 抽取失败，于是去改 Prompt，")
    print("      或者给这几百条加人工处理")
    print()


# ============================================================
# 主流程
# ============================================================

def main():
    print()
    print("阶段 1 · 第 2 课：让模型输出可靠的数据")
    print()

    results = run_batch(TEST_CASES)
    print_summary(results)

    demo_retry_mechanism()
    show_silent_failure_demo()

    print("=" * 68)
    print("该看出什么")
    print("=" * 68)
    print("""
1. 你刚才做的事，名字叫「评测」。虽然只有 6 条样本，但结构是对的：
   固定输入 → 明确判定标准 → 统计成功率 → 单独列出失败样本。

   阶段 2 会把它扩展成 20 题、50 题的正式测试集。区别只是样本数量，
   方法论完全一样。你现在做的这个，就是最小版的评测系统。

2. 「重试通过」那一栏值得盯住。演示 1 里你会看到：同一个模型、
   同一个任务，弱 Prompt 需要两次才通过 —— 而如果没有反馈重试，
   它第一次的输出就是一条被丢弃的失败样本。

3. 边界样本（"还行吧"、"太差了"这种）才是真正暴露问题的。
   只用标准样本测，成功率永远是 100%，你会以为自己做得很好。

4. 演示 2 是这一课最重要的：
   「兜底」比「崩溃」更危险。崩溃你马上知道有问题；
   兜底会安静地把错误数据喂进下游，直到某天你发现所有结论都是错的。

5. 演示 1 里还藏着一个反直觉的教训：
   模型第 2 次尝试输出 sentiment = 'mixed'，被你的枚举挡回去了。
   但「褒贬都有」本来就是真实存在的一类情绪 —— 缺的是你的 schema。

   当模型反复试图输出一个你没允许的值时，先别急着骂它。
   那往往是它在告诉你：**你的数据结构没覆盖真实世界**。
   这类反馈，只有在你认真看失败原因时才会发现。
""")

    print("=" * 68)
    print("你的练习")
    print("=" * 68)
    print("""
1. 把 extract() 的 max_retries 改成 0，再跑演示 1。
   看那个弱 Prompt 的输出会变成什么结果。

2. 在 TEST_CASES 里加一条真正刁钻的样本，比如：
   "东西不错，但是物流太慢了，不过客服补偿了优惠券，下次还会买。"
   （同时有正面和负面，看模型怎么定 sentiment）

3. 故意把 FIELD_SPEC 里 rating 的描述改成「1-10 的整数」，
   但 validate() 里的范围判断不改。观察 Prompt 和校验打架时会发生什么 ——
   这就是「同一份约定写在两个地方」会出的问题。

4. 想一想：如果这批数据要处理 1 万条，失败率 5% 意味着什么？
   你打算怎么处理这 500 条失败样本？
   （提示：这是阶段 4「产品化」要回答的问题之一）
""")


if __name__ == "__main__":
    main()
