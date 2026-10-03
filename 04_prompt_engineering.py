"""
阶段 1 · 第 1 课：Prompt 对比实验
================================================

同一个任务，五种 Prompt 写法，看结果差多少。

【任务】从一段用户评价里抽取结构化信息。

【为什么选这个任务】
因为它有**客观的成败标准**。而且这里有两层标准，必须分清：

    第一层：输出能不能被 json.loads() 解析 —— 格式对不对
    第二层：解析出来的字段名、类型、取值范围对不对 —— 内容对不对

只测第一层会漏掉大量问题。很多「看起来成功了」的输出，其实是
字段名变成了中文、布尔值变成了字符串、数字变成了文字 ——
程序一读就崩，但你的检查脚本会说「通过」。

这是 Prompt 工程、也是整个 AI 工程里最核心的一课：
**你得先能测出来什么叫「对」，才有资格谈优化。**

【运行方式】在项目目录里：
    source .venv/bin/activate
    python 04_prompt_engineering.py

依赖 llm.py 模块。
"""

import json
import time

import llm

# ============================================================
# 待处理的评价文本
# ============================================================
REVIEW = """上周在你们店里买了那款降噪耳机，等了四天才到货，有点慢。
不过音质是真的好，戴一整天耳朵也不疼，续航也够用。
就是包装盒被压扁了一角，客服态度倒是不错，二话不说就补发了。
整体还是挺满意的，会推荐给朋友，但物流希望能改进。"""


# ============================================================
# 期望的数据结构（这就是你的「验收标准」）
# ============================================================
EXPECTED_SCHEMA = {
    "product": str,
    "rating": int,
    "pros": list,
    "issues": list,
    "recommend": bool,
    "sentiment": str,
}

ALLOWED_SENTIMENTS = {"positive", "neutral", "negative"}


# ============================================================
# 五个 Prompt 变体，从最随便到最规范
# ============================================================

V1 = {
    "name": "V1 极简",
    "note": "只说要做什么，没说怎么做",
    "system": None,
    "user": f"提取这段评价里的信息。\n\n{REVIEW}",
    "extra": None,
}

V2 = {
    "name": "V2 指定格式",
    "note": "提了 JSON，但没说字段和约束",
    "system": None,
    "user": f"提取这段评价里的信息，用 JSON 格式输出。\n\n{REVIEW}",
    "extra": None,
}

V3_SYSTEM = "你是一个信息抽取助手。你只输出 JSON，不输出任何其他文字。"

V3_USER = f"""从下面的用户评价中抽取信息。

必须输出以下字段：
- product: 商品名称，字符串
- rating: 用户满意度评分，1-5 的整数
- pros: 优点列表，字符串数组
- issues: 问题列表，字符串数组
- recommend: 是否愿意推荐给朋友，布尔值（true / false）
- sentiment: 整体情绪，只能是 "positive" / "neutral" / "negative" 之一

要求：
1. 只输出 JSON 对象本身。不要用 ```json 代码块包裹，不要加任何解释文字
2. 评价中没有提到的信息用 null，不要编造
3. 严格使用上面的字段名，不要新增、删除或改名
4. rating 与 recommend 必须是 JSON 数字和布尔值，不要写成字符串

评价：
{REVIEW}"""

V3 = {
    "name": "V3 规范",
    "note": "角色 + 字段定义 + 类型 + 输出约束",
    "system": V3_SYSTEM,
    "user": V3_USER,
    "extra": None,
}

V4_EXAMPLE = """示例：

评价：
这个键盘手感很好，但是用了两周就有一个键失灵了，联系客服也没人回。很失望。

输出：
{"product": "键盘", "rating": 2, "pros": ["手感好"], "issues": ["键位失灵", "客服无响应"], "recommend": false, "sentiment": "negative"}"""

V4 = {
    "name": "V4 规范+示例",
    "note": "V3 再加一个输入输出示例",
    "system": V3_SYSTEM,
    "user": f"{V4_EXAMPLE}\n\n现在请处理下面这条评价。\n\n评价：\n{REVIEW}",
    "extra": None,
}

V5 = {
    "name": "V5 规范+JSON模式",
    "note": "不改 Prompt，改用 API 的强制 JSON 特性",
    "system": V3_SYSTEM,
    "user": V3_USER,
    "extra": {"response_format": {"type": "json_object"}},
}


VARIANTS = [V1, V2, V3, V4, V5]


# ============================================================
# 判定层
# ============================================================
def strip_code_fence(text):
    """
    去掉模型可能加上的 markdown 代码块包装。
    真实项目里，这个清理动作几乎每次都要做。
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    return cleaned


def check_parseable(text):
    """第一层：能不能解析成 JSON。"""
    try:
        return True, json.loads(strip_code_fence(text)), None
    except json.JSONDecodeError as error:
        return False, None, f"{error.msg}（位置 {error.pos}）"


def check_schema(data):
    """
    第二层：解析出来的东西，是不是我们真正想要的。

    返回 (是否通过, 问题列表)。问题列表会具体指出哪里不对。
    """
    problems = []

    if not isinstance(data, dict):
        return False, [f"顶层应是对象，实际是 {type(data).__name__}"]

    # 1. 字段是否齐全、类型是否正确
    for key, expected_type in EXPECTED_SCHEMA.items():
        if key not in data:
            problems.append(f"缺少字段 `{key}`")
            continue
        value = data[key]
        # 注意：Python 里 isinstance(True, int) 是 True，
        # 所以布尔值会被误判为整数，必须单独排除。
        if expected_type is int and isinstance(value, bool):
            problems.append(f"`{key}` 应为整数，实际是布尔值 {value}")
        elif expected_type is bool and not isinstance(value, bool):
            problems.append(f"`{key}` 应为布尔值，实际是 {type(value).__name__}：{value!r}")
        elif not isinstance(value, expected_type):
            problems.append(f"`{key}` 应为 {expected_type.__name__}，实际是 {type(value).__name__}")

    # 2. 有没有多出不该有的字段
    for key in data:
        if key not in EXPECTED_SCHEMA:
            problems.append(f"多出未定义的字段 `{key}`")

    # 3. 取值范围
    rating = data.get("rating")
    if isinstance(rating, int) and not isinstance(rating, bool):
        if not 1 <= rating <= 5:
            problems.append(f"`rating` 超出 1-5 范围：{rating}")

    sentiment = data.get("sentiment")
    if isinstance(sentiment, str) and sentiment not in ALLOWED_SENTIMENTS:
        problems.append(f"`sentiment` 取值非法：{sentiment!r}（只允许 {sorted(ALLOWED_SENTIMENTS)}）")

    # 4. 数组里应该都是字符串
    for key in ("pros", "issues"):
        value = data.get(key)
        if isinstance(value, list) and not all(isinstance(x, str) for x in value):
            problems.append(f"`{key}` 数组里混入了非字符串元素")

    return len(problems) == 0, problems


def judge(text):
    """综合两层判定，返回结果字典。"""
    parsed_ok, data, parse_error = check_parseable(text)

    if not parsed_ok:
        return {
            "parse_ok": False,
            "schema_ok": False,
            "parse_error": parse_error,
            "schema_problems": ["（上一步就没通过，无法检查字段）"],
            "data": None,
        }

    schema_ok, problems = check_schema(data)
    return {
        "parse_ok": True,
        "schema_ok": schema_ok,
        "parse_error": None,
        "schema_problems": problems,
        "data": data,
    }


def run_variant(variant):
    messages = []
    if variant["system"]:
        messages.append({"role": "system", "content": variant["system"]})
    messages.append({"role": "user", "content": variant["user"]})

    started = time.time()
    text, usage = llm.chat(messages, temperature=0, extra=variant["extra"])
    elapsed = time.time() - started

    return {
        "text": text,
        "tokens": usage.get("total_tokens", 0),
        "cost": llm.estimate_cost(usage),
        "seconds": elapsed,
        **judge(text),
    }


# ============================================================
# 主流程
# ============================================================
def main():
    print("=" * 66)
    print("Prompt 对比实验：同一任务，五种写法")
    print("=" * 66)
    print()
    print("两层判定标准：")
    print("  第一层  格式：输出能否被 json.loads() 解析")
    print("  第二层  内容：字段名 / 类型 / 取值范围是否完全符合约定")
    print()

    results = []

    for variant in VARIANTS:
        print("─" * 66)
        print(f"{variant['name']}    （{variant['note']}）")
        print("─" * 66)

        try:
            result = run_variant(variant)
        except llm.LLMError as error:
            print(f"  调用失败：{error}\n")
            results.append({**variant, "parse_ok": False, "schema_ok": False,
                            "text": "", "tokens": 0, "cost": 0.0, "seconds": 0.0,
                            "schema_problems": [str(error)], "parse_error": str(error)})
            continue

        results.append({**variant, **result})

        preview = result["text"].strip()
        if len(preview) > 360:
            preview = preview[:360] + "\n  ...（已截断）"
        for line in preview.split("\n"):
            print(f"  {line}")

        print()
        print(f"  格式：{'✔ 可解析' if result['parse_ok'] else '✘ 无法解析'}")
        if not result["parse_ok"]:
            print(f"        原因：{result['parse_error']}")
        else:
            print(f"  内容：{'✔ 符合约定' if result['schema_ok'] else '✘ 不符合约定'}")
            for problem in result["schema_problems"][:6]:
                print(f"        · {problem}")
            if len(result["schema_problems"]) > 6:
                print(f"        · ...还有 {len(result['schema_problems']) - 6} 项")
        print(f"  耗时 {result['seconds']:.1f}s · {result['tokens']} token"
              f" · 约 {result['cost']:.6f} 元")
        print()

    # ---------- 汇总 ----------
    print("=" * 66)
    print("结果汇总")
    print("=" * 66)
    print()
    header = f"{'写法':<18} {'格式':<8} {'内容':<8} {'token':<8} {'成本(元)':<12}"
    print(header)
    print("-" * 66)
    for item in results:
        fmt = "✔" if item["parse_ok"] else "✘"
        sem = "✔" if item["schema_ok"] else "✘"
        print(f"{item['name']:<18} {fmt + ' 可解析':<8} {sem + ' 合规':<8} "
              f"{item['tokens']:<8} {item['cost']:<12.6f}")

    fully_ok = sum(1 for item in results if item["schema_ok"])
    parse_only = sum(1 for item in results if item["parse_ok"])
    print("-" * 66)
    print(f"格式可解析：{parse_only} / {len(results)}"
          f"      真正可用：{fully_ok} / {len(results)}")
    print()

    gap = parse_only - fully_ok
    if gap > 0:
        print(f"⚠️  注意这 {gap} 个的差距：它们「看起来成功了」，但字段或类型不对。")
        print("   如果只测第一层，你会以为已经跑通 —— 直到线上报 KeyError。")
        print()

    print(f"本轮实验总花费：约 {sum(i['cost'] for i in results):.6f} 元")
    print()

    # ---------- 结论 ----------
    print("=" * 66)
    print("该看出什么")
    print("=" * 66)
    print("""
1. V1 输出了漂亮的中文条目列表 —— 人看着挺好，程序完全没法用。
   最常见的错误：用「给人看」的标准去写「给程序用」的接口。

2. V2 是最危险的一种。它输出了合法 JSON，所以第一层判定会通过。
   但字段名变成了中文、布尔值变成了字符串、还多送了几个字段。
   你的下游代码 data["product"] 会直接 KeyError。
   → 「格式对」不等于「内容对」，这两层必须分开测。

3. V3 写清「角色 + 字段 + 类型 + 约束」后，两层都通过。
   这四样是结构化输出的最小可用集合。

4. V4 加一个示例。当输出格式比较特殊、或要求难以用语言描述时，
   一个示例胜过一百字说明。

5. V5 用的是 V3 的 Prompt，只多了 response_format 参数。
   Prompt 能「提高概率」，但有些问题不该用概率解决 ——
   该用确定性机制解决。生产环境里两者是配合使用的：
   能用 API 特性保证的，就不要靠求模型听话。
""")

    print("=" * 66)
    print("你的练习")
    print("=" * 66)
    print("""
1. 把 REVIEW 换成一段你自己的真实评价（比如吐槽某次外卖），重跑，
   看哪几个变体在第二层判定上翻车。

2. 把 temperature 从 0 改成 1.5，连跑三次。观察哪些变体开始不稳定。
   结构化输出任务，temperature 建议就用 0。

3. 给 V3 故意删掉「不要用 ```json 代码块包裹」这一条，看第一层会不会挂。

4. 只保留 V3，把字段从 6 个减到 2 个（只留 product 和 rating），
   看要多少字才能稳定通过 —— 这是「Prompt 精简」，生产环境里省 token 就是省钱。

5. 进阶思考：如果要处理 1 万条评价，你希望第一层失败率高还是第二层？
   为什么？（提示：想想两种失败在线上分别会造成什么后果）
""")


if __name__ == "__main__":
    main()
