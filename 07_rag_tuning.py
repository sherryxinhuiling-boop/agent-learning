"""
阶段 2 · 第 3 课：用数据决定参数
================================================

上一课建立了基线，但配置（400 字一块 / 重叠 80 / 取前 5）是我随手定的。
这一课把参数扫一遍，看真实的最优值在哪。

【这一课真正教的东西】
不是「400 字好还是 600 字好」，而是**「怎么决定一个参数该取多少」**。

RAG 里有大量这类参数：块大小、重叠长度、检索条数、相似度阈值……
它们没有普适的正确答案，网上抄来的推荐值通常不适用于你的数据。

唯一的办法是：**固定评测方法，扫一遍参数，用数字选。**

【为什么要固定评测方法】
你有 20 题测试集、三档命中率指标。这就是一套可复现的尺子。
参数扫描的全部意义在于：**尺子不变，只换参数。**
如果每次都用不同的题目或不同的判定标准，扫出来的结果毫无意义。

【运行方式】
    source .venv/bin/activate
    python 07_rag_tuning.py

会跑十几组配置，大约需要十几秒。
"""

import json
from pathlib import Path

import rag

PROJECT_DIR = Path(__file__).resolve().parent
KNOWLEDGE_DIR = PROJECT_DIR / "knowledge"
TESTSET_PATH = PROJECT_DIR / "testset.json"

# 要扫描的块大小
CHUNK_SIZES = [150, 250, 400, 550, 750, 1000]

# 重叠比例（相对于块大小）。0 表示不留重叠
OVERLAP_RATIOS = [0.0, 0.2]

# 判定用的最大档位
MAX_K = 5


def load_testset():
    with open(TESTSET_PATH, encoding="utf-8") as f:
        return json.load(f)["questions"]


def hit_at(retrieved, phrases, k):
    """前 k 名的文本合并后，是否包含全部答案要点。"""
    merged = "\n".join(chunk["text"] for chunk, _ in retrieved[:k])
    return all(p in merged for p in phrases)


def evaluate(index, questions, max_k=MAX_K):
    """
    拿固定的测试集去测一个索引。

    注意这里所有配置都用同一个 questions 列表、同一个 hit_at 判定、
    同一个 MAX_K —— 尺子不变，只换参数。
    """
    answerable = [q for q in questions if q["answerable"]]
    hits = {1: 0, 3: 0, 5: 0}
    ranks = []

    for q in answerable:
        retrieved = index.search(q["question"], top_k=max_k)
        for k in (1, 3, 5):
            if hit_at(retrieved, q["must_contain"], k):
                hits[k] += 1
        first = next(
            (i for i in range(1, max_k + 1)
             if hit_at(retrieved, q["must_contain"], i)),
            None,
        )
        if first:
            ranks.append(first)

    total = len(answerable)
    return {
        "total": total,
        "h1": hits[1],
        "h3": hits[3],
        "h5": hits[5],
        "avg_rank": sum(ranks) / len(ranks) if ranks else 0,
    }


def evaluate_at(index, questions, k):
    """
    只测「前 k 名」这一个档位的命中情况。

    上一课的 evaluate 一次给出 1/3/5 三档，适合做基线报告；
    但做 top_k 扫描时不行 —— 那样打印出来的永远是 k=5 那个数字，
    换个 top_k 也不会变，看起来就像「top_k 对结果没影响」。
    """
    answerable = [q for q in questions if q["answerable"]]
    hits = 0
    total_chars = 0
    for q in answerable:
        retrieved = index.search(q["question"], top_k=k)
        if hit_at(retrieved, q["must_contain"], k):
            hits += 1
        total_chars += sum(len(c["text"]) for c, _ in retrieved)
    return hits, len(answerable), total_chars / len(answerable)


def bar(value, total, width=24):
    filled = round(value / total * width) if total else 0
    return "█" * filled + "·" * (width - filled)


def main():
    print()
    print("=" * 78)
    print("阶段 2 第 3 课：参数扫描 —— 用数据决定参数")
    print("=" * 78)
    print()

    questions = load_testset()
    answerable = [q for q in questions if q["answerable"]]
    print(f"  测试集：{len(answerable)} 道可回答题（判定标准固定不变）")
    print()

    # ============================================================
    # 扫描一：块大小 × 重叠比例
    # ============================================================
    print("=" * 78)
    print("扫描一：块大小 × 重叠比例")
    print("=" * 78)
    print()
    print(f"  {'块大小':<8} {'重叠':<6} {'块数':<6} {'top-1':<8} {'':<14} {'top-3':<8} {'top-5':<8} {'平均位次'}")
    print("  " + "-" * 74)

    results = []
    for size in CHUNK_SIZES:
        for ratio in OVERLAP_RATIOS:
            overlap = int(size * ratio)
            index, chunks, _ = rag.build_index_from_folder(
                KNOWLEDGE_DIR, chunk_size=size, overlap=overlap
            )
            metrics = evaluate(index, questions)

            results.append({
                "size": size,
                "ratio": ratio,
                "overlap": overlap,
                "chunks": len(chunks),
                **metrics,
            })

            print(f"  {size:<8} {overlap:<6} {len(chunks):<6} "
                  f"{metrics['h1']:>2}/{metrics['total']:<5} "
                  f"{bar(metrics['h1'], metrics['total'], 12):<14} "
                  f"{metrics['h3']:>2}/{metrics['total']:<5} "
                  f"{metrics['h5']:>2}/{metrics['total']:<5} "
                  f"{metrics['avg_rank']:.2f}")
    print()

    # ============================================================
    # 找出最优
    # ============================================================
    print("=" * 78)
    print("最优配置")
    print("=" * 78)
    print()

    # 排序规则：优先 top-1，其次 top-3，再其次平均位次
    best = max(results, key=lambda r: (r["h1"], r["h3"], -r["avg_rank"]))
    baseline = next(r for r in results if r["size"] == 400 and r["overlap"] == 80)

    print(f"  {'配置':<22} {'top-1':<10} {'top-3':<10} {'top-5':<10} {'块数'}")
    print("  " + "-" * 62)
    print(f"  {'基线 400/80':<22} "
          f"{baseline['h1']:>2}/{baseline['total']:<7} "
          f"{baseline['h3']:>2}/{baseline['total']:<7} "
          f"{baseline['h5']:>2}/{baseline['total']:<7} {baseline['chunks']}")
    print(f"  {'最优 ' + str(best['size']) + '/' + str(best['overlap']):<22} "
          f"{best['h1']:>2}/{best['total']:<7} "
          f"{best['h3']:>2}/{best['total']:<7} "
          f"{best['h5']:>2}/{best['total']:<7} {best['chunks']}")
    print()

    delta = best["h1"] - baseline["h1"]
    if delta > 0:
        print(f"  → 换个块大小，top-1 命中从 {baseline['h1']} 提升到 {best['h1']}"
              f"（+{delta} 题，{delta / baseline['total']:.0%}）。")
        print("    这说明参数确实值得调，而且答案不在直觉里。")
    elif delta == 0:
        print(f"  → 块大小在当前语料上的影响不明显（都是 {best['h1']}/{best['total']}）。")
        print("    这本身也是有价值的信息：当语料短、结构清晰时，切块不是瓶颈。")
    else:
        print(f"  → 基线反而更好。这提醒我们：不要凭直觉改参数。")
    print()

    # 样本量提醒 —— 这一步很重要，否则很容易把噪声当成结论
    per_question = 1 / best["total"]
    print(f"  ⚠️ 样本量提醒：可回答题只有 {best['total']} 道，")
    print(f"     所以 1 道题 = {per_question:.1%}。")
    print(f"     top-1 从 {baseline['h1']} 到 {best['h1']} 相差 {abs(delta)} 道题 ——")
    if abs(delta) <= 2:
        print(f"     这个幅度的差异**很可能来自随机波动**，不足以支撑「550 更好」的结论。")
        print(f"     要确认它稳定，得把测试集扩大后再复现一次。")
    else:
        print(f"     差异幅度较大，相对可信，但仍建议扩大测试集复核。")
    print()
    print("     这就是小测试集的代价：**它只能发现大问题，分不清小差别。**")
    print()

    # 重叠到底有没有用
    print("  重叠的作用：")
    for size in CHUNK_SIZES:
        no_overlap = next(r for r in results if r["size"] == size and r["overlap"] == 0)
        with_overlap = next(r for r in results if r["size"] == size and r["overlap"] > 0)
        diff = with_overlap["h1"] - no_overlap["h1"]
        if diff != 0:
            sign = "▲" if diff > 0 else "▼"
            print(f"    块大小 {size}：加重叠后 top-1 {no_overlap['h1']} → "
                  f"{with_overlap['h1']}　{sign}{abs(diff)}")
    print("    （如果变化不大，说明这块语料的答案很少横跨块边界 ——")
    print("      换成结构松散的长文档，重叠的作用会明显得多）")
    print()

    # ============================================================
    # 扫描二：检索条数 top_k
    # ============================================================
    print("=" * 78)
    print("扫描二：检索取回几条（top_k）")
    print("=" * 78)
    print()

    index, chunks, _ = rag.build_index_from_folder(
        KNOWLEDGE_DIR, chunk_size=best["size"], overlap=best["overlap"]
    )

    print(f"  {'top_k':<12} {'命中':<10} {'命中率':<10} {'送入模型的字符数(均值)'}")
    print("  " + "-" * 60)

    for k in (1, 2, 3, 5, 8, 12, len(chunks)):
        hits, total, avg_chars = evaluate_at(index, questions, k)
        label = f"{k}" + ("（全部）" if k == len(chunks) else "")
        print(f"  {label:<12} {hits:>2}/{total:<7} "
              f"{hits / total:>6.0%}     {avg_chars:>8.0f}")
    print()

    print("  → 看这两列怎么变化：命中率上升，送进模型的字符数也在上升。")
    print("    最后一行完全命中，但代价是每次请求要处理一万多字符 ——")
    print("    这已经不是检索了，而是「把所有资料一股脑丢给模型」。")
    print()
    print("    **一个可以轻易刷到 100% 的指标，必须用资源消耗来约束。**")
    print("    选 top_k 的本质是「召回率」和「成本」之间的取舍。")
    print("    从表里挑那个「命中率已经到位、字符数还没起飞」的拐点。")
    print()

    # ============================================================
    # 结论
    # ============================================================
    print("=" * 78)
    print("该看出什么")
    print("=" * 78)
    print("""
1. 你今天做的这件事，叫「参数调优」，但它和「凭感觉调参」有本质区别：
   你有一个固定的测试集、一个固定的判定标准，只换了参数。

   **没有固定评测的调参，只是换一种方式猜。**

2. 最优值往往不极端。块太小会切碎语义，太大则会稀释相关性 ——
   所以曲线通常是中间高、两头低。但这个「中间」在哪，
   取决于你的语料长度和结构，抄不来。

3. 注意「重叠」那一栏。如果加重叠没什么提升，说明当前语料里
   答案很少横跨块边界 —— 那么重叠就是白花的存储成本，可以去掉。

4. 最后那张 top_k 表是这一课最重要的：
   指标必须和成本一起看。只看命中率的「优化」会把系统推向
   「把所有资料都塞给模型」这种既贵又慢的极端。

5. 停下来想一想：如果参数扫描的结果显示，几乎所有配置的分数都差不多，
   那说明什么？
   （提示：说明瓶颈不在你现在调的这个环节上。继续调它是在浪费时间，
     该去找真正的瓶颈。）
""")

    print("=" * 78)
    print("你的练习")
    print("=" * 78)
    print("""
1. 在 CHUNK_SIZES 里加一个 30，再跑一次。看极端小的块会发生什么。
   记录它带来的块数量和叠加成本 —— 小块是「精度换成本」。

2. 现在最优配置已经被扫出来了。把它写回 rag.py 的 DEFAULT_CHUNK_SIZE，
   然后重跑 06_rag_baseline.py，确认数字对得上。
   （这一步叫「把结论固化到代码里」，否则下次又从默认值开始）

3. 想一想：如果我要处理的是 1000 篇文档而不是 10 篇，
   这个扫描还要跑多久？你会怎么优化这个过程？
   （提示：这是「评测成本」问题，阶段 4 会正面遇到）

4. 把测试集从 20 题扩到 30 题，重跑扫描。观察最优配置有没有变 ——
   如果变了，说明 20 题的样本量还不足以稳定地定出参数。
   这是评测集规模带来的不确定性，值得亲手感受一次。
""")


if __name__ == "__main__":
    main()
