"""
阶段 2 · 第 2 课：跑出 RAG 检索的基线
================================================

这一课只做一件事：**给检索效果一个可信的数字。**

【第一版指标失败了，这一版是修正后的】

第一版我用「答案所在的文件有没有被检索到」当判定标准，结果是 15/15、
召回率 100%。100% 听起来很好，实际上说明**指标失效了** ——
一共才 10 篇文档，取前 5 名，正确答案的文件几乎必然在里面。
一个测不出差别的指标，无法指导任何优化。

修正后改用**段落级判定**：每题指定若干「答案要点」短语，
必须能在检索结果的原文里原样找到，才算命中。
这比「文件出现」严格得多。

【分档看指标】
同时报告 top-1 / top-3 / top-5 三档命中率。
只报一个 top-5 会把问题藏起来 —— 真正反映排序质量的是 top-1。

【关于「无法回答题」的分数】
理想情况下，语料里没有答案的问题，检索最高分应该明显偏低。
但实测发现两个分布**严重重叠**，说明 BM25 的原始分不能直接当拒答依据。
脚本里用「阈值扫描」把这个结论量化出来，而不是拍一个数字。

【运行方式】
    source .venv/bin/activate
    python 06_rag_baseline.py
"""

import json
import statistics
from pathlib import Path

import rag

PROJECT_DIR = Path(__file__).resolve().parent
KNOWLEDGE_DIR = PROJECT_DIR / "knowledge"
TESTSET_PATH = PROJECT_DIR / "testset.json"

MAX_K = 5


def load_testset():
    with open(TESTSET_PATH, encoding="utf-8") as f:
        return json.load(f)["questions"]


# ============================================================
# 判定：段落级命中
# ============================================================

def contains_all(text, phrases):
    """所有要点短语都能在文本里找到，才算命中。"""
    return all(p in text for p in phrases)


def hit_at(retrieved, phrases, k):
    """前 k 名的文本合并后，是否包含全部答案要点。"""
    merged = "\n".join(chunk["text"] for chunk, _ in retrieved[:k])
    return contains_all(merged, phrases)


# ============================================================
# 阈值扫描：拒答阈值到底能不能用
# ============================================================

def scan_threshold(answerable_scores, unanswerable_scores):
    """
    扫描所有可能的阈值，找出最优的那一个。

    对每个候选阈值：
      假接受 = 无法回答题里，分数高于阈值（系统会去硬答 → 应该拒答却没拒）
      假拒绝 = 可回答题里，分数低于阈值（系统会拒答 → 本该回答却没答）

    返回 (最优阈值, 最小错误数, 明细列表)
    """
    candidates = sorted(set(answerable_scores + unanswerable_scores))
    best = None

    for threshold in candidates:
        false_accept = sum(1 for s in unanswerable_scores if s >= threshold)
        false_reject = sum(1 for s in answerable_scores if s < threshold)
        total = false_accept + false_reject
        rows = (threshold, false_accept, false_reject, total)
        if best is None or total < best[3]:
            best = rows

    return best


# ============================================================
# 主流程
# ============================================================

def main():
    print()
    print("=" * 70)
    print("阶段 2 第 2 课：建立检索基线")
    print("=" * 70)
    print()

    index, chunks, doc_count = rag.build_index_from_folder(KNOWLEDGE_DIR)
    lengths = [len(c["text"]) for c in chunks]
    print(f"  语料：{doc_count} 篇文档  →  {len(chunks)} 个块"
          f"（平均 {sum(lengths) // len(lengths)} 字）")
    print(f"  配置：chunk_size={rag.DEFAULT_CHUNK_SIZE}  "
          f"overlap={rag.DEFAULT_OVERLAP}  检索取前 {MAX_K} 名")
    print()

    questions = load_testset()
    answerable = [q for q in questions if q["answerable"]]
    unanswerable = [q for q in questions if not q["answerable"]]

    # ---------- 逐题检索 ----------
    rows = []
    for q in questions:
        retrieved = index.search(q["question"], top_k=MAX_K)
        top1 = retrieved[0][1] if retrieved else 0.0

        record = {
            "q": q,
            "retrieved": retrieved,
            "top1_score": top1,
        }
        if q["answerable"]:
            record["h1"] = hit_at(retrieved, q["must_contain"], 1)
            record["h3"] = hit_at(retrieved, q["must_contain"], 3)
            record["h5"] = hit_at(retrieved, q["must_contain"], 5)
            # 首次命中位次
            record["rank"] = next(
                (i for i in range(1, MAX_K + 1)
                 if hit_at(retrieved, q["must_contain"], i)),
                None,
            )
        rows.append(record)

    # ---------- 逐题明细 ----------
    print("=" * 70)
    print("逐题明细（○ = 该档位命中，× = 未命中）")
    print("=" * 70)
    print()
    print(f"  {'题目':<6} {'题型':<10} {'top1':<6} {'top3':<6} {'top5':<6} {'最高分':<8} 问题")
    print("  " + "-" * 66)

    for r in rows:
        q = r["q"]
        short = q["question"][:26]
        if q["answerable"]:
            marks = "".join("○" if r[k] else "×" for k in ("h1", "h3", "h5"))
            m1, m3, m5 = marks[0], marks[1], marks[2]
            print(f"  {q['id']:<6} {q['type']:<10} {m1:<6} {m3:<6} {m5:<6} "
                  f"{r['top1_score']:<8.2f} {short}")
        else:
            print(f"  {q['id']:<6} {'无法回答':<10} {'—':<6} {'—':<6} {'—':<6} "
                  f"{r['top1_score']:<8.2f} {short}")
    print()

    # ---------- 基线 ----------
    ans_rows = [r for r in rows if r["q"]["answerable"]]

    print("=" * 70)
    print("基线数字")
    print("=" * 70)
    print()

    for k, key in ((1, "h1"), (3, "h3"), (5, "h5")):
        got = sum(1 for r in ans_rows if r[key])
        bar = "█" * round(got / len(ans_rows) * 30)
        print(f"  命中 @ top-{k}   {got:>2} / {len(ans_rows)}  "
              f"{got / len(ans_rows):>4.0%}  {bar}")
    print()

    ranked = [r["rank"] for r in ans_rows if r["rank"]]
    if ranked:
        print(f"  平均首次命中位次：第 {statistics.mean(ranked):.1f} 位")
        print(f"  第一位即命中    ：{sum(1 for x in ranked if x == 1)} 题")
    print()

    print("  按题型看 top-1 命中：")
    by_type = {}
    for r in ans_rows:
        by_type.setdefault(r["q"]["type"], []).append(r)
    for type_name, items in sorted(by_type.items()):
        got = sum(1 for i in items if i["h1"])
        print(f"    {type_name:<10} {got} / {len(items)}")
    print()

    # ---------- 未命中明细 ----------
    missed = [r for r in ans_rows if not r["h5"]]
    missed1 = [r for r in ans_rows if not r["h1"] and r["h5"]]
    if missed1:
        print("=" * 70)
        print("top-1 没命中、但 top-5 命中的题（排序问题，不是召回问题）")
        print("=" * 70)
        print()
        for r in missed1:
            q = r["q"]
            print(f"  {q['id']}  {q['question']}")
            print(f"      要点 {q['must_contain']} 在第 {r['rank']} 位才出现")
            print(f"      top-1 实际是：{r['retrieved'][0][0]['source']}")
            print()
    if missed:
        print(f"  另有 {len(missed)} 题在 top-5 内完全没命中 —— 这些是真正需要优化的。")
        print()

    # ---------- 分数分布与拒答阈值 ----------
    print("=" * 70)
    print("拒答阈值：能不能用分数判断「该不该回答」")
    print("=" * 70)
    print()

    ans_scores = [r["top1_score"] for r in ans_rows]
    unans_scores = [r["top1_score"] for r in rows if not r["q"]["answerable"]]

    print(f"  可回答题   最高分：{min(ans_scores):.2f} ~ {max(ans_scores):.2f}"
          f"   平均 {statistics.mean(ans_scores):.2f}")
    print(f"  无法回答题 最高分：{min(unans_scores):.2f} ~ {max(unans_scores):.2f}"
          f"   平均 {statistics.mean(unans_scores):.2f}")
    print()

    overlap_low = max(min(unans_scores), min(ans_scores))
    overlap_high = min(max(unans_scores), max(ans_scores))
    ans_over = [s for s in unans_scores if s > min(ans_scores)]
    print(f"  两个区间重叠在 {overlap_low:.2f} ~ {overlap_high:.2f} 之间，"
          f"有 {len(ans_over)} / {len(unans_scores)} 道无法回答题的分数"
          f"高于可回答题的最低分。")
    print()

    threshold, fa, fr, total = scan_threshold(ans_scores, unans_scores)
    print(f"  阈值扫描结果（所有可能阈值里最优的那个）：")
    print(f"    阈值 = {threshold:.2f}")
    print(f"    假接受（该拒答却答了）：{fa} / {len(unans_scores)}")
    print(f"    假拒绝（该回答却拒了）：{fr} / {len(ans_scores)}")
    print(f"    合计错误：{total} / {len(ans_scores) + len(unans_scores)}")
    print()
    print("  → 结论：即使取最优阈值，也一定会犯错。")
    print("    单靠「检索分数低于某值就拒答」是不可靠的。")
    print("    更稳的做法是两条腿走路：一是让模型在 Prompt 里被明确要求")
    print("    「资料没有就说未提及」，二是把「无法回答题」的准确率")
    print("    作为一项独立指标持续监控。")
    print()

    # ---------- 完整链路演示 ----------
    print("=" * 70)
    print("完整链路：检索 → 生成")
    print("=" * 70)
    print()

    demo = [ans_rows[0], rows[-1]]      # 一题可回答 + 一题无法回答
    for r in demo:
        q = r["q"]
        print("-" * 70)
        print(f"问题：{q['question']}")
        print(f"类型：{q['type']}")
        print()
        try:
            text, usage = rag.answer(q["question"], r["retrieved"])
            print(f"模型回答：{text.strip()}")
            print(f"（消耗 {usage.get('total_tokens', 0)} token）")
        except Exception as error:
            print(f"[生成失败] {error}")
        print()
        print(f"参考答案：{q['answer']}")
        print()

    # ---------- 结论 ----------
    print("=" * 70)
    print("该看出什么")
    print("=" * 70)
    print("""
1. 先看三档命中率的差距。如果 top-5 很高但 top-1 明显低，
   说明「东西都找到了，但排序不够准」—— 这是重排（rerank）要解决的问题。
   如果 top-5 也不高，那是召回本身有问题，该动切块或换检索方式。

2. 我第一版用「文件级」判定得出 100%，那是个失败指标。
   **指标饱和和指标错误一样危险** —— 它不会报警，只会让你以为一切正常。
   建测试集时一定要检查：这个指标能不能区分「好」和「差」？

3. 阈值扫描那一栏是全篇最实用的结论：
   **不要用单一的检索分数来决定拒答。** 与其拍一个阈值，
   不如把「资料中没有就不回答」写进 Prompt，并把无法回答题
   当成一项独立指标长期跟踪。

4. 看最后那个演示。无法回答的那道题，模型答的是「资料中未提及」——
   它没有编造。这说明 Prompt 层的约束起作用了。
   但请注意：这次成功不代表下次成功，所以才需要指标持续监控。
""")

    print("=" * 70)
    print("你的练习")
    print("=" * 70)
    print("""
1. 改 rag.py 里的 DEFAULT_CHUNK_SIZE：200 → 400 → 800，
   各跑一次，记录三档命中率。画出一条曲线，看看最优值在哪。
   （提示：块越小定位越准，但容易切碎；块越大上下文越全，但噪声更多）

2. 把 DEFAULT_OVERLAP 改成 0，重跑。有没有题从命中变成未命中？
   那些题就是「答案跨块」的题 —— 它们就是需要重叠的理由。

3. 看 top-1 命中率低的那些题，打开对应文档，找出答案所在的段落，
   想想为什么它没排到第一。（提示：BM25 只看词频，不看语序和语义）

4. 进阶：给 testset.json 再加 5 道「无法回答题」，但要用更刁钻的问法 ——
   比如问一个和语料话题很接近、但确实没写的内容。
   看模型的拒答能力会不会下降。
""")


if __name__ == "__main__":
    main()
