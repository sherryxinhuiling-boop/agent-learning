"""
rag.py —— RAG 检索模块
================================================

把「文档 → 切块 → 建索引 → 检索 → 生成回答」这条链路封装起来。
和 llm.py 一样，别的脚本只要 import 它。

【当前用的是 BM25 关键词检索】

为什么先用 BM25 而不是向量检索？
    DeepSeek 没有 embedding 接口，做稠密向量检索需要额外的 embedding 服务
    或本地模型（要装 2GB 的 torch）。先用零依赖的 BM25 把全链路跑通，
    得到第一个基线数字，再决定要不要引入向量检索。

    这个顺序本身就是工程上的正确做法：**先有基线，再谈优化。**
    没有基线的「优化」只是换了一种猜测。

【BM25 是什么】
    一个经典的检索打分算法，核心思想有两条：
      1. 一个词在当前文档里出现得越多，文档越相关（词频）
      2. 一个词在所有文档里越少见，它的区分度越高（逆文档频率）
         比如「的」「是」到处都有，区分度接近零；
         而「重排」「召回率」只在少数文档出现，一旦命中就很说明问题
    在此基础上还做了长度归一，避免长文档仅因为字多而占优。

【中文的特殊问题】
    BM25 是按「词」统计的，英文靠空格自然分词，中文没有空格。
    所以需要 jieba 先把句子切成分词 —— 这一步叫分词，是中文检索的必要预处理。
"""

import json
import re
from pathlib import Path

import jieba
import llm
from rank_bm25 import BM25Okapi

# ============================================================
# 配置
# ============================================================

# 切块参数（这两个数字对检索效果影响很大，后面会用测试集实测）
DEFAULT_CHUNK_SIZE = 400      # 每块目标字符数
DEFAULT_OVERLAP = 80          # 相邻块的重叠字符数

# 检索参数
DEFAULT_TOP_K = 5


# ============================================================
# 第 1 步：加载文档
# ============================================================

def load_documents(folder):
    """
    读取文件夹下所有 .md 文件，返回文档列表。

    每篇文档是一个字典：
        text     正文
        source   文件名（用于引用溯源，必须保留）
        title    标题（取第一个 # 开头的行）
    """
    folder = Path(folder)
    documents = []

    for path in sorted(folder.glob("*.md")):
        text = path.read_text(encoding="utf-8")

        # 找第一个一级标题当作文档标题
        title = path.stem
        for line in text.splitlines():
            if line.startswith("# "):
                title = line[2:].strip()
                break

        documents.append({
            "text": text,
            "source": path.name,
            "title": title,
        })

    return documents


# ============================================================
# 第 2 步：切块
# ============================================================

def split_paragraphs(text):
    """
    按空行把文档切成段落。

    这是「结构优先」的切块思路：Markdown 和多数文档都有天然的结构
    （标题、段落），沿着结构边界切不会把一句话拦腰截断。
    只有在段落本身超长时，才退而求其次按标点硬切。
    """
    blocks = []
    for block in re.split(r"\n\s*\n", text):
        block = block.strip()
        if block:
            blocks.append(block)
    return blocks


def split_long_block(block, chunk_size):
    """
    段落太长时按句子边界切开，避免一块里塞进太多内容。
    优先在句末标点处断开。
    """
    sentences = re.split(r"(?<=[。！？；\n])", block)
    pieces = []
    current = ""
    for sentence in sentences:
        if len(current) + len(sentence) <= chunk_size:
            current += sentence
        else:
            if current:
                pieces.append(current)
            current = sentence
    if current:
        pieces.append(current)
    return pieces


def chunk_document(document, chunk_size=DEFAULT_CHUNK_SIZE, overlap=DEFAULT_OVERLAP):
    """
    把一篇文档切成若干块。

    策略：结构优先 —— 先按段落切，再把相邻的小段落合并到接近 chunk_size。
    相邻块之间保留 overlap 个字符的重叠。

    为什么要重叠：如果答案刚好横跨两个块的边界，
    没有重叠就会两边都拿不全；有重叠则至少有一个块包含完整信息。

    每块都带上 source / title / position 元数据 —— 没有元数据就无法引用溯源。
    """
    blocks = split_paragraphs(document["text"])

    # 先把所有段落展开成不超过 chunk_size 的小块
    expanded = []
    for block in blocks:
        if len(block) <= chunk_size:
            expanded.append(block)
        else:
            expanded.extend(split_long_block(block, chunk_size))

    # 再把小块合并成接近 chunk_size 的块
    chunks = []
    current = ""
    for block in expanded:
        if len(current) + len(block) + 2 <= chunk_size:
            current = f"{current}\n\n{block}".strip()
        else:
            if current:
                chunks.append(current)
            current = block
    if current:
        chunks.append(current)

    # 加重叠
    if overlap > 0:
        overlapped = []
        for index, chunk in enumerate(chunks):
            if index == 0:
                overlapped.append(chunk)
            else:
                tail = chunks[index - 1][-overlap:]
                overlapped.append(f"{tail}\n\n{chunk}")
        chunks = overlapped

    # 附加元数据
    return [
        {
            "text": chunk,
            "source": document["source"],
            "title": document["title"],
            "position": index,          # 在文档中的第几块，便于定位
        }
        for index, chunk in enumerate(chunks)
    ]


def build_chunks(documents, chunk_size=DEFAULT_CHUNK_SIZE, overlap=DEFAULT_OVERLAP):
    """把所有文档切成一个大块列表。"""
    all_chunks = []
    for document in documents:
        all_chunks.extend(chunk_document(document, chunk_size, overlap))
    return all_chunks


# ============================================================
# 第 3 步：建索引
# ============================================================

def tokenize(text):
    """
    中文分词 + 清理。

    BM25 按词统计，中文必须先分词。jieba.lcut 会切出词和标点，
    这里把标点和空白过滤掉 —— 它们对区分文档没有帮助，只会干扰打分。
    """
    words = jieba.lcut(text)
    return [w.strip().lower() for w in words if w.strip() and not re.fullmatch(r"[\W_]+", w)]


class BM25Index:
    """
    BM25 索引。

    用法：
        index = BM25Index(chunks)
        results = index.search("Agent 由哪几部分组成", top_k=5)
    """

    def __init__(self, chunks):
        self.chunks = chunks
        self.corpus_tokens = [tokenize(c["text"]) for c in chunks]
        self.bm25 = BM25Okapi(self.corpus_tokens)

    def search(self, query, top_k=DEFAULT_TOP_K):
        """
        检索与问题最相关的块。

        返回列表，每项是 (块字典, 分数)，按分数从高到低。
        分数是 BM25 原始分，没有归一化 —— 不同问题之间的绝对值不可直接比较，
        但同一问题内的排序是有意义的。
        """
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)

        # 按分数排序，取前 top_k
        ranked = sorted(
            zip(self.chunks, scores),
            key=lambda pair: pair[1],
            reverse=True,
        )
        return [(chunk, float(score)) for chunk, score in ranked[:top_k]]


# ============================================================
# 第 4 步：把检索结果拼成给模型的上下文
# ============================================================

def build_context(results, max_chars=3000):
    """
    把检索到的块拼成一段上下文，每块标注编号和出处。

    标注出处不是可选项：RAG 的输出必须能指回原文，
    用户看不到出处就无法判断该不该相信这个答案。
    """
    parts = []
    used = 0
    for number, (chunk, _score) in enumerate(results, start=1):
        piece = f"[片段 {number}] 出处：{chunk['source']}\n{chunk['text']}"
        if used + len(piece) > max_chars:
            break
        parts.append(piece)
        used += len(piece)
    return "\n\n---\n\n".join(parts)


SYSTEM_PROMPT = """你是一个严格基于资料回答问题的助手。

规则：
1. 只使用提供的资料回答。资料里没有的信息，直接回答「资料中未提及」。
2. 绝对不要根据常识或猜测补充资料里没有的内容。
3. 回答后在末尾标注引用的片段编号，格式：[来源: 片段 N]
4. 回答简洁，不要重复问题。"""


def build_answer_prompt(question, context):
    return f"""请根据下面的资料回答问题。

资料：
{context}

问题：{question}"""


def answer(question, results, model=None, temperature=0):
    """
    基于检索结果生成回答。

    注意 temperature=0：问答任务要的是稳定和忠实，
    不是创造性。这个参数沿用第 1 课学到的原则。
    """
    context = build_context(results)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_answer_prompt(question, context)},
    ]
    text, usage = llm.chat(messages, model=model, temperature=temperature)
    return text, usage


# ============================================================
# 一站式入口
# ============================================================

def build_index_from_folder(folder, chunk_size=DEFAULT_CHUNK_SIZE, overlap=DEFAULT_OVERLAP):
    """从文件夹直接建好索引，返回 (索引, 所有块, 文档数)。"""
    documents = load_documents(folder)
    chunks = build_chunks(documents, chunk_size, overlap)
    return BM25Index(chunks), chunks, len(documents)


if __name__ == "__main__":
    # 直接运行本文件：看看语料被切成什么样，以及一次检索的实际结果
    project = Path(__file__).resolve().parent
    index, chunks, doc_count = build_index_from_folder(project / "knowledge")

    print("=" * 64)
    print("语料库切块结果")
    print("=" * 64)
    print(f"  文档数：{doc_count}")
    print(f"  切块数：{len(chunks)}")
    lengths = [len(c["text"]) for c in chunks]
    print(f"  块长度：最短 {min(lengths)} / 平均 {sum(lengths)//len(lengths)} / 最长 {max(lengths)}")
    print()
    for chunk in chunks[:3]:
        preview = chunk["text"][:70].replace("\n", " ")
        print(f"  [{chunk['source']} #{chunk['position']}] {preview}...")
    print()

    print("=" * 64)
    print("试一次检索")
    print("=" * 64)
    query = "一个 Agent 由哪几个部分组成"
    print(f"  问题：{query}\n")
    for number, (chunk, score) in enumerate(index.search(query, top_k=3), start=1):
        print(f"  第 {number} 名  分数 {score:.3f}  出处 {chunk['source']} #{chunk['position']}")
        print(f"    {chunk['text'][:80].replace(chr(10), ' ')}...")
        print()
