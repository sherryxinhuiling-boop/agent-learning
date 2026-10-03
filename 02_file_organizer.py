"""
阶段 0 · 项目 0.1：文件整理脚本
================================================

这个脚本做一件真实有用的事：
    把一个文件夹里的文件，按扩展名分类，复制到不同的子文件夹里。

    sandbox/source/                     sandbox/organized/
        会议纪要.md          →              文档/会议纪要.md
        截图.png             →              图片/截图.png
        数据.csv             →              表格/数据.csv
        笔记.txt             →              文档/笔记.txt

【安全设计】只复制，绝不移动或删除原文件。程序跑完，源目录一个字节都不会变。

【运行方式】在项目目录里：
    source .venv/bin/activate
    python 02_file_organizer.py

第一次运行会自动创建一个演示沙盒（sandbox/），里面有几份假文件，
所以你不需要用自己真实的文件来冒险。

【整理你自己的文件夹】先看清楚它要做什么，再用绝对路径：
    python 02_file_organizer.py /path/to/你的目录 /path/to/输出目录
"""

from pathlib import Path      # 处理路径的现代写法，比字符串拼接可靠得多
import shutil                 # 专门负责复制文件
import sys


# ============================================================
# 配置区：改这里就能调整整理规则
# ============================================================
# 这是一个字典：键是扩展名，值是归类的名字。
# 想加新类型，就在下面加一行，比如 ".pptx": "演示文稿",

CATEGORIES = {
    ".jpg": "图片",
    ".jpeg": "图片",
    ".png": "图片",
    ".gif": "图片",
    ".pdf": "文档",
    ".docx": "文档",
    ".md": "文档",
    ".txt": "文档",
    ".xlsx": "表格",
    ".csv": "表格",
    ".pptx": "演示文稿",
    ".mp3": "音频",
    ".mp4": "视频",
    ".zip": "压缩包",
    ".py": "代码",
    ".js": "代码",
}

OTHER = "其他"        # 不在上面清单里的扩展名，都归到这里


# ============================================================
# 第 1 个函数：判断一个文件属于哪一类
# ============================================================
def category_of(file_path):
    """
    输入一个文件路径，返回它应该被归入的类别名。

    .suffix 能拿到扩展名，比如 Path("a.png").suffix 得到 ".png"
    .lower() 转成小写，这样 ".PNG" 和 ".png" 能被当成同一类
    """
    extension = file_path.suffix.lower()
    # 字典的 .get(键, 默认值)：有就用对应的值，没有就用默认值
    return CATEGORIES.get(extension, OTHER)


# ============================================================
# 第 2 个函数：创建演示沙盒
# ============================================================
def prepare_demo(source_dir):
    """
    在 source_dir 里放几份演示文件，方便安全地测试脚本。
    如果目录里已经有文件，就什么都不做。
    """
    source_dir.mkdir(parents=True, exist_ok=True)

    demo_files = [
        "会议纪要.md",
        "学习笔记.txt",
        "季度数据.csv",
        "项目截图.png",
        "产品原型.jpg",
        "研究报告.pdf",
        "培训视频.mp4",
        "背景音乐.mp3",
        "源码示例.py",
        "没有扩展名的文件",          # 用来测试兜底逻辑
        "奇怪的格式.xyz",            # 也不在分类表里
    ]

    created = 0
    for file_name in demo_files:
        target = source_dir / file_name      # / 运算符用来拼路径
        if not target.exists():
            target.write_text(f"这是演示文件：{file_name}\n", encoding="utf-8")
            created += 1

    if created > 0:
        print(f"已创建演示沙盒：{source_dir}")
        print(f"（生成了 {created} 个假文件，可以放心测试）\n")


# ============================================================
# 第 3 个函数：核心整理逻辑
# ============================================================
def organize(source_dir, target_dir):
    """
    把 source_dir 里的文件按类别复制到 target_dir 下的子文件夹。
    返回一个统计字典，记录每个类别复制了几个文件。
    """
    # 先检查源目录是否真的存在，不存在就直接返回
    if not source_dir.is_dir():
        print(f"错误：找不到目录 {source_dir}")
        return None

    stats = {}          # 用来记录「类别 → 复制了几个文件」
    copied = 0
    skipped = 0

    # .iterdir() 会列出目录里的所有条目；sorted() 让输出顺序稳定，方便阅读
    for item in sorted(source_dir.iterdir()):

        # 跳过子目录，只处理文件。
        # 注意：这里用 continue 而不是 break —— break 会直接结束循环，
        # 那样后面的文件就都不处理了。
        if not item.is_file():
            print(f"  跳过目录：{item.name}")
            skipped += 1
            continue

        # 跳过隐藏文件（以 . 开头的，比如 .DS_Store）
        if item.name.startswith("."):
            print(f"  跳过隐藏文件：{item.name}")
            skipped += 1
            continue

        # 决定去哪个子文件夹
        category = category_of(item)
        dest_dir = target_dir / category
        dest_dir.mkdir(parents=True, exist_ok=True)

        # 处理重名：如果目标文件已存在，就加个序号，不覆盖
        dest_file = dest_dir / item.name
        counter = 1
        while dest_file.exists():
            dest_file = dest_dir / f"{item.stem}_{counter}{item.suffix}"
            counter += 1

        try:
            # copy2 会连同文件的修改时间等属性一起复制
            shutil.copy2(item, dest_file)
            print(f"  {item.name}  →  {category}/{dest_file.name}")
            copied += 1
            stats[category] = stats.get(category, 0) + 1
        except OSError as error:
            # 万一某个文件复制失败（权限、磁盘满等），
            # 不要让整个程序崩溃，打印原因后继续处理下一个
            print(f"  复制失败：{item.name}（原因：{error}）")
            skipped += 1

    return {"stats": stats, "copied": copied, "skipped": skipped}


# ============================================================
# 主流程
# ============================================================
def main():
    # 命令行参数：如果用户传了路径就用他的，否则用默认的沙盒目录
    if len(sys.argv) >= 3:
        source_dir = Path(sys.argv[1]).expanduser().resolve()
        target_dir = Path(sys.argv[2]).expanduser().resolve()
    else:
        project_dir = Path(__file__).parent          # 脚本所在目录
        source_dir = project_dir / "sandbox" / "source"
        target_dir = project_dir / "sandbox" / "organized"
        prepare_demo(source_dir)

    print("=" * 56)
    print(f"源目录：{source_dir}")
    print(f"目标目录：{target_dir}")
    print("=" * 56)
    print("\n开始整理（只复制，不动原文件）：")

    result = organize(source_dir, target_dir)

    if result is None:
        print("\n整理中止。请检查上面的源目录路径是否正确。")
        return

    print("\n" + "=" * 56)
    print("整理完成")
    print("=" * 56)
    print(f"共复制 {result['copied']} 个文件，跳过 {result['skipped']} 个\n")

    if result["stats"]:
        print("按类别统计：")
        for category, count in sorted(result["stats"].items()):
            print(f"  {category}：{count} 个")
    else:
        print("（没有文件被复制）")

    print(f"\n原文件仍然完好地在：{source_dir}")


# 这行判断的意思是：只有「直接运行这个文件」时才执行 main()。
# 如果这个文件被别的程序 import，就不会自动跑起来。
if __name__ == "__main__":
    main()
