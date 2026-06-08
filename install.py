#!/usr/bin/env python3
"""
chexie-knowledge 跨平台一键安装脚本
支持 Windows / macOS / Linux

用法:
    python install.py                      # 使用默认路径
    python install.py --root D:\chexie     # Windows 自定义路径
    python install.py --root ~/chexie      # macOS / Linux 自定义路径
    python install.py --skip-data          # 跳过数据下载（已有数据时）
    python install.py --skip-dep           # 跳过依赖安装
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

# ── 配置 ────────────────────────────────────────────────

GITHUB_RELEASE = (
    "https://github.com/3469915488-oss/chexie.skill/"
    "releases/download/v3.0.0/chexie_data.tar.gz"
)
REQUIRED_FILES = ["faiss_index.bin", "faiss_meta.jsonl", "chexie_fts.db"]


def default_root() -> Path:
    """根据操作系统返回默认安装路径。"""
    system = platform.system()
    if system == "Windows":
        return Path(os.environ.get("APPDATA", str(Path.home()))) / "chexie-knowledge"
    elif system == "Darwin":
        return Path.home() / "Library" / "Application Support" / "chexie-knowledge"
    else:
        return Path("/opt/chexie-knowledge")


def print_step(msg: str) -> None:
    print(f"\n  → {msg}")


def check_python() -> bool:
    """检查 Python 版本 >= 3.9。"""
    v = sys.version_info
    if v.major < 3 or (v.major == 3 and v.minor < 9):
        print(f"  ⚠ Python 3.9+ 需要，当前版本: {v.major}.{v.minor}")
        return False
    print(f"  ✓ Python {v.major}.{v.minor}.{v.micro}")
    return True


def install_deps(root: Path) -> bool:
    """安装 Python 依赖。"""
    req_file = root / "requirements.txt"
    if not req_file.exists():
        print("  ⚠ 未找到 requirements.txt，跳过依赖安装")
        return True

    print_step("安装 Python 依赖...")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-r", str(req_file)],
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        print("  ✓ 依赖安装完成")
        return True
    except subprocess.CalledProcessError:
        print("  ✗ 依赖安装失败，请手动运行:")
        print(f"    pip install -r {req_file}")
        return False


def download_data(root: Path) -> bool:
    """下载并解压数据包。"""
    if all((root / f).exists() for f in REQUIRED_FILES):
        print_step("数据文件已存在，跳过下载")
        return True

    print_step(f"下载数据包...\n    {GITHUB_RELEASE}")

    tarball = root / "chexie_data.tar.gz"
    root.mkdir(parents=True, exist_ok=True)

    try:
        urllib.request.urlretrieve(GITHUB_RELEASE, str(tarball))
    except Exception as e:
        print(f"  ✗ 下载失败: {e}")
        print("  请检查网络连接，或手动从 GitHub Release 下载后解压到目标目录")
        return False

    print_step("解压数据包...")
    try:
        with tarfile.open(tarball, "r:gz") as tar:
            tar.extractall(path=str(root))
        tarball.unlink()
        print("  ✓ 数据解压完成")
    except Exception as e:
        print(f"  ✗ 解压失败: {e}")
        if tarball.exists():
            print(f"  请手动解压: {tarball}")
        return False

    return True


def copy_scripts(root: Path) -> bool:
    """Copy scripts/ directory to install root."""
    src = Path("scripts")
    dst = root / "scripts"
    if not src.exists():
        print("  ⚠ 未找到 scripts/ 目录（需要从项目仓库运行 install.py）")
        return False
    print_step("复制检索脚本...")
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.is_file() and f.suffix == ".py":
            shutil.copy2(f, dst / f.name)
    print(f"  ✓ 已复制 {sum(1 for f in dst.iterdir() if f.suffix == '.py')} 个脚本")
    return True


def build_bm25(root: Path) -> bool:
    """构建 BM25 索引（可选）。"""
    bm25_file = root / "bm25_index.jsonl"
    if bm25_file.exists():
        print("  BM25 索引已存在")
        return True

    build_script = root / "build_bm25.py"
    if not build_script.exists():
        print("  ⚠ 未找到 build_bm25.py，跳过 BM25 构建")
        return True

    print_step("构建 BM25 索引（约 2-3 分钟）...")
    try:
        subprocess.check_call(
            [sys.executable, str(build_script)],
            cwd=str(root),
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        print("  ✓ BM25 索引构建完成")
        return True
    except subprocess.CalledProcessError:
        print("  ✗ BM25 构建失败（不影响基础搜索，可稍后重试）")
        return False


def verify(root: Path) -> bool:
    """验证安装完整性。"""
    print_step("验证安装...")
    missing = [f for f in REQUIRED_FILES if not (root / f).exists()]
    if missing:
        print(f"  ✗ 缺少文件: {missing}")
        return False

    # 统计条目数
    meta_path = root / "faiss_meta.jsonl"
    count = 0
    with meta_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1

    index_path = root / "faiss_index.bin"
    size_mb = index_path.stat().st_size / (1024 * 1024)

    print(f"  ✓ 索引条目: {count:,}")
    print(f"  ✓ 索引大小: {size_mb:.1f} MB")
    return True


def print_env_guide(root: Path) -> None:
    """打印环境变量设置指引。"""
    system = platform.system()
    print("\n" + "=" * 56)
    print("  安装完成！请设置以下环境变量：")
    print("=" * 56)

    if system == "Windows":
        print(f"""
  CMD (管理员):
      setx CHEXIE_ROOT "{root}"

  PowerShell:
      [Environment]::SetEnvironmentVariable('CHEXIE_ROOT', '{root}', 'User')
""")
    else:
        shell = os.environ.get("SHELL", "/bin/bash").split("/")[-1]
        rcfile = ".zshrc" if "zsh" in shell else ".bashrc"
        print(f"""
  在 ~/{rcfile} 中添加:
      export CHEXIE_ROOT="{root}"

  然后执行:
      source ~/{rcfile}
""")

    # 模型目录（可选）
    print(f"  （可选）如需自定义模型缓存目录:")
    if system == "Windows":
        print(f'  setx CHEXIE_MODEL_DIR "C:\\models"')
    else:
        print(f'  export CHEXIE_MODEL_DIR="/path/to/models"')
    print()
    print("  验证安装:")
    print(f"    python {root / 'scripts' / 'search_chexie.py'} --info")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="chexie-knowledge 跨平台安装脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python install.py                        # 默认路径
    python install.py --root D:\\chexie       # 自定义路径
    python install.py --skip-data            # 已有数据，只装依赖
    python install.py --no-bm25              # 跳过 BM25 构建
""",
    )
    parser.add_argument("--root", type=str, help="安装根目录")
    parser.add_argument("--skip-data", action="store_true", help="跳过数据下载")
    parser.add_argument("--skip-dep", action="store_true", help="跳过依赖安装")
    parser.add_argument("--no-bm25", action="store_true", help="跳过 BM25 索引构建")
    args = parser.parse_args()

    root = Path(args.root) if args.root else default_root()

    print(f"  chexie-knowledge 安装程序")
    print(f"  OS:      {platform.system()} {platform.release()}")
    print(f"  安装目录: {root}")
    print(f"  Python:  {sys.executable}")

    if not check_python():
        sys.exit(1)

    # 1. 安装依赖
    if not args.skip_dep:
        # 先找 requirements.txt — 可能在当前目录（已 clone 项目）或 root 目录
        if Path("requirements.txt").exists():
            install_deps(Path("."))
        elif (root / "requirements.txt").exists():
            install_deps(root)
        else:
            print("  ⚠ 未找到 requirements.txt，请手动安装依赖")

    # 2. 下载数据
    if not args.skip_data:
        if not download_data(root):
            sys.exit(1)

    # 2.5. 复制脚本
    copy_scripts(root)

    # 3. 构建 BM25
    if not args.no_bm25:
        build_bm25(root)

    # 4. 验证
    if not verify(root):
        print("\n  ⚠ 验证未通过，但可尝试手动修复")
        sys.exit(1)

    # 5. 打印环境变量指引
    print_env_guide(root)

    print("  🎉 全部完成！\n")


if __name__ == "__main__":
    main()
