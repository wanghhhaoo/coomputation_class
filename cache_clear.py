#!/usr/bin/env python3
"""清空 __pycache__ 缓存脚本"""

import os
import shutil
import sys


def clear_pycache(root_dir="."):
    count = 0
    for dirpath, dirnames, _ in os.walk(root_dir):
        for dirname in dirnames:
            if dirname == "__pycache__":
                full_path = os.path.join(dirpath, dirname)
                shutil.rmtree(full_path)
                print(f"已删除: {full_path}")
                count += 1
    print(f"\n完成！共删除 {count} 个 __pycache__ 目录。")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "."
    print(f"正在清理目录: {os.path.abspath(target)}\n")
    clear_pycache(target)