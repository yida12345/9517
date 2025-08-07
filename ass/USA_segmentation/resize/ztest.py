#!/usr/bin/env python3
import os
import sys

def print_tree(root, prefix=''):
    """
    类似 tree 命令，递归打印所有子目录，但每个目录只显示最多两个文件。
    root: 当前处理的目录路径
    prefix: 用于输出的前缀
    """
    try:
        entries = os.listdir(root)
    except PermissionError:
        print(f"{prefix}[权限不足]")
        return

    # 将子项分为目录和文件
    dirs = [e for e in entries if os.path.isdir(os.path.join(root, e))]
    files = [e for e in entries if os.path.isfile(os.path.join(root, e))]

    # 按名称排序
    dirs.sort()
    files.sort()

    # 限制文件数量为前两项
    limited_files = files[:2]

    # 合并：目录先，然后文件
    combined = dirs + limited_files
    total = len(combined)

    for idx, name in enumerate(combined):
        path = os.path.join(root, name)
        connector = '└── ' if idx == total - 1 else '├── '
        print(f"{prefix}{connector}{name}")
        # 如果是目录，递归所有子目录
        if os.path.isdir(path):
            extension = '    ' if idx == total - 1 else '│   '
            print_tree(path, prefix + extension)


def main():
    # 支持命令行指定根目录，默认为当前目录
    root_dir = sys.argv[1] if len(sys.argv) > 1 else '.'
    print(root_dir)
    print_tree(root_dir)

if __name__ == '__main__':
    main()
