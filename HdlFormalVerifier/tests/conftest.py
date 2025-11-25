"""
pytest 配置文件
自动设置 Python 路径，使 tests 可以导入 src
"""
import sys
import os
from pathlib import Path

# 获取项目根目录（conftest.py 的父目录的父目录）
project_root = Path(__file__).parent.parent.absolute()

# 添加到 Python 路径
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

print(f"✓ Added to sys.path: {project_root}")