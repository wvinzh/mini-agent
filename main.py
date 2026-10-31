"""等价于 `python -m mini_agent`，方便直接 `python main.py --mock` 运行。"""
import sys

from mini_agent.__main__ import main

if __name__ == "__main__":
    sys.exit(main())
