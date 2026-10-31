"""测试包。运行：python -m unittest discover -s tests -t . -v"""
import logging

# 测试里会故意制造很多错误/重试，屏蔽框架的警告日志让输出更干净
logging.getLogger("mini_agent").setLevel(logging.CRITICAL)
