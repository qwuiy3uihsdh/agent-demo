"""
日志工具模块。

全项目共用同一个名为 "agent" 的 logger，好处是：
  - 控制台和日志文件用的是同一套格式，方便对照排查；
  - 因为绑定了 FileHandler，控制台里被刷屏刷掉的信息，
    仍然能在 logs/agent_YYYYMMDD.log 里翻到。

级别约定：
  - 控制台（console_level）默认 INFO：只显示关键流程，保持终端清爽；
  - 文件（file_level）默认 DEBUG：连检索到的完整提示词这种大块内容也落盘，
    需要深挖时去 logs/ 目录看。
"""

import datetime
import logging
import os

from utils.path_tool import get_abs_path


# 日志目录固定放在项目根目录下的 logs/，不存在就自动创建
LOG_ROOT = get_abs_path("logs")
os.makedirs(LOG_ROOT, exist_ok=True)

# 统一格式：时间 - logger名 - 级别 - 文件名:行号 - 消息
# 带上"文件名:行号"是为了看到日志就能直接定位到代码位置
DEFAULT_LOG_FORMAT = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
)


def get_logger(
        name: str = "agent",
        console_level: int = logging.INFO,
        file_level: int = logging.DEBUG,
        log_file: str = None,
) -> logging.Logger:
    """
    获取（或创建）一个 logger。

    同名 logger 在同一个进程里只会被配置一次：如果它已经带了 handler，
    就直接复用返回，避免重复 addHandler 导致同一条日志被打印多遍。

    :param name: logger 名称，默认 "agent"
    :param console_level: 控制台输出的最低级别
    :param file_level: 日志文件写入的最低级别
    :param log_file: 日志文件路径；不传则按 "logs/{name}_YYYYMMDD.log" 自动生成
    :return: 配置好的 Logger 实例
    """
    logger = logging.getLogger(name=name)
    # logger 自身设为 DEBUG，真正的过滤交给下面两个 handler，
    # 这样控制台和文件可以用不同的级别
    logger.setLevel(logging.DEBUG)

    # 已经初始化过（例如模块被多次 import），直接返回，防止日志重复输出
    if logger.handlers:
        return logger

    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(DEFAULT_LOG_FORMAT)
    logger.addHandler(console_handler)

    # 未指定文件路径时，按天生成日志文件，避免单个文件无限膨胀
    if not log_file:
        log_file = os.path.join(LOG_ROOT, f"{name}_{datetime.datetime.now().strftime('%Y%m%d')}.log")

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(file_level)
    file_handler.setFormatter(DEFAULT_LOG_FORMAT)
    logger.addHandler(file_handler)

    return logger


# 模块级单例：其他模块直接 `from utils.logger_handler import logger` 即可使用
logger = get_logger()


if __name__ == "__main__":
    # 自查用：应当能在控制台看到 info/error/warning，在日志文件里额外看到 debug
    logger.info("信息日志")
    logger.error("错误日志")
    logger.warning("警告日志")
    logger.debug("调试日志")
