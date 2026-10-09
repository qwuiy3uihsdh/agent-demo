"""
提示词加载模块。

项目里的提示词不写在代码里，而是放在 prompts/ 目录下的 .txt 文件中，
由 config/prompts.yml 配置相对路径。这样改提示词不用动 Python 代码。

一共三类提示词：
  - main_prompt.txt     主 Agent 的系统提示词（定义人设、工具清单、调用规则）
  - report_prompt.txt   生成使用报告时动态切换过去的系统提示词
  - rag_summarize.txt   RAG 检索后"根据参考资料总结"的子链提示词
"""

from utils.config_handler import prompts_conf
from utils.path_tool import get_abs_path
from utils.logger_handler import logger


def _load_prompt_file(conf_key: str, scene_name: str) -> str:
    """
    从配置中取出路径并读取提示词文本（内部公共实现）。

    :param conf_key: config/prompts.yml 中的配置项名
    :param scene_name: 出错时打印的场景名，便于定位
    :return: 提示词全文
    """
    try:
        prompt_path = get_abs_path(prompts_conf[conf_key])
    except KeyError as e:
        logger.error(f"[{scene_name}]配置项{conf_key}缺失")
        raise e

    try:
        # 用 with 确保文件句柄及时释放
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logger.error(f"[{scene_name}]解析提示词{prompt_path}出错: {str(e)}")
        raise e


def load_system_prompts() -> str:
    """加载主 Agent 的系统提示词。"""
    return _load_prompt_file("main_prompt_path", "load_system_prompts")


def load_rag_summarize_prompts() -> str:
    """加载 RAG 总结子链的提示词。"""
    return _load_prompt_file("rag_summarize_prompt_path", "load_rag_summarize_prompts")


def load_report_prompts() -> str:
    """加载报告生成场景的系统提示词。"""
    return _load_prompt_file("report_prompt_path", "load_report_prompts")


if __name__ == "__main__":
    print(load_report_prompts())
