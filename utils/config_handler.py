"""
配置加载模块。

统一读取 config/ 目录下的 4 个 YAML 配置文件，并在模块导入时各加载一次，
其他模块直接 import 对应的字典变量即可，不用重复读盘：

    from utils.config_handler import rag_conf, chroma_conf

注意：这些字典在进程启动时就固定下来了。改了 YAML 需要重启程序才生效。
"""

import yaml

from utils.path_tool import get_abs_path


def _load_yaml(config_path: str, encoding: str = "utf-8") -> dict:
    """
    读取并解析一个 YAML 文件（内部公共实现）。

    :param config_path: YAML 文件的绝对路径
    :param encoding: 文件编码
    :return: 解析后的字典
    """
    # 用 with 确保文件句柄及时释放
    with open(config_path, "r", encoding=encoding) as f:
        # FullLoader 支持 YAML 的大部分常用语法，同时比 UnsafeLoader 安全
        return yaml.load(f, Loader=yaml.FullLoader)


def load_rag_config(config_path: str = get_abs_path("config/rag.yml"), encoding: str = "utf-8") -> dict:
    """加载模型配置（对话模型名、embedding 模型名）。"""
    return _load_yaml(config_path, encoding)


def load_chroma_config(config_path: str = get_abs_path("config/chroma.yml"), encoding: str = "utf-8") -> dict:
    """加载向量库配置（集合名、持久化目录、分片参数、检索条数等）。"""
    return _load_yaml(config_path, encoding)


def load_prompts_config(config_path: str = get_abs_path("config/prompts.yml"), encoding: str = "utf-8") -> dict:
    """加载提示词文件路径配置。"""
    return _load_yaml(config_path, encoding)


def load_agent_config(config_path: str = get_abs_path("config/agent.yml"), encoding: str = "utf-8") -> dict:
    """加载 Agent 相关配置（外部数据文件路径等）。"""
    return _load_yaml(config_path, encoding)


# ---- 模块级单例：进程内只读一次盘 ----
rag_conf = load_rag_config()
chroma_conf = load_chroma_config()
prompts_conf = load_prompts_config()
agent_conf = load_agent_config()


if __name__ == "__main__":
    print(rag_conf["chat_model_name"])
