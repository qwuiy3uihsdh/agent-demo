"""
路径工具模块。

项目中所有对配置文件、提示词、知识库的读取都必须经过这里，
以保证无论从哪个目录启动程序（例如从项目根目录跑 app.py，
或从 rag/ 目录直接跑某个脚本），拿到的都是同一个绝对路径。
"""

import os


def get_project_root() -> str:
    """
    获取项目根目录的绝对路径。

    本文件的路径是 <项目根>/utils/path_tool.py，
    因此向上一级是 utils/，再向上一级才是项目根。

    :return: 项目根目录的绝对路径，例如 "D:/code/vscode/agent-demo"
    """
    cur_file = os.path.abspath(__file__)
    cur_file_dir = os.path.dirname(cur_file)
    project_root = os.path.dirname(cur_file_dir)

    return project_root


def get_abs_path(relative_path: str) -> str:
    """
    把项目内的相对路径转换成绝对路径。

    :param relative_path: 相对于项目根目录的路径，例如 "config/rag.yml"
    :return: 转换后的绝对路径
    """
    project_root = get_project_root()
    return os.path.join(project_root, relative_path)


if __name__ == "__main__":
    # 修正点：原代码写的是 "config\config.py"，其中 \c 是无效转义序列，
    # Python 3.12 起会触发 SyntaxWarning。这里用 os.path.join 拼接，跨平台也安全。
    print(get_abs_path(os.path.join("config", "rag.yml")))
