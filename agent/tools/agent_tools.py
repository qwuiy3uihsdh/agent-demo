"""
Agent 工具集模块。

这里定义的每个 @tool 函数都是主 Agent 可以自主调用的"能力"。
Agent 看不到这些函数的具体实现，只能看到函数名、参数和 description，
因此 **description 写得准不准，直接决定了模型会不会在正确的时机调用它**。

注意：除 rag_summarize 外，天气/位置/用户ID/月份这几个都是 mock 实现
（返回随机值或固定字符串），只是为了让 demo 流程跑通，接入真实系统时替换即可。
"""

import os
import random

from langchain_core.tools import tool

from rag.rag_service import RagSummarizeService
from utils.config_handler import agent_conf
from utils.logger_handler import logger
from utils.path_tool import get_abs_path


# ---------------------------------------------------------------------------
# RAG 服务采用懒加载
#
# 修正点：原代码在模块顶层直接 `rag_summarize_service = RagSummarizeSevice()`，
# 只要有人 import 这个模块，就会立刻去连 Chroma、初始化 embedding 模型，
# 既拖慢启动，也会在没配 DASHSCOPE_API_KEY 时直接崩在 import 阶段。
# 改成第一次真正调用该工具时才创建，失败也只影响这一个工具。
# ---------------------------------------------------------------------------
_rag_summarize_service: RagSummarizeService | None = None


def _get_rag_summarize_service() -> RagSummarizeService:
    """获取（首次调用时创建）全局唯一的 RAG 服务实例。"""
    global _rag_summarize_service
    if _rag_summarize_service is None:
        logger.info("[rag_summarize]首次调用，正在初始化 RAG 服务（向量库 + embedding 模型）")
        _rag_summarize_service = RagSummarizeService()
    return _rag_summarize_service


@tool(description=(
    "从向量存储中检索参考资料，入参 query 为贴合用户问题的核心检索词（纯文本字符串）。"
    "凡涉及产品保养、故障排查、使用技巧、环境适配、选购对比等专业问题，"
    "必须先调用 rag_summarize 检索资料，再基于检索结果作答"
))
def rag_summarize(query: str) -> str:
    """检索专业知识库并总结，返回一段基于参考资料的答复。"""
    return _get_rag_summarize_service().rag_summarize(query=query)


@tool(description="获取指定城市的天气，以消息字符串形式返回")
def get_weather(city: str) -> str:
    """（mock）获取指定城市天气。"""
    return f"城市{city}天气为晴天，气温26摄氏度，空气湿度60%，南风1级，最近6小时降雨概率低"


@tool(description="获取用户所在城市的名称，以纯字符串形式返回")
def get_user_location() -> str:
    """（mock）获取用户所在城市。"""
    return random.choice(["深圳", "合肥", "杭州"])


@tool(description="获取用户的ID，以纯字符串形式返回")
def get_user_id() -> str:
    """（mock）获取用户 ID。"""
    return random.choice(["1001", "1002", "1003", "1004", "1005", "1006", "1007", "1008", "1009", "1010"])


@tool(description="获取当前月份，以纯字符串形式返回")
def get_current_month() -> str:
    """（mock）获取当前月份，格式 YYYY-MM。"""
    return random.choice(
        ["2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06",
         "2025-07", "2025-08", "2025-09", "2025-10", "2025-11", "2025-12"]
    )


@tool(description="从外部系统中获取指定用户在指定月份的使用记录，以纯字符串形式返回，如果未检索到返回空字符串")
def fetch_external_data(user_id: str, month: str) -> str:
    """（mock）从 CSV 中读取指定用户、指定月份的使用记录。"""
    # 首次调用时才真正读盘，之后复用内存里的 external_data
    generate_external_data()

    try:
        return external_data[user_id][month]
    except KeyError:
        logger.warning(f"[fetch_external_data]未能检索到用户{user_id}在月份{month}的使用记录数据")
        return ""


# 形如 {user_id: {month: {"特征":..., "效率":..., "耗材":..., "对比":...}}}
external_data = {}


def generate_external_data():
    """
    把 data/external/records.csv 读进内存（只读一次）。

    CSV 各列含义：用户ID, 特征, 效率, 耗材, 对比, 月份。
    """
    if not external_data:
        external_data_path = get_abs_path(agent_conf["external_data_path"])

        if not os.path.exists(external_data_path):
            raise FileNotFoundError(f"外部数据{external_data_path}不存在")

        with open(external_data_path, "r", encoding="utf-8") as f:
            # 跳过表头
            for line in f.readlines()[1:]:
                arr: list[str] = line.strip().split(",")

                if len(arr) < 6:
                    # 脏数据直接跳过，避免下标越界把整个流程带崩
                    logger.warning(f"[generate_external_data]跳过格式异常的行: {line.strip()}")
                    continue

                # CSV 里的字段被引号包裹，这里统一去掉
                user_id: str = arr[0].replace('"', "")
                feature: str = arr[1].replace('"', "")
                efficiency: str = arr[2].replace('"', "")
                consumables: str = arr[3].replace('"', "")
                comparison: str = arr[4].replace('"', "")
                time: str = arr[5].replace('"', "")

                if user_id not in external_data:
                    external_data[user_id] = {}

                external_data[user_id][time] = {
                    "特征": feature,
                    "效率": efficiency,
                    "耗材": consumables,
                    "对比": comparison,
                }


@tool(description="无入参无返回值，调用后触发中间件自动为报告生成的场景动态注入上下文信息，为后续提示词切换提供上下文信息")
def fill_context_for_report() -> str:
    """
    报告生成的"信号工具"。

    它本身不做事，真正的逻辑在 middleware.monitor_tool 里：
    拦截到这次调用后会把 runtime.context["report"] 置为 True，
    于是后续的模型调用会切换到 report_prompt.txt 那套系统提示词。
    """
    return "fill_context_for_report已调用"


if __name__ == "__main__":
    print(fetch_external_data("1005", "2025-09"))
