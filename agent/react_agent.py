"""
主 Agent 模块。

用 LangChain 1.x 的 create_agent 组装出一个 ReAct 风格的 Agent：
模型 + 工具集 + 中间件，形成一个"思考 → 调用工具 → 观察结果 → 再思考"的循环，
直到模型给出最终回答为止。

对外只暴露一个 execute_stream()，以流式方式把回答吐给调用方（app.py 的聊天界面）。
"""

from langchain.agents import create_agent
from langchain_core.messages import AIMessage

from agent.tools.agent_tools import (
    fetch_external_data,
    fill_context_for_report,
    get_current_month,
    get_user_id,
    get_user_location,
    get_weather,
    rag_summarize,
)
from agent.tools.middleware import log_before_model, monitor_tool, report_prompt_switch
from model.factory import chat_model
from utils.logger_handler import logger
from utils.prompt_loader import load_system_prompts


class ReactAgent:
    """扫地机器人智能客服 Agent。"""

    def __init__(self):
        """
        组装 Agent。

        - model：底层对话模型，负责思考与决定调用哪个工具
        - tools：可自主调用的能力清单，模型只能按 description 选择，看不到实现
        - middleware：插在流程关键节点的钩子，见 middleware.py
        - system_prompt：定义人设、工具清单与调用规则
        """
        self.agent = create_agent(
            model=chat_model,
            system_prompt=load_system_prompts(),
            tools=[rag_summarize, get_weather, get_user_location,
                   get_user_id, get_current_month, fetch_external_data, fill_context_for_report],
            middleware=[monitor_tool, log_before_model, report_prompt_switch],
        )

    def execute_stream(self, query: str):
        """
        执行一次问答，并以流式方式逐段产出**最终回答**的文本。

        关于 stream_mode 的选型（这是本项目最关键的一处修正）：
        ------------------------------------------------------------------
        原实现用的是 stream_mode="values"，该模式在**每个节点执行完**都会把
        整份 state 的最后一条消息吐出来。于是一次带工具调用的问答会依次产出：

            HumanMessage(用户原问题) → AIMessage(思考过程) → ToolMessage(工具原始返回值) → AIMessage(最终回答)

        而 app.py 是 `write_stream()` 直接把这些内容全部拼进同一个气泡的，
        结果就是聊天框里混进了用户自己的问题、以及"合肥""2025-07"这种裸工具结果。

        改成 stream_mode="messages" 后，拿到的是 (消息片段, 元数据) 二元组，
        粒度是 token 级增量，配合下面的过滤即可只保留要展示给用户的内容。
        ------------------------------------------------------------------

        :param query: 用户输入的问题
        :return: 生成器，逐段产出回答文本
        """
        input_dict = {
            "messages": [
                {"role": "user", "content": query},
            ]
        }

        # context 是本次运行的运行时上下文，report=False 表示默认走普通问答提示词，
        # 后续一旦 Agent 调用了 fill_context_for_report，中间件会把它改成 True。
        for chunk, metadata in self.agent.stream(
            input_dict,
            stream_mode="messages",
            context={"report": False},
        ):
            # 过滤一：工具节点的输出（ToolMessage）是给模型看的原始结果，
            # 不应该直接展示给用户
            if metadata.get("langgraph_node") == "tools":
                continue

            # 过滤二：只保留 AI 产出的内容，排除 HumanMessage / ToolMessage / SystemMessage
            # （AIMessageChunk 是 AIMessage 的子类，所以这一条同时覆盖流式与非流式两种情形）
            if not isinstance(chunk, AIMessage):
                continue

            # 过滤三：附带 tool_calls 的 AIMessage 是"决定调工具"的那一步，
            # 其正文只是给日志看的思考过程，不混进最终回答
            if getattr(chunk, "tool_calls", None):
                continue

            content = chunk.content

            # content 可能是 None 或内容块列表，非字符串一律跳过，避免写进界面时报错
            if not isinstance(content, str) or not content:
                continue

            yield content


if __name__ == "__main__":
    # 命令行自查：不启动 Streamlit，直接看流式输出效果
    try:
        for piece in ReactAgent().execute_stream("给我生成我的使用报告"):
            print(piece, end="", flush=True)
    except Exception as e:
        logger.error(f"[ReactAgent]执行失败: {str(e)}", exc_info=True)
        raise
