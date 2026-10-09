"""
Agent 中间件模块。

中间件是插在 Agent 执行流程关键节点上的钩子，本项目用了三个：

  1. monitor_tool      —— 包裹每一次工具调用，负责打日志、捕获异常，
                          并在 fill_context_for_report 被调用时切换"报告模式"。
  2. log_before_model  —— 每次调模型之前，打印当前消息列表的概况，方便排查上下文。
  3. report_prompt_switch —— 动态决定这次用哪套系统提示词（普通问答 / 报告生成）。

这三个都是通过装饰器注册的，只要在 create_agent(middleware=[...]) 里传进去即可生效。
"""

from typing import Callable

from langchain.agents import AgentState
from langchain.agents.middleware import (
    ModelRequest,
    before_model,
    dynamic_prompt,
    wrap_tool_call,
)
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import BaseMessage, ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Command

from utils.logger_handler import logger
from utils.prompt_loader import load_report_prompts, load_system_prompts


def _preview_message(message: BaseMessage, max_len: int = 200) -> str:
    """
    把一条消息压成适合打日志的一行文本。

    为什么要单独写这个函数：AIMessage 在"发起工具调用"时，content 往往是空串，
    某些模型（如 Claude）的 content 还可能是内容块列表而不是字符串，
    直接 `.strip()` 会抛 AttributeError。这里统一做了兜底和截断。

    :param message: 任意 LangChain 消息对象
    :param max_len: 最大保留字符数，超出部分截断，避免日志被长文本刷屏
    :return: 可直接拼进日志的单行描述
    """
    content = message.content

    if content is None:
        content = ""
    elif not isinstance(content, str):
        # 内容块列表等非字符串形态，转成字符串展示
        content = str(content)

    content = content.strip()

    if not content:
        # 没有文本，但可能有工具调用意图，这个信息对排查很关键，不能丢
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            names = [tc.get("name") for tc in tool_calls]
            return f"<无文本内容，发起工具调用: {names}>"
        return "<无文本内容>"

    if len(content) > max_len:
        content = content[:max_len] + "…（已截断）"

    return content


@wrap_tool_call
def monitor_tool(
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], ToolMessage | Command],
) -> ToolMessage | Command:
    """
    工具调用监控中间件。

    包在真实工具外面：先记录"要调谁、传什么参"，再放行给 handler 执行，
    执行成功/失败都记一笔，最后处理报告模式的上下文切换。

    :param request: 本次工具调用请求，包含工具名、参数、运行时上下文
    :param handler: 真正执行工具的处理器，必须调用它才会真正执行
    :return: 工具执行结果
    """
    logger.info(f"[tool_monitor]执行工具: {request.tool_call['name']}")
    logger.info(f"[tool_monitor]传入参数: {request.tool_call['args']}")

    try:
        result = handler(request)
        logger.info(f"[tool_monitor]工具: {request.tool_call['name']}调用成功")

        # 报告模式的开关：Agent 一旦调用了 fill_context_for_report，
        # 就把 report 标记写进运行时上下文，report_prompt_switch 会读它来换提示词。
        # 注意必须写在 handler 之后，否则工具本身报错时也会误切到报告提示词。
        if request.tool_call['name'] == "fill_context_for_report":
            request.runtime.context['report'] = True
            logger.info("[tool_monitor]检测到 fill_context_for_report，已切换至报告生成模式")

        return result
    except Exception as e:
        # 记完日志再原样抛出，让 Agent 自己知道这步失败了，而不是被静默吞掉
        logger.error(f"[tool_monitor]工具{request.tool_call['name']}调用失败，原因: {str(e)}")
        raise e


@before_model
def log_before_model(
    state: AgentState,
    runtime: Runtime,
):
    """
    模型调用前日志中间件。

    每轮模型调用前都能看到"当前上下文里有多少条消息、最后一条是什么"，
    排查多轮工具调用时非常有用。

    :param state: 当前 Agent 状态，messages 是完整的消息列表
    :param runtime: 运行时对象（此处未使用，但中间件签名要求保留）
    :return: None 表示不修改状态，直接放行
    """
    messages = state['messages']

    logger.info(f"[log_before_model]即将调用模型，带有{len(messages)}条消息")

    if messages:
        last_message = messages[-1]
        logger.info(
            f"[log_before_model]{type(last_message).__name__} | {_preview_message(last_message)}"
        )

    return None


@dynamic_prompt
def report_prompt_switch(request: ModelRequest):
    """
    动态提示词中间件：决定本次调用使用哪套系统提示词。

    默认用 main_prompt.txt；一旦 report 标记被点亮（用户要生成使用报告），
    就切换到 report_prompt.txt，该提示词里只保留报告生成相关的工具说明和输出格式要求。

    :param request: 模型调用请求，可通过 request.runtime.context 读到运行时上下文
    :return: 本次要使用的系统提示词字符串
    """
    is_report = request.runtime.context.get("report", False)

    if is_report:
        # 原代码这里是一句 print("*"*10)，会把非日志内容混进控制台，改为走 logger
        logger.info("[report_prompt_switch]检测到报告模式，切换为报告生成提示词")
        return load_report_prompts()

    return load_system_prompts()
