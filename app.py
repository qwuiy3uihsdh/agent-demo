"""
应用入口：基于 Streamlit 的扫地机器人智能客服聊天界面。

启动方式（在项目根目录执行）：
    streamlit run app.py

说明：Agent 本身是无状态的，每轮问答只把**当前这一句**用户输入交给它，
历史对话仅用于界面展示，模型并不会看到之前的上下文
（若要支持多轮记忆，需要给 create_agent 配 checkpointer 并传入 thread_id）。
"""

import streamlit as sl

from agent.react_agent import ReactAgent
from utils.logger_handler import logger

sl.title("扫地机器人智能客服")
sl.divider()

# session_state 是 Streamlit 里"每个浏览器会话各存一份"的状态容器，
# 每次交互（rerun）都会重新执行整个脚本，所以这些对象必须存进 session_state，
# 否则每一轮都会被重新创建：Agent 会反复初始化、历史消息会被清空。
if "agent" not in sl.session_state:
    sl.session_state["agent"] = ReactAgent()

if "message" not in sl.session_state:
    # 形如 [{"role": "user"/"assistant", "content": "..."}]
    sl.session_state["message"] = []

# 重放历史消息：脚本每次 rerun 都会清空页面，需要把之前的对话重新画一遍
for message in sl.session_state["message"]:
    sl.chat_message(message["role"]).write(message["content"])

prompt = sl.chat_input()

if prompt:
    # 先立刻把用户这句话显示出来并记入历史
    sl.chat_message("user").write(prompt)
    sl.session_state["message"].append({"role": "user", "content": prompt})

    # 生成回答期间失败不应让整个页面崩掉，这里统一兜住并给出提示
    full_response = None
    try:
        with sl.spinner("智能客服思考中"):
            # execute_stream() 产出的是纯文本增量片段，
            # write_stream() 会边收边渲染，并把拼接后的完整字符串返回
            res_stream = sl.session_state["agent"].execute_stream(prompt)
            full_response = sl.chat_message("assistant").write_stream(res_stream)
    except Exception as e:
        logger.error(f"[app]生成回答失败: {str(e)}", exc_info=True)
        sl.error(f"抱歉，回答生成失败：{e}")

    # 只有成功拿到回答才入库，避免把半截内容记进历史
    if full_response:
        sl.session_state["message"].append({"role": "assistant", "content": full_response})
