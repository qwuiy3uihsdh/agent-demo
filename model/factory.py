"""
模型工厂模块。

用工厂模式把"聊天模型"和"向量化模型"的创建过程包起来，
其他模块只 import 模块级的两个实例即可，不用关心构造参数：

    from model.factory import chat_model, embedddings_model

两者都通过 DashScope（阿里云百炼）的 OpenAI 兼容接口访问，
API Key 从环境变量 DASHSCOPE_API_KEY 读取，不在代码里硬编码。
"""

import os
from abc import ABC, abstractmethod
from typing import Optional

from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from utils.config_handler import rag_conf


def _get_api_key() -> str:
    """
    读取 DashScope API Key。

    原实现直接 `os.environ["DASHSCOPE_API_KEY"]`，缺变量时只抛一个 KeyError，
    看不出到底缺的是什么。这里换成带明确指引的报错，启动阶段就能发现问题。

    :return: API Key 字符串
    """
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "未读取到环境变量 DASHSCOPE_API_KEY。请先设置该环境变量再启动程序，"
            "例如：export DASHSCOPE_API_KEY=sk-xxxx（Windows CMD: set DASHSCOPE_API_KEY=sk-xxxx）"
        )
    return api_key


class BaseModelFactory(ABC):
    """模型工厂抽象基类：所有具体工厂都要实现 generator()。"""

    @abstractmethod
    def generator(self) -> Optional[BaseChatModel | Embeddings]:
        """创建并返回一个模型实例。"""
        pass


class ChatModelFactory(BaseModelFactory):
    """对话模型工厂，产出 Agent 用来思考、调用工具、生成回答的大模型。"""

    def generator(self) -> Optional[BaseChatModel | Embeddings]:
        return ChatOpenAI(
            # 模型名来自 config/rag.yml，默认 qwen-max
            model=rag_conf["chat_model_name"],
            api_key=_get_api_key(),
            # 走 OpenAI 兼容模式，所以可以直接复用 ChatOpenAI
            base_url="https://ws-slyo0j61dbkx5gs6.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
        )


class EmbeddingsFactory(BaseModelFactory):
    """向量化模型工厂，产出把文本转成向量、供 Chroma 检索用的 embedding 模型。"""

    def generator(self) -> Optional[BaseChatModel | Embeddings]:
        return DashScopeEmbeddings(
            model=rag_conf["embedding_model_name"],  # 按你账号支持填 v2/v3/v4
            dashscope_api_key=_get_api_key(),        # 或不传，靠环境变量 DASHSCOPE_API_KEY
        )


# 模块级单例：进程内只创建一次，避免每次用到都重新建连接
chat_model = ChatModelFactory().generator()
embedddings_model = EmbeddingsFactory().generator()
