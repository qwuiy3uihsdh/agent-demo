"""
RAG 总结服务模块。

这里定义的是 Agent 的一个"工具"背后的实现：给一个 query，
先到 Chroma 向量库里召回若干资料片段，再把「query + 参考资料」交给
一个独立的总结子链，产出一段"基于资料"的答复，最终作为工具结果返回给主 Agent。

之所以要做成"检索 + 总结"两段式，而不是把原始片段直接丢给主 Agent，
是为了让 context 更短、更聚焦，同时由子链的提示词严格约束"不许编造"。
"""

import time

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from model.factory import chat_model
from rag.vector_store import VectorStoreService
from utils.logger_handler import logger
from utils.prompt_loader import load_rag_summarize_prompts


class RagSummarizeService(object):
    """
    RAG 检索 + 总结服务。

    构造时会建好向量库连接、检索器、提示词模板和 LangChain 链，
    因此比较重，整个进程里复用同一个实例即可（见 agent/tools/agent_tools.py）。
    """

    def __init__(self):
        self.vector_store = VectorStoreService()
        self.retriever = self.vector_store.get_retriever()
        self.prompt_text = load_rag_summarize_prompts()
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        self.model = chat_model
        self.chain = self._init_chain()

    def _init_chain(self):
        """
        组装 LCEL 链：填充提示词 → 记录提示词 → 调用模型 → 取出纯文本。

        中间插入 _log_prompt 是为了把最终送给模型的完整提示词写进日志文件，
        方便排查"到底检索到了什么、提示词长什么样"。
        """
        return self.prompt_template | self._log_prompt | self.model | StrOutputParser()

    def _log_prompt(self, prompt):
        """
        透传钩子：把完整提示词落到日志文件（DEBUG 级，只在文件里，不刷控制台）。

        :param prompt: 上一步产出的 PromptValue
        :return: 原样返回，不改变链的数据流
        """
        logger.debug(f"[RAG提示词]\n{prompt.to_string()}")
        return prompt

    def retriever_docs(self, query: str) -> list[Document]:
        """
        按语义相似度从向量库召回资料片段。

        :param query: 检索词
        :return: 召回的 Document 列表（可能为空）
        """
        docs = self.retriever.invoke(query)

        # 把召回情况打出来：这是判断"RAG 到底跑没跑、召回了什么"最直接的依据
        sources = [doc.metadata.get("source", "未知来源") for doc in docs]
        logger.info(f"[retriever_docs]检索词: {query!r}，召回 {len(docs)} 个分片，来源: {sources}")

        return docs

    def rag_summarize(self, query: str) -> str:
        """
        检索并总结，返回一段可直接作为工具结果的文本。

        :param query: 检索词（贴合用户问题的核心关键词）
        :return: 基于参考资料生成的概括性回答
        """
        start = time.perf_counter()

        context_docs = self.retriever_docs(query=query)

        # 把召回的分片拼成带编号的参考资料块，编号方便模型引用和人工核对
        context = ""
        count = 0
        for doc in context_docs:
            count += 1
            # 修正点：原代码这里写成了 "[参考资料1]: 参考资料:xxx"，重复了措辞
            context += f"[参考资料{count}]: {doc.page_content} | 资料元数据：{doc.metadata}\n"

        if not context_docs:
            # 明确告诉子模型"没检索到"，避免它对着空上下文硬编
            logger.warning(f"[rag_summarize]检索词{query!r}未召回任何资料，将基于空参考资料作答")
            context = "（未检索到相关参考资料）"

        answer = self.chain.invoke(
            {
                "input": query,
                "context": context,
            }
        )

        cost = time.perf_counter() - start
        logger.info(
            f"[rag_summarize]完成，耗时{cost:.2f}s，"
            f"参考资料{count}条，参考资料总长度{len(context)}字符"
        )

        return answer


if __name__ == "__main__":
    # 独立自查：直接从命令行验证"检索 → 总结"这条链路是否正常
    print(RagSummarizeService().rag_summarize("小户型适合那些扫地机器人"))
