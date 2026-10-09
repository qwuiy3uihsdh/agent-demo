"""
向量存储模块。

职责有两块：
  1. load_document()：扫描 data/ 目录 → 解析 txt/pdf → 分片 → 向量化 → 存入 Chroma。
     用 MD5 去重，已经入过库的文件不会重复写入。
  2. get_retriever()：产出一个检索器，供 RAG 子链按语义相似度召回资料。

Chroma 的数据持久化在 chroma_db/ 目录，进程重启后知识库仍在。
"""

import os

from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from model.factory import embedddings_model
from utils.config_handler import chroma_conf
from utils.file_handler import get_file_MD5, listdir_with_allowed_type, pdf_loader, txt_loader
from utils.logger_handler import logger
from utils.path_tool import get_abs_path


class VectorStoreService:
    """封装 Chroma 向量库的加载与检索。"""

    def __init__(self):
        # 连接（不存在则创建）指定的集合。embedding 函数决定了文本怎么变成向量，
        # 必须和入库时用的模型保持一致，否则检索出来的相似度没有意义。
        self.vectore_store = Chroma(
            collection_name=chroma_conf["collection_name"],
            embedding_function=embedddings_model,
            persist_directory=chroma_conf["persist_directory"],
        )

        # 文本分片器：先把长文档切成带少量重叠的小块，再逐块向量化。
        # overlap 是为了避免一句话正好被切断、语义丢失。
        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_conf["chunk_size"],
            chunk_overlap=chroma_conf["chunk_overlap"],
            separators=chroma_conf["separators"],
            length_function=len,
        )

    def get_retriever(self):
        """返回检索器，k 表示每次召回最相似的 k 个分片。"""
        return self.vectore_store.as_retriever(search_kwargs={"k": chroma_conf["k"]})

    def load_document(self):
        """把 data/ 目录下所有允许类型的文件增量写入向量库（已入库的自动跳过）。"""

        def check_md5_hex(md5_str: str) -> bool:
            """
            检查某个 MD5 是否已经记录在去重文件中。

            文件不存在时会先创建空文件，再返回 False（视为"尚无记录"）。

            :param md5_str: 待检查的 MD5 十六进制字符串
            :return: True 表示已存在（应跳过入库），False 表示未见过
            """
            md5_store_path = get_abs_path(chroma_conf["md5_hex_store"])

            # 去重文件还不存在：创建空文件，说明知识库是空的
            if not os.path.exists(md5_store_path):
                with open(md5_store_path, "w", encoding="utf-8"):
                    pass
                return False

            # 逐行比对，命中即已入库。用 with 确保句柄释放
            with open(md5_store_path, "r", encoding="utf-8") as f:
                for line in f.readlines():
                    if line.strip() == md5_str:
                        return True

            return False

        def save_md5(md5_str: str):
            """
            把新入库内容的 MD5 追加写入去重文件，供后续 check_md5_hex 比对。

            :param md5_str: 已成功入库内容的 MD5
            """
            with open(get_abs_path(chroma_conf["md5_hex_store"]), "a", encoding="utf-8") as f:
                f.write(md5_str + "\n")

        def get_file_documents(read_path: str):
            """按后缀选择对应的解析器，把文件变成 Document 列表。"""
            if read_path.lower().endswith("txt"):
                return txt_loader(read_path)

            if read_path.lower().endswith("pdf"):
                return pdf_loader(read_path)

            return []

        allowed_files_path: list[str] = listdir_with_allowed_type(
            get_abs_path(chroma_conf["data_path"]),
            tuple(chroma_conf["allow_knowledge_file_type"]),
        )

        if not allowed_files_path:
            logger.warning(f"[加载知识库]{chroma_conf['data_path']}下没有找到可加载的文件")

        for path in allowed_files_path:
            md5_hex = get_file_MD5(path)

            # 算不出 MD5（文件读不了）时直接跳过，避免把 None 写进去重文件
            if not md5_hex:
                logger.warning(f"[加载知识库]{path}无法计算MD5，跳过")
                continue

            if check_md5_hex(md5_hex):
                logger.info(f"[加载知识库]{path}内容已经存在与知识库中，跳过")
                continue

            try:
                documents = get_file_documents(path)

                if not documents:
                    logger.warning(f"[加载知识库]{path}内没有有效文本，跳过")
                    continue

                spilt_documents = self.spliter.split_documents(documents=documents)

                if not spilt_documents:
                    logger.warning(f"[加载知识库]{path}分片后没有有效文本，跳过")
                    continue

                self.vectore_store.add_documents(spilt_documents)

                # 只有真正写入成功了才记 MD5，保证去重记录与库内容一致
                save_md5(md5_hex)

                logger.info(f"[加载知识库]{path}内容加载完成，共{len(spilt_documents)}个分片")
            except Exception as e:
                # 单个文件失败不影响其他文件
                logger.error(f"[加载知识库]{path}加载失败, {str(e)}", exc_info=True)
                continue


if __name__ == "__main__":
    vs = VectorStoreService()

    vs.load_document()

    retriever = vs.get_retriever()

    res = retriever.invoke("迷路")

    for r in res:
        print(r.page_content)
        print("-" * 20)
