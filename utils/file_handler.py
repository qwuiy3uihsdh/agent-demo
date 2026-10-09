"""
文件处理工具模块。

负责知识库入库前的三件事：
  1. 计算文件 MD5 —— 用于判断文件是否已经入过库，避免重复向量化；
  2. 按后缀过滤出知识库目录里允许加载的文件；
  3. 把 txt / pdf 解析成 LangChain 的 Document 列表。

注意：listdir_with_allowed_type 只扫描一层，不递归子目录，
所以 data/external/ 这种子目录不会被当成知识库内容加载。
"""

import hashlib
import os

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

from utils.logger_handler import logger


def get_file_MD5(filepath: str) -> str | None:
    """
    计算文件的 MD5 值。

    大文件不会一次性读进内存，而是按 4096 字节分块喂给 md5 对象。

    :param filepath: 文件路径
    :return: 32 位十六进制 MD5 字符串；文件不存在、不是文件或读取失败时返回 None
    """
    if not os.path.exists(filepath):
        logger.error(f"[md5计算]文件{filepath}不存在")
        return None

    if not os.path.isfile(filepath):
        logger.error(f"[md5计算]路径{filepath}不是文件")
        return None

    md5_obj = hashlib.md5()

    chunk_size = 4096
    try:
        with open(filepath, "rb") as f:
            # 海象运算符：边读边判断，读到空串即结束
            while chunk := f.read(chunk_size):
                md5_obj.update(chunk)
            md5_hex = md5_obj.hexdigest()
            return md5_hex
    except Exception as e:
        logger.error(f"[md5计算]文件{filepath}计算md5失败, {str(e)}")
        return None


def listdir_with_allowed_type(dirpath: str, allowed_types: tuple[str]) -> tuple[str, ...]:
    """
    列出目录下所有后缀符合要求的文件的完整路径（不递归子目录）。

    :param dirpath: 待扫描的目录
    :param allowed_types: 允许的后缀元组，例如 ("txt", "pdf")
    :return: 匹配到的文件绝对路径元组；目录不合法时为**空元组**
    """
    files = []

    # 修正点：原代码这里 return 的是 allowed_types 本身（形如 ("txt","pdf")），
    # 调用方会把它当成文件路径列表去遍历，导致拿 "txt" 去算 MD5、去解析文档。
    # 目录不可用时正确语义是"没有找到任何文件"，必须返回空元组。
    if not os.path.isdir(dirpath):
        logger.error(f"[获取允许的文件列表]{dirpath}不是文件夹")
        return ()

    for f in os.listdir(dirpath):
        # 转小写比较，避免 "A.TXT" 这类后缀被漏掉
        if f.lower().endswith(allowed_types):
            files.append(os.path.join(dirpath, f))

    return tuple(files)


def pdf_loader(filepath: str, password=None) -> list[Document]:
    """
    解析 PDF 文件，逐页转成 Document。

    :param filepath: PDF 路径
    :param password: 加密 PDF 的密码，不需要时传 None
    :return: Document 列表，每个元素对应 PDF 的一页
    """
    return PyPDFLoader(file_path=filepath, password=password).load()


def txt_loader(filepath: str) -> list[Document]:
    """
    解析 txt 文件，整个文件作为一个 Document。

    :param filepath: txt 路径
    :return: 只含一个元素的 Document 列表
    """
    return TextLoader(file_path=filepath, encoding="utf-8").load()
