"""
文档加载器 —— 支持 TXT / PDF，工厂模式按扩展名分发。
"""
import os
from typing import List
from langchain_core.documents import Document


class DocumentLoader:
    """统一的文档加载入口，内部按扩展名分发到具体加载器"""

    LOADERS = {}  # ext → handler 注册表

    @classmethod
    def register(cls, ext: str):
        """装饰器：注册加载器到指定扩展名"""
        def decorator(loader_cls):
            for e in ext if isinstance(ext, (list, tuple)) else [ext]:
                cls.LOADERS[e.lower()] = loader_cls
            return loader_cls
        return decorator

    @classmethod
    def load(cls, file_path: str) -> List[Document]:
        ext = os.path.splitext(file_path)[1].lower()
        loader_cls = cls.LOADERS.get(ext)
        if loader_cls is None:
            raise ValueError(f"不支持的文件格式: {ext}")
        return loader_cls().load(file_path)

    @classmethod
    def load_directory(cls, directory: str) -> List[Document]:
        """加载目录下所有支持的文件"""
        all_docs = []
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if not os.path.isfile(file_path):
                continue
            ext = os.path.splitext(filename)[1].lower()
            if ext not in cls.LOADERS:
                print(f"  [跳过] 不支持格式: {filename}")
                continue
            try:
                docs = cls.load(file_path)
                all_docs.extend(docs)
                print(f"  [加载] {filename} → {len(docs)} 个文档片段")
            except Exception as e:
                print(f"  [失败] {filename}: {e}")
        return all_docs


@DocumentLoader.register([".txt", ".text"])
class TxtLoader:
    def load(self, file_path: str) -> List[Document]:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        filename = os.path.basename(file_path)
        return [Document(page_content=content, metadata={"source": filename})]


@DocumentLoader.register(".pdf")
class PdfLoader:
    def load(self, file_path: str) -> List[Document]:
        try:
            from langchain_community.document_loaders import PyPDFLoader
        except ImportError:
            raise ImportError("请安装 pypdf: pip install pypdf")
        loader = PyPDFLoader(file_path)
        return loader.load()
