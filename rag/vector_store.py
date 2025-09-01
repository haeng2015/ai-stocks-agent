from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader, TextLoader
import os
import shutil
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

class VectorStoreManager:
    """向量存储管理器，用于加载文档、创建向量存储和进行检索"""
    
    def __init__(self, vector_store_path=None, embedding_model=None):
        """
        初始化向量存储管理器
        
        Args:
            vector_store_path: 向量存储的保存路径
            embedding_model: 使用的嵌入模型名称
        """
        self.vector_store_path = vector_store_path or os.getenv('VECTOR_STORE_PATH', 'data/vector_store')
        self.embedding_model = embedding_model or os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
        self.chunk_size = int(os.getenv('CHUNK_SIZE', '1000'))
        self.chunk_overlap = int(os.getenv('CHUNK_OVERLAP', '200'))
        
        # 初始化嵌入模型
        try:
            # 尝试使用本地模型，设置local_files_only=True以强制使用本地缓存
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True},
                cache_folder=os.getenv('CACHE_DIR', './cache'),
                local_files_only=True  # 强制使用本地文件，不尝试从Hugging Face下载
            )
            print(f"成功加载本地嵌入模型: {self.embedding_model}")
        except Exception as e:
            # 如果本地模型加载失败，提供友好的错误消息和指导
            print("\n" + "="*80)
            print("错误: 无法加载嵌入模型。这可能是因为您没有预先下载模型。")
            # print(f"模型名称: {self.embedding_model}")
            # print("\n解决方案:")
            # print("1. 确保您已连接到互联网，首次运行时会自动下载模型")
            # print("2. 或预先下载模型并放到本地缓存目录")
            # print("3. 或在.env文件中设置EMBEDDING_MODEL环境变量为已下载的本地模型路径")
            # print("="*80 + "\n")
            
            # 尝试在离线模式下创建一个简单的嵌入模型
            try:
                print("尝试创建简单的本地嵌入模型...")
                from langchain_community.embeddings import GPT4AllEmbeddings
                self.embeddings = GPT4AllEmbeddings()
                print("成功创建GPT4AllEmbeddings作为替代")
            except:
                print("警告: 无法创建替代嵌入模型。请确保已正确配置环境。")
                # 创建一个继承自Embeddings基类的模拟嵌入类
                from langchain_core.embeddings import Embeddings
                class SimpleMockEmbeddings(Embeddings):
                    def __init__(self):
                        pass
                    
                    def embed_documents(self, texts):
                        # 返回固定长度的随机向量作为模拟
                        import random
                        return [[random.random() for _ in range(384)] for _ in texts]
                    
                    def embed_query(self, text):
                        # 返回固定长度的随机向量作为模拟
                        import random
                        return [random.random() for _ in range(384)]
                
                self.embeddings = SimpleMockEmbeddings()
                print("已创建简单的模拟嵌入模型用于测试")
        
        # 初始化向量存储
        self.vector_store = None
    
    def load_documents(self, data_dir='data'):
        """
        从目录加载文档
        
        Args:
            data_dir: 数据目录路径
            
        Returns:
            加载的文档列表
        """
        # 在较新版本的LangChain中，encoding参数应传递给TextLoader而不是DirectoryLoader
        loader = DirectoryLoader(
            data_dir,
            glob="*.txt",
            loader_cls=lambda path: TextLoader(path, encoding='utf-8')
        )
        documents = loader.load()
        print(f"加载了 {len(documents)} 个文档")
        return documents
    
    def split_documents(self, documents):
        """
        将文档分割成小块
        
        Args:
            documents: 文档列表
            
        Returns:
            分割后的文档块列表
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        splits = text_splitter.split_documents(documents)
        print(f"文档分割成 {len(splits)} 个块")
        return splits
    
    def create_vector_store(self, splits):
        """
        创建向量存储
        
        Args:
            splits: 文档块列表
            
        Returns:
            创建的向量存储
        """
        # 创建向量存储
        self.vector_store = FAISS.from_documents(splits, self.embeddings)
        
        # 保存向量存储
        if not os.path.exists(self.vector_store_path):
            os.makedirs(self.vector_store_path)
        self.vector_store.save_local(self.vector_store_path)
        print(f"向量存储已保存到 {self.vector_store_path}")
        
        return self.vector_store
    
    def load_vector_store(self):
        """
        加载已保存的向量存储
        
        Returns:
            加载的向量存储
        """
        # 检查向量存储目录是否存在以及是否包含必要的文件
        index_faiss_path = os.path.join(self.vector_store_path, 'index.faiss')
        index_pkl_path = os.path.join(self.vector_store_path, 'index.pkl')
        
        if os.path.exists(index_faiss_path) and os.path.exists(index_pkl_path):
            try:
                self.vector_store = FAISS.load_local(
                    self.vector_store_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                print(f"向量存储已从 {self.vector_store_path} 加载")
            except Exception as e:
                print(f"加载向量存储时出错: {str(e)}")
                print("将重新创建向量存储...")
                # 尝试重新创建向量存储
                documents = self.load_documents()
                if documents:
                    splits = self.split_documents(documents)
                    self.create_vector_store(splits)
        else:
            print(f"向量存储文件不存在: {self.vector_store_path}")
            # 确保向量存储目录存在
            if not os.path.exists(self.vector_store_path):
                os.makedirs(self.vector_store_path)
                print(f"已创建向量存储目录: {self.vector_store_path}")
            
            # 尝试从数据目录创建向量存储
            documents = self.load_documents()
            if documents:
                splits = self.split_documents(documents)
                self.create_vector_store(splits)
            else:
                print("警告: 没有找到文档来创建向量存储。请确保data目录下有文本文件。")
                # 创建一个空的向量存储作为占位符
                self._create_empty_vector_store()
        
        return self.vector_store
        
    def _create_empty_vector_store(self):
        """
        创建一个空的向量存储作为占位符
        """
        try:
            # 创建一个简单的文档来初始化向量存储
            from langchain_core.documents import Document
            empty_docs = [Document(page_content="占位文档，用于初始化向量存储", metadata={"source": "empty"})]
            
            # 使用模拟嵌入或实际嵌入创建向量存储
            if hasattr(self.embeddings, 'embed_documents'):
                self.vector_store = FAISS.from_documents(empty_docs, self.embeddings)
                self.vector_store.save_local(self.vector_store_path)
                print(f"已创建空向量存储作为占位符: {self.vector_store_path}")
        except Exception as e:
            print(f"创建空向量存储时出错: {str(e)}")
            print("将创建一个简单的模拟向量存储类")
            
            # 创建一个模拟向量存储类
            class MockVectorStore:
                def similarity_search(self, query, k=4):
                    return []
                
                def save_local(self, path):
                    pass
            
            self.vector_store = MockVectorStore()
    
    def retrieve(self, query, k=4):
        """
        检索与查询相关的文档
        
        Args:
            query: 查询文本
            k: 返回的文档数量
            
        Returns:
            相关文档列表
        """
        if not self.vector_store:
            self.load_vector_store()
        
        if self.vector_store:
            relevant_docs = self.vector_store.similarity_search(query, k=k)
            return relevant_docs
        else:
            print("向量存储未初始化，无法进行检索")
            return []
    
    def update_vector_store(self, data_dir='data'):
        """
        更新向量存储
        
        Args:
            data_dir: 数据目录路径
            
        Returns:
            更新后的向量存储
        """
        documents = self.load_documents(data_dir)
        splits = self.split_documents(documents)
        return self.create_vector_store(splits)

# 示例用法
if __name__ == "__main__":
    # 初始化向量存储管理器
    vector_store_manager = VectorStoreManager()
    
    # 加载或创建向量存储
    vector_store = vector_store_manager.load_vector_store()
    
    # 如果向量存储不存在或需要更新，则创建或更新
    if not vector_store or input("是否更新向量存储？(y/n): ").lower() == 'y':
        vector_store_manager.update_vector_store()
    
    # 测试检索功能
    query = "什么是股票的市盈率？"
    relevant_docs = vector_store_manager.retrieve(query)
    
    print(f"\n检索到的相关文档 ({len(relevant_docs)}):")
    for i, doc in enumerate(relevant_docs):
        print(f"\n文档 {i+1}: ")
        print(f"内容: {doc.page_content[:200]}...")
        print(f"来源: {doc.metadata['source']}")