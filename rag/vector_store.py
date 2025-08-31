from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os
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
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
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
        loader = DirectoryLoader(data_dir, glob="*.txt", loader_cls=TextLoader, encoding='utf-8')
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
        if os.path.exists(self.vector_store_path):
            self.vector_store = FAISS.load_local(
                self.vector_store_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            print(f"向量存储已从 {self.vector_store_path} 加载")
        else:
            print(f"向量存储文件不存在: {self.vector_store_path}")
            # 尝试从数据目录创建向量存储
            documents = self.load_documents()
            if documents:
                splits = self.split_documents(documents)
                self.create_vector_store(splits)
        
        return self.vector_store
    
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