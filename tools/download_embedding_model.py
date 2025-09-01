#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
下载Hugging Face嵌入模型的工具脚本

该脚本用于预先下载嵌入模型到本地缓存，解决VectorStoreManager无法加载本地模型的问题。
"""

import os
import sys
import argparse
from transformers import AutoTokenizer, AutoModel
from langchain_huggingface import HuggingFaceEmbeddings
from tqdm import tqdm

def download_model(model_name, cache_dir="./cache"):
    """
    下载Hugging Face模型到本地缓存
    
    Args:
        model_name: 模型名称或路径
        cache_dir: 缓存目录路径
    """
    try:
        print(f"\n开始下载模型: {model_name}")
        print(f"缓存目录: {os.path.abspath(cache_dir)}")
        
        # 确保缓存目录存在
        os.makedirs(cache_dir, exist_ok=True)
        
        # 下载tokenizer
        print("\n下载tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )
        
        # 下载模型
        print("\n下载模型...")
        model = AutoModel.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )
        
        # 保存模型以确保完全缓存
        print("\n保存模型到缓存...")
        tokenizer.save_pretrained(os.path.join(cache_dir, model_name.split('/')[-1]))
        model.save_pretrained(os.path.join(cache_dir, model_name.split('/')[-1]))
        
        print(f"\n✅ 模型下载完成: {model_name}")
        return True
        
    except Exception as e:
        print(f"❌ 下载模型时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_model(model_name, cache_dir="./cache"):
    """
    测试下载的模型是否能被LangChain正确加载
    
    Args:
        model_name: 模型名称或路径
        cache_dir: 缓存目录路径
    """
    try:
        print(f"\n测试模型加载: {model_name}")
        
        # 使用local_files_only=True测试本地模型加载
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True},
            cache_folder=cache_dir,
            local_files_only=True  # 强制使用本地文件
        )
        
        # 测试嵌入功能
        test_text = "这是一个测试句子，用于验证嵌入模型是否正常工作"
        embedding = embeddings.embed_query(test_text)
        
        print(f"✅ 模型加载成功!")
        print(f"嵌入向量维度: {len(embedding)}")
        print(f"嵌入向量示例: {embedding[:5]}")
        return True
        
    except Exception as e:
        print(f"❌ 测试模型加载时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    # 命令行参数解析
    parser = argparse.ArgumentParser(description="下载Hugging Face嵌入模型工具")
    parser.add_argument(
        "--model", 
        type=str, 
        # default="sentence-transformers/all-MiniLM-L6-v2",
        default="thenlper/gte-large-zh",
        help=f"要下载的模型名称，默认为: 'thenlper/gte-large-zh'")
    parser.add_argument(
        "--cache-dir", 
        type=str, 
        default="./cache",
        help="模型缓存目录，默认为'./cache'")
    parser.add_argument(
        "--test", 
        action="store_true",
        help="下载后测试模型加载")
    
    args = parser.parse_args()
    
    # 显示欢迎信息
    print("="*80)
    print("            Hugging Face 嵌入模型下载工具            ")
    print("="*80)
    print("这个工具可以帮助您预先下载嵌入模型到本地缓存，")
    print("解决VectorStoreManager无法加载本地模型的问题。")
    print("="*80)
    
    # 下载模型
    success = download_model(args.model, args.cache_dir)
    
    # 如果下载成功且需要测试，则测试模型
    if success and args.test:
        test_model(args.model, args.cache_dir)
    
    # 提供后续使用指南
    print("\n" + "="*80)
    print("使用指南:")
    print(f"1. 模型已下载到: {os.path.abspath(args.cache_dir)}")
    print("2. 您可以在.env文件中设置以下环境变量:")
    print(f"   EMBEDDING_MODEL={args.model}")
    print(f"   CACHE_DIR={os.path.abspath(args.cache_dir)}")
    print("3. 或者直接运行main.py，程序现在应该能够使用本地模型了")
    print("="*80)

if __name__ == "__main__":
    main()