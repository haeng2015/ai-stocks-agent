#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AI Stocks Agent 主程序

这是股票智能体的主入口文件，整合了所有功能模块，包括：
- 模型推理（Ollama和VLLM）
- RAG检索增强生成
- LangGraph工作流

提供命令行交互界面，支持股票分析、市场查询等功能。
"""

import os
import sys
import argparse
import json
from dotenv import load_dotenv

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain.model_selector import ModelSelector
from rag.vector_store import VectorStoreManager
from rag.rag_chain import RAGChain
from langgraph.stock_analysis_workflow import StockAnalysisWorkflow

class AIStocksAgent:
    """股票智能体主类，整合所有功能模块"""
    
    def __init__(self, model_type='ollama'):
        """
        初始化股票智能体
        
        Args:
            model_type: 使用的模型类型，可选值为'ollama'或'vllm'
        """
        print("正在初始化AI Stocks Agent...")
        
        # 加载环境变量
        load_dotenv()
        
        # 初始化模型选择器
        self.model_selector = ModelSelector(default_model_type=model_type)
        
        # 初始化向量存储管理器
        self.vector_store_manager = VectorStoreManager()
        
        # 初始化RAG链
        self.rag_chain = None
        
        # 初始化工作流
        self.workflow = None
        
        print(f"AI Stocks Agent 初始化完成，使用 {model_type} 模型")
    
    def init_rag(self):
        """初始化RAG链"""
        if self.rag_chain is None:
            print("正在初始化RAG链...")
            model = self.model_selector.get_model()
            self.rag_chain = RAGChain(model, self.vector_store_manager)
            print("RAG链初始化完成")
        return self.rag_chain
    
    def init_workflow(self):
        """初始化LangGraph工作流"""
        if self.workflow is None:
            print("正在初始化股票分析工作流...")
            self.workflow = StockAnalysisWorkflow(self.model_selector)
            print("工作流初始化完成")
        return self.workflow
    
    def simple_query(self, query):
        """简单查询，直接使用模型回答"""
        model = self.model_selector.get_model()
        response = model.invoke(query)
        return response
    
    def rag_query(self, query):
        """使用RAG增强的查询"""
        rag_chain = self.init_rag()
        response = rag_chain.invoke(query)
        return response
    
    def workflow_analysis(self, query):
        """使用工作流进行综合分析"""
        workflow = self.init_workflow()
        result = workflow.invoke(query)
        return result
    
    def update_vector_store(self):
        """更新向量存储"""
        print("正在更新向量存储...")
        self.vector_store_manager.update_vector_store()
        print("向量存储更新完成")
        
        # 如果RAG链已初始化，重新初始化它
        if self.rag_chain is not None:
            self.rag_chain = None
            self.init_rag()
    
    def switch_model(self, model_type):
        """切换模型类型"""
        if model_type in ['ollama', 'vllm']:
            print(f"正在切换到 {model_type} 模型...")
            self.model_selector = ModelSelector(default_model_type=model_type)
            
            # 重置RAG链和工作流，以便使用新模型
            self.rag_chain = None
            self.workflow = None
            
            print(f"模型已切换为 {model_type}")
            return True
        else:
            print(f"不支持的模型类型: {model_type}，支持的类型有: ollama, vllm")
            return False
    
    def interactive_mode(self):
        """交互式命令行模式"""
        print("\n===== AI Stocks Agent 交互式模式 =====")
        print("输入 'help' 查看可用命令")
        print("输入 'exit' 退出程序")
        
        while True:
            try:
                command = input("\n> ").strip()
                
                if command.lower() == 'exit' or command.lower() == 'quit':
                    print("谢谢使用AI Stocks Agent，再见！")
                    break
                
                elif command.lower() == 'help':
                    self._show_help()
                
                elif command.lower().startswith('model '):
                    # 切换模型
                    _, model_type = command.split(' ', 1)
                    self.switch_model(model_type.strip())
                
                elif command.lower() == 'update vector':
                    # 更新向量存储
                    self.update_vector_store()
                
                elif command.lower() == 'status':
                    # 显示状态信息
                    self._show_status()
                
                elif command.lower().startswith('simple '):
                    # 简单查询
                    _, query = command.split(' ', 1)
                    response = self.simple_query(query)
                    print("\n[简单查询结果]:")
                    print(response)
                
                elif command.lower().startswith('rag '):
                    # RAG查询
                    _, query = command.split(' ', 1)
                    response = self.rag_query(query)
                    print("\n[RAG增强查询结果]:")
                    print(response)
                
                elif command.lower().startswith('workflow '):
                    # 工作流分析
                    _, query = command.split(' ', 1)
                    result = self.workflow_analysis(query)
                    print("\n[工作流分析结果]:")
                    print(result["final_report"])
                
                else:
                    # 默认使用RAG查询
                    response = self.rag_query(command)
                    print("\n[RAG增强查询结果]:")
                    print(response)
                    
            except KeyboardInterrupt:
                print("\n谢谢使用AI Stocks Agent，再见！")
                break
            except Exception as e:
                print(f"错误: {str(e)}")
    
    def _show_help(self):
        """显示帮助信息"""
        print("\n可用命令:")
        print("  exit/quit           - 退出程序")
        print("  help                - 显示此帮助信息")
        print("  model <type>        - 切换模型类型 (ollama 或 vllm)")
        print("  update vector       - 更新向量存储")
        print("  status              - 显示当前状态")
        print("  simple <query>      - 使用简单查询模式")
        print("  rag <query>         - 使用RAG增强查询模式")
        print("  workflow <query>    - 使用工作流进行综合分析")
        print("  <query>             - 默认使用RAG增强查询模式")
    
    def _show_status(self):
        """显示当前状态信息"""
        print("\n当前状态:")
        print(f"  模型类型: {self.model_selector.default_model_type}")
        print(f"  RAG链: {'已初始化' if self.rag_chain else '未初始化'}")
        print(f"  工作流: {'已初始化' if self.workflow else '未初始化'}")
        print(f"  向量存储路径: {self.vector_store_manager.vector_store_path}")
        print(f"  嵌入模型: {self.vector_store_manager.embedding_model}")

def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='AI Stocks Agent - 股票智能分析助手')
    parser.add_argument('--model', type=str, default='ollama', choices=['ollama', 'vllm'],
                        help='选择使用的模型类型 (默认: ollama)')
    parser.add_argument('--query', type=str, help='直接进行查询')
    parser.add_argument('--rag', action='store_true', help='使用RAG增强查询模式')
    parser.add_argument('--workflow', action='store_true', help='使用工作流进行综合分析')
    parser.add_argument('--update-vector', action='store_true', help='更新向量存储')
    parser.add_argument('--interactive', action='store_true', help='进入交互式模式')
    
    args = parser.parse_args()
    
    # 初始化AI Stocks Agent
    agent = AIStocksAgent(model_type=args.model)
    
    # 根据参数执行相应操作
    if args.update_vector:
        agent.update_vector_store()
    
    elif args.query:
        if args.workflow:
            # 使用工作流进行综合分析
            result = agent.workflow_analysis(args.query)
            print(result["final_report"])
        elif args.rag:
            # 使用RAG增强查询
            response = agent.rag_query(args.query)
            print(response)
        else:
            # 简单查询
            response = agent.simple_query(args.query)
            print(response)
    
    else:
        # 默认进入交互式模式
        agent.interactive_mode()

if __name__ == "__main__":
    main()