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

# 调整sys.path以确保优先导入项目中的模块
# 首先将项目根目录添加到sys.path的最前面
project_root = os.path.abspath('.')
if project_root in sys.path:
    sys.path.remove(project_root)
sys.path.insert(0, project_root)

# 获取虚拟环境的site-packages路径
site_packages = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.venv', 'Lib', 'site-packages')

# 如果site-packages路径存在，添加到sys.path
if os.path.exists(site_packages) and site_packages not in sys.path:
    sys.path.append(site_packages)

import argparse
import json
from dotenv import load_dotenv

# 导入日志工具
from utils.logger import setup_logging, get_logger

# 设置日志记录
setup_logging()
logger = get_logger('main')

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
        logger.debug("正在初始化AI Stocks Agent...")
        
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
        
        logger.debug(f"AI Stocks Agent 初始化完成，使用 {model_type} 模型")
    
    def init_rag(self):
        """初始化RAG链"""
        if self.rag_chain is None:
            logger.debug("正在初始化RAG链...")
            model = self.model_selector.get_model()
            self.rag_chain = RAGChain(model, self.vector_store_manager)
            logger.debug("RAG链初始化完成")
        return self.rag_chain
    
    def init_workflow(self):
        """初始化LangGraph工作流"""
        if self.workflow is None:
            logger.debug("正在初始化股票分析工作流...")
            self.workflow = StockAnalysisWorkflow(self.model_selector)
            logger.debug("工作流初始化完成")
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
        logger.debug("正在更新向量存储...")
        self.vector_store_manager.update_vector_store()
        logger.debug("向量存储更新完成")
        
        # 如果RAG链已初始化，重新初始化它
        if self.rag_chain is not None:
            self.rag_chain = None
            self.init_rag()
    
    def switch_model(self, model_type):
        """切换模型类型"""
        if model_type in ['ollama', 'vllm']:
            logger.debug(f"正在切换到 {model_type} 模型...")
            self.model_selector = ModelSelector(default_model_type=model_type)
            
            # 重置RAG链和工作流，以便使用新模型
            self.rag_chain = None
            self.workflow = None
            
            logger.debug(f"模型已切换为 {model_type}")
            return True
        else:
            logger.warning(f"不支持的模型类型: {model_type}，支持的类型有: ollama, vllm")
            return False
    
    def interactive_mode(self):
        """交互式命令行模式"""
        interactive_banner = "\n===== AI Stocks Agent 交互式模式 ====="
        help_prompt = "输入 'help' 查看可用命令"
        exit_prompt = "输入 'exit' 退出程序"
        
        logger.info(interactive_banner)
        logger.info(help_prompt)
        logger.info(exit_prompt)
        
        logger.debug("进入交互式模式")
        
        while True:
            try:
                command = input("\n> ").strip()
                logger.debug(f"接收到命令: {command}")
                
                if command.lower() == 'exit' or command.lower() == 'quit':
                    exit_message = "谢谢使用AI Stocks Agent，再见！"
                    logger.info(exit_message)
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
                    logger.debug(f"执行简单查询: {query}")
                    response = self.simple_query(query)
                    logger.info("\n[简单查询结果]:")
                    logger.info(response)
                
                elif command.lower().startswith('rag '):
                    # RAG查询
                    _, query = command.split(' ', 1)
                    logger.debug(f"执行RAG查询: {query}")
                    response = self.rag_query(query)
                    logger.info("\n[RAG增强查询结果]:")
                    logger.info(response)
                
                elif command.lower().startswith('workflow '):
                    # 工作流分析
                    _, query = command.split(' ', 1)
                    logger.debug(f"执行工作流分析: {query}")
                    result = self.workflow_analysis(query)
                    logger.info("\n[工作流分析结果]:")
                    logger.info(result["final_report"])
                
                else:
                    # 默认使用RAG查询
                    logger.debug(f"执行默认RAG查询: {command}")
                    response = self.rag_query(command)
                    logger.info("\n[RAG增强查询结果]:")
                    logger.info(response)
                    
            except KeyboardInterrupt:
                exit_message = "\n谢谢使用AI Stocks Agent，再见！"
                logger.debug(exit_message)
                break
            except Exception as e:
                error_message = f"错误: {str(e)}"
                logger.error(error_message)
    
    def _show_help(self):
        """显示帮助信息"""
        logger.debug("\n可用命令:")
        logger.debug("  exit/quit           - 退出程序")
        logger.debug("  help                - 显示此帮助信息")
        logger.debug("  model <type>        - 切换模型类型 (ollama 或 vllm)")
        logger.debug("  update vector       - 更新向量存储")
        logger.debug("  status              - 显示当前状态")
        logger.debug("  simple <query>      - 使用简单查询模式")
        logger.debug("  rag <query>         - 使用RAG增强查询模式")
        logger.debug("  workflow <query>    - 使用工作流进行综合分析")
        logger.debug("  <query>             - 默认使用RAG增强查询模式")
        logger.debug("显示帮助信息")
    
    def _show_status(self):
        """显示当前状态信息"""
        status_info = f"\n当前状态:\n  模型类型: {self.model_selector.default_model_type}\n  RAG链: {'已初始化' if self.rag_chain else '未初始化'}\n  工作流: {'已初始化' if self.workflow else '未初始化'}\n  向量存储路径: {self.vector_store_manager.vector_store_path}\n  嵌入模型: {self.vector_store_manager.embedding_model}"
        logger.debug(status_info)
        logger.debug(f"显示状态信息: 模型类型={self.model_selector.default_model_type}, RAG链={'已初始化' if self.rag_chain else '未初始化'}, 工作流={'已初始化' if self.workflow else '未初始化'}")

def main():
    """主函数"""
    logger.info("启动AI Stocks Agent")
    
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
    
    # 记录命令行参数
    logger.debug(f"命令行参数: {vars(args)}")
    
    # 初始化AI Stocks Agent
    agent = AIStocksAgent(model_type=args.model)
    
    # 根据参数执行相应操作
    if args.update_vector:
        logger.debug("执行向量存储更新操作")
        agent.update_vector_store()
    
    elif args.query:
        logger.debug(f"执行查询操作: {args.query}")
        if args.workflow:
            logger.debug("使用工作流进行综合分析")
            # 使用工作流进行综合分析
            result = agent.workflow_analysis(args.query)
            logger.info(result["final_report"])
        elif args.rag:
            logger.debug("使用RAG增强查询模式")
            # 使用RAG增强查询
            response = agent.rag_query(args.query)
            logger.info(response)
        else:
            logger.debug("使用简单查询模式")
            # 简单查询
            response = agent.simple_query(args.query)
            logger.info(response)
    
    else:
        logger.debug("默认进入交互式模式")
        # 默认进入交互式模式
        agent.interactive_mode()
    
    logger.info("AI Stocks Agent 程序结束")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"程序异常退出: {str(e)}", exc_info=True)
        sys.exit(1)