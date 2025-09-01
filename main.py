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

# 获取虚拟环境的site-packages路径
site_packages = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.venv', 'Lib', 'site-packages')

# 确保site-packages路径优先于项目目录
sys.path.insert(0, site_packages)

# 然后将项目根目录添加到sys.path
project_root = os.path.abspath('.')
if project_root in sys.path:
    sys.path.remove(project_root)
sys.path.insert(1, project_root)

import argparse
import json
from dotenv import load_dotenv

# 导入日志工具
from utils.logger import setup_logging, get_logger

# 设置日志记录
setup_logging()
logger = get_logger('main')

# 导入项目自定义的stocksmith模块（替代原langsmith模块以避免名称冲突）
from stocksmith import get_manager

# 导入项目自定义的模型选择器模块
from stockschain.model_selector import ModelSelector
from rag.vector_store import VectorStoreManager
from rag.rag_chain import RAGChain
from stocksgraph.stock_analysis_workflow import StockAnalysisWorkflow

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
                
                elif command.lower() == 'eval':
                    # 运行LangSmith评估
                    logger.debug("执行LangSmith评估")
                    self.run_langsmith_evaluation()
                    
                elif command.lower() == 'viz':
                    # 生成评估结果可视化
                    logger.debug("生成评估结果可视化")
                    self.generate_visualizations()
                    
                elif command.lower() == 'analysis':
                    # 运行完整评估与可视化分析
                    logger.debug("运行完整评估与可视化分析")
                    self.run_full_analysis()
                    
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
    
    def run_langsmith_evaluation(self):
        """运行LangSmith评估"""
        try:
            logger.info("正在准备LangSmith评估...")
            
            # 创建LangSmith管理器
            langsmith_manager = get_manager(self.model_selector)
            
            # 检查LangSmith配置
            if not langsmith_manager.is_langsmith_configured():
                logger.warning("警告: LangSmith未配置，评估结果将仅保存在本地")
                logger.info("提示: 如需使用LangSmith平台功能，请在.env文件中配置LANGCHAIN_API_KEY、LANGCHAIN_TRACING_V2和LANGCHAIN_PROJECT")
            
            # 创建示例测试用例
            logger.info("正在创建示例测试用例...")
            import os
            import json
            sample_test_cases = [
                {
                    "query": "分析苹果公司(AAPL)的投资价值",
                    "reference_answer": "苹果公司(AAPL)拥有强大的品牌价值、稳健的财务状况和持续创新的能力。公司在智能手机市场占据主导地位，同时在服务业务方面持续增长。从技术面看，股价保持上升趋势，支撑位在210美元左右，阻力位在230美元左右。综合来看，苹果公司具有长期投资价值。"
                },
                {
                    "query": "特斯拉(TSLA)的股票技术面如何？",
                    "reference_answer": "特斯拉(TSLA)近期股价波动较大，目前处于上升趋势中。从技术指标看，MACD指标显示买入信号，RSI指标处于中性区域。支撑位在180美元左右，阻力位在210美元左右。成交量有所增加，表明市场关注度提高。"
                }
            ]
            
            test_cases_file = "temp_test_cases.json"
            with open(test_cases_file, 'w', encoding='utf-8') as f:
                json.dump(sample_test_cases, f, ensure_ascii=False, indent=2)
            
            logger.info("正在运行评估...")
            results = langsmith_manager.run_evaluation(
                test_cases_file=test_cases_file,
                project_name=f"ai-stocks-agent-{self.model_selector.default_model_type}",
                output_file=f"evaluation_results_{self.model_selector.default_model_type}.json"
            )
            
            # 清理临时文件
            if os.path.exists(test_cases_file):
                os.remove(test_cases_file)
            
            logger.info("\n评估完成！")
            logger.info(f"评估结果已保存到: evaluation_results_{self.model_selector.default_model_type}.json")
            logger.info("\n您可以使用 'viz' 命令生成可视化结果，或使用 'analysis' 命令运行完整的评估与可视化分析。")
            
        except Exception as e:
            logger.error(f"运行评估时出错: {str(e)}", exc_info=True)
    
    def generate_visualizations(self):
        """生成评估结果可视化"""
        try:
            logger.info("正在准备生成评估结果可视化...")
            
            # 创建LangSmith管理器
            langsmith_manager = get_manager(self.model_selector)
            
            # 检查评估结果文件是否存在
            results_file = f"evaluation_results_{self.model_selector.default_model_type}.json"
            import os
            if not os.path.exists(results_file):
                logger.warning(f"未找到评估结果文件: {results_file}")
                logger.info("请先使用 'eval' 命令运行评估。")
                return
            
            # 生成可视化
            output_dir = f"visualizations_{self.model_selector.default_model_type}"
            logger.info(f"正在生成可视化结果到 {output_dir}...")
            langsmith_manager.generate_visualizations(
                results_file=results_file,
                output_dir=output_dir
            )
            
            logger.info("\n可视化完成！")
            logger.info(f"可视化结果已保存到: {output_dir}")
            logger.info(f"请查看 {os.path.join(output_dir, 'evaluation_report.html')} 获取详细报告")
            
        except Exception as e:
            logger.error(f"生成可视化时出错: {str(e)}", exc_info=True)
    
    def run_full_analysis(self):
        """运行完整的评估和可视化分析"""
        try:
            logger.info("正在准备完整的评估和可视化分析...")
            
            # 创建LangSmith管理器
            langsmith_manager = get_manager(self.model_selector)
            
            # 检查LangSmith配置
            if not langsmith_manager.is_langsmith_configured():
                logger.warning("警告: LangSmith未配置，评估结果将仅保存在本地")
                logger.info("提示: 如需使用LangSmith平台功能，请在.env文件中配置LANGCHAIN_API_KEY、LANGCHAIN_TRACING_V2和LANGCHAIN_PROJECT")
            
            # 创建示例测试用例
            logger.info("正在创建示例测试用例...")
            import os
            import json
            sample_test_cases = [
                {
                    "query": "分析苹果公司(AAPL)的投资价值",
                    "reference_answer": "苹果公司(AAPL)拥有强大的品牌价值、稳健的财务状况和持续创新的能力。公司在智能手机市场占据主导地位，同时在服务业务方面持续增长。从技术面看，股价保持上升趋势，支撑位在210美元左右，阻力位在230美元左右。综合来看，苹果公司具有长期投资价值。"
                },
                {
                    "query": "特斯拉(TSLA)的股票技术面如何？",
                    "reference_answer": "特斯拉(TSLA)近期股价波动较大，目前处于上升趋势中。从技术指标看，MACD指标显示买入信号，RSI指标处于中性区域。支撑位在180美元左右，阻力位在210美元左右。成交量有所增加，表明市场关注度提高。"
                },
                {
                    "query": "微软(MSFT)的基本面分析",
                    "reference_answer": "微软(MSFT)拥有强大的软件生态系统，Azure云服务业务增长迅速。公司财务状况稳健，收入和利润持续增长。市盈率处于合理水平，股息收益率稳定。公司在AI领域持续投入，未来增长潜力巨大。"
                }
            ]
            
            test_cases_file = "temp_test_cases.json"
            with open(test_cases_file, 'w', encoding='utf-8') as f:
                json.dump(sample_test_cases, f, ensure_ascii=False, indent=2)
            
            # 运行完整分析
            output_dir = f"visualizations_{self.model_selector.default_model_type}"
            logger.info(f"正在运行评估和生成可视化结果到 {output_dir}...")
            results = langsmith_manager.run_full_analysis(
                test_cases_file=test_cases_file,
                project_name=f"ai-stocks-agent-{self.model_selector.default_model_type}",
                output_dir=output_dir
            )
            
            # 清理临时文件
            if os.path.exists(test_cases_file):
                os.remove(test_cases_file)
            
            logger.info("\n完整分析完成！")
            logger.info(f"可视化结果已保存到: {output_dir}")
            logger.info(f"请查看 {os.path.join(output_dir, 'evaluation_report.html')} 获取详细报告")
            
        except Exception as e:
            logger.error(f"运行完整分析时出错: {str(e)}", exc_info=True)
    
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
        logger.debug("  eval                - 运行LangSmith评估")
        logger.debug("  viz                 - 生成评估结果可视化")
        logger.debug("  analysis            - 运行完整评估与可视化分析")
        logger.debug("显示帮助信息")
    
    def _show_status(self):
        """显示当前状态信息"""
        status_info = f"\n当前状态:\n  模型类型: {self.model_selector.default_model_type}\n  RAG链: {'已初始化' if self.rag_chain else '未初始化'}\n  工作流: {'已初始化' if self.workflow else '未初始化'}\n  向量存储路径: {self.vector_store_manager.vector_store_path}\n  嵌入模型: {self.vector_store_manager.embedding_model}"
        
        # 检查LangSmith配置状态
        try:
            import os
            langsmith_configured = False
            if os.environ.get('LANGCHAIN_API_KEY') and os.environ.get('LANGCHAIN_TRACING_V2') == 'true':
                langsmith_configured = True
            status_info += f"\n  LangSmith配置: {'已配置' if langsmith_configured else '未配置'}"
        except Exception:
            status_info += "\n  LangSmith配置: 检查失败"
        
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