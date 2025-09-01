#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LangSmith集成示例用法"""

import os
import sys
import argparse
from typing import Dict, Any

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 导入必要的模块
from stockschain.model_selector import ModelSelector
from utils.logger import get_logger
from stocksmith import get_manager, StocksEvaluationRunner, StocksVisualizer

# 创建日志实例
logger = get_logger('stocksmith_example')

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='AI Stocks Agent - LangSmith评估与可视化工具')
    
    # 模型相关参数
    parser.add_argument('--model-type', type=str, default='ollama', 
                      choices=['ollama', 'vllm', 'api'],
                      help='使用的模型类型')
    
    # 评估相关参数
    parser.add_argument('--run-evaluation', action='store_true',
                      help='运行评估')
    parser.add_argument('--test-cases', type=str,
                      help='测试用例文件路径')
    parser.add_argument('--project-name', type=str, 
                      default='ai-stocks-agent-evaluation',
                      help='LangSmith项目名称')
    
    # 可视化相关参数
    parser.add_argument('--generate-visualizations', action='store_true',
                      help='生成可视化结果')
    parser.add_argument('--results-file', type=str, 
                      default='evaluation_results.json',
                      help='评估结果文件路径')
    
    # 综合分析参数
    parser.add_argument('--run-full-analysis', action='store_true',
                      help='运行完整的评估和可视化分析')
    parser.add_argument('--output-dir', type=str, 
                      default='./visualizations',
                      help='可视化输出目录')
    
    return parser.parse_args()

def create_sample_test_cases(file_path: str = "sample_test_cases.json") -> None:
    """创建示例测试用例文件"""
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
        },
        {
            "query": "亚马逊(AMZN)的市场竞争格局分析",
            "reference_answer": "亚马逊(AMZN)在电子商务领域占据主导地位，但面临来自沃尔玛、阿里巴巴等竞争对手的挑战。在云计算领域，AWS是市场领导者，但微软Azure和谷歌云也在快速增长。公司的Prime会员服务为其带来了稳定的收入和用户粘性。总体来看，亚马逊在多个领域都面临激烈竞争，但仍然保持着强大的市场地位。"
        },
        {
            "query": "谷歌(GOOGL)的增长前景如何？",
            "reference_answer": "谷歌(GOOGL)在搜索引擎和在线广告领域保持领先地位。公司的YouTube平台是全球最大的视频分享网站之一。在AI领域，谷歌通过Gemini系列模型和各种AI应用展示了其技术实力。云计算业务Google Cloud也在快速增长。公司的多元化业务布局和持续的研发投入使其具有良好的长期增长前景。"
        }
    ]
    
    with open(file_path, 'w', encoding='utf-8') as f:
        import json
        json.dump(sample_test_cases, f, ensure_ascii=False, indent=2)
    
    logger.info(f"示例测试用例已创建: {file_path}")

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_arguments()
    
    # 初始化模型选择器
    logger.info(f"初始化模型选择器，使用模型类型: {args.model_type}")
    model_selector = ModelSelector(default_model_type=args.model_type)
    
    # 初始化LangSmith管理器
    langsmith_manager = get_manager(model_selector)
    
    # 检查LangSmith配置状态
    if langsmith_manager.is_langsmith_configured():
        logger.info("LangSmith已成功配置，可以上传评估结果到LangSmith平台")
    else:
        logger.warning("LangSmith未配置，评估结果将仅保存在本地")
        logger.warning("提示: 如需使用LangSmith平台功能，请在.env文件中配置LANGCHAIN_API_KEY、LANGCHAIN_TRACING_V2和LANGCHAIN_PROJECT")
    
    # 根据命令行参数执行相应操作
    if args.run_full_analysis:
        # 运行完整的评估和可视化分析
        logger.info("运行完整的评估和可视化分析...")
        
        # 如果没有指定测试用例文件，创建示例测试用例
        if not args.test_cases:
            create_sample_test_cases("sample_test_cases.json")
            args.test_cases = "sample_test_cases.json"
        
        results = langsmith_manager.run_full_analysis(
            test_cases_file=args.test_cases,
            project_name=args.project_name,
            output_dir=args.output_dir
        )
        
        # 显示评估摘要
        if "error" not in results:
            logger.info("评估和可视化分析完成！")
            logger.info(f"可视化结果已保存到: {args.output_dir}")
            logger.info(f"请查看 {os.path.join(args.output_dir, 'evaluation_report.html')} 获取详细报告")
    
    elif args.run_evaluation:
        # 只运行评估
        logger.info("运行评估...")
        
        # 如果没有指定测试用例文件，创建示例测试用例
        if not args.test_cases:
            create_sample_test_cases("sample_test_cases.json")
            args.test_cases = "sample_test_cases.json"
        
        results = langsmith_manager.run_evaluation(
            test_cases_file=args.test_cases,
            project_name=args.project_name,
            output_file=args.results_file
        )
        
        # 显示评估摘要
        if "error" not in results:
            logger.info("评估完成！")
            logger.info(f"评估结果已保存到: {args.results_file}")
    
    elif args.generate_visualizations:
        # 只生成可视化
        logger.info("生成可视化结果...")
        
        langsmith_manager.generate_visualizations(
            results_file=args.results_file,
            output_dir=args.output_dir
        )
    
    else:
        # 默认：展示简单的评估和可视化示例
        logger.info("运行LangSmith集成示例...")
        
        # 创建示例测试用例
        create_sample_test_cases("sample_test_cases.json")
        
        # 运行评估
        logger.info("运行示例评估...")
        results = langsmith_manager.run_evaluation(
            test_cases_file="sample_test_cases.json",
            project_name=args.project_name,
            output_file="example_evaluation_results.json"
        )
        
        # 生成可视化
        if "error" not in results:
            logger.info("生成示例可视化...")
            langsmith_manager.generate_visualizations(
                results_file="example_evaluation_results.json",
                output_dir="./example_visualizations"
            )
            
            logger.info("LangSmith集成示例运行完成！")
            logger.info("示例可视化结果已保存到: ./example_visualizations")
            logger.info("请查看 ./example_visualizations/evaluation_report.html 获取详细报告")
            
            # 提供更多使用信息
            logger.info("\n更多使用方式：")
            logger.info("  1. 运行完整分析: python example_usage.py --run-full-analysis")
            logger.info("  2. 仅运行评估: python example_usage.py --run-evaluation")
            logger.info("  3. 仅生成可视化: python example_usage.py --generate-visualizations")
            logger.info("  4. 自定义模型类型: python example_usage.py --model-type vllm")
            logger.info("  5. 指定测试用例: python example_usage.py --test-cases your_test_cases.json")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"程序运行出错: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)