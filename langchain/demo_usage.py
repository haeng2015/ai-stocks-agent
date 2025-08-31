from model_selector import ModelSelector
import time

"""
股票智能体模型推理示例

此文件演示了如何使用Ollama和VLLM两种本地模型推理方式来处理股票相关的查询
"""

# 示例查询列表
stock_queries = [
    "什么是股票的市盈率(P/E ratio)？它有什么意义？",
    "请分析一下技术分析中的移动平均线指标。",
    "什么是价值投资？它的核心原则是什么？",
    "如何解读公司的资产负债表？"
]

# 系统提示词
system_prompt = "你是一位专业的金融分析师，拥有丰富的股票市场知识和经验。请用专业但易懂的语言回答用户的问题。"

def run_demo():
    """运行模型推理示例"""
    print("===== 股票智能体模型推理示例 =====\n")
    
    # 初始化模型选择器
    model_selector = ModelSelector()
    
    # 测试Ollama模型
    print("【测试Ollama模型】")
    try:
        start_time = time.time()
        for query in stock_queries[:2]:  # 只测试前两个查询，避免耗时过长
            print(f"\n问: {query}")
            response = model_selector.invoke(query, model_type='ollama', system_prompt=system_prompt)
            print(f"答: {response}")
            print("-" * 50)
        ollama_time = time.time() - start_time
        print(f"\nOllama模型完成测试，耗时: {ollama_time:.2f}秒\n")
    except Exception as e:
        print(f"Ollama模型测试失败: {str(e)}")
    
    # 测试VLLM模型
    print("【测试VLLM模型】")
    try:
        start_time = time.time()
        for query in stock_queries[:2]:  # 只测试前两个查询，避免耗时过长
            print(f"\n问: {query}")
            response = model_selector.invoke(query, model_type='vllm', system_prompt=system_prompt)
            print(f"答: {response}")
            print("-" * 50)
        vllm_time = time.time() - start_time
        print(f"\nVLLM模型完成测试，耗时: {vllm_time:.2f}秒\n")
    except Exception as e:
        print(f"VLLM模型测试失败: {str(e)}")
        print("请确保vllm服务已启动，可以使用以下命令启动:")
        print("python -m vllm.entrypoints.api_server --model meta-llama/Llama-3-8b-instruct --port 8000")
    
    # 比较两种模型的性能
    if 'ollama_time' in locals() and 'vllm_time' in locals():
        print("【性能比较】")
        if ollama_time < vllm_time:
            print(f"Ollama模型比VLLM模型快 {vllm_time/ollama_time:.2f}倍")
        else:
            print(f"VLLM模型比Ollama模型快 {ollama_time/vllm_time:.2f}倍")
    
    print("\n===== 测试完成 =====")

if __name__ == "__main__":
    run_demo()