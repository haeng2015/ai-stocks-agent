from langchain.model_selector import ModelSelector

# 初始化模型选择器
try:
    print("正在初始化模型选择器...")
    model_selector = ModelSelector(default_model_type='ollama')
    print("模型选择器初始化成功")
    
    # 测试模型调用
    print("\n正在测试模型调用...")
    response = model_selector.invoke("什么是股票技术分析？")
    print("模型调用成功!")
    print(f"响应结果: {response}")
except Exception as e:
    print(f"错误: {str(e)}")
    import traceback
    traceback.print_exc()