# -*- coding: utf-8 -*-
"""
测试模型选择器的修复是否成功
"""
import os
import sys

# 调整sys.path以确保优先导入项目中的模块
project_root = os.path.abspath('.')
sys.path.insert(0, project_root)

from langchain.model_selector import ModelSelector

def test_model_selector():
    """测试模型选择器是否能够正常工作"""
    try:
        # 初始化模型选择器
        print("初始化模型选择器...")
        model_selector = ModelSelector(default_model_type='ollama')
        print("模型选择器初始化成功")
        
        # 尝试调用模型
        print("\n尝试调用模型...")
        try:
            response = model_selector.invoke("解释一下什么是股票技术分析？")
            print("模型调用成功！")
            print(f"响应结果: {response}")
        except Exception as e:
            print(f"模型调用出错: {str(e)}")
            
    except Exception as e:
        print(f"测试失败: {str(e)}")

if __name__ == "__main__":
    test_model_selector()