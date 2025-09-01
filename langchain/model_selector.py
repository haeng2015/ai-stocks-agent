from .ollama.ollama_llm import OllamaLLM
from .vllm.vllm_llm import VLLM
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

class ModelSelector:
    """模型选择器，用于在不同的模型推理后端之间进行选择"""
    
    # 支持的模型类型
    MODEL_TYPES = {
        'ollama': OllamaLLM,
        'vllm': VLLM
    }
    
    def __init__(self, default_model_type=None):
        """
        初始化模型选择器
        
        Args:
            default_model_type: 默认使用的模型类型，可选值为'ollama'或'vllm'
        """
        self.default_model_type = default_model_type or os.getenv('DEFAULT_MODEL_TYPE', 'ollama')
        
        # 确保默认模型类型有效
        if self.default_model_type not in self.MODEL_TYPES:
            raise ValueError(f"不支持的模型类型: {self.default_model_type}，支持的类型有: {list(self.MODEL_TYPES.keys())}")
        
        # 存储已初始化的模型实例
        self.models = {}
    
    def get_model(self, model_type=None, **kwargs):
        """
        获取指定类型的模型实例
        
        Args:
            model_type: 模型类型，可选值为'ollama'或'vllm'
            **kwargs: 传递给模型初始化的参数
            
        Returns:
            模型实例
        """
        model_type = model_type or self.default_model_type
        
        # 检查模型类型是否有效
        if model_type not in self.MODEL_TYPES:
            raise ValueError(f"不支持的模型类型: {model_type}，支持的类型有: {list(self.MODEL_TYPES.keys())}")
        
        # 如果模型实例已存在且没有提供新的参数，则返回已存在的实例
        key = f"{model_type}_{hash(frozenset(kwargs.items()))}"
        if key not in self.models:
            # 创建新的模型实例
            model_class = self.MODEL_TYPES[model_type]
            self.models[key] = model_class(**kwargs)
        
        return self.models[key]
    
    def invoke(self, prompt, model_type=None, system_prompt=None, **kwargs):
        """
        调用指定类型的模型生成文本
        
        Args:
            prompt: 用户输入文本
            model_type: 模型类型
            system_prompt: 系统提示词
            **kwargs: 传递给模型invoke方法的参数
            
        Returns:
            模型生成的文本
        """
        # 确定实际使用的模型类型
        actual_model_type = model_type or self.default_model_type
        model = self.get_model(model_type)
        
        # 根据模型类型调用相应的方法
        if actual_model_type == 'ollama':
            # 对于Ollama模型，使用direct_invoke方法
            return model.direct_invoke(prompt, system_prompt=system_prompt, **kwargs)
        else:
            # 对于其他模型，保持原有调用方式
            return model.invoke(prompt, system_prompt=system_prompt, **kwargs)
    
    def batch_invoke(self, prompts, model_type=None, system_prompt=None, **kwargs):
        """
        批量调用指定类型的模型生成文本
        
        Args:
            prompts: 输入文本列表
            model_type: 模型类型
            system_prompt: 系统提示词
            **kwargs: 传递给模型batch_invoke方法的参数
            
        Returns:
            模型生成的文本列表
        """
        model = self.get_model(model_type)
        return model.batch_invoke(prompts, system_prompt=system_prompt, **kwargs)

# 示例用法
if __name__ == "__main__":
    # 初始化模型选择器
    model_selector = ModelSelector(default_model_type='ollama')
    
    # 使用默认模型(ollama)进行调用
    try:
        response_ollama = model_selector.invoke("解释一下什么是股票技术分析？")
        print("Ollama模型调用结果:")
        print(response_ollama)
        print("\n" + "="*50 + "\n")
    except Exception as e:
        print(f"Ollama调用失败: {str(e)}")
    
    # 使用vllm模型进行调用
    try:
        response_vllm = model_selector.invoke("解释一下什么是股票技术分析？", model_type='vllm')
        print("VLLM模型调用结果:")
        print(response_vllm)
    except Exception as e:
        print(f"VLLM调用失败: {str(e)}")