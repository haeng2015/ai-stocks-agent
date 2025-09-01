import requests
import json
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

class VLLM:
    """VLLM本地模型推理封装类"""
    
    def __init__(self, api_url=None, model=None, temperature=0.7, max_tokens=1024):
        """
        初始化VLLM实例
        
        Args:
            api_url: VLLM API的URL，默认为环境变量中的VLLM_API_URL
            model: 使用的模型名称，默认为环境变量中的VLLM_MODEL
            temperature: 生成文本的随机性，默认为0.7
            max_tokens: 最大生成的token数，默认为1024
        """
        self.api_url = api_url or os.getenv('VLLM_API_URL', 'http://localhost:8000/generate')
        self.model = model or os.getenv('VLLM_MODEL', 'meta-llama/Llama-3-8b-instruct')
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # 检查VLLM服务是否可用
        try:
            self._check_service()
            print(f"VLLM服务已连接: {self.api_url}")
        except Exception as e:
            print(f"警告: 无法连接到VLLM服务 - {str(e)}")
            print("请确保vllm服务已启动，可以使用以下命令启动:\n", 
                  f"python -m vllm.entrypoints.api_server --model {self.model} --port 8000")
    
    def _check_service(self):
        """检查VLLM服务是否可用"""
        try:
            response = requests.get(self.api_url.replace('generate', 'health'), timeout=5)
            response.raise_for_status()
        except requests.exceptions.RequestException:
            # 如果health端点不可用，尝试发送一个简单的请求
            try:
                headers = {"Content-Type": "application/json"}
                payload = {
                    "prompt": "test",
                    "model": self.model,
                    "temperature": 0,
                    "max_tokens": 1
                }
                response = requests.post(self.api_url, headers=headers, data=json.dumps(payload), timeout=5)
                response.raise_for_status()
            except requests.exceptions.RequestException as e:
                raise ConnectionError(f"无法连接到VLLM服务: {str(e)}")
    
    def invoke(self, prompt, system_prompt=None, temperature=None, max_tokens=None):
        """
        调用VLLM生成文本
        
        Args:
            prompt: 用户输入文本
            system_prompt: 系统提示词
            temperature: 生成文本的随机性，如果为None则使用初始化时的值
            max_tokens: 最大生成的token数，如果为None则使用初始化时的值
            
        Returns:
            模型生成的文本
        """
        # 构建完整的提示
        if system_prompt:
            # 对于Llama-3风格的模型，使用<|start_header_id|>system<|end_header_id|>格式
            full_prompt = f"<|start_header_id|>system<|end_header_id|>\n{system_prompt}<|eot_id|>\n<|start_header_id|>user<|end_header_id|>\n{prompt}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>"
        else:
            full_prompt = f"<|start_header_id|>user<|end_header_id|>\n{prompt}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>"
        
        # 准备请求参数
        headers = {"Content-Type": "application/json"}
        payload = {
            "prompt": full_prompt,
            "model": self.model,
            "temperature": temperature or self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
            "stop": ["<|eot_id|>"]  # 确保生成在合适的位置停止
        }
        
        # 发送请求
        try:
            response = requests.post(self.api_url, headers=headers, data=json.dumps(payload), timeout=60)
            response.raise_for_status()
            
            # 解析响应
            result = response.json()
            return result.get("text", "")
        except requests.exceptions.RequestException as e:
            raise Exception(f"VLLM调用失败: {str(e)}")
        except json.JSONDecodeError:
            raise Exception(f"VLLM响应解析失败: {response.text}")
    
    def batch_invoke(self, prompts, system_prompt=None, temperature=None, max_tokens=None):
        """
        批量调用VLLM生成文本
        
        Args:
            prompts: 输入文本列表
            system_prompt: 系统提示词，将应用于所有输入
            temperature: 生成文本的随机性
            max_tokens: 最大生成的token数
            
        Returns:
            模型生成的文本列表
        """
        results = []
        for prompt in prompts:
            results.append(self.invoke(prompt, system_prompt, temperature, max_tokens))
        return results

# 示例用法
if __name__ == "__main__":
    # 初始化VLLM
    vllm = VLLM()
    
    # 测试基本调用
    try:
        response = vllm.invoke("什么是股票基本面分析？")
        print("基本调用结果:")
        print(response)
        print("\n" + "="*50 + "\n")
        
        # 测试带系统提示的调用
        system_prompt = "你是一位专业的金融分析师，请用简单易懂的语言解释复杂的金融概念。"
        response_with_system = vllm.invoke("什么是股票基本面分析？", system_prompt)
        print("带系统提示的调用结果:")
        print(response_with_system)
    except Exception as e:
        print(f"错误: {str(e)}")