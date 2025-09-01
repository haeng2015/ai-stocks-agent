# -*- coding: utf-8 -*-
"""
日志工具模块
负责配置和提供日志记录功能，将日志输出到控制台和文件
"""
import os
import logging
from datetime import datetime
import sys

class Logger:
    """日志记录器类，提供统一的日志记录功能"""
    
    def __init__(self, name='ai-stocks-agent', log_dir='logs', log_level=logging.INFO):
        """
        初始化日志记录器
        
        Args:
            name: 日志记录器名称
            log_dir: 日志文件保存目录
            log_level: 日志级别
        """
        # 确保日志目录存在
        self.log_dir = log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # 获取当前日期，用于日志文件名
        today = datetime.now().strftime('%Y-%m-%d')
        log_file = os.path.join(log_dir, f'{name}_{today}.log')
        
        # 创建logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level)
        
        # 避免重复添加handler
        if not self.logger.handlers:
            # 创建控制台handler
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(log_level)
            
            # 创建文件handler
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(log_level)
            
            # 定义日志格式
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            
            # 设置handler的格式
            console_handler.setFormatter(formatter)
            file_handler.setFormatter(formatter)
            
            # 添加handler到logger
            self.logger.addHandler(console_handler)
            self.logger.addHandler(file_handler)
    
    def debug(self, message):
        """记录debug级别的日志"""
        self.logger.debug(message)
    
    def info(self, message):
        """记录info级别的日志"""
        self.logger.info(message)
    
    def warning(self, message):
        """记录warning级别的日志"""
        self.logger.warning(message)
    
    def error(self, message):
        """记录error级别的日志"""
        self.logger.error(message)
    
    def critical(self, message):
        """记录critical级别的日志"""
        self.logger.critical(message)

# 全局日志实例字典，用于缓存已创建的日志记录器
_loggers = {}

# 创建全局日志实例
def get_logger(name='ai-stocks-agent'):
    """获取日志记录器实例，避免重复创建"""
    if name not in _loggers:
        _loggers[name] = Logger(name).logger
    return _loggers[name]

# 配置全局日志
def setup_logging():
    """设置全局日志配置，只配置根日志记录器的基本属性"""
    # 确保logs目录存在
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # 配置根日志记录器的级别，避免重复添加handler
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # 确保根日志记录器没有默认的handler（防止重复日志）
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

# 创建一个简单的日志装饰器
def log_function(func):
    """记录函数调用的装饰器"""
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        logger.debug(f"调用函数: {func.__name__}")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"函数 {func.__name__} 执行成功")
            return result
        except Exception as e:
            logger.error(f"函数 {func.__name__} 执行失败: {str(e)}")
            raise
    return wrapper