# 日志记录工具文件
import logging
import os
import time
from typing import Optional


def get_logger(name: str, log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """
    获取日志记录器
    
    参数:
        name: str - 日志记录器名称
        log_file: Optional[str] - 日志文件路径，默认为None（仅控制台输出）
        level: int - 日志级别，默认为logging.INFO
        
    返回:
        logging.Logger - 日志记录器实例
    """
    # 创建日志记录器
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 避免重复添加处理器
    if not logger.handlers:
        # 创建格式化器
        formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # 文件处理器（如果指定了日志文件）
        if log_file:
            # 确保日志目录存在
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)
            
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    
    return logger


class Logger:
    """
    日志记录器类，提供更简洁的日志记录方法
    """
    
    def __init__(self, name: str, log_file: Optional[str] = None, level: int = logging.INFO):
        """
        初始化日志记录器
        
        参数:
            name: str - 日志记录器名称
            log_file: Optional[str] - 日志文件路径，默认为None（仅控制台输出）
            level: int - 日志级别，默认为logging.INFO
        """
        self.logger = get_logger(name, log_file, level)
    
    def info(self, msg: str, *args, **kwargs) -> None:
        """记录INFO级别日志"""
        self.logger.info(msg, *args, **kwargs)
    
    def debug(self, msg: str, *args, **kwargs) -> None:
        """记录DEBUG级别日志"""
        self.logger.debug(msg, *args, **kwargs)
    
    def warning(self, msg: str, *args, **kwargs) -> None:
        """记录WARNING级别日志"""
        self.logger.warning(msg, *args, **kwargs)
    
    def error(self, msg: str, *args, **kwargs) -> None:
        """记录ERROR级别日志"""
        self.logger.error(msg, *args, **kwargs)
    
    def critical(self, msg: str, *args, **kwargs) -> None:
        """记录CRITICAL级别日志"""
        self.logger.critical(msg, *args, **kwargs)


# 默认日志记录器
_default_logger = Logger(__name__)


# 便捷函数
def info(msg: str, *args, **kwargs) -> None:
    """便捷INFO日志记录"""
    _default_logger.info(msg, *args, **kwargs)


def debug(msg: str, *args, **kwargs) -> None:
    """便捷DEBUG日志记录"""
    _default_logger.debug(msg, *args, **kwargs)


def warning(msg: str, *args, **kwargs) -> None:
    """便捷WARNING日志记录"""
    _default_logger.warning(msg, *args, **kwargs)


def error(msg: str, *args, **kwargs) -> None:
    """便捷ERROR日志记录"""
    _default_logger.error(msg, *args, **kwargs)


def critical(msg: str, *args, **kwargs) -> None:
    """便捷CRITICAL日志记录"""
    _default_logger.critical(msg, *args, **kwargs)
