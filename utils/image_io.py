# 图像处理工具文件
import cv2
import numpy as np
import os
from typing import Optional, List
from utils.logger import get_logger

# 获取日志记录器
logger = get_logger(__name__)


def read_image(image_path: str, color_mode: str = 'bgr') -> Optional[np.ndarray]:
    """
    读取图像文件
    
    参数:
        image_path: str - 图像文件路径
        color_mode: str - 颜色模式，可选值：'bgr'（默认）、'rgb'、'gray'
        
    返回:
        Optional[np.ndarray] - 图像数组，如果读取失败则返回None
    """
    if not os.path.exists(image_path):
        logger.error(f"图像文件不存在: {image_path}")
        return None
    
    try:
        # 读取图像，支持中文路径
        with open(image_path, 'rb') as f:
            img_data = f.read()
        
        # 使用cv2.imdecode读取图像数据
        img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            logger.error(f"无法读取图像文件: {image_path}")
            return None
        
        # 转换颜色模式
        if color_mode == 'rgb':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif color_mode == 'gray':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif color_mode != 'bgr':
            logger.warning(f"未知的颜色模式: {color_mode}，将使用默认的BGR模式")
        
        logger.debug(f"成功读取图像: {image_path}，尺寸: {img.shape}")
        return img
    except Exception as e:
        logger.error(f"读取图像时出错: {image_path}，错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def save_image(image: np.ndarray, save_path: str, color_mode: str = 'bgr', input_color_mode: str = 'rgb') -> bool:
    """
    保存图像文件
    
    参数:
        image: np.ndarray - 图像数组
        save_path: str - 保存路径
        color_mode: str - 保存时使用的颜色模式，可选值：'bgr'（默认）、'rgb'、'gray'
        input_color_mode: str - 输入图像的实际颜色模式，可选值：None（自动检测）、'bgr'、'rgb'
        
    返回:
        bool - 保存成功返回True，否则返回False
    """
    try:
        # 确保图像是numpy数组
        if not isinstance(image, np.ndarray):
            logger.error("输入不是numpy数组")
            return False
        
        # 转换颜色模式
        img_to_save = image.copy()
        
        # 确定输入图像的实际颜色模式
        actual_input_mode = input_color_mode
        
        # 首先将图像转换为目标颜色模式
        if color_mode == 'gray':
            # 如果需要保存为灰度图
            if len(img_to_save.shape) == 3 and img_to_save.shape[2] == 3:
                if actual_input_mode == 'rgb':
                    # 输入是RGB，转换为灰度
                    img_to_save = cv2.cvtColor(img_to_save, cv2.COLOR_RGB2GRAY)
                else:
                    # 输入是BGR，转换为灰度
                    img_to_save = cv2.cvtColor(img_to_save, cv2.COLOR_BGR2GRAY)
        else:
            # 对于RGB或BGR模式，确保图像是3通道的
            if len(img_to_save.shape) == 2:
                # 灰度图转换为3通道
                img_to_save = cv2.cvtColor(img_to_save, cv2.COLOR_GRAY2BGR)
            elif img_to_save.shape[2] != 3:
                # 确保是3通道图像
                img_to_save = cv2.cvtColor(img_to_save, cv2.COLOR_RGBA2BGR)
            
            # cv2.imwrite总是期望BGR格式，所以确保输入转换为BGR
            if actual_input_mode == 'rgb':
                # 输入是RGB，转换为BGR用于cv2.imwrite
                img_to_save = cv2.cvtColor(img_to_save, cv2.COLOR_RGB2BGR)
        
        # 确保保存目录存在
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # 保存图像，支持中文路径
        try:
            # 使用cv2.imencode将图像编码为字节流
            _, buffer = cv2.imencode('.png', img_to_save)
            img_bytes = buffer.tobytes()
            
            # 使用open函数保存，支持中文路径
            with open(save_path, 'wb') as f:
                f.write(img_bytes)
            
            logger.debug(f"成功保存图像: {save_path}")
            return True
        except Exception as e:
            logger.error(f"无法保存图像: {save_path}，错误: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    except Exception as e:
        logger.error(f"保存图像时出错: {save_path}，错误: {str(e)}")
        return False


def resize_image(image: np.ndarray, target_size: tuple, interpolation: int = cv2.INTER_LINEAR) -> Optional[np.ndarray]:
    """
    调整图像大小
    
    参数:
        image: np.ndarray - 输入图像
        target_size: tuple - 目标尺寸 (width, height)
        interpolation: int - 插值方法，默认cv2.INTER_LINEAR
        
    返回:
        Optional[np.ndarray] - 调整大小后的图像，失败则返回None
    """
    try:
        resized = cv2.resize(image, target_size, interpolation=interpolation)
        logger.debug(f"图像大小调整成功，从 {image.shape[:2][::-1]} 调整到 {target_size}")
        return resized
    except Exception as e:
        logger.error(f"调整图像大小时出错: {str(e)}")
        return None


def normalize_image(image: np.ndarray, mean: tuple = (0, 0, 0), std: tuple = (1, 1, 1)) -> Optional[np.ndarray]:
    """
    归一化图像
    
    参数:
        image: np.ndarray - 输入图像
        mean: tuple - 均值，默认(0, 0, 0)
        std: tuple - 标准差，默认(1, 1, 1)
        
    返回:
        Optional[np.ndarray] - 归一化后的图像，失败则返回None
    """
    try:
        # 转换为float32类型
        img_float = image.astype(np.float32)
        
        # 归一化
        img_normalized = (img_float - np.array(mean)) / np.array(std)
        
        logger.debug("图像归一化成功")
        return img_normalized
    except Exception as e:
        logger.error(f"归一化图像时出错: {str(e)}")
        return None


def batch_read_images(image_paths: List[str], color_mode: str = 'bgr') -> List[Optional[np.ndarray]]:
    """
    批量读取图像文件
    
    参数:
        image_paths: List[str] - 图像文件路径列表
        color_mode: str - 颜色模式，可选值：'bgr'（默认）、'rgb'、'gray'
        
    返回:
        List[Optional[np.ndarray]] - 图像数组列表，读取失败的位置为None
    """
    images = []
    for path in image_paths:
        img = read_image(path, color_mode)
        images.append(img)
    
    logger.info(f"批量读取完成，共 {len(image_paths)} 张图像，成功读取 {sum(1 for img in images if img is not None)} 张")
    return images


def convert_color_space(image: np.ndarray, src_mode: str, dst_mode: str) -> Optional[np.ndarray]:
    """
    转换图像颜色空间
    
    参数:
        image: np.ndarray - 输入图像
        src_mode: str - 源颜色模式，可选值：'bgr'、'rgb'、'gray'
        dst_mode: str - 目标颜色模式，可选值：'bgr'、'rgb'、'gray'
        
    返回:
        Optional[np.ndarray] - 转换后的图像，失败则返回None
    """
    try:
        # 构建转换映射
        conversion_map = {
            ('bgr', 'rgb'): cv2.COLOR_BGR2RGB,
            ('rgb', 'bgr'): cv2.COLOR_RGB2BGR,
            ('bgr', 'gray'): cv2.COLOR_BGR2GRAY,
            ('rgb', 'gray'): cv2.COLOR_RGB2GRAY,
            ('gray', 'bgr'): cv2.COLOR_GRAY2BGR,
            ('gray', 'rgb'): cv2.COLOR_GRAY2RGB
        }
        
        # 检查是否支持的转换
        if (src_mode, dst_mode) in conversion_map:
            converted = cv2.cvtColor(image, conversion_map[(src_mode, dst_mode)])
            logger.debug(f"颜色空间转换成功: {src_mode} -> {dst_mode}")
            return converted
        elif src_mode == dst_mode:
            # 无需转换
            return image
        else:
            logger.error(f"不支持的颜色空间转换: {src_mode} -> {dst_mode}")
            return None
    except Exception as e:
        logger.error(f"转换颜色空间时出错: {str(e)}")
        return None


def ensure_image_format(image: np.ndarray, target_mode: str = 'bgr', target_dtype: type = np.uint8) -> Optional[np.ndarray]:
    """
    确保图像格式符合要求
    
    参数:
        image: np.ndarray - 输入图像
        target_mode: str - 目标颜色模式，可选值：'bgr'（默认）、'rgb'、'gray'
        target_dtype: type - 目标数据类型，默认np.uint8
        
    返回:
        Optional[np.ndarray] - 处理后的图像，失败则返回None
    """
    try:
        # 确保是numpy数组
        if not isinstance(image, np.ndarray):
            logger.error("输入不是numpy数组")
            return None
        
        # 确定当前模式
        if len(image.shape) == 2:
            current_mode = 'gray'
        elif len(image.shape) == 3 and image.shape[2] == 3:
            current_mode = 'bgr'  # 假设输入是BGR格式
        else:
            logger.error(f"不支持的图像维度: {image.shape}")
            return None
        
        # 转换颜色空间
        if current_mode != target_mode:
            image = convert_color_space(image, current_mode, target_mode)
            if image is None:
                return None
        
        # 转换数据类型
        if image.dtype != target_dtype:
            if target_dtype == np.uint8:
                # 确保值在合理范围内
                image = np.clip(image, 0, 255).astype(target_dtype)
            else:
                image = image.astype(target_dtype)
            
        logger.debug(f"图像格式确保成功，模式: {target_mode}，数据类型: {target_dtype}")
        return image
    except Exception as e:
        logger.error(f"确保图像格式时出错: {str(e)}")
        return None
