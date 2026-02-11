import cv2
import numpy as np
import os


def extract_main_structure(
    img_bgr,
    canny_low=100,    # 增加默认低阈值
    canny_high=200,  # 增加默认高阈值
    min_area_ratio=0.02  # 增加默认最小面积比例
):
    """
    从图像中提取主体结构，用于跨版本匹配
    
    输入:
        img_bgr: numpy.ndarray - BGR格式的输入图像
        canny_low: int - Canny边缘检测的低阈值
        canny_high: int - Canny边缘检测的高阈值
        min_area_ratio: float - 最小连通域面积占比（相对于图像总面积）
        
    输出:
        numpy.ndarray - 结构二值图（uint8, 0/255）
        
    处理流程:
        1. 灰度化
        2. 高斯滤波（去噪声）
        3. Canny边缘提取
        4. 形态学操作（连接断裂边缘）
        5. 连通域分析（提取主体结构）
    """
    # 输入验证
    if img_bgr is None:
        raise ValueError("输入图像不能为空")
    
    if len(img_bgr.shape) != 3:
        raise ValueError("输入图像必须是3通道BGR格式")
    
    h, w = img_bgr.shape[:2]
    
    # 确保图像尺寸有效
    if h < 10 or w < 10:
        raise ValueError("输入图像尺寸过小，无法提取结构")
    
    # 1. 转换为灰度图
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # 2. 高斯滤波（去噪声）
    blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    
    # 3. 边缘提取 - 尝试使用Scharr算子获取更强的边缘
    # 使用Scharr算子计算梯度
    grad_x = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=-1)
    grad_y = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=-1)
    
    # 转换为绝对值并归一化
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    
    # 合并梯度
    edges = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    
    # 再使用Canny边缘检测增强
    edges = cv2.Canny(edges, canny_low, canny_high)
    
    # 4. 形态学操作（连接断裂边缘）
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # 先膨胀再腐蚀，增强边缘连接
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_DILATE, kernel, iterations=1)
    edges_closed = cv2.morphologyEx(edges_closed, cv2.MORPH_ERODE, kernel, iterations=1)
    
    # 5. 连通域分析，提取主体结构
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        edges_closed,
        connectivity=8  # 8-连通域
    )
    
    min_area = h * w * min_area_ratio
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # 计算所有连通域的面积
    areas = [stats[i, cv2.CC_STAT_AREA] for i in range(1, num_labels)]
    
    if areas:
        # 只保留面积最大的前N个连通域
        max_area = max(areas)
        # 保留面积大于最大面积一半的连通域
        threshold_area = max_area * 0.5
        
        for i in range(1, num_labels):  # 跳过背景（标签0）
            area = stats[i, cv2.CC_STAT_AREA]
            if area > threshold_area:
                mask[labels == i] = 255
    else:
        # 如果没有连通域，直接使用Canny边缘结果
        mask = edges
    
    return mask


def visualize_structure(img, structure_mask):
    """
    可视化结构提取结果
    
    输入:
        img: numpy.ndarray - BGR格式的原始图像
        structure_mask: numpy.ndarray - 结构二值图
        
    输出:
        numpy.ndarray - 可视化结果图像
    """
    if img is None or structure_mask is None:
        raise ValueError("输入图像或结构掩码不能为空")
    
    if img.shape[:2] != structure_mask.shape[:2]:
        raise ValueError("输入图像和结构掩码尺寸不匹配")
    
    overlay = img.copy()
    overlay[structure_mask > 0] = (0, 255, 0)  # 结构区域标记为绿色
    return cv2.addWeighted(img, 0.7, overlay, 0.3, 0)


def batch_extract_structure(image_paths):
    """
    批量处理图像，提取主体结构
    
    输入:
        image_paths: list - 图像文件路径列表
        
    输出:
        dict - 以图像路径为键，结构掩码为值的字典
    """
    if not isinstance(image_paths, list):
        raise ValueError("image_paths必须是列表类型")
    
    results = {}
    for p in image_paths:
        if not os.path.exists(p):
            print(f"警告：图像文件不存在 - {p}")
            continue
        
        img = cv2.imread(p)
        if img is None:
            print(f"警告：无法读取图像文件 - {p}")
            continue
        
        try:
            mask = extract_main_structure(img)
            results[p] = mask
        except Exception as e:
            print(f"警告：处理图像文件时出错 {p} - {str(e)}")
            continue
    
    return results


# 导入必要的模块
import os