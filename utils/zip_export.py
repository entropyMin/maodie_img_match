# 结果导出工具文件
import os
import zipfile
import json
from typing import List, Dict, Any
import numpy as np
from utils.logger import get_logger
from utils.image_io import save_image

# 获取日志记录器
logger = get_logger(__name__)


def export_results(matches: List[Dict[str, Any]], images_a: List[np.ndarray], 
                   images_b: List[np.ndarray], output_dir: str, 
                   export_format: str = 'zip') -> str:
    """
    导出匹配结果
    
    参数:
        matches: List[Dict[str, Any]] - 匹配结果列表
        images_a: List[np.ndarray] - 第一批图像列表
        images_b: List[np.ndarray] - 第二批图像列表
        output_dir: str - 输出目录
        export_format: str - 导出格式，可选值：'zip'（默认）、'folder'
        
    返回:
        str - 导出文件路径或目录路径
    """
    try:
        # 确保输出目录存在
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 创建结果目录
        result_dir = os.path.join(output_dir, "match_results")
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        
        # 保存匹配信息
        match_info_path = os.path.join(result_dir, "matches.json")
        with open(match_info_path, 'w', encoding='utf-8') as f:
            json.dump(matches, f, ensure_ascii=False, indent=2)
        
        # 保存图像对
        pairs_dir = os.path.join(result_dir, "image_pairs")
        if not os.path.exists(pairs_dir):
            os.makedirs(pairs_dir)
        
        for i, match in enumerate(matches):
            a_idx = match.get('a_index')
            b_idx = match.get('b_index')
            score = match.get('score', 0.0)
            
            if a_idx is not None and b_idx is not None:
                # 确保索引在有效范围内
                if a_idx < len(images_a) and b_idx < len(images_b):
                    img_a = images_a[a_idx]
                    img_b = images_b[b_idx]
                    
                    if img_a is not None and img_b is not None:
                        # 保存图像对
                        pair_dir = os.path.join(pairs_dir, f"pair_{i}")
                        if not os.path.exists(pair_dir):
                            os.makedirs(pair_dir)
                        
                        # 保存A组图像
                        a_path = os.path.join(pair_dir, f"a_{a_idx}.png")
                        save_image(img_a, a_path)
                        
                        # 保存B组图像
                        b_path = os.path.join(pair_dir, f"b_{b_idx}.png")
                        save_image(img_b, b_path)
                        
                        # 保存匹配信息
                        pair_info = {
                            "a_index": a_idx,
                            "b_index": b_idx,
                            "score": score,
                            "a_path": f"a_{a_idx}.png",
                            "b_path": f"b_{b_idx}.png"
                        }
                        info_path = os.path.join(pair_dir, "info.json")
                        with open(info_path, 'w', encoding='utf-8') as f:
                            json.dump(pair_info, f, ensure_ascii=False, indent=2)
        
        # 导出为zip文件
        if export_format == 'zip':
            zip_path = os.path.join(output_dir, "match_results.zip")
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # 添加所有文件到zip
                for root, _, files in os.walk(result_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, result_dir)
                        zipf.write(file_path, arcname)
            
            logger.info(f"匹配结果已导出为zip文件: {zip_path}")
            return zip_path
        else:
            logger.info(f"匹配结果已导出到目录: {result_dir}")
            return result_dir
    except Exception as e:
        logger.error(f"导出结果时出错: {str(e)}")
        return ""


def export_match_statistics(statistics: Dict[str, Any], output_path: str) -> bool:
    """
    导出匹配统计信息
    
    参数:
        statistics: Dict[str, Any] - 统计信息字典
        output_path: str - 输出文件路径
        
    返回:
        bool - 导出成功返回True，否则返回False
    """
    try:
        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 保存统计信息
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(statistics, f, ensure_ascii=False, indent=2)
        
        logger.info(f"统计信息已导出: {output_path}")
        return True
    except Exception as e:
        logger.error(f"导出统计信息时出错: {str(e)}")
        return False


def generate_statistics(matches: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    生成匹配统计信息
    
    参数:
        matches: List[Dict[str, Any]] - 匹配结果列表
        
    返回:
        Dict[str, Any] - 统计信息字典
    """
    try:
        total_matches = len(matches)
        successful_matches = sum(1 for match in matches if match.get('b_index') is not None)
        low_confidence_matches = sum(1 for match in matches if match.get('low_confidence', False))
        
        # 计算平均得分
        scores = [match.get('score', 0.0) for match in matches if match.get('b_index') is not None]
        avg_score = sum(scores) / len(scores) if scores else 0.0
        
        statistics = {
            "total_matches": total_matches,
            "successful_matches": successful_matches,
            "successful_rate": successful_matches / total_matches if total_matches > 0 else 0.0,
            "low_confidence_matches": low_confidence_matches,
            "low_confidence_rate": low_confidence_matches / successful_matches if successful_matches > 0 else 0.0,
            "average_score": avg_score,
            "scores": scores
        }
        
        logger.info(f"生成统计信息: {statistics}")
        return statistics
    except Exception as e:
        logger.error(f"生成统计信息时出错: {str(e)}")
        return {}
