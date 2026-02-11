# 调度入口文件
import sys
import cv2
import numpy as np
import os
import yaml
from typing import List, Dict, Tuple

import logging

# 导入项目核心组件
from algorithms.traditional.matcher import TraditionalMatcher
from core.pipeline import MatchPipeline
from utils.logger import get_logger

# 获取日志记录器
logger = get_logger(__name__, level=logging.DEBUG)

class Controller:
    """应用控制器类"""
    
    def __init__(self):
        """初始化控制器"""
        # 延迟初始化的属性
        self._config = None
        self._matcher = None
        self._pipeline = None
        
        # 保存匹配结果，用于后续打包
        self.matched_images = []
        self.original_batch_A = []
        self.original_batch_B = []
        self.matched_image_filenames = []
    
    @property
    def config(self) -> Dict:
        """获取配置（懒加载）"""
        if self._config is None:
            self._config = self._load_config()
        return self._config
    
    @property
    def matcher(self):
        """获取匹配器（懒加载）"""
        if self._matcher is None:
            self._init_matcher()
        return self._matcher
    
    @property
    def pipeline(self):
        """获取匹配管道（懒加载）"""
        if self._pipeline is None:
            self._init_pipeline()
        return self._pipeline
    
    def _load_config(self) -> Dict:
        """
        加载配置文件
        
        返回:
            Dict: 配置字典
        """
        # 获取项目根目录（支持打包后的环境）
        if getattr(sys, 'frozen', False):
            # 打包后的环境
            base_path = sys._MEIPASS
        else:
            # 开发环境
            base_path = os.path.dirname(os.path.dirname(__file__))
        
        config_path = os.path.join(base_path, "config", "traditional.yaml")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"成功加载配置文件: {config_path}")
            return config
        except Exception as e:
            logger.error(f"加载配置文件失败: {str(e)}")
            # 返回默认配置
            return {
                'phash': {'hash_size': 8, 'highfreq_factor': 4},
                'structure': {'canny_threshold1': 50, 'canny_threshold2': 150, 'min_contour_area': 0.01},
                'matcher': {'similarity_threshold': 0.8, 'max_results': 10}
            }
    
    def _init_matcher(self):
        """初始化匹配器（懒加载）"""
        struct_config = self.config.get('structure', {})
        
        self._matcher = TraditionalMatcher(
            canny_low=struct_config.get('canny_threshold1', 50),
            canny_high=struct_config.get('canny_threshold2', 150),
            min_area_ratio=struct_config.get('min_contour_area', 0.01)
        )
        logger.info("匹配器初始化完成")
    
    def _init_pipeline(self):
        """初始化匹配管道（懒加载）"""
        matcher_config = self.config.get('matcher', {})
        
        self._pipeline = MatchPipeline(
            matcher=self.matcher,
            use_assignment=True,
            confidence_threshold=matcher_config.get('similarity_threshold', 0.05)
        )
        
        logger.info("匹配管道初始化完成")
    
    def process_batches(self, batch_A: List, batch_B: List) -> Tuple[List[np.ndarray], str]:
        """
        处理两批图片的匹配
        
        参数:
            batch_A: 第一批图片列表（原图，有顺序）
            batch_B: 第二批图片列表（重绘图，乱序）
            
        返回:
            Tuple[List[np.ndarray], str]: 
                - 按第一批顺序匹配后的第二批图片
                - 匹配信息字符串
        """
        if not batch_A or not batch_B:
            return [], "错误：请至少上传一张图片"
        
        try:
            logger.info(f"开始处理匹配任务，A组图片数量: {len(batch_A)}, B组图片数量: {len(batch_B)}")
            
            # 调试：打印输入数据类型
            logger.debug(f"A组第一张图片类型: {type(batch_A[0])}")
            logger.debug(f"B组第一张图片类型: {type(batch_B[0])}")
            
            # 保存原始图片路径
            self.original_batch_A = batch_A.copy()
            self.original_batch_B = batch_B.copy()
            
            # 处理Gradio传递的文件路径和Flask传递的元组，同时保存原始索引映射
            def process_image_list(image_list):
                """处理图片列表，转换为numpy数组，同时保存原始索引"""
                valid_images = []
                valid_indices = []  # 保存有效图片对应的原始索引
                for i, item in enumerate(image_list):
                    try:
                        if item is None:
                            logger.debug(f"图片 {i} 为None，跳过")
                            continue
                        
                        img = None
                        
                        # 处理元组格式 (numpy数组, 文件名) - Flask上传
                        if isinstance(item, tuple) and len(item) >= 2:
                            if isinstance(item[0], np.ndarray):
                                logger.debug(f"图片 {i} 是元组格式(numpy数组, 文件名)，直接使用，形状: {item[0].shape}")
                                img = item[0]
                        # 处理numpy数组 - 直接传递
                        elif isinstance(item, np.ndarray):
                            logger.debug(f"图片 {i} 是numpy数组，直接使用，形状: {item.shape}")
                            img = item
                        # 处理文件路径字符串
                        elif isinstance(item, str) and os.path.exists(item):
                            logger.debug(f"图片 {i} 是文件路径，读取图片: {item}")
                            from utils.image_io import read_image
                            img = read_image(item, color_mode='bgr')
                        # 处理Gradio Gallery返回的元组格式 (file_path, caption)
                        elif isinstance(item, (list, tuple)) and len(item) > 0 and isinstance(item[0], str):
                            first_item = item[0]
                            if isinstance(first_item, str) and os.path.exists(first_item):
                                logger.debug(f"图片 {i} 是Gradio元组格式(file_path, caption)，读取图片: {first_item}")
                                from utils.image_io import read_image
                                img = read_image(first_item, color_mode='bgr')
                        
                        if img is not None:
                            valid_images.append(img)
                            valid_indices.append(i)  # 保存原始索引
                            logger.debug(f"图片 {i} 处理成功，形状: {img.shape}")
                        else:
                            logger.warning(f"无法处理图片 {i}: {type(item)} - {item}")
                    except Exception as e:
                        logger.warning(f"处理图片 {i} 时出错: {str(e)}")
                        import traceback
                        traceback.print_exc()
                        continue
                return valid_images, valid_indices
            
            # 处理图片列表，获取有效图片和原始索引映射
            valid_A, valid_indices_A = process_image_list(batch_A)
            valid_B, valid_indices_B = process_image_list(batch_B)
            
            logger.info(f"处理完成，A组有效图片: {len(valid_A)}, B组有效图片: {len(valid_B)}")
            
            if not valid_A or not valid_B:
                return [], "错误：无法处理输入图片，请检查图片格式"
            
            # 执行匹配流程
            results = self.pipeline.run(valid_A, valid_B)
            
            # 处理匹配结果，传递原始索引映射
            output_images, match_info = self._process_results(results, batch_A, batch_B, valid_indices_A)
            
            # 保存匹配结果，用于后续打包
            self.matched_images = output_images.copy()
            
            # 格式化匹配信息
            info_str = "\n".join(match_info)
            logger.info(f"匹配任务完成，成功匹配 {len(output_images)} 对图片")
            
            return output_images, info_str
            
        except Exception as e:
            logger.error(f"处理匹配任务失败: {str(e)}")
            return [], f"处理错误：{str(e)}"
    
    def _filter_valid_images(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """
        过滤无效图片
        
        参数:
            images: List[np.ndarray] - 输入图片列表
            
        返回:
            List[np.ndarray] - 有效图片列表
        """
        valid_images = []
        
        for i, img in enumerate(images):
            # 跳过None值
            if img is None:
                logger.warning(f"图片 {i} 为None值，跳过")
                continue
                
            # 确保img是numpy数组
            if not isinstance(img, np.ndarray):
                try:
                    # 检查是否为PIL Image对象
                    try:
                        from PIL import Image
                        if isinstance(img, Image.Image):
                            logger.debug(f"图片 {i} 是PIL Image对象，转换为numpy数组")
                            img = np.array(img)
                        else:
                            # 尝试直接转换
                            img = np.array(img)
                    except ImportError:
                        # 没有PIL，尝试直接转换
                        img = np.array(img)
                    logger.debug(f"图片 {i} 成功转换为numpy数组，形状: {img.shape}")
                except Exception as e:
                    logger.warning(f"图片 {i} 无法转换为numpy数组: {str(e)}")
                    continue
            
            # 检查图片形状是否合理
            if img.shape == (2,) or (len(img.shape) > 1 and any(dim == 0 for dim in img.shape)):
                logger.warning(f"图片 {i} 形状不合理: {img.shape}，跳过")
                continue
            
            logger.debug(f"图片 {i} 形状: {img.shape}, 数据类型: {img.dtype}")
            
            # 检查图像维度
            if img.ndim == 3:
                # 确保图像通道数正确
                if img.shape[2] == 4:
                    # RGBA格式，转换为RGB
                    try:
                        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
                        logger.debug(f"图片 {i} 从RGBA转换为RGB")
                    except cv2.error as e:
                        logger.warning(f"图片 {i} 从RGBA转换为RGB失败: {str(e)}")
                        continue
                
                # 确保图像数据类型正确
                if img.dtype != np.uint8:
                    img = img.astype(np.uint8)
                    logger.debug(f"图片 {i} 数据类型转换为uint8")
                
                # 将RGB转换为BGR（OpenCV默认格式）
                try:
                    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    valid_images.append(bgr)
                    logger.debug(f"图片 {i} 成功添加到有效列表")
                except cv2.error as e:
                    logger.warning(f"图片 {i} 转换颜色空间失败: {str(e)}")
                    # 尝试直接使用原图，不进行颜色转换
                    try:
                        valid_images.append(img)
                        logger.warning(f"图片 {i} 直接添加到有效列表")
                    except Exception as e2:
                        logger.error(f"图片 {i} 添加失败: {str(e2)}")
                        continue
            elif img.ndim == 2:
                # 灰度图，转换为BGR
                try:
                    bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    valid_images.append(bgr)
                    logger.debug(f"灰度图 {i} 成功转换为BGR并添加到有效列表")
                except cv2.error as e:
                    logger.warning(f"灰度图 {i} 转换颜色空间失败: {str(e)}")
                    continue
            else:
                logger.warning(f"图片 {i} 维度不支持: {img.ndim}")
                continue
        
        logger.info(f"过滤完成，输入 {len(images)} 张图片，有效图片 {len(valid_images)} 张")
        return valid_images
    
    def _process_results(self, results: Dict[int, Dict], 
                        batch_A: List, 
                        batch_B: List, 
                        valid_indices_A: List) -> Tuple[List, List[str]]:
        """
        处理匹配结果，只输出最优的匹配结果，数量不超过B组图片数量，且保持A组图片文件名
        
        参数:
            results: Dict[int, Dict] - 匹配结果字典
            batch_A: List - 原始A组图片
            batch_B: List - 原始B组图片
            valid_indices_A: List - 有效图片对应的原始A组索引
            
        返回:
            Tuple[List, List[str]] - 输出图片列表和匹配信息列表
        """
        from utils.image_io import read_image
        import os
        
        # 1. 整理所有匹配结果，按分数排序
        all_matches = []
        for a_idx in results:
            result = results[a_idx]
            b_idx = result.get('b_index')
            score = result.get('score', 0.0)
            
            if b_idx is not None and a_idx < len(valid_indices_A):
                # 获取原始A组索引
                original_a_idx = valid_indices_A[a_idx]
                all_matches.append({
                    'a_idx': a_idx,
                    'original_a_idx': original_a_idx,
                    'b_idx': b_idx,
                    'score': score,
                    'result': result
                })
        
        # 2. 按分数降序排序，保留最高分的匹配
        all_matches.sort(key=lambda x: x['score'], reverse=True)
        
        # 3. 确保每个B组图片只被匹配一次，最多输出B组图片数量的结果
        used_b_indices = set()
        best_matches = []
        
        for match in all_matches:
            if match['b_idx'] not in used_b_indices and len(best_matches) < len(batch_B):
                best_matches.append(match)
                used_b_indices.add(match['b_idx'])
        
        # 4. 按原始A组图片顺序排序，确保输出顺序与A组一致
        best_matches.sort(key=lambda x: x['original_a_idx'])
        
        # 调试：打印best_matches
        logger.debug(f"best_matches: {best_matches}")
        logger.debug(f"batch_A长度: {len(batch_A)}")
        logger.debug(f"valid_indices_A: {valid_indices_A}")
        
        # 5. 处理匹配结果，生成输出图片和匹配信息
        output_images = []
        match_info = []
        matched_filenames = []
        
        for match in best_matches:
            a_idx = match['a_idx']
            original_a_idx = match['original_a_idx']
            b_idx = match['b_idx']
            score = match['score']
            result = match['result']
            low_conf = result.get('low_confidence', False)
            
            logger.debug(f"处理匹配结果: A索引={a_idx}, 原始A索引={original_a_idx}, B索引={b_idx}")
            
            # 获取第一批图片的文件名（保持与A组图片同名，确保特殊字符完整保留）
            a_filename = f"matched_{original_a_idx:03d}.png"  # 默认文件名
            
            # 确保original_a_idx在有效范围内
            if 0 <= original_a_idx < len(batch_A):
                a_item = batch_A[original_a_idx]
                
                # 直接使用原始路径，不进行任何额外处理
                raw_path = ""
                filename = ""
                
                # 处理各种可能的输入格式
                if isinstance(a_item, str):
                    # 处理文件路径字符串
                    raw_path = a_item
                elif isinstance(a_item, tuple):
                    # 处理Flask上传的元组格式 (numpy数组, 文件名)
                    if len(a_item) >= 2 and isinstance(a_item[1], str):
                        filename = a_item[1]
                    # 处理Gradio Gallery返回的元组格式，通常是 (file_path, caption)
                    else:
                        for item in a_item:
                            if isinstance(item, str):
                                raw_path = item
                                break
                
                if raw_path:
                    # 使用os.path.basename获取文件名，确保特殊字符完整保留
                    a_filename = os.path.basename(raw_path)
                elif filename:
                    # 直接使用从Flask元组中获取的文件名
                    a_filename = filename
                
                # 确保文件名有效
                if not a_filename or a_filename == "":
                    a_filename = f"matched_{original_a_idx:03d}.png"
            
            # 获取匹配的B组图片
            b_img = batch_B[b_idx]
            output_img = None
            
            # 处理元组类型
            if isinstance(b_img, tuple) and len(b_img) > 0:
                logger.debug(f"处理元组类型图片: {b_img}")
                
                # 处理Flask/HTML上传的元组格式 (numpy数组, 文件名)
                if isinstance(b_img[0], np.ndarray):
                    logger.debug(f"处理HTML上传的元组格式 (numpy数组, 文件名)")
                    img = b_img[0]
                    # 检查并转换颜色空间
                    if img.ndim == 3:
                        if img.shape[2] == 3:
                            try:
                                # 检查是否为BGR格式（app.py中cv2.imdecode返回的是BGR格式）
                                output_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            except cv2.error as e:
                                logger.warning(f"转换图片 {b_idx} 从BGR到RGB失败: {str(e)}")
                                # 直接使用原图
                                output_img = img
                        elif img.shape[2] == 4:
                            try:
                                # RGBA格式，转换为RGB
                                output_img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
                            except cv2.error as e:
                                logger.warning(f"转换图片 {b_idx} 从RGBA到RGB失败: {str(e)}")
                                # 直接使用原图
                                output_img = img
                        else:
                            # 其他通道数，直接使用
                            output_img = img
                    elif img.ndim == 2:
                        # 灰度图，转换为RGB
                        try:
                            output_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                        except cv2.error as e:
                            logger.warning(f"转换图片 {b_idx} 从灰度到RGB失败: {str(e)}")
                            # 直接使用原图
                            output_img = img
                    else:
                        # 其他维度，直接使用
                        output_img = img
                # 处理Gradio Gallery返回的元组格式，通常是 (file_path, caption)
                elif isinstance(b_img[0], str):
                    img_path = b_img[0]
                    if os.path.exists(img_path) and os.path.isfile(img_path):
                        # 从文件路径读取图片
                        img = read_image(img_path, color_mode='rgb')
                        if img is not None:
                            output_img = img
                        else:
                            logger.error(f"无法从文件路径读取图片: {img_path}")
                            output_img = None
                    else:
                        logger.error(f"文件路径不存在或不是文件: {img_path}")
                        output_img = None
                else:
                    logger.warning(f"无法处理元组类型图片 {b_idx}: {type(b_img[0])}")
                    output_img = None
            elif isinstance(b_img, np.ndarray):
                # 直接处理numpy数组
                if b_img.ndim == 3:
                    if b_img.shape[2] == 3:
                        try:
                            # 检查是否为BGR格式（通过pipeline处理后的图片是BGR格式）
                            output_img = cv2.cvtColor(b_img, cv2.COLOR_BGR2RGB)
                        except cv2.error as e:
                            logger.warning(f"转换图片 {b_idx} 从BGR到RGB失败: {str(e)}")
                            # 直接使用原图
                            output_img = b_img
                    elif b_img.shape[2] == 4:
                        try:
                            # RGBA格式，转换为RGB
                            output_img = cv2.cvtColor(b_img, cv2.COLOR_RGBA2RGB)
                        except cv2.error as e:
                            logger.warning(f"转换图片 {b_idx} 从RGBA到RGB失败: {str(e)}")
                            # 直接使用原图
                            output_img = b_img
                    else:
                        # 其他通道数，直接使用
                        output_img = b_img
                elif b_img.ndim == 2:
                    # 灰度图，转换为RGB
                    try:
                        output_img = cv2.cvtColor(b_img, cv2.COLOR_GRAY2RGB)
                    except cv2.error as e:
                        logger.warning(f"转换图片 {b_idx} 从灰度到RGB失败: {str(e)}")
                        # 直接使用原图
                        output_img = b_img
                else:
                    # 其他维度，直接使用
                    output_img = b_img
            elif isinstance(b_img, str) and os.path.exists(b_img) and os.path.isfile(b_img):
                # 处理字符串类型的文件路径
                logger.debug(f"处理字符串类型图片路径: {b_img}")
                # 从文件路径读取图片
                img = read_image(b_img, color_mode='rgb')
                if img is not None:
                    output_img = img
                else:
                    logger.error(f"无法从文件路径读取图片: {b_img}")
                    output_img = None
            else:
                # 其他类型，无法处理
                logger.warning(f"无法处理图片 {b_idx}: {type(b_img)}")
                output_img = None
            
            if output_img is not None:  # 只添加有效的匹配结果
                output_images.append(output_img)
                matched_filenames.append(a_filename)  # 保存A组图片的文件名
                
                # 生成匹配信息
                info = f"A[{original_a_idx}] → B[{b_idx}] | score={score:.4f}"
                if low_conf:
                    info += " ⚠️低置信度"
                match_info.append(info)
        
        # 保存匹配文件名，用于后续打包
        self.matched_image_filenames = matched_filenames.copy()
        
        logger.info(f"处理匹配结果完成，生成 {len(output_images)} 张输出图片")
        return output_images, match_info
    
    def export_matched_images(self) -> str:
        """
        将匹配后的图片打包成zip文件
        
        返回:
            str - zip文件路径，如果打包失败则返回空字符串
        """
        logger.info(f"开始导出匹配图片，共有 {len(self.matched_images)} 张图片")
        
        if not self.matched_images:
            logger.warning("没有可导出的匹配图片")
            return ""
        
        try:
            import tempfile
            import os
            import zipfile
            import io
            from utils.image_io import save_image
            
            # 创建临时zip文件，使用更高效的压缩方式
            temp_file = tempfile.NamedTemporaryFile(suffix='.zip', delete=False)
            zip_path = temp_file.name
            temp_file.close()
            logger.info(f"创建临时zip文件: {zip_path}")
            
            # 使用更高效的压缩级别，平衡速度和压缩率
            compression = zipfile.ZIP_STORED  # 对于图片文件，存储模式比压缩更快，因为图片已经是压缩格式
            
            with zipfile.ZipFile(zip_path, 'w', compression) as zipf:
                # 直接将图片写入zip文件，避免中间文件
                logger.info(f"开始将图片写入zip文件，共有 {len(self.matched_images)} 张图片")
                
                for i, (img, original_filename) in enumerate(zip(self.matched_images, self.matched_image_filenames)):
                    logger.info(f"处理图片 {i}: {type(img)}")
                    if img is not None and isinstance(img, np.ndarray):
                        logger.info(f"图片 {i} 是numpy数组，形状: {img.shape}")
                        
                        # 将图片直接编码为字节流，避免写入磁盘
                        try:
                            # 确保图片是RGB格式
                            if len(img.shape) == 3 and img.shape[2] == 3:
                                # RGB格式，转换为BGR供cv2.imencode使用
                                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                            elif len(img.shape) == 3 and img.shape[2] == 4:
                                # RGBA格式，转换为BGR
                                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
                            elif len(img.shape) == 2:
                                # 灰度图，转换为BGR
                                img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                            else:
                                img_bgr = img
                            
                            # 编码为PNG
                            _, buffer = cv2.imencode('.png', img_bgr)
                            img_bytes = buffer.tobytes()
                            
                            # 将字节流直接写入zip文件
                            zipf.writestr(original_filename, img_bytes)
                            logger.info(f"成功将图片 {i} 写入zip文件，使用原始文件名: {original_filename}")
                        except Exception as e:
                            logger.error(f"处理图片 {i} 时出错: {str(e)}")
                            continue
                    else:
                        logger.warning(f"跳过非numpy数组图片 {i}: {type(img)}")
            
            # 检查zip文件内容
            if os.path.exists(zip_path):
                logger.info(f"匹配图片已成功打包为zip文件: {zip_path}")
                # 检查zip文件中的内容
                with zipfile.ZipFile(zip_path, 'r') as zipf:
                    logger.info(f"zip文件包含 {len(zipf.namelist())} 个文件")
                    # 只记录前10个文件名，避免日志过大
                    for file_name in zipf.namelist()[:10]:
                        logger.info(f"zip文件包含: {file_name}")
                    if len(zipf.namelist()) > 10:
                        logger.info(f"... 还有 {len(zipf.namelist()) - 10} 个文件")
                return zip_path
            else:
                logger.error("打包匹配图片失败或文件不存在")
                return ""
            
        except Exception as e:
            logger.error(f"导出匹配图片时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return ""
    
    def process_trainset(self, folderA: str, folderB: str, outputFolder: str) -> List[str]:
        """
        处理trainset模式的匹配
        
        参数:
            folderA: A组图片文件夹地址
            folderB: B组图片文件夹地址
            outputFolder: 输出文件夹C地址
            
        返回:
            List[str]: 匹配信息列表
        """
        logger.info(f"开始Trainset模式匹配，A组文件夹: {folderA}, B组文件夹: {folderB}, 输出文件夹: {outputFolder}")
        
        try:
            import os
            from utils.image_io import read_image, save_image
            
            # 1. 获取A、B文件夹中的图片文件
            def get_image_files(folder):
                """获取文件夹中的图片文件列表"""
                image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
                image_files = []
                
                for filename in os.listdir(folder):
                    file_path = os.path.join(folder, filename)
                    if os.path.isfile(file_path):
                        ext = os.path.splitext(filename)[1].lower()
                        if ext in image_extensions:
                            image_files.append((file_path, filename))
                
                return image_files
            
            # 获取A、B组图片
            imagesA = get_image_files(folderA)
            imagesB = get_image_files(folderB)
            
            logger.info(f"A组图片数量: {len(imagesA)}, B组图片数量: {len(imagesB)}")
            
            if not imagesA or not imagesB:
                return ["错误：A或B组文件夹中没有图片文件"]
            
            # 2. 读取图片数据，转换为与process_batches兼容的格式
            batch_A = []
            for img_path, filename in imagesA:
                img = read_image(img_path, color_mode='bgr')
                if img is not None:
                    batch_A.append((img, filename))
            
            batch_B = []
            for img_path, filename in imagesB:
                img = read_image(img_path, color_mode='bgr')
                if img is not None:
                    batch_B.append((img, filename))
            
            logger.info(f"成功读取A组图片: {len(batch_A)}, B组图片: {len(batch_B)}")
            
            if not batch_A or not batch_B:
                return ["错误：无法读取图片文件，请检查图片格式"]
            
            # 3. 处理图片列表，获取有效图片和原始索引映射
            valid_A, valid_indices_A = self._process_image_list(batch_A)
            valid_B, valid_indices_B = self._process_image_list(batch_B)
            
            # 执行匹配流程
            results = self.pipeline.run(valid_A, valid_B)
            
            # 4. 创建输出文件夹结构
            control_folder = os.path.join(outputFolder, 'control')
            target_folder = os.path.join(outputFolder, 'target')
            
            # 创建文件夹
            os.makedirs(control_folder, exist_ok=True)
            os.makedirs(target_folder, exist_ok=True)
            
            logger.info(f"创建输出文件夹: {control_folder}, {target_folder}")
            
            # 5. 处理匹配结果，获取A->B的映射，确保每个B组图片只被匹配一次
            # 整理所有匹配结果，按分数排序
            all_matches = []
            for a_idx in results:
                result = results[a_idx]
                b_idx = result.get('b_index')
                score = result.get('score', 0.0)
                
                if b_idx is not None and a_idx < len(valid_indices_A) and b_idx < len(valid_indices_B):
                    all_matches.append({
                        'a_idx': a_idx,
                        'b_idx': b_idx,
                        'score': score,
                        'result': result
                    })
            
            # 按分数降序排序
            all_matches.sort(key=lambda x: x['score'], reverse=True)
            
            # 确保每个B组图片只被匹配一次，最多输出B组图片数量的结果
            used_b_indices = set()
            best_matches = []
            
            for match in all_matches:
                if match['b_idx'] not in used_b_indices and len(best_matches) < len(valid_B):
                    best_matches.append(match)
                    used_b_indices.add(match['b_idx'])
            
            # 处理匹配结果，获取A->B的映射
            a_to_b_map = {}
            for match in best_matches:
                a_idx = match['a_idx']
                b_idx = match['b_idx']
                score = match['score']
                
                # 获取原始A组索引
                original_a_idx = valid_indices_A[a_idx]
                # 获取原始B组索引
                original_b_idx = valid_indices_B[b_idx]
                a_to_b_map[original_a_idx] = {
                    'b_idx': original_b_idx,
                    'score': score
                }
            
            # 6. 保存图片
            saved_count = 0
            for a_idx, a_info in a_to_b_map.items():
                b_idx = a_info['b_idx']
                score = a_info['score']
                
                # 确保索引有效
                if 0 <= a_idx < len(imagesA) and 0 <= b_idx < len(imagesB):
                    # 获取原图路径
                    a_img_path, a_filename = imagesA[a_idx]
                    b_img_path, b_filename = imagesB[b_idx]
                    
                    # 读取原图，使用bgr模式（OpenCV默认）
                    a_img = read_image(a_img_path, color_mode='bgr')
                    b_img = read_image(b_img_path, color_mode='bgr')
                    
                    if a_img is not None and b_img is not None:
                        # 保存到对应文件夹，明确指定输入颜色模式为bgr
                        save_image(a_img, os.path.join(control_folder, a_filename), color_mode='bgr', input_color_mode='bgr')
                        
                        # target文件夹中的图片使用A组图片的文件名（保持与control对应）
                        # 获取A组文件名的扩展名
                        a_file_ext = os.path.splitext(a_filename)[1]
                        # 使用A组文件名（包括扩展名）保存B组图片
                        save_image(b_img, os.path.join(target_folder, a_filename), color_mode='bgr', input_color_mode='bgr')
                        saved_count += 1
            
            logger.info(f"成功保存 {saved_count} 对匹配图片到输出文件夹")
            
            # 7. 生成匹配信息
            match_info = [
                f"Trainset模式匹配完成",
                f"A组图片文件夹: {folderA}",
                f"B组图片文件夹: {folderB}",
                f"输出文件夹: {outputFolder}",
                f"A组图片数量: {len(imagesA)}",
                f"B组图片数量: {len(imagesB)}",
                f"成功匹配并保存: {saved_count} 对图片",
                f"control文件夹: {control_folder}",
                f"target文件夹: {target_folder}"
            ]
            
            return match_info
            
        except Exception as e:
            logger.error(f"Trainset模式匹配失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return [f"Trainset模式匹配失败: {str(e)}"]
    
    def _process_image_list(self, image_list):
        """处理图片列表，转换为numpy数组，同时保存原始索引"""
        valid_images = []
        valid_indices = []  # 保存有效图片对应的原始索引
        
        for i, item in enumerate(image_list):
            try:
                if item is None:
                    logger.debug(f"图片 {i} 为None，跳过")
                    continue
                
                img = None
                
                # 处理元组格式 (numpy数组, 文件名) - Flask上传
                if isinstance(item, tuple) and len(item) >= 2:
                    if isinstance(item[0], np.ndarray):
                        logger.debug(f"图片 {i} 是元组格式(numpy数组, 文件名)，直接使用，形状: {item[0].shape}")
                        img = item[0]
                # 处理numpy数组 - 直接传递
                elif isinstance(item, np.ndarray):
                    logger.debug(f"图片 {i} 是numpy数组，直接使用，形状: {item.shape}")
                    img = item
                # 处理文件路径字符串
                elif isinstance(item, str) and os.path.exists(item):
                    logger.debug(f"图片 {i} 是文件路径，读取图片: {item}")
                    from utils.image_io import read_image
                    img = read_image(item, color_mode='bgr')
                # 处理Gradio Gallery返回的元组格式 (file_path, caption)
                elif isinstance(item, (list, tuple)) and len(item) > 0 and isinstance(item[0], str):
                    first_item = item[0]
                    if isinstance(first_item, str) and os.path.exists(first_item):
                        logger.debug(f"图片 {i} 是Gradio元组格式(file_path, caption)，读取图片: {first_item}")
                        from utils.image_io import read_image
                        img = read_image(first_item, color_mode='bgr')
                
                if img is not None:
                    valid_images.append(img)
                    valid_indices.append(i)  # 保存原始索引
                    logger.debug(f"图片 {i} 处理成功，形状: {img.shape}")
                else:
                    logger.warning(f"无法处理图片 {i}: {type(item)} - {item}")
            except Exception as e:
                logger.warning(f"处理图片 {i} 时出错: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        return valid_images, valid_indices