# 传统算法匹配器主类
import numpy as np
import cv2
from core.matcher_base import BaseMatcher, MatchResult
from algorithms.traditional.structure import extract_main_structure
from algorithms.traditional.hash_matcher import TraditionalHashMatcher


class TraditionalMatcher(BaseMatcher):
    """
    传统图像匹配器，整合结构提取和哈希匹配
    """
    
    def __init__(self, canny_low=50, canny_high=150, min_area_ratio=0.01, 
                 weights=(0.3, 0.3, 0.4), topk=3):
        """
        初始化传统匹配器
        
        参数:
            canny_low: int - Canny边缘检测低阈值
            canny_high: int - Canny边缘检测高阈值
            min_area_ratio: float - 最小连通域面积比例
            weights: tuple - 三种哈希的权重
            topk: int - 返回前k个匹配结果
        """
        super().__init__()
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.min_area_ratio = min_area_ratio
        self.weights = weights
        self.topk = topk
        
        # 子匹配器
        self.hash_matcher = TraditionalHashMatcher(weights=weights, topk=topk)
        
        # 存储中间结果（与TraditionalHashMatcher保持一致的命名）
        self.structs_a = None
        self.structs_b = None
        self.structs_A = None  # 用于pipeline访问，格式：{"a_0": struct0, "a_1": struct1, ...}
        self.structs_B = None  # 用于pipeline访问，格式：{"b_0": struct0, "b_1": struct1, ...}
    
    def prepare(self, images_a: list, images_b: list, structs_a=None, structs_b=None) -> None:
        """
        前处理阶段：保存输入、提取结构
        
        参数:
            images_a: list - A组图像列表
            images_b: list - B组图像列表
            structs_a: list - A组结构掩码列表（可选）
            structs_b: list - B组结构掩码列表（可选）
        """
        super().prepare(images_a, images_b)
        
        # 如果提供了结构掩码，则直接使用，否则提取
        if structs_a is not None and structs_b is not None:
            self.structs_a = structs_a
            self.structs_b = structs_b
        else:
            # 提取结构掩码
            self.structs_a = []
            self.structs_b = []
            
            # 处理A组图像
            for img in images_a:
                struct = self._extract_structure(img)
                self.structs_a.append(struct)
            
            # 处理B组图像
            for img in images_b:
                struct = self._extract_structure(img)
                self.structs_b.append(struct)
        
        # 准备哈希匹配器
        self.hash_matcher.prepare(images_a, images_b, self.structs_a, self.structs_b)
        
        # 为pipeline准备结构掩码（格式与TraditionalHashMatcher保持一致）
        self.structs_A = {f"a_{i}": struct for i, struct in enumerate(self.structs_a)}
        self.structs_B = {f"b_{i}": struct for i, struct in enumerate(self.structs_b)}
    
    def _extract_structure(self, img: np.ndarray) -> np.ndarray:
        """
        提取单个图像的结构掩码
        
        参数:
            img: numpy.ndarray - 输入图像
            
        返回:
            numpy.ndarray - 结构掩码
        """
        if img is None:
            return np.zeros((64, 64), dtype=np.uint8)
        
        # 确保图像是BGR格式
        if len(img.shape) == 3 and img.shape[2] == 3:
            # 已经是BGR格式
            pass
        elif len(img.shape) == 3 and img.shape[2] == 4:
            # RGBA格式，转换为BGR
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        elif len(img.shape) == 2:
            # 灰度图，转换为BGR
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            # 其他格式，调整为BGR
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # 提取结构
        try:
            struct = extract_main_structure(
                img, 
                canny_low=self.canny_low, 
                canny_high=self.canny_high, 
                min_area_ratio=self.min_area_ratio
            )
            return struct
        except Exception as e:
            # 提取失败，返回空掩码
            return np.zeros((64, 64), dtype=np.uint8)
    
    def match(self) -> dict[int, MatchResult]:
        """
        执行匹配逻辑
        
        返回:
            Dict[int, MatchResult] - 匹配结果字典
        """
        # 使用哈希匹配器执行匹配
        results = self.hash_matcher.match()
        
        # 确保结果完整性
        for i in range(self.num_a):
            if i not in results:
                results[i] = self._empty_result(method="traditional")
        
        return results
    
    def get_structures(self) -> tuple[list, list]:
        """
        获取提取的结构掩码
        
        返回:
            tuple[list, list] - (A组结构掩码列表, B组结构掩码列表)
        """
        return self.structs_a, self.structs_b
