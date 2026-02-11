import cv2
import numpy as np


def normalize_structure_mask(mask, size=64):
    """
    将结构 mask 转成 hash 友好的格式
    
    参数:
        mask: numpy.ndarray - 结构掩码（uint8, 0/255）
        size: int - 归一化后的尺寸
        
    返回:
        numpy.ndarray - 归一化后的掩码
    """
    if mask is None:
        raise ValueError("输入掩码不能为空")
    
    if len(mask.shape) != 2:
        raise ValueError("输入掩码必须是2D数组")
    
    # 使用最近邻插值，避免破坏边缘
    mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
    return mask


def ahash(img):
    """
    均值哈希算法
    
    参数:
        img: numpy.ndarray - 输入图像（归一化后的掩码）
        
    返回:
        numpy.ndarray - 哈希值（0/1数组）
    """
    img = img.astype(np.float32)
    mean = img.mean()
    return (img > mean).astype(np.uint8).flatten()


def dhash(img):
    """
    差分哈希算法
    
    参数:
        img: numpy.ndarray - 输入图像（归一化后的掩码）
        
    返回:
        numpy.ndarray - 哈希值（0/1数组）
    """
    diff = img[:, 1:] > img[:, :-1]
    return diff.astype(np.uint8).flatten()


def phash(img, hash_size=8):
    """
    感知哈希算法（简化DCT版，结构专用）
    
    参数:
        img: numpy.ndarray - 输入图像（归一化后的掩码）
        hash_size: int - 哈希大小
        
    返回:
        numpy.ndarray - 哈希值（0/1数组）
    """
    # 调整为DCT友好的尺寸
    img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_NEAREST)
    img = np.float32(img)
    
    # 应用DCT变换
    dct = cv2.dct(img)
    
    # 提取低频部分
    dct_low = dct[:hash_size, :hash_size]
    
    # 使用中位数作为阈值
    median = np.median(dct_low)
    return (dct_low > median).astype(np.uint8).flatten()


def hamming_distance(h1, h2):
    """
    计算两个哈希值之间的汉明距离
    
    参数:
        h1: numpy.ndarray - 第一个哈希值
        h2: numpy.ndarray - 第二个哈希值
        
    返回:
        int - 汉明距离
    """
    if h1.shape != h2.shape:
        raise ValueError("两个哈希值的形状必须相同")
    
    return np.count_nonzero(h1 != h2)


def combined_distance(mask1, mask2, weights=(0.3, 0.3, 0.4)):
    """
    多Hash融合距离计算
    
    参数:
        mask1: numpy.ndarray - 第一个结构掩码
        mask2: numpy.ndarray - 第二个结构掩码
        weights: tuple - 三种哈希的权重
        
    返回:
        float - 归一化的融合距离（0~1，越小越相似）
    """
    if len(weights) != 3:
        raise ValueError("权重必须包含三个元素")
    
    # 归一化结构掩码
    m1 = normalize_structure_mask(mask1)
    m2 = normalize_structure_mask(mask2)
    
    # 计算三种哈希
    h1_a, h2_a = ahash(m1), ahash(m2)
    h1_d, h2_d = dhash(m1), dhash(m2)
    h1_p, h2_p = phash(m1), phash(m2)
    
    # 计算归一化的汉明距离（0~1）
    da = hamming_distance(h1_a, h2_a) / len(h1_a)
    dd = hamming_distance(h1_d, h2_d) / len(h1_d)
    dp = hamming_distance(h1_p, h2_p) / len(h1_p)
    
    # 加权融合
    return weights[0] * da + weights[1] * dd + weights[2] * dp


def match_batches(structs_A, structs_B, topk=3):
    """
    批量匹配结构掩码
    
    参数:
        structs_A: dict - 第一批结构掩码（键: 图像路径, 值: 结构掩码）
        structs_B: dict - 第二批结构掩码（键: 图像路径, 值: 结构掩码）
        topk: int - 返回前k个匹配结果
        
    返回:
        dict - 匹配结果（键: A中的图像路径, 值: 匹配的B图像路径和距离列表）
    """
    if not structs_A:
        raise ValueError("第一批结构掩码不能为空")
    
    if not structs_B:
        raise ValueError("第二批结构掩码不能为空")
    
    results = {}
    
    for pa, ma in structs_A.items():
        scores = []
        for pb, mb in structs_B.items():
            try:
                d = combined_distance(ma, mb)
                scores.append((pb, d))
            except Exception as e:
                print(f"计算距离时出错 {pa} -> {pb}: {str(e)}")
                continue
        
        # 按距离升序排序（距离越小越相似）
        scores.sort(key=lambda x: x[1])
        results[pa] = scores[:topk]
    
    return results


def convert_to_match_result(a_paths, b_paths, match_results):
    """
    将匹配结果转换为与BaseMatcher兼容的格式
    
    参数:
        a_paths: list - A组图像的路径列表（按输入顺序）
        b_paths: list - B组图像的路径列表
        match_results: dict - 批量匹配结果
        
    返回:
        list - 匹配结果列表，每个元素包含index_b和score
    """
    # 创建B路径到索引的映射
    b_path_to_index = {path: idx for idx, path in enumerate(b_paths)}
    
    results = []
    
    for path_a in a_paths:
        if path_a not in match_results:
            # 没有匹配结果
            results.append({
                'b_index': None,
                'score': 0.0
            })
            continue
        
        # 获取最佳匹配
        top_match = match_results[path_a][0]
        path_b, distance = top_match
        
        # 转换距离为相似度分数（距离越小，相似度越高）
        # 这里简单地使用1 - distance作为相似度
        score = 1.0 - distance
        
        results.append({
            'b_index': b_path_to_index.get(path_b, None),
            'score': score
        })
    
    return results


# 以下是传统算法版本的Matcher类，实现BaseMatcher接口
from core.matcher_base import BaseMatcher, MatchResult


class TraditionalHashMatcher(BaseMatcher):
    """
    基于多哈希融合的传统图像匹配器
    """
    
    def __init__(self, weights=(0.3, 0.3, 0.4), topk=3):
        """
        初始化传统哈希匹配器
        
        参数:
            weights: tuple - 三种哈希的权重
            topk: int - 返回前k个匹配结果
        """
        super().__init__()
        self.weights = weights
        self.topk = topk
        self.structs_A = None  # A组结构掩码
        self.structs_B = None  # B组结构掩码
        self.b_paths = None    # B组图像路径列表
        
    def prepare(self, images_a, images_b, structs_a=None, structs_b=None):
        """
        准备匹配器
        
        参数:
            images_a: list - A组图像列表
            images_b: list - B组图像列表
            structs_a: list - A组结构掩码列表
            structs_b: list - B组结构掩码列表
        """
        if structs_a is None or structs_b is None:
            raise ValueError("必须提供结构掩码")
        
        if len(images_a) != len(structs_a):
            raise ValueError("A组图像和结构掩码数量不匹配")
        
        if len(images_b) != len(structs_b):
            raise ValueError("B组图像和结构掩码数量不匹配")
        
        # 存储结构掩码和路径
        self.structs_A = {f"a_{i}": struct for i, struct in enumerate(structs_a)}
        self.structs_B = {f"b_{i}": struct for i, struct in enumerate(structs_b)}
        self.b_paths = [f"b_{i}" for i in range(len(images_b))]
        
        return True
    
    def match(self):
        """
        执行匹配
        
        返回:
            dict - 匹配结果字典，键为a_index，值为MatchResult类型
        """
        if self.structs_A is None or self.structs_B is None:
            raise ValueError("匹配器未准备好")
        
        # 执行批量匹配
        match_results = match_batches(self.structs_A, self.structs_B, self.topk)
        
        # 转换为MatchResult格式
        a_paths = list(self.structs_A.keys())
        converted_results = convert_to_match_result(a_paths, self.b_paths, match_results)
        
        # 创建MatchResult字典
        results = {}
        for i, result in enumerate(converted_results):
            match_result = {
                "a_index": i,
                "b_index": result['b_index'],
                "score": result['score'],
                "method": "multi_hash",
                "meta": {
                    'topk_matches': match_results[a_paths[i]]
                }
            }
            results[i] = match_result
        
        return results