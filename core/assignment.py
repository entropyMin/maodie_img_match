import numpy as np
from scipy.optimize import linear_sum_assignment

class AssignmentSolver:
    """
    顺序重排 & 冲突消解求解器
    
    负责将距离矩阵转换为一一对应的映射关系，确保：
    1. 一一对应（no many-to-one）
    2. 总距离最小
    3. 尽量保持局部最优
    """
    
    def __init__(self, confidence_threshold=0.05):
        """
        初始化AssignmentSolver
        
        参数:
            confidence_threshold: float - 低置信度阈值
        """
        self.confidence_threshold = confidence_threshold
        self.distance_matrix = None
        self.matches = None
        self.confidences = None
        self.flags = None
    
    def build_distance_matrix(self, structs_A, structs_B, distance_func):
        """
        构建距离矩阵
        
        参数:
            structs_A: dict - A组结构掩码
            structs_B: dict - B组结构掩码
            distance_func: callable - 距离计算函数
        
        返回:
            numpy.ndarray - 距离矩阵
            list - A组键列表
            list - B组键列表
        """
        A_keys = list(structs_A.keys())
        B_keys = list(structs_B.keys())
        
        n, m = len(A_keys), len(B_keys)
        D = np.zeros((n, m), dtype=np.float32)
        
        for i, ka in enumerate(A_keys):
            for j, kb in enumerate(B_keys):
                D[i, j] = distance_func(structs_A[ka], structs_B[kb])
        
        return D, A_keys, B_keys
    
    def hungarian_match(self, D=None):
        """
        使用匈牙利算法进行全局最优匹配
        
        参数:
            D: numpy.ndarray - 距离矩阵（可选）
        
        返回:
            list - 匹配对列表 [(i, j), ...]
        """
        if D is not None:
            self.distance_matrix = D
        
        if self.distance_matrix is None:
            raise ValueError("距离矩阵未设置")
        
        # 使用SciPy的线性总和分配算法（匈牙利算法）
        row_ind, col_ind = linear_sum_assignment(self.distance_matrix)
        self.matches = list(zip(row_ind, col_ind))
        
        return self.matches
    
    def compute_confidence(self, row, assigned_j):
        """
        计算单个匹配的置信度
        
        参数:
            row: numpy.ndarray - 距离矩阵的一行
            assigned_j: int - 分配的列索引
        
        返回:
            float - 置信度分数
        """
        sorted_scores = np.sort(row)
        best = sorted_scores[0]
        
        # 如果只有一个选项，置信度为最大值
        if len(sorted_scores) < 2:
            return float('inf')
        
        second = sorted_scores[1]
        return second - best
    
    def analyze_confidence(self):
        """
        分析所有匹配的置信度
        
        返回:
            list - 置信度列表
            list - 低置信度标记列表
        """
        if self.matches is None:
            raise ValueError("尚未进行匹配")
        
        self.confidences = []
        self.flags = []
        
        for i, j in self.matches:
            conf = self.compute_confidence(self.distance_matrix[i], j)
            self.confidences.append(conf)
            self.flags.append(conf < self.confidence_threshold)
        
        return self.confidences, self.flags
    
    def reorder_B_by_A(self, structs_A, structs_B, distance_func):
        """
        将B组按A组顺序重新排列
        
        参数:
            structs_A: dict - A组结构掩码
            structs_B: dict - B组结构掩码
            distance_func: callable - 距离计算函数
        
        返回:
            list - 按A顺序排列的B组键
            list - 匹配信息
        """
        # 构建距离矩阵
        D, A_keys, B_keys = self.build_distance_matrix(structs_A, structs_B, distance_func)
        
        # 执行匈牙利匹配
        self.hungarian_match(D)
        
        # 分析置信度
        self.analyze_confidence()
        
        # 构建结果，保留所有匹配结果，包括低置信度的
        ordered_B = []
        match_info = []
        
        for idx, ((i, j), conf, flag) in enumerate(zip(self.matches, self.confidences, self.flags)):
            ordered_B.append(B_keys[j])
            match_info.append({
                "A": A_keys[i],
                "B": B_keys[j],
                "score": self.distance_matrix[i, j],
                "confidence": conf,
                "low_confidence": flag
            })
        
        return ordered_B, match_info
    
    def detect_low_confidence(self, threshold=None):
        """
        检测低置信度匹配
        
        参数:
            threshold: float - 可选的阈值，覆盖默认值
        
        返回:
            list - 低置信度标记列表
        """
        if threshold is None:
            threshold = self.confidence_threshold
        
        if self.matches is None:
            raise ValueError("尚未进行匹配")
        
        flags = []
        for i, j in self.matches:
            conf = self.compute_confidence(self.distance_matrix[i], j)
            flags.append(conf < threshold)
        
        return flags
    
    def get_statistics(self):
        """
        获取匹配统计信息
        
        返回:
            dict - 统计信息
        """
        if self.matches is None:
            raise ValueError("尚未进行匹配")
        
        if self.confidences is None:
            self.analyze_confidence()
        
        num_matches = len(self.matches)
        num_low_confidence = sum(self.flags)
        avg_confidence = np.mean(self.confidences)
        avg_score = np.mean([self.distance_matrix[i, j] for i, j in self.matches])
        
        return {
            "num_matches": num_matches,
            "num_low_confidence": num_low_confidence,
            "low_confidence_ratio": num_low_confidence / num_matches if num_matches > 0 else 0,
            "avg_confidence": avg_confidence,
            "avg_score": avg_score
        }

# 便捷函数
def solve_assignment(structs_A, structs_B, distance_func, confidence_threshold=0.05):
    """
    便捷函数：直接求解分配问题
    
    参数:
        structs_A: dict - A组结构掩码
        structs_B: dict - B组结构掩码
        distance_func: callable - 距离计算函数
        confidence_threshold: float - 低置信度阈值
    
    返回:
        tuple: (ordered_B, match_info, statistics)
    """
    solver = AssignmentSolver(confidence_threshold)
    ordered_B, match_info = solver.reorder_B_by_A(structs_A, structs_B, distance_func)
    statistics = solver.get_statistics()
    
    return ordered_B, match_info, statistics

def build_distance_matrix(structs_A, structs_B, distance_func):
    """
    便捷函数：构建距离矩阵
    
    参数:
        structs_A: dict - A组结构掩码
        structs_B: dict - B组结构掩码
        distance_func: callable - 距离计算函数
    
    返回:
        tuple: (D, A_keys, B_keys)
    """
    solver = AssignmentSolver()
    return solver.build_distance_matrix(structs_A, structs_B, distance_func)