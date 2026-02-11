from typing import Dict
from core.matcher_base import BaseMatcher, MatchResult
from core.assignment import AssignmentSolver


class MatchPipeline:
    """
    匹配流程调度器
    只负责编排流程，不包含任何算法逻辑
    """

    def __init__(self, matcher: BaseMatcher, use_assignment=True, confidence_threshold=0.05):
        """
        初始化匹配流程调度器
        
        参数:
            matcher: BaseMatcher - 匹配器实例
            use_assignment: bool - 是否使用AssignmentSolver进行顺序重排和冲突消解
            confidence_threshold: float - 低置信度阈值
        """
        self.matcher = matcher
        self.use_assignment = use_assignment
        self.confidence_threshold = confidence_threshold
        self.assignment_solver = None
        
        if use_assignment:
            self.assignment_solver = AssignmentSolver(confidence_threshold)

    def run(self, images_a: list, images_b: list, structs_a=None, structs_b=None) -> Dict[int, MatchResult]:
        """
        执行完整匹配流程
        
        参数:
            images_a: list - A组图像列表
            images_b: list - B组图像列表
            structs_a: list - A组结构掩码列表（可选）
            structs_b: list - B组结构掩码列表（可选）
            
        返回:
            Dict[int, MatchResult] - 匹配结果字典
        """
        if not images_a:
            raise ValueError("images_a is empty")
        if not images_b:
            raise ValueError("images_b is empty")

        # 1. 前处理 / 特征准备
        self.matcher.prepare(images_a, images_b, structs_a, structs_b)

        # 2. 执行匹配
        results = self.matcher.match()

        # 3. 结果完整性校验（非常重要）
        self._validate_results(results, len(images_a))
        
        # 4. 顺序重排 & 冲突消解（Step 5）
        if self.use_assignment and hasattr(self.matcher, 'structs_A') and hasattr(self.matcher, 'structs_B'):
            results = self._run_assignment(results, len(images_a), len(images_b))

        return results

    def _validate_results(self, results: Dict[int, MatchResult], num_a: int):
        """
        防御式校验，提前暴露算法错误
        """
        if not isinstance(results, dict):
            raise TypeError("match() must return a dict")

        for i in range(num_a):
            if i not in results:
                raise ValueError(f"Missing match result for a_index={i}")

            r = results[i]
            if "b_index" not in r or "score" not in r or "method" not in r:
                raise ValueError(f"Invalid MatchResult format at a_index={i}")
    
    def _run_assignment(self, initial_results, num_a, num_b):
        """
        执行顺序重排和冲突消解
        
        参数:
            initial_results: Dict[int, MatchResult] - 初始匹配结果
            num_a: int - A组图像数量
            num_b: int - B组图像数量
            
        返回:
            Dict[int, MatchResult] - 重排后的匹配结果
        """
        from algorithms.traditional.hash_matcher import combined_distance
        
        # 获取结构掩码
        structs_A = self.matcher.structs_A
        structs_B = self.matcher.structs_B
        
        if structs_A is None or structs_B is None:
            print("警告: 无法执行AssignmentSolver，缺少结构掩码")
            return initial_results
        
        # 执行重排
        ordered_B, match_info = self.assignment_solver.reorder_B_by_A(
            structs_A, 
            structs_B, 
            combined_distance
        )
        
        # 创建新的结果字典，初始化为原始结果
        new_results = initial_results.copy()
        
        # 映射A索引到B索引
        for idx, info in enumerate(match_info):
            # 从A键名中提取索引
            a_idx = int(info["A"].split("_")[1])
            # 从B键名中提取索引
            b_idx = int(info["B"].split("_")[1])
            
            # 获取原始结果
            original_result = initial_results[a_idx]
            
            # 更新匹配结果
            new_results[a_idx] = {
                "b_index": b_idx,
                "score": original_result["score"],
                "method": original_result["method"],
                "meta": original_result.get("meta", {}),
                "confidence": info["confidence"],
                "low_confidence": info["low_confidence"]
            }
        

        
        return new_results