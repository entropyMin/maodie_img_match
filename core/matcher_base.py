from abc import ABC, abstractmethod
from typing import Dict, Optional, Any, TypedDict


class MatchResult(TypedDict):
    """
    单个 A 图像的匹配结果
    """
    b_index: Optional[int]   # 匹配到的 B 索引，None 表示未匹配
    score: float             # 置信度 / 相似度，越大越好
    method: str              # 使用的方法标识，如 'phash', 'clip'
    meta: Optional[Dict[str, Any]]  # 可选调试信息


class BaseMatcher(ABC):
    """
    所有匹配算法的统一接口
    """

    def __init__(self):
        self.images_a = None
        self.images_b = None
        self.num_a = 0
        self.num_b = 0

    @abstractmethod
    def prepare(self, images_a: list, images_b: list) -> None:
        """
        前处理阶段：
        - 保存输入
        - 特征提取
        - 中间结果缓存

        ⚠️ 不允许在这里做匹配决策
        """
        self.images_a = images_a
        self.images_b = images_b
        self.num_a = len(images_a)
        self.num_b = len(images_b)

    @abstractmethod
    def match(self) -> Dict[int, MatchResult]:
        """
        执行匹配逻辑

        return:
        {
            a_index: {
                "b_index": int | None,
                "score": float,
                "method": str,
                "meta": dict | None
            }
        }

        约定：
        - 必须覆盖所有 a_index（即使未匹配）
        - score 统一为“越大越可信”
        """
        pass

    def _empty_result(self, score: float = 0.0, method: str = "none") -> MatchResult:
        """
        标准未匹配结果，供子类复用
        """
        return {
            "b_index": None,
            "score": score,
            "method": method,
            "meta": None
        }
