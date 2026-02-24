#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
retrieval_planner.py
負責把 turn_state + user_query + retrieval_state → RetrievalPlan

注意：
- 以下數據結構仍在使用：RetrievalPlan, RetrievalMode, TopicPolicy, 
  RetrievalState, PlanDecisionTrace
- RetrievalPlanner 類別已刪除（未使用）
- 目前使用的規劃器：DSTBasedRetrievalPlanner (dst_based_planner.py)
"""

from __future__ import annotations  # 啟用延遲評估，解決前向引用問題

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

# 頂部導入，避免在方法內部導入導致的相對導入問題
try:
    from .topic_extractor import extract_explicit_topics
    from .topic_ontology import default_ontology
except ImportError:
    # 如果相對導入失敗（例如在測試中），嘗試絕對導入
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from topic_extractor import extract_explicit_topics
    from topic_ontology import default_ontology


class RetrievalMode(Enum):
    """檢索模式（泛用）"""
    TOPIC_FOCUSED = "TOPIC_FOCUSED"  # 只在指定 topic 範圍內找
    GLOBAL_OVERVIEW = "GLOBAL_OVERVIEW"  # 整體概覽/全域搜尋
    # 保留舊模式（向後兼容）
    REPORT_OVERVIEW = "REPORT_OVERVIEW"
    DOMAIN = "DOMAIN"
    SCORE = "SCORE"
    META = "META"


class TopicPolicy(Enum):
    """主題策略（泛用）"""
    SOFT_FOCUS = "SOFT_FOCUS"  # 主題聚焦但可少量補充
    MIX_TOPK = "MIX_TOPK"  # 多主題混合配額
    # 保留舊策略（向後兼容）
    LOCK_TOP1 = "LOCK_TOP1"
    TOP2_UNION = "TOP2_UNION"
    UNLOCKED = "UNLOCKED"
    SOFT_TOP1 = "SOFT_TOP1"
    UNLOCK_GLOBAL = "UNLOCK_GLOBAL"


# 向後兼容：DomainPolicy 作為 TopicPolicy 的別名
DomainPolicy = TopicPolicy


@dataclass
class RetrievalState:
    """跨輪檢索狀態（泛用）"""
    turn_index: int = 0
    active_mode: Optional[str] = None
    active_topics: List[str] = field(default_factory=list)  # 當前活躍的主題列表
    active_domain: Optional[str] = None  # 保留（向後兼容）
    active_section_ids: List[str] = field(default_factory=list)
    active_item_ids: List[str] = field(default_factory=list)
    assistant_anchor_text: str = ""
    
    def reset(self):
        """重置狀態（用於 shift_hard）"""
        self.active_topics = []
        self.active_domain = None
        self.active_section_ids = []
        self.active_item_ids = []
        # 保留 assistant_anchor_text 供 rerank 使用


@dataclass
class PlanDecisionTrace:
    """檢索計劃決策追蹤（可回答「哪些 DST 欄位影響了哪些檢索決策」）"""
    topics_source: str  # "explicit" / "prev_active" / "distribution"
    mode_reason: str  # 為什麼選擇這個 mode
    policy_reason: str  # 為什麼選擇這個 policy
    quota_reason: str  # 配額分配的邏輯
    section_weight_reason: str  # Section 權重的邏輯
    confidence_used: float  # 使用的信心值
    explicit_topics: List[str] = field(default_factory=list)  # 明確提到的主題
    active_topics: List[str] = field(default_factory=list)  # 最終使用的活躍主題
    # 新增 routing 決策追蹤欄位
    mentioned_topics: List[str] = field(default_factory=list)  # 明確提到的主題（高 precision）
    topics_by_mass: List[str] = field(default_factory=list)  # 基於累積機率選取的主題
    selected_topics: List[str] = field(default_factory=list)  # 最終選取的主題
    topic_k: int = 0  # 動態 K 值
    p1: float = 0.0  # Top1 機率
    p2: float = 0.0  # Top2 機率
    margin: float = 0.0  # p1 - p2
    normalized_entropy: float = 0.0  # 正規化熵
    cumP: float = 0.0  # 累積機率
    overview_intent_hit: bool = False  # 是否命中概覽意圖
    list_intent_hit: bool = False  # 是否命中列舉意圖
    ambiguous_flag: bool = False  # 是否模糊
    routing_action: str = ""  # FOCUS/MIX/GLOBAL/CLARIFY
    routing_reason: str = ""  # routing_action 的原因


@dataclass
class RetrievalPlan:
    """檢索計劃（泛用）"""
    mode: RetrievalMode
    topic_policy: TopicPolicy  # 新的泛用策略
    topics: List[str]  # 要檢索的主題列表（替代 domains）
    graph_hops: int  # 2~5
    k_per_subdomain: int  # 每個 subdomain 取幾個 group/section
    k_items: int  # item 取幾個
    use_sections: List[str]  # assessment/observation/training/suggestion
    rerank_weights: Dict[str, float]  # sim_q / sim_anchor / graph_prox
    ask_clarify: bool
    reasons: List[str]  # debug log
    # 新增欄位（泛用）
    topic_alloc: Dict[str, int] = field(default_factory=dict)  # 每個 topic 的配額
    section_type_weights: Dict[str, float] = field(default_factory=dict)  # Section type 權重
    plan_decision_trace: Optional[PlanDecisionTrace] = None  # 決策追蹤
    # 新增智慧 routing 欄位
    routing_action: str = ""  # "FOCUS" | "MIX" | "GLOBAL" | "CLARIFY"
    # 兩層分配：subdomain → section type
    two_level_quota: Dict[str, Dict[str, int]] = field(default_factory=dict)  # {"subdomain": {"section_type": quota}}
    # 保留舊欄位（向後兼容）
    domain_policy: Optional[TopicPolicy] = None
    domains: List[str] = field(default_factory=list)
    domain_alloc: Dict[str, int] = field(default_factory=dict)
    allow_parent_domain_expand: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典"""
        return {
            "mode": self.mode.value,
            "domain_policy": self.domain_policy.value,
            "domains": self.domains,
            "graph_hops": self.graph_hops,
            "k_per_subdomain": self.k_per_subdomain,
            "k_items": self.k_items,
            "use_sections": self.use_sections,
            "rerank_weights": self.rerank_weights,
            "ask_clarify": self.ask_clarify,
            "reasons": self.reasons,
            "domain_alloc": self.domain_alloc,
            "allow_parent_domain_expand": self.allow_parent_domain_expand,
        }
