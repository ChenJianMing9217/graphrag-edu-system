# semantic_flow_module_v2.py
# 清晰、模塊化的語義流程追蹤系統

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import json

from .embedding import TextEncoder
from .domain_router import DomainRouter, DomainResult
from .context_similarity import ContextSimilarity, ContextSimConfig
from .multi_topic_tracker import MultiTopicTracker, MultiTopicConfig
from .dst_policy import DSTPolicyConfig, decide_policy, action_to_predicted_flow
from .task_scope_classifier import TaskScopeClassifier, PredictResult


# ============================================================================
# 結果數據結構
# ============================================================================

@dataclass
class DomainAnalysis:
    """領域分析結果"""
    top_domain: str
    top_prob: float
    entropy: float
    distribution: Dict[str, float] = field(default_factory=dict)
    active_domains: List[str] = field(default_factory=list)
    active_domain_probs: Dict[str, float] = field(default_factory=dict)
    is_multi_domain: bool = False
    fused_distribution: Optional[Dict[str, float]] = None  # 模糊延續融合後的分布（如果觸發）
    is_overview_query: bool = False  # 是否為整體查詢
    overview_distribution: Optional[Dict[str, float]] = None  # 整體查詢時的領域分布（如果觸發）


@dataclass
class ContextAnalysis:
    """上下文相似度分析"""
    similarity_score: float  # C
    source: str  # "first_turn" | "prev_user" | "prev_bot"
    is_first_turn: bool


@dataclass
class TopicAnalysis:
    """主題延續分析"""
    is_continuing: bool
    overlap_score: float  # MT
    reason: str
    prev_top_domain: Optional[str] = None
    cur_top_domain: Optional[str] = None
    prev_dist: Optional[Dict[str, float]] = None  # 上一輪的領域分布（更新前）
    prev_active_domains: Optional[List[str]] = None  # 上一輪的活躍領域列表（更新前）
    tv_distance: Optional[float] = None  # TV 距離（Total Variation Distance）


@dataclass
class PolicyDecision:
    """策略決策結果（簡化版：移除 D_level）"""
    context_level: str  # "high" | "low"
    is_ambiguous: bool
    policy_case: str  # e.g., "CH_MTH_NARROW_MD"
    retrieval_action: str  # "NARROW_GRAPH" | "CONTEXT_FIRST" | etc
    semantic_flow: str  # "continue" | "shift_soft" | "shift_hard"


@dataclass
class FlowResult:
    """完整的語義流程分析結果"""
    turn_index: int
    
    # 三層分析
    domain_analysis: DomainAnalysis
    context_analysis: ContextAnalysis
    topic_analysis: TopicAnalysis
    policy_decision: PolicyDecision
    
    # 可選：任務/範圍分類
    task_label: Optional[str] = None
    task_dist: Optional[Dict[str, float]] = None
    scope_label: Optional[str] = None
    scope_dist: Optional[Dict[str, float]] = None

    def to_dict(self) -> dict:
        """轉換為字典格式（完整分析結果）"""
        result = {
            "turn_index": self.turn_index,
            "domain_analysis": {
                "top_domain": self.domain_analysis.top_domain,
                "top_prob": float(self.domain_analysis.top_prob),
                "entropy": float(self.domain_analysis.entropy),
                "distribution": {k: float(v) for k, v in self.domain_analysis.distribution.items()},
                "active_domains": list(self.domain_analysis.active_domains),
                "active_domain_probs": {k: float(v) for k, v in self.domain_analysis.active_domain_probs.items()},
                "is_multi_domain": self.domain_analysis.is_multi_domain,
                "is_overview_query": self.domain_analysis.is_overview_query,
            },
            "context_analysis": {
                "similarity_score": float(self.context_analysis.similarity_score),
                "source": self.context_analysis.source,
                "is_first_turn": self.context_analysis.is_first_turn,
            },
            "topic_analysis": {
                "is_continuing": self.topic_analysis.is_continuing,
                "overlap_score": float(self.topic_analysis.overlap_score),
                "reason": self.topic_analysis.reason,
                "prev_top_domain": self.topic_analysis.prev_top_domain,
                "cur_top_domain": self.topic_analysis.cur_top_domain,
                "tv_distance": float(self.topic_analysis.tv_distance) if self.topic_analysis.tv_distance is not None else None,
                "prev_dist": {k: float(v) for k, v in self.topic_analysis.prev_dist.items()} if self.topic_analysis.prev_dist else None,
                "prev_active_domains": list(self.topic_analysis.prev_active_domains) if self.topic_analysis.prev_active_domains else None,
            },
            "policy_decision": {
                "context_level": self.policy_decision.context_level,
                "is_ambiguous": self.policy_decision.is_ambiguous,
                "policy_case": self.policy_decision.policy_case,
                "retrieval_action": self.policy_decision.retrieval_action,
                "semantic_flow": self.policy_decision.semantic_flow,
            },
        }
        
        # 添加任務/範圍分類
        if self.task_label:
            result["task_label"] = self.task_label
        if self.task_dist:
            result["task_dist"] = {k: float(v) for k, v in self.task_dist.items()}
        if self.scope_label:
            result["scope_label"] = self.scope_label
        if self.scope_dist:
            result["scope_dist"] = {k: float(v) for k, v in self.scope_dist.items()}
        
        # 添加融合後的分布（如果存在）
        if self.domain_analysis.fused_distribution:
            result["domain_analysis"]["fused_distribution"] = {
                k: float(v) for k, v in self.domain_analysis.fused_distribution.items()
            }
        
        # 添加整體分布（如果存在）
        if self.domain_analysis.overview_distribution:
            result["domain_analysis"]["overview_distribution"] = {
                k: float(v) for k, v in self.domain_analysis.overview_distribution.items()
            }
        
        return result

    def to_json(self) -> str:
        """轉換為 JSON 字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    def __str__(self) -> str:
        """簡潔的文本表示"""
        lines = [
            f"[Turn {self.turn_index}] {self.policy_decision.semantic_flow.upper()} | "
            f"{self.policy_decision.retrieval_action}",
            f"  Domain: {self.domain_analysis.top_domain} "
            f"(p={self.domain_analysis.top_prob:.3f}, entropy={self.domain_analysis.entropy:.3f})",
            f"  Context: C={self.context_analysis.similarity_score:.3f} ({self.context_analysis.source})",
            f"  Topic: continuing={self.topic_analysis.is_continuing} "
            f"(overlap={self.topic_analysis.overlap_score:.3f}, {self.topic_analysis.reason})",
            f"  Policy: {self.policy_decision.policy_case}",
        ]
        if self.task_label:
            lines.append(f"  Task: {self.task_label}")
        if self.scope_label:
            lines.append(f"  Scope: {self.scope_label}")
        return "\n".join(lines)


# ============================================================================
# 主分類器
# ============================================================================

class SemanticFlowClassifier:
    """
    語義流程分類器 - 整合多個模組進行對話狀態追蹤
    
    流程：
    1. 領域路由：判斷涉及的領域及其機率分布
    2. 上下文相似度：計算當前輸入與歷史的相似度
    3. 主題追蹤：判斷多領域主題是否延續
    4. 策略決策：基於 C+MT 進行四象限決策，選擇檢索策略
    5. (可選) 任務/範圍分類：分類用戶的查詢類型和範圍
    """

    def __init__(
        self,
        *,
        text_encoder: TextEncoder,
        domain_router: DomainRouter,
        context_similarity: Optional[ContextSimilarity] = None,
        topic_tracker: Optional[MultiTopicTracker] = None,
        policy_cfg: Optional[DSTPolicyConfig] = None,
        enable_task_scope: bool = False,
        task_scope_clf: TaskScopeClassifier = None,
    ):
        """
        初始化語義流程分類器
        
        Args:
            text_encoder: 文本編碼器
            domain_router: 領域路由器
            context_similarity: 上下文相似度計算器 (可選，使用預設配置)
            topic_tracker: 主題追蹤器 (可選，使用預設配置)
            policy_cfg: 策略配置 (可選，使用預設配置)
            enable_task_scope: 是否啟用任務/範圍分類
            task_scope_clf: 任務/範圍分類器 (如果 enable_task_scope=True 需提供)
        """
        self.text_encoder = text_encoder
        self.domain_router = domain_router

        # 初始化各子模組
        self.context_similarity = context_similarity or ContextSimilarity(
            encoder=self.text_encoder,
            cfg=ContextSimConfig(),
        )
        self.topic_tracker = topic_tracker or MultiTopicTracker(MultiTopicConfig())
        self.policy_cfg = policy_cfg or DSTPolicyConfig()

        # 狀態追蹤
        self.turn_index = 0

        # 任務/範圍分類
        self.enable_task_scope = enable_task_scope
        self.task_scope_clf = task_scope_clf

    def reset(self) -> None:
        """重置對話狀態"""
        self.turn_index = 0
        self.context_similarity.reset()
        self.topic_tracker.reset()
    
    def save_state(self, user_id: int, child_id: int, state_dir: str = "dialogue_states") -> bool:
        """
        保存當前對話狀態到文件
        
        Args:
            user_id: 用戶 ID
            child_id: 兒童 ID
            state_dir: 狀態文件保存目錄
        
        Returns:
            是否成功保存
        """
        try:
            from .state_persistence import save_dialogue_state
            return save_dialogue_state(
                user_id, child_id,
                self.context_similarity,
                self.topic_tracker,
                self.turn_index,
                state_dir
            )
        except Exception as e:
            print(f"[DST] 保存狀態失敗: {e}")
            return False
    
    def load_state(self, user_id: int, child_id: int, state_dir: str = "dialogue_states") -> bool:
        """
        從文件加載對話狀態
        
        Args:
            user_id: 用戶 ID
            child_id: 兒童 ID
            state_dir: 狀態文件保存目錄
        
        Returns:
            是否成功加載
        """
        try:
            from .state_persistence import load_dialogue_state
            turn_idx = load_dialogue_state(
                user_id, child_id,
                self.context_similarity,
                self.topic_tracker,
                state_dir
            )
            if turn_idx is not None:
                self.turn_index = turn_idx
                return True
            return False
        except Exception as e:
            print(f"[DST] 加載狀態失敗: {e}")
            return False

    def _analyze_domain(self, user_query: str) -> DomainAnalysis:
        """領域分析"""
        dr: DomainResult = self.domain_router.predict(user_query)
        
        return DomainAnalysis(
            top_domain=dr.top_domain,
            top_prob=float(dr.top_prob),
            entropy=float(dr.entropy),
            distribution=dict(dr.dist),
            active_domains=list(dr.active_domains),
            active_domain_probs=dict(dr.active_domain_probs),
            is_multi_domain=len(dr.active_domains) >= 2,
        )

    def _analyze_context(self, user_query: str) -> ContextAnalysis:
        """上下文相似度分析"""
        if self.turn_index == 0:
            C_info = {
                "C": self.context_similarity.cfg.neutral_first_turn,
                "source": "first_turn",
            }
        else:
            C_info = self.context_similarity.compute(user_query)
        
        C = float(C_info["C"])
        C = max(0.0, min(1.0, C))
        
        return ContextAnalysis(
            similarity_score=C,
            source=str(C_info.get("source", "")),
            is_first_turn=(self.turn_index == 0),
        )

    def _analyze_topic(
        self, 
        domain_dist: Dict[str, float], 
        top_domain: str, 
        cur_active_domains: List[str],
    ) -> TopicAnalysis:
        """主題延續分析"""
        # 在更新之前保存上一輪的分布和活躍領域（用於模糊延續）
        prev_dist_before_update = dict(self.topic_tracker.state.prev_dist) if self.topic_tracker.state.prev_dist else None
        prev_active_domains_before_update = list(self.topic_tracker.state.prev_active_domains) if self.topic_tracker.state.prev_active_domains else None
        
        topic_info = self.topic_tracker.check_topic_continuation(
            cur_dist=domain_dist,
            cur_raw_top_domain=top_domain,
            confidence=0.0,  # 不再使用 confidence，傳入 0.0 保持接口兼容
            cur_active_domains=cur_active_domains,
            prev_active_domains=self.topic_tracker.state.prev_active_domains,
        )
        
        return TopicAnalysis(
            is_continuing=bool(topic_info.get("topic_continue", True)),
            overlap_score=float(topic_info.get("topic_overlap", 0.0)),  # 這是綜合 MT 分數
            reason=str(topic_info.get("reason", "")),
            prev_top_domain=topic_info.get("prev_raw_top_domain"),
            cur_top_domain=topic_info.get("cur_raw_top_domain"),
            prev_dist=prev_dist_before_update,  # 保存更新前的上一輪分布
            prev_active_domains=prev_active_domains_before_update,  # 保存更新前的上一輪活躍領域
            tv_distance=topic_info.get("tv_distance"),  # TV 距離
        )

    def _get_all_domains(self) -> List[str]:
        """
        獲取所有領域列表
        
        Returns:
            List[str]: 所有領域名稱列表
        """
        from .domain_anchors import DOMAINS
        return DOMAINS.copy()
    
    def _get_memory_domains(self) -> Dict[str, float]:
        """
        獲取記憶中的領域分布
        
        Returns:
            Dict[str, float]: 記憶中的領域分布
        """
        return dict(self.topic_tracker.state.memory_dist) if self.topic_tracker.state.memory_dist else {}
    
    def _get_overview_distribution(
        self,
        domain: DomainAnalysis,
        strategy: str = "memory"
    ) -> Dict[str, float]:
        """
        獲取整體查詢時的領域分布
        
        Args:
            domain: 領域分析結果
            strategy: 策略
                - "all": 所有領域均勻分布
                - "memory": 使用記憶中的領域（memory_dist）
                - "active": 使用當前活躍領域（active_domains）
                - "hybrid": 混合策略（記憶 + 當前活躍）
        
        Returns:
            Dict[str, float]: 整體查詢時的領域分布
        """
        if strategy == "all":
            # 所有領域均勻分布
            all_domains = self._get_all_domains()
            return {d: 1.0 / len(all_domains) for d in all_domains}
        
        elif strategy == "memory":
            # 使用記憶中的領域
            mem_dist = self._get_memory_domains()
            if mem_dist:
                # 過濾掉機率太低的領域（< 0.05）
                filtered = {d: prob for d, prob in mem_dist.items() if prob >= 0.05}
                if filtered:
                    # 重新正規化
                    total = sum(filtered.values())
                    return {d: v / total for d, v in filtered.items()}
            # 如果記憶為空，回退到所有領域
            all_domains = self._get_all_domains()
            return {d: 1.0 / len(all_domains) for d in all_domains}
        
        elif strategy == "active":
            # 使用當前活躍領域
            if domain.active_domains:
                return {d: 1.0 / len(domain.active_domains) for d in domain.active_domains}
            # 如果沒有活躍領域，回退到所有領域
            all_domains = self._get_all_domains()
            return {d: 1.0 / len(all_domains) for d in all_domains}
        
        elif strategy == "hybrid":
            # 混合策略：記憶 + 當前活躍
            mem_dist = self._get_memory_domains()
            active_set = set(domain.active_domains) if domain.active_domains else set()
            
            # 合併領域
            combined_domains = set()
            if mem_dist:
                combined_domains.update([d for d, prob in mem_dist.items() if prob >= 0.05])
            combined_domains.update(active_set)
            
            if combined_domains:
                # 計算權重：記憶領域權重 0.6，活躍領域權重 0.4
                overview_dist = {}
                for d in combined_domains:
                    mem_weight = mem_dist.get(d, 0.0) * 0.6 if mem_dist else 0.0
                    active_weight = (1.0 / len(active_set) * 0.4) if d in active_set and active_set else 0.0
                    overview_dist[d] = mem_weight + active_weight
                
                # 重新正規化
                total = sum(overview_dist.values())
                if total > 0:
                    return {d: v / total for d, v in overview_dist.items()}
            
            # 如果合併後為空，回退到所有領域
            all_domains = self._get_all_domains()
            return {d: 1.0 / len(all_domains) for d in all_domains}
        
        else:
            # 默認：所有領域
            all_domains = self._get_all_domains()
            return {d: 1.0 / len(all_domains) for d in all_domains}
    
    def _decide_policy(
        self,
        domain: DomainAnalysis,
        context: ContextAnalysis,
        topic: TopicAnalysis,
        user_query: str,  # 新增參數：用於檢測整體查詢
    ) -> PolicyDecision:
        """決策層"""
        from .dst_policy import compute_MT, predicted_flow_from_C_MT
        
        # 初始化調整後的參數
        adjusted_topic_continue = topic.is_continuing
        adjusted_topic_overlap = topic.overlap_score
        fused_distribution = None  # 記錄融合後的分布（若有）
        
        # 檢測整體查詢
        try:
            from retrieval_module.task_scope_classifier_helper import is_overview_intent
            is_overview_query = is_overview_intent(user_query)
        except ImportError:
            # Fallback：簡單實現
            overview_keywords = ["整體", "概覽", "總結", "全部", "整份", "有哪些領域", "評估了哪些"]
            is_overview_query = any(kw in user_query for kw in overview_keywords)
        
        # 標記整體查詢
        domain.is_overview_query = is_overview_query
        
        # 如果整體查詢，生成整體領域分布
        if is_overview_query:
            # 使用所有領域均勻分布策略（所有領域都要看）
            overview_dist = self._get_overview_distribution(domain, strategy="all")
            domain.overview_distribution = overview_dist
            print(f"  [整體查詢] 檢測到整體查詢")
            print(f"    - 所有領域: {self._get_all_domains()}")
            print(f"    - 記憶領域: {list(self._get_memory_domains().keys())}")
            print(f"    - 當前活躍領域: {domain.active_domains}")
            print(f"    - 整體分布 top5: {sorted(overview_dist.items(), key=lambda x: x[1], reverse=True)[:5]}")
        
        # 模糊延續邏輯：如果模糊且上一輪領域存在，直接回退到上一輪分布
        # 注意：整體查詢時跳過模糊延續
        # 簡化版：只用 entropy 判斷模糊度，只要模糊就直接回退，不檢查 top domain 是否相同
        if self.policy_cfg.enable_ambiguous_continuation and not is_overview_query:
            is_ambiguous = domain.entropy >= self.policy_cfg.ambiguous_continuation_entropy_th
            
            # 使用 topic.prev_top_domain（這是更新前的值，真正的上一輪領域）
            # 而不是 self.topic_tracker.state.prev_raw_top_domain（已經被更新為當前輪）
            prev_top_domain = topic.prev_top_domain
            
            # 調試輸出
            if self.turn_index > 0:
                print(f"[模糊延續調試] Turn {self.turn_index}:")
                print(f"  - is_ambiguous: {is_ambiguous} (entropy={domain.entropy:.4f} >= {self.policy_cfg.ambiguous_continuation_entropy_th})")
                print(f"  - prev_top_domain: {prev_top_domain} (來自 TopicAnalysis)")
                print(f"  - current_top_domain: {domain.top_domain}")
                print(f"  - turn_index > 0: {self.turn_index > 0}")
            
            # 簡化：只要模糊就直接回退，不管 top domain 是否相同
            # 但需要檢查 prev_dist 是否存在，否則無法回退
            if is_ambiguous and prev_top_domain and self.turn_index > 0:
                # 檢查 prev_dist 是否存在
                prev_dist = topic.prev_dist  # 使用 TopicAnalysis 中保存的更新前的 prev_dist
                prev_active_domains = topic.prev_active_domains  # 使用 TopicAnalysis 中保存的更新前的 prev_active_domains
                
                if prev_dist:
                    # prev_dist 存在，可以觸發模糊延續
                    should_continue = True
                    print(f"  [模糊延續] 觸發模糊延續（entropy={domain.entropy:.4f} >= {self.policy_cfg.ambiguous_continuation_entropy_th}）")
                    
                    print(f"  [模糊延續] 調整前：topic_continue={topic.is_continuing}, topic_overlap={topic.overlap_score:.4f}")
                    # 調整 MT 相關參數以傾向延續
                    # 1. 提升 topic_overlap（假設有延續性）
                    adjusted_topic_overlap = max(
                        topic.overlap_score,
                        self.policy_cfg.ambiguous_continuation_min_overlap
                    )
                    
                    # 2. 如果原本不是延續，改為延續
                    if not topic.is_continuing:
                        adjusted_topic_continue = True

                    # 同步回 TopicAnalysis，讓輸出反映調整後的狀態
                    topic.is_continuing = adjusted_topic_continue
                    topic.overlap_score = adjusted_topic_overlap
                    if topic.reason:
                        topic.reason = f"{topic.reason} | ambiguous_continuation_adjusted"
                    else:
                        topic.reason = "ambiguous_continuation_adjusted"
                    
                    # 模糊延續：直接使用上一輪的分布，不融合本輪
                    # 直接使用上一輪的分布作為融合後的分布
                    fused_distribution = dict(prev_dist)
                    
                    # 更新記憶：使用上一輪的分布和活躍領域（讓記憶「回退」到上一輪狀態）
                    # 這樣如果下一輪還是模糊，會繼續延續上一輪的領域
                    if self.turn_index > 0:
                        # 將記憶更新為上一輪的分布
                        self.topic_tracker.state.memory_dist = dict(prev_dist)
                        # 同時更新 prev_dist，讓下一輪的計算更準確
                        self.topic_tracker.state.prev_dist = dict(prev_dist)
                        # 回退 prev_active_domains，保持一致性
                        if prev_active_domains:
                            self.topic_tracker.state.prev_active_domains = list(prev_active_domains)
                        else:
                            self.topic_tracker.state.prev_active_domains = []
                    
                    print(f"  [模糊延續] 直接使用上一輪分布：prev_domain={prev_top_domain}")
                    print(f"    - 本輪分布 top3: {sorted(domain.distribution.items(), key=lambda x: x[1], reverse=True)[:3]}")
                    print(f"    - 使用上一輪分布 top3: {sorted(prev_dist.items(), key=lambda x: x[1], reverse=True)[:3]}")
                    if prev_active_domains:
                        print(f"    - 回退 prev_active_domains: {prev_active_domains}")
                    
                    print(f"  [模糊延續] 調整後：topic_continue={adjusted_topic_continue}, topic_overlap={adjusted_topic_overlap:.4f}")
                else:
                    # prev_dist 為空，無法回退，不觸發模糊延續
                    print(f"  [模糊延續] 跳過：prev_dist 為空，無法回退到上一輪分布")
            else:
                if self.turn_index > 0:
                    print(f"  [模糊延續] 未觸發：is_ambiguous={is_ambiguous}, prev_top_domain={prev_top_domain}, turn_index={self.turn_index}")
        
        # 使用調整後的參數進行決策（簡化版：移除 D 和 margin）
        C_level, ambig, policy_case, action = decide_policy(
            C=context.similarity_score,
            normalized_entropy=domain.entropy,
            topic_continue=adjusted_topic_continue,  # 使用調整後的值
            topic_overlap=adjusted_topic_overlap,    # 使用調整後的值
            is_multi_domain=domain.is_multi_domain,
            cfg=self.policy_cfg,
        )
        
        # 使用調整後的 MT 計算 semantic_flow
        MT = compute_MT(adjusted_topic_continue, adjusted_topic_overlap)
        semantic_flow = predicted_flow_from_C_MT(
            C=context.similarity_score,
            MT=MT,
            cfg=self.policy_cfg
        )
        
        # 如果進行了記憶融合，更新 domain 的 fused_distribution
        if fused_distribution is not None:
            domain.fused_distribution = fused_distribution
        
        # 如果整體查詢，優先使用整體分布
        if domain.is_overview_query and domain.overview_distribution:
            domain.fused_distribution = domain.overview_distribution
            print(f"  [整體查詢] 使用整體分布作為 fused_distribution")
        
        return PolicyDecision(
            context_level=C_level,
            is_ambiguous=ambig,
            policy_case=policy_case,
            retrieval_action=action,
            semantic_flow=semantic_flow,
        )

    def _classify_task_scope(self, user_query: str) -> tuple:
        """任務/範圍分類"""
        if not self.enable_task_scope:
            return None, None, None, None
        
        try:
            task_result = self.task_scope_clf.predict_task(user_query)
            scope_result = self.task_scope_clf.predict_scope(user_query)
            
            task_label = task_result.label if hasattr(task_result, "label") else str(task_result)
            task_dist = task_result.dist if hasattr(task_result, "dist") else None
            scope_label = scope_result.label if hasattr(scope_result, "label") else str(scope_result)
            scope_dist = scope_result.dist if hasattr(scope_result, "dist") else None
            
            return task_label, task_dist, scope_label, scope_dist
        except Exception:
            return None, None, None, None

    def predict(
        self,
        user_query: str,
        assistant_reply: Optional[str] = None,
    ) -> FlowResult:
        """
        進行完整的語義流程分析
        
        Args:
            user_query: 用戶輸入
            assistant_reply: 助手回應 (用於更新上下文記憶)
        
        Returns:
            FlowResult: 包含所有分析層的完整結果
        """
        user_query = (user_query or "").strip()
        turn_idx = self.turn_index

        # 逐層分析
        domain = self._analyze_domain(user_query)
        context = self._analyze_context(user_query)
        topic = self._analyze_topic(
            domain.distribution, 
            domain.top_domain, 
            domain.active_domains,  # 傳遞 active_domains
        )
        policy = self._decide_policy(domain, context, topic, user_query)  # 傳遞 user_query

        # 任務/範圍分類
        task_label, task_dist, scope_label, scope_dist = self._classify_task_scope(user_query)

        # 構建結果
        result = FlowResult(
            turn_index=turn_idx,
            domain_analysis=domain,
            context_analysis=context,
            topic_analysis=topic,
            policy_decision=policy,
            task_label=task_label,
            task_dist=task_dist,
            scope_label=scope_label,
            scope_dist=scope_dist,
        )

        # 更新記憶狀態
        if assistant_reply is None:
            self.context_similarity.update(user_query)
        else:
            self.context_similarity.update(user_query, assistant_reply)

        self.turn_index += 1
        return result


# ============================================================================
# 便利函數
# ============================================================================

def format_flow_result(result: FlowResult) -> str:
    """格式化流程結果為可讀的文本"""
    return str(result)


def batch_analyze(
    classifier: SemanticFlowClassifier,
    dialogue: List[Dict[str, str]],
) -> List[FlowResult]:
    """
    批量分析一段對話序列
    
    Args:
        classifier: 分類器實例
        dialogue: 對話列表，每個元素是 {"user": "...", "assistant": "..."}
    
    Returns:
        FlowResult 列表
    """
    results = []
    for turn in dialogue:
        user_query = turn.get("user", "")
        assistant_reply = turn.get("assistant", None)
        result = classifier.predict(user_query, assistant_reply)
        results.append(result)
    return results
