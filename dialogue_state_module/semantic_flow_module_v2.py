# semantic_flow_module_v2.py
# 清晰、模塊化的語義流程追蹤系統

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import json
import numpy as np

from .embedding import TextEncoder, score_overview_similarity
from .domain_router import DomainRouter, DomainResult
from .context_similarity import ContextSimilarity, ContextSimConfig
from .multi_topic_tracker import MultiTopicTracker, MultiTopicConfig
from .dst_policy import DSTPolicyConfig, decide_policy
from .task_scope_classifier import TaskScopeClassifier, PredictResult
from .utils.region_extractor import extract_region


# 控制 DST 模組是否輸出詳細除錯訊息
DST_DEBUG_VERBOSE: bool = True


# ============================================================================
# 結果數據結構
# ============================================================================

@dataclass
class DomainAnalysis: # 領域分析結果
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
class ContextAnalysis: # 上下文相似度分析
    similarity_score: float  # C
    source: str  # "first_turn" | "prev_user" | "prev_bot"
    is_first_turn: bool


@dataclass
class TopicAnalysis: # 主題延續分析
    is_continuing: bool
    overlap_score: float  # MT
    reason: str
    prev_top_domain: Optional[str] = None
    cur_top_domain: Optional[str] = None
    prev_dist: Optional[Dict[str, float]] = None  # 上一輪的領域分布（更新前）
    prev_active_domains: Optional[List[str]] = None  # 上一輪的活躍領域列表（更新前）
    tv_distance: Optional[float] = None  # TV 距離（Total Variation Distance）
    active_domain_coverage: Optional[float] = None  # active_domains Jaccard 覆蓋度
    continuation_mode: Optional[str] = None  # "strong" | "soft" | "shift"


@dataclass
class PolicyDecision: # 策略決策結果
    context_level: str  # "high" | "low"
    is_ambiguous: bool
    policy_case: str  # e.g., "CH_MTH_NARROW_MD"
    retrieval_action: str  # "NARROW_GRAPH" | "CONTEXT_FIRST" | etc
    semantic_flow: str  # "continue" | "shift_soft" | "shift_hard"


@dataclass
class FlowResult: # 完整的語義流程分析結果
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
    detected_region: Optional[str] = None  # 偵測到的地區（如：台北市）

    def to_dict(self) -> dict: # 轉換為字典格式（完整分析結果）
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
        
        if self.detected_region:
            result["detected_region"] = self.detected_region
        
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

    def to_json(self) -> str: # 轉換為 JSON 字符串
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    def __str__(self) -> str: # 簡潔的文本表示
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
        if self.detected_region:
            lines.append(f"  Region: {self.detected_region}")
        return "\n".join(lines)


# ============================================================================
# 主分類器
# ============================================================================

class SemanticFlowClassifier: # 語義流程分類器
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
        overview_anchor_vecs: Optional[List[np.ndarray]] = None,
        overview_sim_threshold: float = 0.5,
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
            overview_anchor_vecs: 整體意圖錨點向量列表（用於向量比對，取代關鍵字）
            overview_sim_threshold: 整體意圖相似度門檻
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
        self._prev_scope: Optional[str] = None  # 上一輪 Scope（供模糊沿用與持久化）
        self._prev_was_overview: bool = False  # 上一輪是否為整體（供模糊+整體兩條規則）

        # 整體意圖：向量比對（不用關鍵字）
        self._overview_anchor_vecs = overview_anchor_vecs if overview_anchor_vecs else []
        self._overview_sim_threshold = overview_sim_threshold

        # 任務/範圍分類
        self.enable_task_scope = enable_task_scope
        self.task_scope_clf = task_scope_clf

    def reset(self) -> None:
        """重置對話狀態"""
        self.turn_index = 0
        self._prev_scope = None
        self._prev_was_overview = False
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
                state_dir,
                prev_scope=self._prev_scope,
                prev_was_overview=self._prev_was_overview,
            )
        except Exception as e:
            print(f"[DST] 保存狀態失敗: {e}")
            return False
    
    def load_state(self, user_id: int, child_id: int, state_dir: str = "dialogue_states") -> bool: # 載入前一輪的對話狀態
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
            from .state_persistence import load_dialogue_state # 載入前一輪的對話狀態
            result = load_dialogue_state(
                user_id, child_id,
                self.context_similarity,
                self.topic_tracker,
                state_dir
            )
            if result is not None:
                self.turn_index = result[0]
                self._prev_scope = result[1]
                self._prev_was_overview = result[2] if len(result) > 2 else False
                return True
            return False
        except Exception as e:
            print(f"[DST] 加載狀態失敗: {e}")
            return False

    def _analyze_domain(self, user_query: str) -> DomainAnalysis: # 以本輪使用者的訊息來判斷領域
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

    def _analyze_context(self, user_query: str) -> ContextAnalysis: # 計算本輪與前一輪的語義相似度
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

    def _analyze_topic( # 判斷本輪是否為主題延續
        self, 
        domain_dist: Dict[str, float], 
        top_domain: str, 
        cur_active_domains: List[str],
        is_ambiguous: bool,
    ) -> TopicAnalysis:
        # 在更新之前保存上一輪的分布和活躍領域（用於模糊延續）
        prev_dist_before_update = dict(self.topic_tracker.state.prev_dist) if self.topic_tracker.state.prev_dist else None
        prev_active_domains_before_update = list(self.topic_tracker.state.prev_active_domains) if self.topic_tracker.state.prev_active_domains else None
        
        topic_info = self.topic_tracker.check_topic_continuation(
            cur_dist=domain_dist,
            cur_raw_top_domain=top_domain,
            confidence=0.0,  # 不再使用 confidence，傳入 0.0 保持接口兼容
            cur_active_domains=cur_active_domains,
            prev_active_domains=self.topic_tracker.state.prev_active_domains,
            is_ambiguous=is_ambiguous,
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
            active_domain_coverage=topic_info.get("active_domain_coverage"),
            continuation_mode=topic_info.get("continuation_mode"),
        )

    def _get_all_domains(self) -> List[str]: # 取得所有領域列表
        from .domain_anchors import DOMAINS
        return DOMAINS.copy()
    
    def _get_overview_distribution(self) -> Dict[str, float]: # 取得整體查詢時的領域分布
        all_domains = self._get_all_domains()
        return {d: 1.0 / len(all_domains) for d in all_domains}
    
    def _analyze_overview_query(
        self,
        domain: DomainAnalysis,
        user_query: str,
        is_ambiguous: bool,
    ) -> Tuple[bool, float]:
        """
        判斷本輪是否為「整體查詢」，並在需要時建立 overview_distribution。
        只負責整體相關訊號與分布，不處理記憶或模糊延續。
        """
        overview_sim = 0.0
        if self._overview_anchor_vecs:
            query_vec = self.text_encoder.encode(user_query)
            overview_sim = score_overview_similarity(query_vec, self._overview_anchor_vecs)
        is_overview_by_vector = overview_sim >= self._overview_sim_threshold
        is_overview_query = is_ambiguous and (is_overview_by_vector or self._prev_was_overview)
        if is_overview_query and DST_DEBUG_VERBOSE:
            print(
                f"  [整體查詢] 模糊且(向量像整體或上輪整體) → 判定為整體 "
                f"(ambiguous={is_ambiguous}, sim={overview_sim:.3f}, prev_was_overview={self._prev_was_overview})"
            )
        
        domain.is_overview_query = is_overview_query
        overview_dist = None
        
        # 若本輪為整體或需沿用整體：生成整體領域分布
        if is_overview_query or (
            self.policy_cfg.enable_ambiguous_continuation
            and domain.entropy >= self.policy_cfg.ambiguous_continuation_entropy_th
            and self._prev_was_overview
        ):
            overview_dist = self._get_overview_distribution()
            domain.overview_distribution = overview_dist
            if is_overview_query and DST_DEBUG_VERBOSE:
                print(
                    "[整體查詢] 整體分布 top5: "
                    f"{sorted(overview_dist.items(), key=lambda x: x[1], reverse=True)[:5]}"
                )
        
        return is_overview_query, float(overview_sim)
    
    def _handle_memory_and_fused_distribution(
        self,
        domain: DomainAnalysis,
        topic: TopicAnalysis,
        is_ambiguous: bool,
        is_overview_query: bool,
    ) -> Tuple[bool, float, Optional[Dict[str, float]]]:
        """
        根據已蒐集的訊號（模糊 / 整體 / topic 狀態）決定：
        - 是否觸發整體重啟（規則 A）或整體沿用（規則 B）
        - 是否進行模糊延續並覆寫 topic_continue / topic_overlap
        - 本輪要使用的 fused_distribution（若有）
        
        注意：這裡可以在特殊情況下重置或回退 MultiTopicTracker 的記憶，
        讓所有記憶相關 side-effect 集中在同一個函式中處理。
        """
        adjusted_topic_continue = topic.is_continuing
        adjusted_topic_overlap = topic.overlap_score
        fused_distribution: Optional[Dict[str, float]] = None
        
        # 規則 A：模糊且整體且上一輪不是整體 → 整體、清除記憶、新對話
        if is_ambiguous and is_overview_query and not self._prev_was_overview:
            domain.is_overview_query = True
            if domain.overview_distribution is None:
                domain.overview_distribution = self._get_overview_distribution()
            fused_distribution = dict(domain.overview_distribution)
            domain.fused_distribution = fused_distribution
            self.topic_tracker.reset()
            if DST_DEBUG_VERBOSE:
                print("  [整體查詢] 規則A：模糊+整體，清除記憶、啟用新對話")
        
        # 規則 B：模糊且上一輪是整體 → 整體、有記憶（沿用整體分布）
        elif is_ambiguous and self._prev_was_overview:
            domain.is_overview_query = True
            if domain.overview_distribution is None:
                domain.overview_distribution = self._get_overview_distribution()
            fused_distribution = dict(domain.overview_distribution)
            domain.fused_distribution = fused_distribution
            if DST_DEBUG_VERBOSE:
                print("  [整體查詢] 規則B：模糊+上一輪整體，判定為整體、有記憶")
        
        # 一般模糊延續：模糊且上一輪有領域、非整體情境
        elif self.policy_cfg.enable_ambiguous_continuation and not is_overview_query:
            # 使用 topic.prev_top_domain（這是更新前的值，真正的上一輪領域）
            prev_top_domain = topic.prev_top_domain
            
            # 調試輸出（僅在 verbose 模式下啟用）
            if self.turn_index > 0 and DST_DEBUG_VERBOSE:
                print(f"[模糊延續調試] Turn {self.turn_index}:")
                print(
                    f"  - is_ambiguous: {is_ambiguous} "
                    f"(entropy={domain.entropy:.4f} >= {self.policy_cfg.ambiguous_continuation_entropy_th})"
                )
                print(f"  - prev_top_domain: {prev_top_domain} (來自 TopicAnalysis)")
                print(f"  - current_top_domain: {domain.top_domain}")
                print(f"  - turn_index > 0: {self.turn_index > 0}")
            
            # 簡化：只要模糊就直接回退，不管 top domain 是否相同
            # 但需要檢查 prev_dist 是否存在，否則無法回退
            if is_ambiguous and prev_top_domain and self.turn_index > 0:
                prev_dist = topic.prev_dist
                prev_active_domains = topic.prev_active_domains
                
                if prev_dist:
                    if DST_DEBUG_VERBOSE:
                        print(
                            "  [模糊延續] 觸發模糊延續"
                            f"（entropy={domain.entropy:.4f} >= {self.policy_cfg.ambiguous_continuation_entropy_th}）"
                        )
                        
                        print(
                            "  [模糊延續] 調整前："
                            f"topic_continue={topic.is_continuing}, topic_overlap={topic.overlap_score:.4f}"
                        )
                    # 調整 MT 相關參數以傾向延續
                    adjusted_topic_overlap = max(
                        topic.overlap_score,
                        self.policy_cfg.ambiguous_continuation_min_overlap,
                    )
                    
                    if not topic.is_continuing:
                        adjusted_topic_continue = True

                    # 同步回 TopicAnalysis，讓輸出反映調整後的狀態
                    topic.is_continuing = adjusted_topic_continue
                    topic.overlap_score = adjusted_topic_overlap
                    if topic.reason:
                        topic.reason = f"{topic.reason} | ambiguous_continuation_adjusted"
                    else:
                        topic.reason = "ambiguous_continuation_adjusted"
                    
                    # 模糊延續：直接使用上一輪的分布與 active_domains，不融合本輪
                    fused_distribution = dict(prev_dist)

                    # 若上一輪有 active_domains，沿用作為本輪的 active_domains（語意：模糊但延續上一輪主題池）
                    if prev_active_domains:
                        domain.active_domains = list(prev_active_domains)
                        domain.is_multi_domain = len(domain.active_domains) >= 2
                    
                    # 更新記憶：使用上一輪的分布和活躍領域（讓記憶「回退」到上一輪狀態）
                    if self.turn_index > 0:
                        self.topic_tracker.state.memory_dist = dict(prev_dist)
                        self.topic_tracker.state.prev_dist = dict(prev_dist)
                        if prev_active_domains:
                            self.topic_tracker.state.prev_active_domains = list(prev_active_domains)
                        else:
                            self.topic_tracker.state.prev_active_domains = []
                    
                    if DST_DEBUG_VERBOSE:
                        print(f"  [模糊延續] 直接使用上一輪分布：prev_domain={prev_top_domain}")
                        print(
                            "    - 本輪分布 top3: "
                            f"{sorted(domain.distribution.items(), key=lambda x: x[1], reverse=True)[:3]}"
                        )
                        print(
                            "    - 使用上一輪分布 top3: "
                            f"{sorted(prev_dist.items(), key=lambda x: x[1], reverse=True)[:3]}"
                        )
                        if prev_active_domains:
                            print(f"    - 回退 prev_active_domains: {prev_active_domains}")
                        
                        print(
                            "  [模糊延續] 調整後："
                            f"topic_continue={adjusted_topic_continue}, topic_overlap={adjusted_topic_overlap:.4f}"
                        )
                else:
                    if DST_DEBUG_VERBOSE:
                        print("  [模糊延續] 跳過：prev_dist 為空，無法回退到上一輪分布")
            else:
                if self.turn_index > 0 and DST_DEBUG_VERBOSE:
                    print(
                        "  [模糊延續] 未觸發："
                        f"is_ambiguous={is_ambiguous}, prev_top_domain={prev_top_domain}, turn_index={self.turn_index}"
                    )
        # 非模糊情境：根據多領域追蹤的強/軟/切模式調整 active_domains 與 fused_distribution
        if not is_ambiguous and not is_overview_query:
            mode = getattr(topic, "continuation_mode", None)
            prev_dist = topic.prev_dist
            prev_active = topic.prev_active_domains or []

            # 強延續：偏向上一輪領域
            if mode == "strong":
                if prev_active:
                    domain.active_domains = list(prev_active)
                    domain.is_multi_domain = len(domain.active_domains) >= 2
                if prev_dist:
                    # 讓檢索優先使用上一輪的領域分布
                    fused_distribution = dict(prev_dist)
                    domain.fused_distribution = fused_distribution

            # 軟延續：上一輪 + 本輪混和
            elif mode == "soft":
                # active_domains 取聯集（多領域池內換焦點）
                merged = set(domain.active_domains or []) | set(prev_active or [])
                domain.active_domains = sorted(list(merged)) if merged else list(domain.active_domains)
                domain.is_multi_domain = len(domain.active_domains) >= 2

                # 分布混和：0.5 * prev + 0.5 * cur（若有 prev_dist）
                if prev_dist:
                    alpha = 0.5
                    mixed: Dict[str, float] = {}
                    keys = set(prev_dist.keys()) | set(domain.distribution.keys())
                    for k in keys:
                        mixed[k] = alpha * float(prev_dist.get(k, 0.0)) + (1.0 - alpha) * float(domain.distribution.get(k, 0.0))
                    total = sum(mixed.values())
                    if total > 0:
                        mixed = {k: v / total for k, v in mixed.items()}
                    fused_distribution = mixed
                    domain.fused_distribution = fused_distribution

            # 切換（mode == "shift" 或其他）：保持本輪 active_domains 與分布，不做額外 fused

        return adjusted_topic_continue, adjusted_topic_overlap, fused_distribution
    
    def _decide_policy( # 決策層
        self,
        domain: DomainAnalysis,
        context: ContextAnalysis,
        topic: TopicAnalysis,
        user_query: str,  # 新增參數：用於檢測整體查詢
        task_label: Optional[str] = None,
        detected_region: Optional[str] = None,
    ) -> PolicyDecision:
        from .dst_policy import compute_MT, predicted_flow_from_C_MT
        
        # 是否模糊（需先算，整體僅在「模糊」時才依向量／上一輪整體判定）
        is_ambiguous = self.policy_cfg.enable_ambiguous_continuation and (domain.entropy >= self.policy_cfg.ambiguous_continuation_entropy_th)
        
        # 1. 先以 entropy + 向量相似度判斷是否為整體查詢，並建立 overview_distribution（若需要）
        is_overview_query, _ = self._analyze_overview_query(domain, user_query, is_ambiguous)
        
        # 2. 再根據「模糊 / 整體 / topic 狀態」統一決定記憶與 fused_distribution
        adjusted_topic_continue, adjusted_topic_overlap, fused_distribution = self._handle_memory_and_fused_distribution(
            domain=domain,
            topic=topic,
            is_ambiguous=is_ambiguous,
            is_overview_query=is_overview_query,
        )
        
        # 使用調整後的參數進行決策（簡化版：移除 D 和 margin）
        C_level, ambig, policy_case, action = decide_policy(
            C=context.similarity_score,
            normalized_entropy=domain.entropy,
            topic_continue=adjusted_topic_continue,  # 使用調整後的值
            topic_overlap=adjusted_topic_overlap,    # 使用調整後的值
            is_multi_domain=domain.is_multi_domain,
            cfg=self.policy_cfg,
            task_label=task_label,
            detected_region=detected_region,
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
            if DST_DEBUG_VERBOSE:
                print(f"  [整體查詢] 使用整體分布作為 fused_distribution")
        
        return PolicyDecision(
            context_level=C_level,
            is_ambiguous=ambig,
            policy_case=policy_case,
            retrieval_action=action,
            semantic_flow=semantic_flow,
        )

    def _classify_scope(
        self,
        domain_analysis: DomainAnalysis,
        user_query: str,
    ) -> tuple:
        """
        依規則分類 Scope（要檢索多少主題分支）。
        特殊處理：
        1. 若發生領域模糊觸發對話沿用（fused_distribution 已設）→ Scope 沿用上一輪。
        2. 若首輪且本輪模糊（entropy 高）→ 預設「整體」。
        否則：
        - S_overview: 整體查詢 → 查整張圖
        - S_multi_domain: 多領域 → 查多個特定領域
        - S_domain: 單領域 → 查單一領域
        """
        # 1. 本輪為整體查詢（向量比或規則 A/B）→ 先判整體
        if domain_analysis.is_overview_query:
            label = "S_overview"
            dist = {label: 1.0}
            return label, dist

        # 2. 模糊沿用：只有在「模糊 + 已有 fused 分布」時，Scope 才沿用上一輪
        #    若只是一般 strong/soft 延續導致 fused_distribution 被設，但熵不高，仍應依本輪 active_domains 判斷
        if (
            domain_analysis.fused_distribution is not None
            and self.policy_cfg.enable_ambiguous_continuation
            and domain_analysis.entropy >= self.policy_cfg.ambiguous_continuation_entropy_th
        ):
            label = self._prev_scope or "S_overview"
            dist = {label: 1.0}
            return label, dist

        # 3. 首輪且模糊：無上一輪可沿用，預設整體
        if self.turn_index == 0 and domain_analysis.entropy >= self.policy_cfg.ambiguous_continuation_entropy_th:
            label = "S_overview"
            dist = {label: 1.0}
            return label, dist

        # 4. 一般規則
        if len(domain_analysis.active_domains) >= 2:
            label = "S_multi_domain"
        else:
            label = "S_domain"
        dist = {label: 1.0}
        return label, dist

    def _classify_task_only(self, user_query: str) -> tuple:
        """僅做 Task 分類（Scope 改由規則 _classify_scope 決定）"""
        if not self.enable_task_scope:
            return None, None
        
        try:
            task_result = self.task_scope_clf.predict_task(user_query)
            task_label = task_result.label if hasattr(task_result, "label") else str(task_result)
            task_dist = task_result.dist if hasattr(task_result, "dist") else None
            return task_label, task_dist
        except Exception:
            return None, None

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

        # 先依 entropy 判斷本輪是否為「模糊」狀態，再將此訊號傳給 MultiTopicTracker
        is_ambiguous_for_topic = (
            self.policy_cfg.enable_ambiguous_continuation
            and domain.entropy >= self.policy_cfg.ambiguous_continuation_entropy_th
        )

        topic = self._analyze_topic(
            domain.distribution, 
            domain.top_domain, 
            domain.active_domains,  # 傳遞 active_domains
            is_ambiguous=is_ambiguous_for_topic,
        )
        # 提取地區
        detected_region = extract_region(user_query)

        # Task 分類（分類器）；Scope 分類（規則：整體/單領域/多領域）
        task_label, task_dist = self._classify_task_only(user_query)

        policy = self._decide_policy(
            domain, context, topic, user_query, 
            task_label=task_label, 
            detected_region=detected_region
        )

        scope_label, scope_dist = self._classify_scope(domain, user_query)

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
            detected_region=detected_region,
        )

        # 供下一輪 Scope 沿用與持久化
        self._prev_scope = scope_label
        self._prev_was_overview = (scope_label == "S_overview")

        # 更新記憶狀態
        if assistant_reply is None:
            self.context_similarity.update(user_query)
        else:
            self.context_similarity.update(user_query, assistant_reply)

        self.turn_index += 1
        return result