#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dst_based_planner.py
基於 DST retrieval_action 的新檢索規劃器
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from .retrieval_planner import (
    RetrievalPlan, RetrievalMode, TopicPolicy, RetrievalState, PlanDecisionTrace
)
from .graph_client import SUBDOMAIN_TO_DOMAIN, DIALOGUE_STATE_DOMAINS, _map_subdomain_to_domain
from .topic_ontology import default_ontology


# 配置常數
TOTAL_K = 10  # 總片段數
TASK_THRESHOLD = 0.1  # Task 機率門檻
SUBDOMAIN_PROB_THRESHOLD = 0.03  # Subdomain 機率門檻（用於過濾極小值）


class DSTBasedRetrievalPlanner:
    """基於 DST retrieval_action 的檢索規劃器"""
    
    def __init__(self, ontology=None):
        """
        初始化規劃器
        
        Args:
            ontology: TopicOntology 實例（預設使用 default_ontology）
        """
        self.ontology = ontology or default_ontology
    
    def plan(
        self,
        user_query: str,
        turn_state: Dict[str, Any],
        retrieval_state: RetrievalState,
        doc_id: str
    ) -> RetrievalPlan:
        """
        規劃檢索策略（基於 DST retrieval_action）
        
        Args:
            user_query: 使用者查詢
            turn_state: 對話狀態（來自 dialogue_state_module 的 FlowResult.to_dict()）
            retrieval_state: 跨輪檢索狀態
            doc_id: 文件 ID
        
        Returns:
            RetrievalPlan: 檢索計劃
        """
        reasons = []
        
        # 1. 提取 DST 輸出
        retrieval_action = turn_state.get("retrieval_action", "")
        domain_distribution = turn_state.get("domain_distribution", {})
        task_dist = turn_state.get("task_dist", {})
        task_pred = turn_state.get("task_pred", "")
        scope_pred = turn_state.get("scope_pred", "")
        semantic_flow = turn_state.get("semantic_flow", "continue")
        top_domain = turn_state.get("top_domain", "")
        top_domain_prob = turn_state.get("top_domain_prob", 0.0)
        # DST 提供的領域範圍信息
        active_domains = turn_state.get("active_domains", [])  # 本輪的活躍領域
        prev_active_domains = turn_state.get("prev_active_domains", [])  # 上一輪的活躍領域
        normalized_entropy = turn_state.get("normalized_entropy", 0.0)  # 用於判斷是否模糊
        is_ambiguous = turn_state.get("is_ambiguous", False)  # DST 的模糊判斷
        fused_distribution_used = turn_state.get("fused_distribution_used", False)  # DST 是否觸發了模糊延續
        
        # 如果 entropy 很高（模糊），根據 DST 是否觸發模糊延續來決定處理方式
        ENTROPY_THRESHOLD = 0.8  # 與 DST 的 ambiguous_continuation_entropy_th 一致
        if normalized_entropy >= ENTROPY_THRESHOLD and prev_active_domains and turn_state.get("turn_index", 0) > 0:
            if fused_distribution_used:
                # DST 觸發了模糊延續：直接使用上一輪的領域
                reasons.append(f"entropy={normalized_entropy:.4f} >= {ENTROPY_THRESHOLD}（高模糊度），且 DST 觸發模糊延續，使用上一輪的 prev_active_domains={prev_active_domains}")
                active_domains = prev_active_domains
            else:
                # DST 未觸發模糊延續：合併上一輪和本輪的頂級領域
                merged_domains = set(prev_active_domains)
                if top_domain:
                    merged_domains.add(top_domain)
                active_domains = sorted(list(merged_domains))
                reasons.append(f"entropy={normalized_entropy:.4f} >= {ENTROPY_THRESHOLD}（高模糊度），但 DST 未觸發模糊延續，合併上一輪和本輪領域：prev_active_domains={prev_active_domains}, top_domain={top_domain}, merged={active_domains}")
        
        reasons.append(f"retrieval_action={retrieval_action}")
        reasons.append(f"semantic_flow={semantic_flow}")
        reasons.append(f"normalized_entropy={normalized_entropy:.4f}, is_ambiguous={is_ambiguous}")
        reasons.append(f"DST active_domains={active_domains}（已根據 entropy 調整）")
        reasons.append(f"DST prev_active_domains={prev_active_domains}")
        
        # 2. 決定 active tasks（過門檻即可）
        active_tasks = self._get_active_tasks(task_dist, task_pred, reasons)
        
        # 3. 合併 use_sections 和 section_weights
        use_sections = self._merge_use_sections(active_tasks, scope_pred, reasons)
        section_type_weights = self._merge_section_weights(
            active_tasks, task_dist, reasons
        )
        
        # 4. 根據 retrieval_action 決定 subdomain 範圍（優先使用 DST 提供的 active_domains）
        # 注意：如果 entropy 很高，active_domains 已經調整為 prev_active_domains
        subdomain_quota = self._decide_subdomain_quota(
            retrieval_action,
            domain_distribution,
            top_domain,
            top_domain_prob,
            active_domains,  # 可能已根據 entropy 調整為 prev_active_domains
            prev_active_domains,  # 上一輪的活躍領域
            retrieval_state,
            reasons
        )
        
        # 5. 兩層分配：subdomain → section type
        two_level_quota = self._allocate_two_level_quota(
            subdomain_quota,
            use_sections,
            section_type_weights,
            reasons
        )
        
        # 6. 決定 mode 和 topic_policy
        mode = RetrievalMode.TOPIC_FOCUSED
        topic_policy = TopicPolicy.MIX_TOPK
        
        # 7. 決定 rerank 權重（保持原邏輯）
        topic_overlap = turn_state.get("topic_overlap", 0.0)
        rerank_weights = self._get_rerank_weights(semantic_flow, topic_overlap, reasons)
        
        # 8. 構建 plan_decision_trace
        trace = PlanDecisionTrace(
            topics_source="dst_based",
            mode_reason=f"retrieval_action={retrieval_action}",
            policy_reason=f"topic_policy={topic_policy.value}",
            quota_reason=f"two_level_quota={two_level_quota}",
            section_weight_reason=f"active_tasks={active_tasks}",
            confidence_used=0.0,  # 已移除 confidence，使用 entropy 代替
            active_topics=list(subdomain_quota.keys()),
            routing_action=retrieval_action
        )
        
        # 9. 構建 RetrievalPlan
        # 將 two_level_quota 轉換為 topic_alloc（向後兼容）
        topic_alloc = {}
        for subdomain, section_quota in two_level_quota.items():
            topic_alloc[subdomain] = sum(section_quota.values())
        
        return RetrievalPlan(
            mode=mode,
            topic_policy=topic_policy,
            topics=list(subdomain_quota.keys()),
            graph_hops=3,  # 固定值，可根據需要調整
            k_per_subdomain=2,  # 固定值，可根據需要調整
            k_items=TOTAL_K,
            use_sections=use_sections,
            rerank_weights=rerank_weights,
            ask_clarify=(retrieval_action == "DUAL_OR_CLARIFY" and 
                        turn_state.get("is_ambiguous", False)),
            reasons=reasons,
            topic_alloc=topic_alloc,
            section_type_weights=section_type_weights,
            plan_decision_trace=trace,
            routing_action=retrieval_action,
            two_level_quota=two_level_quota,  # 兩層分配結果
            # 向後兼容
            domain_policy=topic_policy,
            domains=list(subdomain_quota.keys()),
            domain_alloc=topic_alloc,
            allow_parent_domain_expand=False
        )
    
    def _get_active_tasks(
        self,
        task_dist: Dict[str, float],
        task_pred: str,
        reasons: List[str]
    ) -> List[str]:
        """
        從 task_dist 中選取所有過門檻的 tasks
        
        Args:
            task_dist: Task 機率分布
            task_pred: 單一 task 預測（fallback）
            reasons: 原因列表
        
        Returns:
            List[str]: 過門檻的 task labels
        """
        if task_dist:
            active_tasks = [
                task for task, prob in task_dist.items()
                if prob >= TASK_THRESHOLD
            ]
            if active_tasks:
                reasons.append(f"active_tasks (threshold={TASK_THRESHOLD}): {active_tasks}")
                return active_tasks
        
        # Fallback: 使用單一 task_pred
        if task_pred:
            reasons.append(f"使用單一 task_pred: {task_pred}")
            return [task_pred]
        
        return []
    
    def _merge_use_sections(
        self,
        active_tasks: List[str],
        scope_pred: str,
        reasons: List[str]
    ) -> List[str]:
        """
        合併多個 tasks 對應的 use_sections
        
        Args:
            active_tasks: 過門檻的 task labels
            scope_pred: Scope 預測
            reasons: 原因列表
        
        Returns:
            List[str]: 合併後的 section types
        """
        all_sections = ["assessment", "observation", "training", "suggestion"]
        use_sections = set()
        
        # 根據每個 task 添加對應的 sections
        for task in active_tasks:
            if task == "T5_coaching":
                use_sections.update(["training", "suggestion"])
            elif task == "T3_clinical_to_daily":
                use_sections.update(["observation"])
            elif task == "T2_score_interpretation":
                use_sections.update(["assessment"])
            # T1_report_overview 和其他 tasks 預設使用所有 sections
            else:
                use_sections.update(all_sections)
        
        # Scope 也會影響
        if scope_pred == "S3_subskill_context":
            use_sections.update(["training", "suggestion"])
        
        # 如果沒有任何 task 或都未匹配，預設使用所有 sections
        if not use_sections:
            use_sections = set(all_sections)
        
        result = sorted(list(use_sections))
        reasons.append(f"merged use_sections: {result}")
        return result
    
    def _merge_section_weights(
        self,
        active_tasks: List[str],
        task_dist: Dict[str, float],
        reasons: List[str]
    ) -> Dict[str, float]:
        """
        使用 task_dist 的機率加權合併多個 tasks 的 section_weights
        
        Args:
            active_tasks: 過門檻的 task labels
            task_dist: Task 機率分布
            reasons: 原因列表
        
        Returns:
            Dict[str, float]: 合併後的 section_type 權重
        """
        combined_weights = {}
        
        # 只考慮過門檻的 tasks，並 normalize 它們的機率
        active_task_probs = {
            task: prob for task, prob in task_dist.items()
            if task in active_tasks
        }
        
        # Normalize 機率（確保總和為 1）
        total_prob = sum(active_task_probs.values())
        if total_prob <= 0:
            # Fallback: 均勻權重
            default_sections = ["assessment", "observation", "training", "suggestion"]
            result = {sec: 1.0 / len(default_sections) for sec in default_sections}
            reasons.append(f"fallback section_weights: {result}")
            return result
        
        normalized_probs = {
            task: prob / total_prob
            for task, prob in active_task_probs.items()
        }
        
        # 加權合併每個 task 的 section_weights
        for task, weight in normalized_probs.items():
            task_weights = self.ontology.get_section_weights(task)
            for sec_type, sec_weight in task_weights.items():
                combined_weights[sec_type] = (
                    combined_weights.get(sec_type, 0.0) + weight * sec_weight
                )
        
        # 如果某些 section_type 沒有權重，補上預設值
        all_sections = ["assessment", "observation", "training", "suggestion"]
        for sec_type in all_sections:
            if sec_type not in combined_weights:
                combined_weights[sec_type] = 0.0
        
        reasons.append(f"merged section_weights: {combined_weights}")
        return combined_weights
    
    def _decide_subdomain_quota(
        self,
        retrieval_action: str,
        domain_distribution: Dict[str, float],
        top_domain: str,
        top_domain_prob: float,
        active_domains: List[str],  # 新增：DST 提供的本輪活躍領域
        prev_active_domains: List[str],  # 新增：DST 提供的上一輪活躍領域
        retrieval_state: RetrievalState,
        reasons: List[str]
    ) -> Dict[str, int]:
        """
        根據 retrieval_action 決定 subdomain 範圍並分配 quota
        
        Args:
            retrieval_action: DST 的檢索動作
            domain_distribution: Subdomain 機率分布
            top_domain: 頂級 subdomain
            top_domain_prob: 頂級 subdomain 的機率
            retrieval_state: 檢索狀態
            reasons: 原因列表
        
        Returns:
            Dict[str, int]: Subdomain quota 分配，例如 {"粗大動作": 6, "精細動作": 4}
        """
        # 過濾極小機率
        filtered_dist = {
            subdomain: prob
            for subdomain, prob in domain_distribution.items()
            if prob >= SUBDOMAIN_PROB_THRESHOLD
        }
        
        if not filtered_dist:
            # Fallback: 使用 top_domain
            if top_domain:
                reasons.append(f"fallback: 使用 top_domain={top_domain}")
                return {top_domain: TOTAL_K}
            return {}
        
        # 根據 retrieval_action 決定 subdomain 範圍（優先使用 DST 提供的 active_domains）
        if retrieval_action == "NARROW_GRAPH":
            # NARROW_GRAPH: 使用上一輪和本輪的交集
            subdomains = self._narrow_graph_subdomains(
                filtered_dist, active_domains, prev_active_domains, retrieval_state, reasons
            )
        elif retrieval_action == "CONTEXT_FIRST":
            # CONTEXT_FIRST: 使用本輪的領域（靠上下文延續）
            subdomains = self._context_first_subdomains(
                filtered_dist, active_domains, reasons
            )
        elif retrieval_action == "WIDE_IN_DOMAIN":
            # WIDE_IN_DOMAIN: 使用本輪的領域（領域內廣泛檢索）
            subdomains = self._wide_in_domain_subdomains(
                filtered_dist, active_domains, top_domain, reasons
            )
        elif retrieval_action == "DUAL_OR_CLARIFY":
            # DUAL_OR_CLARIFY: 使用本輪的領域（可能更廣泛）
            subdomains = self._dual_or_clarify_subdomains(
                filtered_dist, active_domains, top_domain, top_domain_prob, reasons
            )
        else:
            # Fallback: 使用 top_domain 或 active_domains
            if active_domains:
                reasons.append(f"unknown retrieval_action={retrieval_action}, 使用 DST active_domains={active_domains}")
                subdomains = active_domains
            else:
                reasons.append(f"unknown retrieval_action={retrieval_action}, 使用 top_domain={top_domain}")
                subdomains = [top_domain] if top_domain else list(filtered_dist.keys())[:1]
        
        # 在選定的 subdomains 內按機率分配 quota
        quota = self._allocate_subdomain_quota(subdomains, filtered_dist, reasons)
        
        return quota
    
    def _narrow_graph_subdomains(
        self,
        domain_distribution: Dict[str, float],
        active_domains: List[str],  # DST 提供的本輪活躍領域
        prev_active_domains: List[str],  # DST 提供的上一輪活躍領域
        retrieval_state: RetrievalState,
        reasons: List[str]
    ) -> List[str]:
        """
        NARROW_GRAPH: 使用上一輪和本輪的交集（優先使用 DST 提供的 active_domains）
        
        Args:
            domain_distribution: 本輪機率分布
            active_domains: DST 提供的本輪活躍領域
            prev_active_domains: DST 提供的上一輪活躍領域
            retrieval_state: 檢索狀態
            reasons: 原因列表
        
        Returns:
            List[str]: 選定的 subdomains
        """
        # 優先使用 DST 提供的 active_domains
        if active_domains and prev_active_domains:
            prev_set = set(prev_active_domains)
            cur_set = set(active_domains)
            intersection = prev_set & cur_set
            
            if intersection:
                reasons.append(f"NARROW_GRAPH: 使用 DST 交集 {sorted(list(intersection))}")
                return sorted(list(intersection))
            else:
                # 如果沒有交集，使用本輪的 active_domains
                reasons.append(f"NARROW_GRAPH: 無交集，使用本輪 DST active_domains={sorted(active_domains)}")
                return sorted(active_domains)
        
        # Fallback: 使用 retrieval_state（向後兼容）
        prev_active = set(retrieval_state.active_topics) if retrieval_state.active_topics else set()
        cur_high_prob = {
            subdomain for subdomain, prob in domain_distribution.items()
            if prob >= 0.1
        }
        
        intersection = prev_active & cur_high_prob
        if intersection:
            reasons.append(f"NARROW_GRAPH: fallback 使用 retrieval_state 交集 {sorted(list(intersection))}")
            return sorted(list(intersection))
        
        # 最後 fallback: 使用本輪 active_domains 或 top
        if active_domains:
            reasons.append(f"NARROW_GRAPH: fallback 使用 DST active_domains={sorted(active_domains)}")
            return sorted(active_domains)
        
        top_subdomain = max(domain_distribution.items(), key=lambda x: x[1])[0]
        reasons.append(f"NARROW_GRAPH: fallback top_subdomain={top_subdomain}")
        return [top_subdomain]
    
    def _context_first_subdomains(
        self,
        domain_distribution: Dict[str, float],
        active_domains: List[str],  # DST 提供的本輪活躍領域
        reasons: List[str]
    ) -> List[str]:
        """
        CONTEXT_FIRST: 使用本輪的領域（靠上下文延續，不排除上一輪）
        
        語義：C 高但 MT 低，文本看起來像延續但主題池不延續
        策略：不硬縮小，先靠上下文延續，使用本輪的 active_domains
        
        Args:
            domain_distribution: 本輪機率分布
            active_domains: DST 提供的本輪活躍領域
            reasons: 原因列表
        
        Returns:
            List[str]: 選定的 subdomains（優先使用 DST 的 active_domains）
        """
        # 優先使用 DST 提供的 active_domains
        if active_domains:
            reasons.append(f"CONTEXT_FIRST: 使用 DST active_domains={sorted(active_domains)}（靠上下文延續）")
            return sorted(active_domains)
        
        # Fallback: 使用本輪高機率 subdomains（機率 >= 0.1）
        cur_high_prob = {
            subdomain for subdomain, prob in domain_distribution.items()
            if prob >= 0.1
        }
        
        if cur_high_prob:
            reasons.append(f"CONTEXT_FIRST: fallback 使用本輪高機率 subdomains {sorted(cur_high_prob)}")
            return sorted(list(cur_high_prob))
        
        # 最後 fallback: 使用本輪 top
        top_subdomain = max(domain_distribution.items(), key=lambda x: x[1])[0]
        reasons.append(f"CONTEXT_FIRST: fallback top_subdomain={top_subdomain}")
        return [top_subdomain]
    
    def _wide_in_domain_subdomains(
        self,
        domain_distribution: Dict[str, float],
        active_domains: List[str],  # DST 提供的本輪活躍領域
        top_domain: str,
        reasons: List[str]
    ) -> List[str]:
        """
        WIDE_IN_DOMAIN: 同 macro-domain 下的所有相關 subdomains（優先使用 DST 的 active_domains）
        
        Args:
            domain_distribution: 本輪機率分布
            active_domains: DST 提供的本輪活躍領域
            top_domain: 頂級 subdomain
            reasons: 原因列表
        
        Returns:
            List[str]: 選定的 subdomains
        """
        # 優先使用 DST 提供的 active_domains
        if active_domains:
            reasons.append(f"WIDE_IN_DOMAIN: 使用 DST active_domains={sorted(active_domains)}")
            return sorted(active_domains)
        
        if not top_domain or top_domain not in SUBDOMAIN_TO_DOMAIN:
            # Fallback: 使用本輪高機率
            cur_high_prob = [
                subdomain for subdomain, prob in domain_distribution.items()
                if prob >= 0.1
            ]
            if cur_high_prob:
                reasons.append(f"WIDE_IN_DOMAIN: fallback 使用本輪高機率 {cur_high_prob}")
                return cur_high_prob
            return []
        
        # 找到 top_domain 對應的 macro-domain
        macro_domain = _map_subdomain_to_domain(top_domain)
        
        # 找出所有屬於同一 macro-domain 的 subdomains（且機率 >= 門檻）
        same_macro_subdomains = [
            subdomain for subdomain, prob in domain_distribution.items()
            if (subdomain in SUBDOMAIN_TO_DOMAIN and
                _map_subdomain_to_domain(subdomain) == macro_domain and
                prob >= SUBDOMAIN_PROB_THRESHOLD)
        ]
        
        if same_macro_subdomains:
            reasons.append(f"WIDE_IN_DOMAIN: fallback macro_domain={macro_domain}, subdomains={same_macro_subdomains}")
            return sorted(same_macro_subdomains)
        
        # Fallback: 使用 top_domain
        reasons.append(f"WIDE_IN_DOMAIN: fallback top_domain={top_domain}")
        return [top_domain]
    
    def _dual_or_clarify_subdomains(
        self,
        domain_distribution: Dict[str, float],
        active_domains: List[str],  # DST 提供的本輪活躍領域
        top_domain: str,
        top_domain_prob: float,
        reasons: List[str]
    ) -> List[str]:
        """
        DUAL_OR_CLARIFY: 優先使用 DST 的 active_domains
        如果 top domain 機率過門檻，優先使用 top domain
        否則檢索所有過門檻的 subdomains
        
        Args:
            domain_distribution: 本輪機率分布
            active_domains: DST 提供的本輪活躍領域
            top_domain: 頂級 subdomain
            top_domain_prob: 頂級 subdomain 的機率
            reasons: 原因列表
        
        Returns:
            List[str]: 選定的 subdomains
        """
        # 優先使用 DST 提供的 active_domains
        if active_domains:
            reasons.append(f"DUAL_OR_CLARIFY: 使用 DST active_domains={sorted(active_domains)}")
            return sorted(active_domains)
        
        # 門檻：如果 top_prob >= 0.3，優先使用 top_domain
        TOP_DOMAIN_PROB_THRESHOLD = 0.3
        
        # 如果 top domain 機率過門檻，優先使用 top domain
        if top_domain and top_domain_prob >= TOP_DOMAIN_PROB_THRESHOLD:
            reasons.append(
                f"DUAL_OR_CLARIFY: fallback top domain 機率過門檻 "
                f"({top_domain}={top_domain_prob:.4f} >= {TOP_DOMAIN_PROB_THRESHOLD})，優先使用"
            )
            return [top_domain]
        
        # 否則檢索所有過門檻的 subdomains
        all_subdomains = [
            subdomain for subdomain, prob in domain_distribution.items()
            if prob >= SUBDOMAIN_PROB_THRESHOLD
        ]
        
        reasons.append(
            f"DUAL_OR_CLARIFY: fallback top domain 機率未過門檻 "
            f"({top_domain}={top_domain_prob:.4f} < {TOP_DOMAIN_PROB_THRESHOLD})，"
            f"檢索所有 subdomains ({len(all_subdomains)} 個)"
        )
        return sorted(all_subdomains)
    
    def _allocate_subdomain_quota(
        self,
        subdomains: List[str],
        domain_distribution: Dict[str, float],
        reasons: List[str]
    ) -> Dict[str, int]:
        """
        在選定的 subdomains 內按機率分配 quota
        
        Args:
            subdomains: 選定的 subdomains
            domain_distribution: 完整機率分布
            reasons: 原因列表
        
        Returns:
            Dict[str, int]: Subdomain quota 分配
        """
        if not subdomains:
            return {}
        
        # 只考慮選定的 subdomains 的機率
        selected_probs = {
            subdomain: domain_distribution.get(subdomain, 0.0)
            for subdomain in subdomains
        }
        
        # Normalize
        total_prob = sum(selected_probs.values())
        if total_prob <= 0:
            # 均勻分配
            quota_per_subdomain = TOTAL_K // len(subdomains)
            remainder = TOTAL_K % len(subdomains)
            allocation = {subdomain: quota_per_subdomain for subdomain in subdomains}
            for i, subdomain in enumerate(subdomains[:remainder]):
                allocation[subdomain] += 1
            reasons.append(f"均勻分配 quota: {allocation}")
            return allocation
        
        normalized_probs = {
            subdomain: prob / total_prob
            for subdomain, prob in selected_probs.items()
        }
        
        # 按機率分配 quota
        allocation = {}
        allocated_total = 0
        
        # 先保證每個 subdomain 至少 1 個（如果 quota 足夠）
        min_per_subdomain = 1 if TOTAL_K >= len(subdomains) else 0
        
        for subdomain in subdomains:
            if min_per_subdomain > 0:
                allocation[subdomain] = min_per_subdomain
                allocated_total += min_per_subdomain
            else:
                allocation[subdomain] = 0
        
        # 剩餘 quota 按機率分配
        remaining_quota = TOTAL_K - allocated_total
        
        if remaining_quota > 0:
            weighted_allocation = {}
            for subdomain in subdomains:
                weighted_allocation[subdomain] = round(
                    normalized_probs[subdomain] * remaining_quota
                )
            
            # 調整以確保總和正確
            weighted_total = sum(weighted_allocation.values())
            if weighted_total != remaining_quota:
                diff = remaining_quota - weighted_total
                if diff > 0:
                    # 加給機率最大的
                    max_subdomain = max(subdomains, key=lambda s: normalized_probs[s])
                    weighted_allocation[max_subdomain] += diff
                else:
                    # 從機率最小的扣
                    min_subdomain = min(subdomains, key=lambda s: normalized_probs[s])
                    weighted_allocation[min_subdomain] = max(
                        0, weighted_allocation[min_subdomain] + diff
                    )
            
            # 合併到 allocation
            for subdomain in subdomains:
                allocation[subdomain] += weighted_allocation[subdomain]
        
        reasons.append(f"subdomain quota 分配: {allocation}")
        return allocation
    
    def _allocate_two_level_quota(
        self,
        subdomain_quota: Dict[str, int],
        use_sections: List[str],
        section_weights: Dict[str, float],
        reasons: List[str]
    ) -> Dict[str, Dict[str, int]]:
        """
        兩層分配：subdomain → section type
        
        Args:
            subdomain_quota: 第一層分配，例如 {"粗大動作": 6, "精細動作": 4}
            use_sections: Section types，例如 ["training", "suggestion", "observation"]
            section_weights: Section type 權重
            reasons: 原因列表
        
        Returns:
            Dict[str, Dict[str, int]]: 
            例如 {
                "粗大動作": {"training": 3, "suggestion": 2, "observation": 1},
                "精細動作": {"training": 2, "suggestion": 1, "observation": 1}
            }
        """
        result = {}
        
        for subdomain, quota in subdomain_quota.items():
            section_quota = self._allocate_section_quota(
                quota, use_sections, section_weights
            )
            result[subdomain] = section_quota
        
        reasons.append(f"兩層分配完成: {len(result)} 個 subdomains")
        return result
    
    def _allocate_section_quota(
        self,
        subdomain_quota: int,
        use_sections: List[str],
        section_weights: Dict[str, float]
    ) -> Dict[str, int]:
        """
        在單一 subdomain 內，按 section type 分配 quota
        
        Args:
            subdomain_quota: 該 subdomain 的總 quota
            use_sections: 要使用的 section types
            section_weights: Section type 權重
        
        Returns:
            Dict[str, int]: 每個 section type 的 quota
        """
        # 只考慮 use_sections 中的 section types
        relevant_weights = {
            sec_type: section_weights.get(sec_type, 0.0)
            for sec_type in use_sections
        }
        
        # 如果沒有權重，均勻分配
        total_weight = sum(relevant_weights.values())
        if total_weight <= 0:
            quota_per_section = subdomain_quota // len(use_sections)
            remainder = subdomain_quota % len(use_sections)
            allocation = {sec: quota_per_section for sec in use_sections}
            for i, sec in enumerate(use_sections[:remainder]):
                allocation[sec] += 1
            return allocation
        
        # Normalize 權重
        normalized_weights = {
            sec_type: weight / total_weight
            for sec_type, weight in relevant_weights.items()
        }
        
        # 按權重分配 quota
        allocation = {}
        allocated_total = 0
        
        # 先保證每個 section type 至少 1 個（如果 quota 足夠）
        min_per_section = 1 if subdomain_quota >= len(use_sections) else 0
        
        for sec_type in use_sections:
            if min_per_section > 0:
                allocation[sec_type] = min_per_section
                allocated_total += min_per_section
            else:
                allocation[sec_type] = 0
        
        # 剩餘 quota 按權重分配
        remaining_quota = subdomain_quota - allocated_total
        
        if remaining_quota > 0:
            weighted_allocation = {}
            for sec_type in use_sections:
                weighted_allocation[sec_type] = round(
                    normalized_weights[sec_type] * remaining_quota
                )
            
            # 調整以確保總和正確
            weighted_total = sum(weighted_allocation.values())
            if weighted_total != remaining_quota:
                diff = remaining_quota - weighted_total
                if diff > 0:
                    max_sec = max(use_sections, key=lambda s: normalized_weights[s])
                    weighted_allocation[max_sec] += diff
                else:
                    min_sec = min(use_sections, key=lambda s: normalized_weights[s])
                    weighted_allocation[min_sec] = max(
                        0, weighted_allocation[min_sec] + diff
                    )
            
            # 合併到 allocation
            for sec_type in use_sections:
                allocation[sec_type] += weighted_allocation[sec_type]
        
        return allocation
    
    def _get_rerank_weights(
        self,
        semantic_flow: str,
        topic_overlap: float,
        reasons: List[str]
    ) -> Dict[str, float]:
        """
        決定 rerank 權重（保持原邏輯）
        
        Args:
            semantic_flow: 語義流程類型
            topic_overlap: 主題重疊分數
            reasons: 原因列表
        
        Returns:
            Dict[str, float]: Rerank 權重
        """
        # 保持原邏輯（可根據需要調整）
        if semantic_flow == "continue":
            weights = {
                "sim_q": 0.4,
                "sim_anchor": 0.5,
                "graph_prox": 0.1
            }
        elif semantic_flow == "shift_soft":
            weights = {
                "sim_q": 0.5,
                "sim_anchor": 0.3,
                "graph_prox": 0.2
            }
        else:  # shift_hard
            weights = {
                "sim_q": 0.6,
                "sim_anchor": 0.2,
                "graph_prox": 0.2
            }
        
        reasons.append(f"rerank_weights: {weights}")
        return weights

