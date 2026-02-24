#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
retrieval_executor.py
執行檢索計劃，從 Neo4j 取得資料並進行 rerank
"""

from typing import Dict, List, Optional, Any
import numpy as np
from .retrieval_planner import RetrievalPlan, RetrievalMode, DomainPolicy, TopicPolicy, RetrievalState
from .graph_client import GraphClient


class RetrievalExecutor:
    """檢索執行器"""
    
    def __init__(self, graph_client: GraphClient, bge_model: Optional[Any] = None):
        """
        初始化執行器
        
        Args:
            graph_client: Neo4j 圖客戶端
            bge_model: BGE-M3 模型（可選，用於計算 embedding 相似度）
        """
        self.graph_client = graph_client
        self.bge_model = bge_model
    
    def execute(
        self,
        plan: RetrievalPlan,
        doc_id: str,
        user_query: str,
        assistant_anchor: str = ""
    ) -> Dict[str, Any]:
        """
        執行檢索計劃
        
        Args:
            plan: 檢索計劃
            doc_id: 文件 ID
            user_query: 使用者查詢
            assistant_anchor: 助理錨點文字（用於 rerank）
        
        Returns:
            Dict: {
                "sections": [...],  # Section 節點列表
                "items": [...],      # Item 節點列表
                "section_ids": [...],  # 用到的 section IDs
                "item_ids": [...],     # 用到的 item IDs
                "candidates": [...]     # 候選內容（用於回答）
            }
        """
        # CLARIFY 模式：直接返回反問，不進 LLM
        routing_action = getattr(plan, 'routing_action', '')
        if routing_action == "CLARIFY":
            return self._execute_clarify(plan, user_query)
        
        if plan.mode == RetrievalMode.META:
            return self._execute_meta(plan)
        elif plan.mode == RetrievalMode.SCORE:
            return self._execute_score(plan, doc_id, user_query)
        elif plan.mode == RetrievalMode.REPORT_OVERVIEW:
            return self._execute_report_overview(plan, doc_id)
        elif plan.mode == RetrievalMode.GLOBAL_OVERVIEW:
            return self._execute_global_overview(plan, doc_id, user_query, assistant_anchor)
        elif plan.mode == RetrievalMode.TOPIC_FOCUSED:
            return self._execute_topic_focused(plan, doc_id, user_query, assistant_anchor)
        else:  # DOMAIN (向後兼容)
            return self._execute_domain(plan, doc_id, user_query, assistant_anchor)
    
    def _execute_clarify(self, plan: RetrievalPlan, user_query: str) -> Dict[str, Any]:
        """
        執行 CLARIFY 模式：直接返回反問，不進 LLM
        
        Args:
            plan: 檢索計劃
            user_query: 使用者查詢
        
        Returns:
            Dict: 包含反問訊息的結果
        """
        # 從 plan_decision_trace 獲取建議的 topics
        suggested_topics = []
        if plan.plan_decision_trace:
            suggested_topics = plan.plan_decision_trace.selected_topics[:3]
        
        # 如果沒有建議 topics，使用 plan.topics
        if not suggested_topics and plan.topics:
            suggested_topics = plan.topics[:3]
        
        # 構建反問訊息
        if suggested_topics:
            topic_list = " / ".join(suggested_topics)
            clarify_text = (
                f"我目前不確定你想問哪個領域。你是想看：{topic_list}？"
                f"你可以回覆其中一個，或說要我一起整理 {suggested_topics[0]}+{suggested_topics[1] if len(suggested_topics) > 1 else suggested_topics[0]}。"
            )
        else:
            clarify_text = "我目前不確定你想問哪個領域。請告訴我你想了解的主題，例如：粗大動作、精細動作、口語理解等。"
        
        return {
            "sections": [],
            "items": [],
            "section_ids": [],
            "item_ids": [],
            "candidates": [{
                "type": "clarify",
                "text": clarify_text,
                "suggested_topics": suggested_topics,
                "source": None
            }]
        }
    
    def _execute_meta(self, plan: RetrievalPlan) -> Dict[str, Any]:
        """執行 META 模式檢索"""
        return {
            "sections": [],
            "items": [],
            "section_ids": [],
            "item_ids": [],
            "candidates": [{
                "type": "meta",
                "text": "此查詢屬於資源/行政類別，不在報告內容範圍內。建議建立資源知識庫或允許網路查詢。",
                "source": None
            }]
        }
    
    def _execute_score(
        self, 
        plan: RetrievalPlan, 
        doc_id: str, 
        user_query: str
    ) -> Dict[str, Any]:
        """執行 SCORE 模式檢索"""
        # 取得所有 Assessment sections（包含分數資訊）
        sections = self.graph_client.get_score_sections(doc_id, limit=plan.k_items)
        
        # 取得每個 section 的 items
        items = []
        section_ids = []
        item_ids = []
        
        for section in sections:
            section_id = section.get("id")
            section_ids.append(section_id)
            
            # 取得該 section 的 items
            section_items = self.graph_client.fetch_items(
                section_id, 
                "assessment", 
                limit=plan.k_items,
                include_subitems=True
            )
            
            for item in section_items:
                items.append(item)
                item_ids.append(item.get("id"))
                # 包含子項
                if "subitems" in item:
                    for subitem in item["subitems"]:
                        item_ids.append(subitem.get("id"))
        
        # 構建候選內容
        candidates = []
        for section in sections:
            candidates.append({
                "type": "section",
                "section_type": "assessment",
                "text": section.get("text", ""),
                "result": section.get("result", ""),
                "tools": section.get("tools", ""),
                "page_start": section.get("page_start"),
                "page_end": section.get("page_end"),
                "section_id": section.get("id")
            })
        
        for item in items[:plan.k_items]:
            candidates.append({
                "type": "item",
                "item_type": "assessment",
                "text": item.get("text", ""),
                "level": item.get("level", 1),
                "item_id": item.get("id"),
                "subitems": item.get("subitems", [])
            })
        
        return {
            "sections": sections,
            "items": items,
            "section_ids": section_ids,
            "item_ids": item_ids,
            "candidates": candidates
        }
    
    def _execute_report_overview(
        self, 
        plan: RetrievalPlan, 
        doc_id: str
    ) -> Dict[str, Any]:
        """執行 REPORT_OVERVIEW 模式檢索"""
        # 取得報告概覽所需的 sections
        domain_name = plan.domains[0] if plan.domains else None
        overview_data = self.graph_client.get_report_overview_sections(
            doc_id,
            domain_name=domain_name,
            k_per_subdomain=plan.k_per_subdomain
        )
        
        sections = []
        items = []
        section_ids = []
        item_ids = []
        candidates = []
        
        # 檢查是否有找到任何資料
        total_sections_found = sum(
            len(sec_list) 
            for domain_dict in overview_data.values() 
            for subdomain_dict in domain_dict.values() 
            for sec_list in subdomain_dict.values()
        )
        
        # 如果鎖定了 domain 但查不到結果，嘗試 fallback：查詢所有 domains
        if total_sections_found == 0 and domain_name:
            overview_data = self.graph_client.get_report_overview_sections(
                doc_id,
                domain_name=None,  # 查詢所有 domains
                k_per_subdomain=plan.k_per_subdomain
            )
        
        for domain_name, subdomains in overview_data.items():
            for subdomain_name, sec_dict in subdomains.items():
                for sec_type, sec_list in sec_dict.items():
                    if sec_type not in plan.use_sections:
                        continue
                    
                    for section in sec_list:
                        section_id = section.get("id")
                        sections.append(section)
                        section_ids.append(section_id)
                        
                        # 取得該 section 的 items（只取 level=1）
                        section_items = self.graph_client.fetch_items(
                            section_id,
                            sec_type,
                            limit=plan.k_items,
                            include_subitems=False
                        )
                        
                        for item in section_items:
                            items.append(item)
                            item_ids.append(item.get("id"))
                        
                        # 構建候選內容
                        candidates.append({
                            "type": "section",
                            "section_type": sec_type,
                            "domain": domain_name,
                            "subdomain": subdomain_name,
                            "text": section.get("text", ""),
                            "page_start": section.get("page_start"),
                            "page_end": section.get("page_end"),
                            "section_id": section_id,
                            "items": [item.get("text", "") for item in section_items]
                        })
        
        return {
            "sections": sections,
            "items": items,
            "section_ids": section_ids,
            "item_ids": item_ids,
            "candidates": candidates
        }
    
    def _execute_topic_focused(
        self,
        plan: RetrievalPlan,
        doc_id: str,
        user_query: str,
        assistant_anchor: str
    ) -> Dict[str, Any]:
        """
        執行 TOPIC_FOCUSED 模式檢索（泛用邏輯）
        
        規則：
        - topics 非空時，一律 topic-filter，再合併 rerank
        - 按 topic_alloc 配額分配
        """
        sections = []
        items = []
        section_ids = []
        item_ids = []
        items_with_path = []
        candidates = []
        
        # 使用 plan.topics（如果有的話），否則使用 plan.domains（向後兼容）
        topics = plan.topics if hasattr(plan, 'topics') and plan.topics else (plan.domains if plan.domains else [])
        topic_alloc = plan.topic_alloc if hasattr(plan, 'topic_alloc') and plan.topic_alloc else (plan.domain_alloc if plan.domain_alloc else {})
        
        if not topics:
            # 無 topics，返回空結果（不應該發生，但防禦性編程）
            return {
                "sections": [],
                "items": [],
                "section_ids": [],
                "item_ids": [],
                "candidates": []
            }
        
        # 檢查是否有兩層分配（優先使用）
        two_level_quota = getattr(plan, 'two_level_quota', None)
        
        if two_level_quota and len(two_level_quota) > 0:
            # 使用兩層分配：subdomain → section type
            for subdomain_name, section_quota_dict in two_level_quota.items():
                for sec_type, quota in section_quota_dict.items():
                    if quota <= 0:
                        continue
                    
                    # 只查詢該 subdomain 和該 section type
                    domain_data = self.graph_client.get_domain_sections(
                        doc_id,
                        [subdomain_name],
                        [sec_type],  # 只查這個 section type
                        k_per_subdomain=plan.k_per_subdomain,
                        allow_parent_domain_expand=False
                    )
                    
                    # 處理該 subdomain + section type 的 items
                    section_items_count = 0
                    for d_name, subdomains in domain_data.items():
                        for sd_name, sec_dict in subdomains.items():
                            if sec_type not in sec_dict:
                                continue
                            
                            sec_list = sec_dict[sec_type]
                            for section in sec_list:
                                section_id = section.get("id")
                                section_name = section.get("name", "") or self._get_section_type_name(sec_type)
                                sections.append(section)
                                section_ids.append(section_id)
                                
                                # 取得該 section 的 items（限制配額）
                                remaining_quota = quota - section_items_count
                                if remaining_quota <= 0:
                                    continue
                                
                                section_items = self.graph_client.fetch_items(
                                    section_id,
                                    sec_type,
                                    limit=remaining_quota,
                                    include_subitems=True
                                )
                                
                                for item in section_items:
                                    if section_items_count >= quota:
                                        break
                                    item_id = item.get("id")
                                    items.append(item)
                                    item_ids.append(item_id)
                                    section_items_count += 1
                                    
                                    # 為 item 添加路徑信息
                                    item_with_path = item.copy()
                                    item_with_path["path"] = {
                                        "domain": d_name,
                                        "subdomain": sd_name,
                                        "topic": sd_name,
                                        "section_type": sec_type,
                                        "section_name": section_name,
                                        "section_id": section_id,
                                        "page_start": section.get("page_start"),
                                        "page_end": section.get("page_end")
                                    }
                                    items_with_path.append(item_with_path)
                                    
                                    # 處理 subitems（也計入配額）
                                    if "subitems" in item:
                                        for subitem in item["subitems"]:
                                            if section_items_count >= quota:
                                                break
                                            subitem_id = subitem.get("id")
                                            item_ids.append(subitem_id)
                                            section_items_count += 1
                                            
                                            subitem_with_path = subitem.copy()
                                            subitem_with_path["path"] = {
                                                "domain": d_name,
                                                "subdomain": sd_name,
                                                "topic": sd_name,
                                                "section_type": sec_type,
                                                "section_name": section_name,
                                                "section_id": section_id,
                                                "parent_item_id": item_id,
                                                "page_start": section.get("page_start"),
                                                "page_end": section.get("page_end")
                                            }
                                            items_with_path.append(subitem_with_path)
                                
                                if section_items_count >= quota:
                                    break
                            if section_items_count >= quota:
                                break
                        if section_items_count >= quota:
                            break
        
        # 如果使用配額策略（MIX_TOPK, SOFT_FOCUS），按配額查詢（向後兼容）
        elif topic_alloc and len(topic_alloc) > 0:
            # 逐個 topic 查詢，按配額限制
            for topic_name, alloc_quota in topic_alloc.items():
                # 只查詢該 topic（對應 subdomain）
                domain_data = self.graph_client.get_domain_sections(
                    doc_id,
                    [topic_name],
                    plan.use_sections,
                    k_per_subdomain=plan.k_per_subdomain,
                    allow_parent_domain_expand=False  # TOPIC_FOCUSED 不允許展開
                )
                
                # 處理該 topic 的 items，限制數量
                topic_items_count = 0
                for d_name, subdomains in domain_data.items():
                    for subdomain_name, sec_dict in subdomains.items():
                        for sec_type, sec_list in sec_dict.items():
                            for section in sec_list:
                                section_id = section.get("id")
                                section_name = section.get("name", "") or self._get_section_type_name(sec_type)
                                sections.append(section)
                                section_ids.append(section_id)
                                
                                # 取得該 section 的 items（限制配額）
                                remaining_quota = alloc_quota - topic_items_count
                                if remaining_quota <= 0:
                                    continue
                                
                                section_items = self.graph_client.fetch_items(
                                    section_id,
                                    sec_type,
                                    limit=remaining_quota,
                                    include_subitems=True
                                )
                                
                                for item in section_items:
                                    if topic_items_count >= alloc_quota:
                                        break
                                    item_id = item.get("id")
                                    items.append(item)
                                    item_ids.append(item_id)
                                    topic_items_count += 1
                                    
                                    # 為 item 添加路徑信息（包含 topic）
                                    item_with_path = item.copy()
                                    item_with_path["path"] = {
                                        "domain": d_name,
                                        "subdomain": subdomain_name,
                                        "topic": subdomain_name,  # topic 對應 subdomain
                                        "section_type": sec_type,
                                        "section_name": section_name,
                                        "section_id": section_id,
                                        "page_start": section.get("page_start"),
                                        "page_end": section.get("page_end")
                                    }
                                    items_with_path.append(item_with_path)
                                    
                                    # 處理 subitems（也計入配額）
                                    if "subitems" in item:
                                        for subitem in item["subitems"]:
                                            if topic_items_count >= alloc_quota:
                                                break
                                            subitem_id = subitem.get("id")
                                            item_ids.append(subitem_id)
                                            topic_items_count += 1
                                            
                                            subitem_with_path = subitem.copy()
                                            subitem_with_path["path"] = {
                                                "domain": d_name,
                                                "subdomain": subdomain_name,
                                                "topic": subdomain_name,
                                                "section_type": sec_type,
                                                "section_name": section_name,
                                                "section_id": section_id,
                                                "parent_item_id": item_id,
                                                "page_start": section.get("page_start"),
                                                "page_end": section.get("page_end")
                                            }
                                            items_with_path.append(subitem_with_path)
                                
                                if topic_items_count >= alloc_quota:
                                    break
                            if topic_items_count >= alloc_quota:
                                break
                        if topic_items_count >= alloc_quota:
                            break
                    if topic_items_count >= alloc_quota:
                        break
        else:
            # 無配額，均勻查詢所有 topics
            domain_data = self.graph_client.get_domain_sections(
                doc_id,
                topics,
                plan.use_sections,
                k_per_subdomain=plan.k_per_subdomain,
                allow_parent_domain_expand=False
            )
            
            # 構建 items_with_path
            for domain_name, subdomains in domain_data.items():
                for subdomain_name, sec_dict in subdomains.items():
                    for sec_type, sec_list in sec_dict.items():
                        for section in sec_list:
                            section_id = section.get("id")
                            section_name = section.get("name", "") or self._get_section_type_name(sec_type)
                            sections.append(section)
                            section_ids.append(section_id)
                            
                            section_items = self.graph_client.fetch_items(
                                section_id,
                                sec_type,
                                limit=plan.k_items,
                                include_subitems=True
                            )
                            
                            for item in section_items:
                                item_id = item.get("id")
                                items.append(item)
                                item_ids.append(item_id)
                                
                                item_with_path = item.copy()
                                item_with_path["path"] = {
                                    "domain": domain_name,
                                    "subdomain": subdomain_name,
                                    "topic": subdomain_name,
                                    "section_type": sec_type,
                                    "section_name": section_name,
                                    "section_id": section_id,
                                    "page_start": section.get("page_start"),
                                    "page_end": section.get("page_end")
                                }
                                items_with_path.append(item_with_path)
                                
                                if "subitems" in item:
                                    for subitem in item["subitems"]:
                                        subitem_id = subitem.get("id")
                                        item_ids.append(subitem_id)
                                        
                                        subitem_with_path = subitem.copy()
                                        subitem_with_path["path"] = {
                                            "domain": domain_name,
                                            "subdomain": subdomain_name,
                                            "topic": subdomain_name,
                                            "section_type": sec_type,
                                            "section_name": section_name,
                                            "section_id": section_id,
                                            "parent_item_id": item_id,
                                            "page_start": section.get("page_start"),
                                            "page_end": section.get("page_end")
                                        }
                                        items_with_path.append(subitem_with_path)
        
        # 對所有 items 進行 rerank（包含 topic_match_bonus）
        candidates = self._rerank_candidates(
            [], items_with_path, user_query, assistant_anchor, plan
        )
        
        # Diversification 保底：確保每個 topic 至少 1 筆（如果候選足夠）
        if topics and len(candidates) > len(topics):
            final_candidates = []
            used_topics = set()
            # 先加入每個 topic 至少 1 筆
            for topic in topics:
                for candidate in candidates:
                    candidate_topic = candidate.get("path", {}).get("topic") or candidate.get("path", {}).get("subdomain")
                    if candidate_topic == topic and topic not in used_topics:
                        final_candidates.append(candidate)
                        used_topics.add(topic)
                        break
            # 再加入剩餘候選
            for candidate in candidates:
                if candidate not in final_candidates:
                    final_candidates.append(candidate)
            candidates = final_candidates[:plan.k_items]
        else:
            candidates = candidates[:plan.k_items]
        
        return {
            "sections": sections,
            "items": items,
            "section_ids": section_ids,
            "item_ids": item_ids,
            "candidates": candidates
        }
    
    def _execute_global_overview(
        self,
        plan: RetrievalPlan,
        doc_id: str,
        user_query: str,
        assistant_anchor: str
    ) -> Dict[str, Any]:
        """
        執行 GLOBAL_OVERVIEW 模式檢索（改進版）
        
        規則：
        - 只有在 topics 空且 task 真的是全域概覽才允許
        - 查詢所有 topics，但要有 diversification
        - 改進：先用 query embedding/keyword 對 section 做一層輕量 rerank，挑前 N（例如 12）再從其中取 5
        """
        # 查詢所有 topics
        all_domains = self.graph_client.list_domains(doc_id)
        all_topics = []
        for domain in all_domains:
            subdomains = self.graph_client.list_subdomains(doc_id, domain["name"])
            all_topics.extend([sd["name"] for sd in subdomains])
        
        # 查詢所有 sections（不限制數量，稍後 rerank）
        domain_data = self.graph_client.get_domain_sections(
            doc_id,
            all_topics,
            plan.use_sections,
            k_per_subdomain=plan.k_per_subdomain * 2,  # 多取一些，供 rerank 使用
            allow_parent_domain_expand=True
        )
        
        sections = []
        items = []
        section_ids = []
        item_ids = []
        items_with_path = []
        
        # 收集所有 items
        for domain_name, subdomains in domain_data.items():
            for subdomain_name, sec_dict in subdomains.items():
                for sec_type, sec_list in sec_dict.items():
                    for section in sec_list:
                        section_id = section.get("id")
                        section_name = section.get("name", "") or self._get_section_type_name(sec_type)
                        sections.append(section)
                        section_ids.append(section_id)
                        
                        # 取得該 section 的 items
                        section_items = self.graph_client.fetch_items(
                            section_id,
                            sec_type,
                            limit=plan.k_items * 2,  # 多取一些，供 rerank 使用
                            include_subitems=True
                        )
                        
                        for item in section_items:
                            item_id = item.get("id")
                            items.append(item)
                            item_ids.append(item_id)
                            
                            item_with_path = item.copy()
                            item_with_path["path"] = {
                                "domain": domain_name,
                                "subdomain": subdomain_name,
                                "topic": subdomain_name,
                                "section_type": sec_type,
                                "section_name": section_name,
                                "section_id": section_id,
                                "page_start": section.get("page_start"),
                                "page_end": section.get("page_end")
                            }
                            items_with_path.append(item_with_path)
                            
                            if "subitems" in item:
                                for subitem in item["subitems"]:
                                    subitem_id = subitem.get("id")
                                    item_ids.append(subitem_id)
                                    
                                    subitem_with_path = subitem.copy()
                                    subitem_with_path["path"] = {
                                        "domain": domain_name,
                                        "subdomain": subdomain_name,
                                        "topic": subdomain_name,
                                        "section_type": sec_type,
                                        "section_name": section_name,
                                        "section_id": section_id,
                                        "parent_item_id": item_id,
                                        "page_start": section.get("page_start"),
                                        "page_end": section.get("page_end")
                                    }
                                    items_with_path.append(subitem_with_path)
        
        # 對所有 items 進行 rerank（包含 query similarity）
        candidates = self._rerank_candidates(
            [], items_with_path, user_query, assistant_anchor, plan
        )
        
        # 限制最終數量（max_chunks=5）
        max_chunks = 5
        candidates = candidates[:max_chunks]
        
        # Diversification 保底：確保每個 topic 至少 1 筆（如果候選足夠）
        if all_topics and len(candidates) > len(all_topics):
            final_candidates = []
            used_topics = set()
            # 先加入每個 topic 至少 1 筆
            for topic in all_topics[:max_chunks]:  # 只考慮前 max_chunks 個 topics
                for candidate in candidates:
                    candidate_topic = candidate.get("path", {}).get("topic") or candidate.get("path", {}).get("subdomain")
                    if candidate_topic == topic and topic not in used_topics:
                        final_candidates.append(candidate)
                        used_topics.add(topic)
                        break
            # 再加入剩餘候選
            for candidate in candidates:
                if candidate not in final_candidates:
                    final_candidates.append(candidate)
            candidates = final_candidates[:max_chunks]
        else:
            candidates = candidates[:max_chunks]
        
        return {
            "sections": sections,
            "items": items,
            "section_ids": section_ids,
            "item_ids": item_ids,
            "candidates": candidates
        }
    
    def _execute_domain(
        self,
        plan: RetrievalPlan,
        doc_id: str,
        user_query: str,
        assistant_anchor: str
    ) -> Dict[str, Any]:
        """執行 DOMAIN 模式檢索（保留，向後兼容）"""
        if plan.domain_policy == DomainPolicy.UNLOCKED and not plan.domains:
            # 無法確定 domain，fallback：查詢所有 domains
            all_domains = self.graph_client.list_domains(doc_id)
            if all_domains:
                # 使用所有 domains 作為 fallback
                fallback_domains = [d["name"] for d in all_domains]
                plan.domains = fallback_domains
                print(f"[RetrievalExecutor] UNLOCKED 且無指定 domains，使用 fallback：查詢所有 domains {fallback_domains}")
                print(f"[RetrievalExecutor] 注意：這可能導致檢索結果不夠精確，建議在 retrieval_planner 中優先使用 top_domain 映射")
            else:
                # 如果連 domains 都查不到，返回空結果
                return {
                    "sections": [],
                    "items": [],
                    "section_ids": [],
                    "item_ids": [],
                    "candidates": []
                }
        
        sections = []
        items = []
        section_ids = []
        item_ids = []
        items_with_path = []  # 包含路徑信息的 items
        candidates = []  # 初始化 candidates，防止未定義錯誤
        
        # 如果使用新策略（MIX_TOPK, SOFT_TOP1），按配額查詢
        if plan.domain_policy in [DomainPolicy.MIX_TOPK, DomainPolicy.SOFT_TOP1] and plan.domain_alloc and len(plan.domain_alloc) > 0:
            # 逐個 domain/subdomain 查詢，按配額限制
            for domain_name, alloc_quota in plan.domain_alloc.items():
                # 只查詢該 domain/subdomain
                domain_data = self.graph_client.get_domain_sections(
                    doc_id,
                    [domain_name],
                    plan.use_sections,
                    k_per_subdomain=plan.k_per_subdomain,
                    allow_parent_domain_expand=plan.allow_parent_domain_expand
                )
                
                # 處理該 domain 的 items，限制數量
                domain_items_count = 0
                for d_name, subdomains in domain_data.items():
                    for subdomain_name, sec_dict in subdomains.items():
                        for sec_type, sec_list in sec_dict.items():
                            for section in sec_list:
                                section_id = section.get("id")
                                section_name = section.get("name", "") or self._get_section_type_name(sec_type)
                                sections.append(section)
                                section_ids.append(section_id)
                                
                                # 取得該 section 的 items（限制配額）
                                remaining_quota = alloc_quota - domain_items_count
                                if remaining_quota <= 0:
                                    continue
                                
                                section_items = self.graph_client.fetch_items(
                                    section_id,
                                    sec_type,
                                    limit=remaining_quota,
                                    include_subitems=True
                                )
                                
                                for item in section_items:
                                    if domain_items_count >= alloc_quota:
                                        break
                                    item_id = item.get("id")
                                    items.append(item)
                                    item_ids.append(item_id)
                                    domain_items_count += 1
                                    
                                    # 為 item 添加路徑信息
                                    item_with_path = item.copy()
                                    item_with_path["path"] = {
                                        "domain": d_name,
                                        "subdomain": subdomain_name,
                                        "section_type": sec_type,
                                        "section_name": section_name,
                                        "section_id": section_id,
                                        "page_start": section.get("page_start"),
                                        "page_end": section.get("page_end")
                                    }
                                    items_with_path.append(item_with_path)
                                    
                                    # 處理 subitems（也計入配額）
                                    if "subitems" in item:
                                        for subitem in item["subitems"]:
                                            if domain_items_count >= alloc_quota:
                                                break
                                            subitem_id = subitem.get("id")
                                            item_ids.append(subitem_id)
                                            domain_items_count += 1
                                            
                                            subitem_with_path = subitem.copy()
                                            subitem_with_path["path"] = {
                                                "domain": d_name,
                                                "subdomain": subdomain_name,
                                                "section_type": sec_type,
                                                "section_name": section_name,
                                                "section_id": section_id,
                                                "parent_item_id": item_id,
                                                "page_start": section.get("page_start"),
                                                "page_end": section.get("page_end")
                                            }
                                            items_with_path.append(subitem_with_path)
                                
                                if domain_items_count >= alloc_quota:
                                    break
                            if domain_items_count >= alloc_quota:
                                break
                        if domain_items_count >= alloc_quota:
                            break
                    if domain_items_count >= alloc_quota:
                        break
            
            # 新策略：對收集到的 items_with_path 進行 rerank
            candidates = self._rerank_candidates(
                [], items_with_path, user_query, assistant_anchor, plan  # sections 傳空列表
            )
            
            # Diversification 保底：確保每個 domain 至少 1 筆（如果候選足夠）
            if plan.domains:
                final_candidates = []
                used_domains = set()
                # 先加入每個 domain 至少 1 筆
                for domain in plan.domains:
                    for candidate in candidates:
                        if candidate.get("path", {}).get("subdomain") == domain or candidate.get("path", {}).get("domain") == domain:
                            if domain not in used_domains:
                                final_candidates.append(candidate)
                                used_domains.add(domain)
                                break
                # 再加入剩餘候選
                for candidate in candidates:
                    if candidate not in final_candidates:
                        final_candidates.append(candidate)
                candidates = final_candidates[:plan.k_items]
            else:
                candidates = candidates[:plan.k_items]
        else:
            # 舊策略：一次性查詢所有 domains
            domain_data = self.graph_client.get_domain_sections(
                doc_id,
                plan.domains if plan.domains else [],
                plan.use_sections,
                k_per_subdomain=plan.k_per_subdomain,
                allow_parent_domain_expand=plan.allow_parent_domain_expand
            )
            
            # 檢查是否有找到任何資料
            total_sections_found = sum(
                len(sec_list) 
                for domain_dict in domain_data.values() 
                for subdomain_dict in domain_dict.values() 
                for sec_list in subdomain_dict.values()
            )
            
            # 如果查不到結果，嘗試 fallback（僅對舊策略）
            if total_sections_found == 0 and plan.domain_policy in [DomainPolicy.LOCK_TOP1, DomainPolicy.TOP2_UNION, DomainPolicy.UNLOCKED]:
                import warnings
                warnings.warn(f"檢索結果為空，使用 fallback 查詢所有 domains（domain_policy={plan.domain_policy.value}）")
                all_domains = self.graph_client.list_domains(doc_id)
                if all_domains:
                    fallback_domains = [d["name"] for d in all_domains]
                    domain_data = self.graph_client.get_domain_sections(
                        doc_id,
                        fallback_domains,
                        plan.use_sections,
                        k_per_subdomain=plan.k_per_subdomain,
                        allow_parent_domain_expand=True  # fallback 時允許展開
                    )
        
            # 如果不是按配額查詢，使用舊邏輯構建 items
            for domain_name, subdomains in domain_data.items():
                for subdomain_name, sec_dict in subdomains.items():
                    for sec_type, sec_list in sec_dict.items():
                        for section in sec_list:
                            section_id = section.get("id")
                            section_name = section.get("name", "") or self._get_section_type_name(sec_type)
                            sections.append(section)  # 保留 section 用於追蹤
                            section_ids.append(section_id)
                            
                            # 取得該 section 的 items（可包含子項）
                            section_items = self.graph_client.fetch_items(
                                section_id,
                                sec_type,
                                limit=plan.k_items,
                                include_subitems=True
                            )
                            
                            for item in section_items:
                                item_id = item.get("id")
                                items.append(item)
                                item_ids.append(item_id)
                                
                                # 為 item 添加路徑信息
                                item_with_path = item.copy()
                                item_with_path["path"] = {
                                    "domain": domain_name,
                                    "subdomain": subdomain_name,
                                    "section_type": sec_type,
                                    "section_name": section_name,
                                    "section_id": section_id,
                                    "page_start": section.get("page_start"),
                                    "page_end": section.get("page_end")
                                }
                                items_with_path.append(item_with_path)
                                
                                # 處理 subitems，也添加路徑信息
                                if "subitems" in item:
                                    for subitem in item["subitems"]:
                                        subitem_id = subitem.get("id")
                                        item_ids.append(subitem_id)
                                        
                                        # 為 subitem 添加路徑信息
                                        subitem_with_path = subitem.copy()
                                        subitem_with_path["path"] = {
                                            "domain": domain_name,
                                            "subdomain": subdomain_name,
                                            "section_type": sec_type,
                                            "section_name": section_name,
                                            "section_id": section_id,
                                            "parent_item_id": item_id,
                                            "page_start": section.get("page_start"),
                                            "page_end": section.get("page_end")
                                        }
                                        items_with_path.append(subitem_with_path)
        
            # 舊策略：只對 items 和 subitems 進行 rerank（不包含 sections）
            candidates = self._rerank_candidates(
                [], items_with_path, user_query, assistant_anchor, plan  # sections 傳空列表
            )
            candidates = candidates[:plan.k_items]
        
        return {
            "sections": sections,
            "items": items,
            "section_ids": section_ids,
            "item_ids": item_ids,
            "candidates": candidates
        }
    
    def _rerank_candidates(
        self,
        sections: List[Dict],
        items: List[Dict],
        user_query: str,
        assistant_anchor: str,
        plan: RetrievalPlan
    ) -> List[Dict]:
        """
        重新排序候選內容（只包含 items 和 subitems，不包含 sections）
        
        Args:
            sections: Section 節點列表（已廢棄，不再使用）
            items: Item/Subitem 節點列表（包含路徑信息）
            user_query: 使用者查詢
            assistant_anchor: 助理錨點
            plan: 檢索計劃
        
        Returns:
            List[Dict]: 排序後的候選內容（只包含 items 和 subitems）
        """
        candidates = []
        
        # 輔助函數：bigram Jaccard 相似度（回退方案）
        def _bigram_similarity(text1: str, text2: str) -> float:
            """計算兩個文本的 bigram Jaccard 相似度（支持中文）"""
            def get_bigrams(text: str) -> set:
                """提取文本的 bigram（支持中文和英文）"""
                bigrams = set()
                # 處理中文字元（每個字元作為一個 token，生成相鄰字元對）
                for i in range(len(text) - 1):
                    bigrams.add(text[i:i+2])
                # 處理英文單詞（按空格分割，每個單詞生成 bigram）
                words = text.split()
                for word in words:
                    if len(word) > 1:
                        for i in range(len(word) - 1):
                            bigrams.add(word[i:i+2].lower())
                return bigrams
            
            bigrams1 = get_bigrams(text1)
            bigrams2 = get_bigrams(text2)
            
            if not bigrams1 or not bigrams2:
                return 0.0
            
            intersection = bigrams1 & bigrams2
            union = bigrams1 | bigrams2
            return len(intersection) / len(union) if union else 0.0
        
        # 相似度計算：優先使用 bge-m3，否則回退到 bigram Jaccard
        def compute_similarity(text1: str, text2: str) -> float:
            """計算兩個文本的相似度（使用 bge-m3 或 bigram Jaccard）"""
            if not text1 or not text2:
                return 0.0
            
            # 優先使用 bge-m3 embedding 相似度
            if self.bge_model is not None:
                try:
                    # 使用 bge-m3 編碼文本（SentenceTransformer 的 encode 方法）
                    # SentenceTransformer.encode() 接受 normalize_embeddings 參數，不接受 batch_size
                    embeddings = self.bge_model.encode([text1, text2], normalize_embeddings=True)
                    
                    # bge-m3 返回字典，包含 'dense_vecs'（密集向量）
                    if isinstance(embeddings, dict) and 'dense_vecs' in embeddings:
                        vecs = embeddings['dense_vecs']
                    elif isinstance(embeddings, np.ndarray):
                        vecs = embeddings
                    else:
                        # 如果格式不符合預期，回退到 bigram
                        return _bigram_similarity(text1, text2)
                    
                    # 確保是 numpy array
                    vecs = np.asarray(vecs, dtype=np.float32)
                    if vecs.shape[0] < 2:
                        return _bigram_similarity(text1, text2)
                    
                    # L2 正規化
                    vec1 = vecs[0]
                    vec2 = vecs[1]
                    norm1 = np.linalg.norm(vec1)
                    norm2 = np.linalg.norm(vec2)
                    if norm1 > 0:
                        vec1 = vec1 / norm1
                    if norm2 > 0:
                        vec2 = vec2 / norm2
                    
                    # 計算 cosine similarity
                    similarity = float(np.dot(vec1, vec2))
                    return max(0.0, min(1.0, similarity))  # 限制在 [0, 1]
                    
                except Exception as e:
                    # 如果 bge-m3 計算失敗，回退到 bigram
                    print(f"[RetrievalExecutor] BGE-M3 相似度計算失敗，回退到 bigram: {e}")
                    return _bigram_similarity(text1, text2)
            else:
                # 沒有 bge 模型，使用 bigram Jaccard
                return _bigram_similarity(text1, text2)
        
        # 只處理 items 和 subitems（不處理 sections）
        for item in items:
            # 判斷是 item 還是 subitem（根據是否有 parent_item_id）
            path = item.get("path", {})
            is_subitem = "parent_item_id" in path
            
            if is_subitem:
                # 這是 subitem
                text = item.get("text", "")
                if not text:  # 跳過空文本的 subitem
                    continue
                
                sim_q = compute_similarity(user_query, text)
                sim_anchor = compute_similarity(assistant_anchor, text) if assistant_anchor else 0.0
                graph_prox = 0.1  # subitem 稍微降低圖距離分數
                
                # Topic match bonus（泛用）：candidate.topic 在 plan.topics 中時加分
                boost_topic_match = 0.0
                candidate_topic = path.get("topic") or path.get("subdomain")
                plan_topics = getattr(plan, 'topics', None) or plan.domains
                if plan_topics and candidate_topic in plan_topics:
                    boost_topic_match = 0.05  # 比 domain boost 稍高，因為是明確匹配
                
                # Short text penalty（泛用）：太短的文本懲罰
                short_text_penalty = 0.0
                text_length = len(text) if text else 0
                if text_length > 0 and text_length < self._get_short_text_threshold(plan):
                    short_text_penalty = -0.02  # 懲罰太短的文本
                
                # Task-aware section bonus（泛用）：根據 section_type_weights 加分
                boost_section_task = 0.0
                section_type = path.get("section_type", "")
                section_weights = getattr(plan, 'section_type_weights', {})
                if section_weights and section_type in section_weights:
                    # 權重越高，bonus 越大（歸一化到 0-0.03）
                    weight = section_weights[section_type]
                    max_weight = max(section_weights.values()) if section_weights else 1.0
                    if max_weight > 0:
                        boost_section_task = 0.03 * (weight / max_weight)
                
                score = (
                    plan.rerank_weights["sim_q"] * sim_q +
                    plan.rerank_weights["sim_anchor"] * sim_anchor +
                    plan.rerank_weights["graph_prox"] * graph_prox +
                    boost_topic_match +
                    short_text_penalty +
                    boost_section_task
                )
                
                # 構建完整路徑描述
                path_description = (
                    f"{path.get('subdomain', '')}(Subdomain)的"
                    f"{path.get('section_name', '')}(Section)為"
                    f"{text}(Subitem)"
                )
                
                candidates.append({
                    "type": "subitem",
                    "item_type": self._get_item_type(item),
                    "text": text,  # 主要文本內容（來自 subitem）
                    "level": item.get("level", 2),
                    "item_id": item.get("id"),
                    "parent_item_id": path.get("parent_item_id"),
                    "path": path,  # 完整路徑信息
                    "path_description": path_description,  # 路徑描述（用於顯示）
                    "score": score,
                    "sim_q": sim_q,
                    "sim_anchor": sim_anchor
                })
            else:
                # 這是 item（level=1）
                text = item.get("text", "")
                if not text:  # 跳過空文本的 item
                    continue
                
                sim_q = compute_similarity(user_query, text)
                sim_anchor = compute_similarity(assistant_anchor, text) if assistant_anchor else 0.0
                graph_prox = 0.0
                
                # Topic match bonus（泛用）：candidate.topic 在 plan.topics 中時加分
                boost_topic_match = 0.0
                candidate_topic = path.get("topic") or path.get("subdomain")
                plan_topics = getattr(plan, 'topics', None) or plan.domains
                if plan_topics and candidate_topic in plan_topics:
                    boost_topic_match = 0.05
                
                # Short text penalty（泛用）：太短的文本懲罰
                short_text_penalty = 0.0
                text_length = len(text) if text else 0
                if text_length > 0 and text_length < self._get_short_text_threshold(plan):
                    short_text_penalty = -0.02
                
                # Task-aware section bonus（泛用）：根據 section_type_weights 加分
                boost_section_task = 0.0
                section_type = path.get("section_type", "")
                section_weights = getattr(plan, 'section_type_weights', {})
                if section_weights and section_type in section_weights:
                    weight = section_weights[section_type]
                    max_weight = max(section_weights.values()) if section_weights else 1.0
                    if max_weight > 0:
                        boost_section_task = 0.03 * (weight / max_weight)
                
                score = (
                    plan.rerank_weights["sim_q"] * sim_q +
                    plan.rerank_weights["sim_anchor"] * sim_anchor +
                    plan.rerank_weights["graph_prox"] * graph_prox +
                    boost_topic_match +
                    short_text_penalty +
                    boost_section_task
                )
                
                # 構建完整路徑描述
                path_description = (
                    f"{path.get('subdomain', '')}(Subdomain)的"
                    f"{path.get('section_name', '')}(Section)為"
                    f"{text}(Item)"
                )
                
                candidates.append({
                    "type": "item",
                    "item_type": self._get_item_type(item),
                    "text": text,  # 主要文本內容（來自 item）
                    "level": item.get("level", 1),
                    "item_id": item.get("id"),
                    "path": path,  # 完整路徑信息
                    "path_description": path_description,  # 路徑描述（用於顯示）
                    "score": score,
                    "sim_q": sim_q,
                    "sim_anchor": sim_anchor
                })
        
        # 按分數排序
        candidates.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        
        return candidates
    
    def _get_section_type(self, section: Dict) -> str:
        """從 section 節點判斷類型"""
        # 可以從 section 的 keys 或 labels 判斷
        # 簡化版：假設有 type 欄位或從 id 判斷
        section_id = section.get("id", "")
        if ":assessment" in section_id:
            return "assessment"
        elif ":observation" in section_id:
            return "observation"
        elif ":training" in section_id:
            return "training"
        elif ":suggestion" in section_id:
            return "suggestion"
        return "unknown"
    
    def _get_item_type(self, item: Dict) -> str:
        """從 item 節點判斷類型"""
        # 優先從 path 中獲取
        path = item.get("path", {})
        if path.get("section_type"):
            return path["section_type"]
        
        # 否則從 item_id 判斷
        item_id = item.get("id", "")
        if "assessment" in item_id:
            return "assessment"
        elif "observation" in item_id:
            return "observation"
        elif "training" in item_id:
            return "training"
        elif "suggestion" in item_id:
            return "suggestion"
        return "unknown"
    
    def _get_section_type_name(self, section_type: str) -> str:
        """獲取 section 類型的中文名稱"""
        type_names = {
            "assessment": "評估結果",
            "observation": "行為觀察",
            "training": "訓練方向",
            "suggestion": "具體建議"
        }
        return type_names.get(section_type, section_type)
    
    def _get_short_text_threshold(self, plan: RetrievalPlan) -> int:
        """
        獲取短文本懲罰閾值（從 ontology 或使用預設值）
        
        Args:
            plan: 檢索計劃
            
        Returns:
            短文本閾值（字符數）
        """
        # 嘗試從 plan 中獲取 ontology 的 SHORT_TEXT_PENALTY_TH
        # 如果沒有，使用預設值 10
        from .topic_ontology import default_ontology
        return getattr(default_ontology, 'SHORT_TEXT_PENALTY_TH', 10)