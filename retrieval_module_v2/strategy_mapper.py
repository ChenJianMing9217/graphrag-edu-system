from typing import Dict, Any, List, Optional
from .types import SearchStrategy, SearchOperation, SearchOperationType

class StrategyMapper:
    """
    Translates DST FlowResult signals into a SearchStrategy.
    """
    def __init__(self, ontology=None):
        from .topic_ontology import default_ontology
        self.ontology = ontology or default_ontology

    def map_dst_to_strategy(self, turn_state: Dict[str, Any], user_query: str = "") -> SearchStrategy:
        strategy = SearchStrategy()
        reasons = []
        
        retrieval_action = turn_state.get("retrieval_action", "")
        scope_label = turn_state.get("scope_pred", "")
        domain_distribution = turn_state.get("domain_distribution", {})
        task_label = turn_state.get("task_pred", "")
        
        reasons.append(f"DST Action: {retrieval_action}")
        reasons.append(f"Scope: {scope_label}")
        
        # 1. Summary Fetch (If Meta/Summary query)
        if scope_label == "S_overview" or task_label in ["T_overview", "T_status_query"]:
            strategy.operations.append(SearchOperation(
                op_type=SearchOperationType.SUMMARY_FETCH,
                params={"query_type": "summary", "limit": 10}
            ))
            reasons.append("Added SUMMARY_FETCH due to overview scope/task")

        # 2. Meta Fetch (If name/age/date mentioned)
        meta_keywords = ["姓名", "年齡", "日期", "性別", "個案"]
        if any(kw in user_query for kw in meta_keywords) or task_label == "T_meta_query":
            strategy.operations.append(SearchOperation(
                op_type=SearchOperationType.META_FETCH,
                params={}
            ))
            reasons.append("Added META_FETCH due to meta-related keywords or task")

        # 3. Subdomain Fetch (Based on domain distribution and task weights)
        active_domains = turn_state.get("active_domains", [])
        
        # Determine sections to fetch based on task
        section_weights = self.ontology.get_section_weights(task_label)
        use_sections = [sec for sec, weight in section_weights.items() if weight > 0]
        if not use_sections:
            use_sections = ["assessment", "observation", "training", "suggestion"]

        if active_domains:
            for domain in active_domains:
                prob = domain_distribution.get(domain, 0.0)
                # For specific subdomain fetch, we use the prob as weight
                strategy.operations.append(SearchOperation(
                    op_type=SearchOperationType.SUBDOMAIN_FETCH,
                    params={"subdomain": domain, "sections": use_sections},
                    weight=prob if prob > 0 else 1.0
                ))
            reasons.append(f"Added SUBDOMAIN_FETCH for domains: {active_domains} with sections: {use_sections}")

        # 4. MySQL Local Resource Fetch
        if retrieval_action == "LOCAL_RESOURCE_SEARCH":
            region = turn_state.get("detected_region")
            if region:
                # 提取關鍵字（簡單處理：排除地區後的詞，或是直接把 query 傳進去讓 DB filter）
                # 這裡我們先從 query 中提取可能感興趣的詞
                keywords = None
                for kw in ["物理治療", "語言治療", "職能治療", "心理治療", "療育", "評估"]:
                    if kw in user_query:
                        keywords = kw
                        break
                
                strategy.operations.append(SearchOperation(
                    op_type=SearchOperationType.MYSQL_RESOURCE_FETCH,
                    params={"region": region, "keywords": keywords}
                ))
                reasons.append(f"Added MYSQL_RESOURCE_FETCH for region: {region}, keywords: {keywords}")
        
        # 5. Contextual Expansion
        if turn_state.get("semantic_flow") == "continue":
            strategy.operations.append(SearchOperation(
                op_type=SearchOperationType.CONTEXTUAL_EXPANSION,
                params={"hops": 1}
            ))
            reasons.append("Added CONTEXTUAL_EXPANSION due to semantic flow 'continue'")

        # 4. Rerank Config
        strategy.rerank_config = {
            "semantic_weight": 0.6,
            "structural_weight": 0.2,
            "context_weight": 0.2
        }
        
        strategy.reasons = reasons
        return strategy
