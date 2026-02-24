#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
domain_mapper.py
Domain 名稱映射：將 dialogue_state_module 識別的 subdomain 名稱
映射到 Neo4j 圖中的 Domain 名稱
"""

# Domain 到 Subdomain 的映射（根據 load_grouped_tables_to_neo4j_minimal.py）
DOMAIN_TO_SUBDOMAIN = {
    "知覺動作功能": ["粗大動作", "精細動作", "感覺統合"],
    "吞嚥/口腔功能": ["口腔動作", "吞嚥反射"],
    "口語溝通功能": ["口語理解", "口語表達"],
    "認知功能": ["認知功能"],
    "社會情緒功能": ["情緒行為與社會適應功能"],
    "綜合評估": []  # 特殊情況
}

# 反向映射：Subdomain 到 Domain
SUBDOMAIN_TO_DOMAIN = {}
for domain, subdomains in DOMAIN_TO_SUBDOMAIN.items():
    for subdomain in subdomains:
        SUBDOMAIN_TO_DOMAIN[subdomain] = domain

# 對話狀態模組識別的 domain 名稱（實際是 subdomain）
DIALOGUE_STATE_DOMAINS = [
    "粗大動作", "精細動作", "感覺統合",
    "口腔動作", "吞嚥功能",
    "口語理解", "口語表達", "說話",
    "認知功能",
    "情緒行為與社會適應功能"
]


def map_subdomain_to_domain(subdomain_name: str) -> str:
    """
    將 subdomain 名稱映射到 domain 名稱
    
    Args:
        subdomain_name: Subdomain 名稱（如 "粗大動作"）
    
    Returns:
        str: Domain 名稱（如 "知覺動作功能"），如果找不到則返回原名稱
    """
    return SUBDOMAIN_TO_DOMAIN.get(subdomain_name, subdomain_name)


def map_domain_to_subdomains(domain_name: str) -> list[str]:
    """
    將 domain 名稱映射到 subdomain 名稱列表
    
    Args:
        domain_name: Domain 名稱（如 "知覺動作功能"）
    
    Returns:
        List[str]: Subdomain 名稱列表
    """
    return DOMAIN_TO_SUBDOMAIN.get(domain_name, [])


def is_subdomain_name(name: str) -> bool:
    """
    判斷名稱是否為 subdomain（而非 domain）
    
    Args:
        name: 要判斷的名稱
    
    Returns:
        bool: 如果是 subdomain 返回 True
    """
    return name in DIALOGUE_STATE_DOMAINS


def find_matching_domain(
    query_domain: str,
    available_domains: list[str]
) -> str:
    """
    在可用 domains 中尋找匹配的 domain
    
    策略：
    1. 如果 query_domain 是 subdomain，先映射到 domain
    2. 直接匹配
    3. 如果找不到，嘗試模糊匹配
    
    Args:
        query_domain: 查詢的 domain 名稱
        available_domains: 可用的 domain 名稱列表
    
    Returns:
        str: 匹配的 domain 名稱，如果找不到則返回 None
    """
    # 1. 如果 query_domain 是 subdomain，映射到 domain
    if is_subdomain_name(query_domain):
        mapped_domain = map_subdomain_to_domain(query_domain)
        if mapped_domain in available_domains:
            return mapped_domain
    
    # 2. 直接匹配
    if query_domain in available_domains:
        return query_domain
    
    # 3. 模糊匹配（部分匹配）
    for available_domain in available_domains:
        if query_domain in available_domain or available_domain in query_domain:
            return available_domain
    
    return None
