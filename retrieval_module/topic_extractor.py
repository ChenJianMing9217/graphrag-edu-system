#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
topic_extractor.py
從使用者文字中提取明確的主題（explicit_topics）
"""

from typing import List, Set, Optional
from .topic_ontology import TopicOntology, default_ontology
import re


def extract_explicit_topics(
    user_text: str,
    ontology: TopicOntology = default_ontology
) -> List[str]:
    """
    從使用者文字中提取明確提到的主題（explicit_topics）
    
    Args:
        user_text: 使用者輸入文字
        ontology: 主題本體配置
        
    Returns:
        明確提到的主題列表（canonical labels），按出現順序
    """
    if not user_text:
        return []
    
    # 提取的主題（去重，保持順序）
    found_topics: List[str] = []
    seen: Set[str] = set()
    
    # 策略 1: 直接匹配 canonical labels（中文不使用 \b）
    # 按長度排序，先匹配長的主題（避免「動作」匹配到「粗大動作」）
    sorted_topics = sorted(ontology.TOPIC_LABELS, key=len, reverse=True)
    for topic in sorted_topics:
        # 直接檢查是否包含完整詞彙
        if topic in user_text:
            # 對於中文，簡單檢查是否包含完整詞彙即可
            # 按出現順序添加到結果中（保持順序）
            if topic not in seen:
                found_topics.append(topic)
                seen.add(topic)
    
    # 策略 2: 匹配 aliases（同樣處理，按長度排序）
    sorted_aliases = sorted(ontology.TOPIC_ALIASES.items(), key=lambda x: len(x[0]), reverse=True)
    for alias, canonical in sorted_aliases:
        if alias in user_text and canonical not in seen:
            # 直接檢查是否包含 alias
            found_topics.append(canonical)
            seen.add(canonical)
    
    # 策略 3: 模糊匹配（如果前兩種都沒找到）
    if not found_topics:
        for topic in ontology.TOPIC_LABELS:
            if topic in user_text and topic not in seen:
                found_topics.append(topic)
                seen.add(topic)
    
    return found_topics


def normalize_topic_list(
    topics: List[str],
    ontology: TopicOntology = default_ontology
) -> List[str]:
    """
    標準化主題列表（將 aliases 轉換為 canonical labels）
    
    Args:
        topics: 原始主題列表
        ontology: 主題本體配置
        
    Returns:
        標準化後的主題列表（去重）
    """
    normalized = []
    seen = set()
    
    for topic in topics:
        canonical = ontology.normalize_topic(topic)
        if canonical and canonical not in seen:
            normalized.append(canonical)
            seen.add(canonical)
    
    return normalized
