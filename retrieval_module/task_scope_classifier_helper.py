#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
task_scope_classifier_helper.py
提供 intent 判斷輔助函數（不依賴 LLM）
"""

# 概覽意圖關鍵詞
OVERVIEW_KEYWORDS = [
    "整體", "概覽", "總結", "全部", "整份", "有哪些領域", "評估了哪些",
    "整體評估", "整體狀況", "整體表現", "整體發展", "整體來看"
]

# 列舉意圖關鍵詞
LIST_KEYWORDS = [
    "和", "以及", "與", "跟", "同時", "還有", "包含", "、", "/",
    "以及", "還有", "加上", "另外", "此外"
]

# 關係意圖關鍵詞（用於判斷是否需要多領域）
RELATION_KEYWORDS = [
    "影響", "導致", "因為", "所以", "關係", "會不會", "是否", "相關"
]

# 多狀態關鍵詞
MULTI_STATUS_KEYWORDS = [
    "狀況", "結果", "評估", "發展", "表現", "如何"
]


def is_overview_intent(user_text: str) -> bool:
    """
    判斷是否為概覽意圖
    
    Args:
        user_text: 使用者文字
    
    Returns:
        bool: 是否為概覽意圖
    """
    if not user_text:
        return False
    
    user_text_lower = user_text.lower()
    for keyword in OVERVIEW_KEYWORDS:
        if keyword in user_text:
            return True
    
    return False


def is_multi_list_intent(user_text: str) -> bool:
    """
    判斷是否為多領域列舉意圖
    
    Args:
        user_text: 使用者文字
    
    Returns:
        bool: 是否為多領域列舉意圖
    """
    if not user_text:
        return False
    
    # 只檢查列舉關鍵詞（「和」、「以及」、「與」等）
    # 不檢查多狀態關鍵詞，因為「如何」等詞彙在單領域查詢中也會出現
    has_list_keyword = any(keyword in user_text for keyword in LIST_KEYWORDS)
    
    return has_list_keyword
