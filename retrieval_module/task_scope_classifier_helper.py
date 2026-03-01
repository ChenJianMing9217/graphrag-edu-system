#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
task_scope_classifier_helper.py
提供 intent 判斷輔助函數（不依賴 LLM）
"""

import os
import json
from typing import Dict, List, Optional

# 預設關鍵詞設定檔路徑（retrieval_module/config/task_scope_keywords.json）
_CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config")
DEFAULT_KEYWORDS_JSON_PATH = os.path.join(_CONFIG_DIR, "task_scope_keywords.json")

_DEFAULT_KEYWORDS = {
    "overview_keywords": [
        "整體", "概覽", "總結", "全部", "整份", "有哪些領域", "評估了哪些",
        "整體評估", "整體狀況", "整體表現", "整體發展", "整體來看",
    ],
    "list_keywords": [
        "和", "以及", "與", "跟", "同時", "還有", "包含", "、", "/",
        "以及", "還有", "加上", "另外", "此外",
    ],
    "relation_keywords": [
        "影響", "導致", "因為", "所以", "關係", "會不會", "是否", "相關",
    ],
    "multi_status_keywords": [
        "狀況", "結果", "評估", "發展", "表現", "如何",
    ],
}


def load_keywords(keywords_path: Optional[str] = None) -> Dict[str, List[str]]:
    """
    從 JSON 載入關鍵詞表。若路徑為 None 則使用預設路徑；讀檔失敗則回傳程式內建預設。
    """
    path = keywords_path or DEFAULT_KEYWORDS_JSON_PATH
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        out = {}
        for key in ("overview_keywords", "list_keywords", "relation_keywords", "multi_status_keywords"):
            val = data.get(key)
            out[key] = list(val) if isinstance(val, list) else _DEFAULT_KEYWORDS[key]
        return out
    except (FileNotFoundError, json.JSONDecodeError, TypeError):
        return {k: list(v) for k, v in _DEFAULT_KEYWORDS.items()}


_loaded = load_keywords()
OVERVIEW_KEYWORDS: List[str] = _loaded["overview_keywords"]
LIST_KEYWORDS: List[str] = _loaded["list_keywords"]
RELATION_KEYWORDS: List[str] = _loaded["relation_keywords"]
MULTI_STATUS_KEYWORDS: List[str] = _loaded["multi_status_keywords"]


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
