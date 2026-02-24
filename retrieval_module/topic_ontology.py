#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
topic_ontology.py
通用主題本體配置（domain-agnostic）
"""

from typing import Dict, List, Set, Optional
from dataclasses import dataclass, field


@dataclass
class TopicOntology:
    """
    主題本體配置（可替換，適用於不同領域）
    
    - topic: 可對應 subdomain、產品類別、章節、疾病科別、功能模組等
    - section_type: 可對應文件段落類型、節點類型、metadata 類型
    """
    
    # Canonical topic labels（標準主題列表）
    TOPIC_LABELS: List[str] = field(default_factory=lambda: [
        "粗大動作", "精細動作", "感覺統合", "口腔動作", "吞嚥功能",
        "口語理解", "口語表達", "說話", "認知功能", "情緒行為與社會適應功能"
    ])
    
    # Topic aliases（別名映射：alias -> canonical）
    TOPIC_ALIASES: Dict[str, str] = field(default_factory=lambda: {
        "粗大": "粗大動作",
        "大動作": "粗大動作",
        "精細": "精細動作",
        "小動作": "精細動作",
        "感覺": "感覺統合",
        "統合": "感覺統合",
        "口腔": "口腔動作",
        "吞嚥": "吞嚥功能",
        "理解": "口語理解",
        "表達": "口語表達",
        "語言": "說話",
        "認知": "認知功能",
        "情緒": "情緒行為與社會適應功能",
        "社會": "情緒行為與社會適應功能",
        "行為": "情緒行為與社會適應功能"
    })
    
    # Task 到 Section Type 的權重映射
    # task: 使用者意圖類別（coaching/how-to, status/summary, definition, comparison, overview 等）
    # section_type: 文件段落類型（assessment, observation, training, suggestion 等）
    TASK_TO_SECTION_WEIGHTS: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "T1_report_overview": {
            "assessment": 0.3,
            "observation": 0.3,
            "training": 0.2,
            "suggestion": 0.2
        },
        "T2_score_interpretation": {
            "assessment": 0.7,
            "observation": 0.2,
            "training": 0.05,
            "suggestion": 0.05
        },
        "T3_definition": {
            "assessment": 0.4,
            "observation": 0.3,
            "training": 0.15,
            "suggestion": 0.15
        },
        "T4_prioritization": {
            "assessment": 0.4,
            "observation": 0.2,
            "training": 0.2,
            "suggestion": 0.2
        },
        "T5_coaching": {
            "training": 0.5,
            "suggestion": 0.3,
            "assessment": 0.1,
            "observation": 0.1
        },
        "T6_comparison": {
            "assessment": 0.4,
            "observation": 0.3,
            "training": 0.15,
            "suggestion": 0.15
        },
        "T_meta": {
            "assessment": 0.25,
            "observation": 0.25,
            "training": 0.25,
            "suggestion": 0.25
        }
    })
    
    # Policy 超參數
    MAX_TOPICS: int = 4  # 最多同時檢索的主題數
    MIN_PER_TOPIC: int = 1  # 每個主題至少檢索的項目數
    MAIN_TOPIC_RATIO: float = 0.7  # 主要主題的配額比例（SOFT_FOCUS 時）
    
    # 不確定性閾值
    TV_TH: float = 0.15  # Topic variance threshold（主題變異閾值）
    MARGIN_TH: float = 0.1  # Margin threshold（邊際閾值）
    SHORT_TEXT_PENALTY_TH: int = 10  # 短文本懲罰閾值（字符數）
    
    def normalize_topic(self, topic_text: str) -> Optional[str]:
        """
        將主題文本標準化為 canonical label
        
        Args:
            topic_text: 原始主題文本
            
        Returns:
            Canonical topic label，如果找不到則返回 None
        """
        # 先檢查是否已經是 canonical
        if topic_text in self.TOPIC_LABELS:
            return topic_text
        
        # 檢查 alias
        if topic_text in self.TOPIC_ALIASES:
            return self.TOPIC_ALIASES[topic_text]
        
        # 模糊匹配（包含關係）
        topic_lower = topic_text.lower()
        for canonical in self.TOPIC_LABELS:
            if topic_lower in canonical.lower() or canonical.lower() in topic_lower:
                return canonical
        
        return None
    
    def get_section_weights(self, task: str) -> Dict[str, float]:
        """
        根據 task 獲取 section type 權重
        
        Args:
            task: 任務類型
            
        Returns:
            Section type 權重字典，如果找不到則返回均勻權重
        """
        if task in self.TASK_TO_SECTION_WEIGHTS:
            return self.TASK_TO_SECTION_WEIGHTS[task].copy()
        
        # 預設均勻權重
        default_sections = ["assessment", "observation", "training", "suggestion"]
        return {sec: 1.0 / len(default_sections) for sec in default_sections}


# 預設實例（可替換）
default_ontology = TopicOntology()
