# 將文字 encode 成向量模組

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na <= 0 or nb <= 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

@dataclass
class EncoderConfig:
    """
    Embedding Server 設定：
    - url: 伺服器端點
    """
    url: str = ""

    def __post_init__(self):
        # 從主設定檔載入
        try:
            from config import EMBED_CONFIG
            self.url = self.url or EMBED_CONFIG.get('url', "http://192.168.150.136:8080/embed")
        except ImportError:
            self.url = self.url or "http://192.168.150.136:8080/embed"

class TextEncoder:
    """
    TextEncoder 的責任只有一個：
    - 透過遠端 API 把文字 encode 成 embedding 向量（np.ndarray）
    """

    def __init__(self, cfg: Optional[EncoderConfig] = None):
        self.cfg = cfg or EncoderConfig()
        import requests
        self._requests = requests

    def encode(self, text: str) -> np.ndarray:
        text = (text or "").strip()
        if not text:
            # 空字串回傳零向量
            return np.zeros((1024,), dtype=np.float32)

        try:
            res = self._requests.post(self.cfg.url, json={"inputs": text})
            res.raise_for_status()
            data = res.json()
            # 支援不同伺服器回傳格式
            if isinstance(data, dict) and "embedding" in data:
                emb = data["embedding"]
            elif isinstance(data, list):
                emb = data[0]
            else:
                emb = data
            
            return np.asarray(emb, dtype=np.float32)
        except Exception as e:
            print(f"[TextEncoder] Encode error: {e}")
            return np.zeros((1024,), dtype=np.float32)

    def encode_many(self, texts: List[str]) -> np.ndarray:
        results = []
        for t in texts:
            results.append(self.encode(t))
        if not results:
            return np.empty((0, 1024), dtype=np.float32)
        return np.stack(results, axis=0)

def encode_anchors(
    encoder: TextEncoder,
    anchors: Dict[str, List[str]],
    domains: List[str],
) -> Dict[str, List[np.ndarray]]:
    """
    將每個 domain 的多個 anchor 句子 encode 成向量列表並回傳 dict。
    建議在系統啟動時做一次並快取。
    
    Args:
        encoder: TextEncoder 實例
        anchors: 每個領域對應的多個句子列表 {domain: [sentence1, sentence2, ...]}
        domains: 領域列表
    
    Returns:
        每個領域對應的多個向量列表 {domain: [vector1, vector2, ...]}
    """
    # 展平所有句子，同時記錄每個領域的句子索引
    all_texts = []
    domain_indices = {}  # {domain: (start_idx, end_idx)}
    
    for d in domains:
        sentences = anchors.get(d, [])
        if not sentences:
            raise ValueError(f"Domain {d} has no anchor sentences")
        
        start_idx = len(all_texts)
        all_texts.extend(sentences)
        end_idx = len(all_texts)
        domain_indices[d] = (start_idx, end_idx)
    
    # 批量編碼所有句子
    mat = encoder.encode_many(all_texts)
    
    # 按領域分組
    result = {}
    for d in domains:
        start_idx, end_idx = domain_indices[d]
        result[d] = [mat[i] for i in range(start_idx, end_idx)]
    
    return result


def encode_overview_anchors(
    encoder: TextEncoder,
    overview_sentences: List[str],
) -> List[np.ndarray]:
    """
    將「整體」錨點句子 encode 成向量列表。用於與使用者 query 做相似度比對（取代關鍵字判斷）。
    建議在系統啟動時做一次並快取。
    """
    if not overview_sentences:
        return []
    texts = [(t or "").strip() for t in overview_sentences if (t or "").strip()]
    if not texts:
        return []
    mat = encoder.encode_many(texts)
    return [mat[i] for i in range(len(mat))]


def score_overview_similarity(
    query_vec: np.ndarray,
    overview_anchor_vecs: List[np.ndarray],
) -> float:
    """
    計算 query 向量與整體錨點向量的相似度，取最大值（Max Pooling）。
    若 overview_anchor_vecs 為空，回傳 0.0。
    """
    if not overview_anchor_vecs:
        return 0.0
    return float(max(cosine_sim(query_vec, v) for v in overview_anchor_vecs))