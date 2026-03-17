# task_scope_classifier.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional
import os
import numpy as np
import json

# 預設設定檔路徑（與本模組同目錄下的 config/task_scope_prototypes.jsonl）
_CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config")
DEFAULT_PROTOTYPES_JSONL_PATH = os.path.join(_CONFIG_DIR, "task_scope_prototypes.jsonl")

def load_prototypes_from_jsonl(jsonl_path: Optional[str] = None) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Load task and scope prototypes from JSONL file.
    """
    task_prototypes: Dict[str, List[str]] = {}
    scope_prototypes: Dict[str, List[str]] = {}
    path = jsonl_path or DEFAULT_PROTOTYPES_JSONL_PATH
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                label = obj['label']
                examples = obj['examples']
                if obj['type'] == 'task':
                    task_prototypes[label] = examples
                elif obj['type'] == 'scope':
                    scope_prototypes[label] = examples
                    
        if not task_prototypes:
            raise ValueError("No task prototypes found in JSONL")
            
    except Exception as e:
        print(f"[TaskScopeClassifier] Error loading {path}: {e}")
        # 極簡備用方案，確保系統不崩潰
        task_prototypes = {"A": ["重點整理"]}
        scope_prototypes = {"S_overview": ["整體狀況"]}
    
    return task_prototypes, scope_prototypes

TASK_NAME_ZH = {
    "A": "報告總覽與閱讀順序",
    "B": "分數/量表/百分位解讀",
    "C": "臨床觀察與表現解讀",
    "D": "能力剖面（優勢/需求/優先順序）",
    "E": "在家訓練怎麼做",
    "F": "融入日常作息的練習",
    "G": "是否需要早療/成效追蹤",
    "H": "轉介與在地資源",
    "I": "報告分享/隱私與安全",
    "J": "與學校合作",
    "K": "補助/福利/申請",
    "L": "後續追蹤/再評估",
    "M": "家長情緒支持與家庭協作",
}

SCOPE_NAME_ZH = {
    "S_overview": "Overview(整體)",
    "S_domain": "Domain(單領域)",
    "S_multi_domain": "Multi-Domain(多領域)",
}

def _l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    denom = np.linalg.norm(x, axis=axis, keepdims=True)
    denom = np.maximum(denom, eps)
    return x / denom

def _embed_texts(embedder: Any, texts: List[str]) -> np.ndarray:
    """
    Try common embedding APIs.
    """
    if hasattr(embedder, "encode_many"):
        vec = embedder.encode_many(texts)
    elif hasattr(embedder, "encode"):
        try:
            vec = embedder.encode(texts)
        except (AttributeError, TypeError):
            if hasattr(embedder, "encode_many"):
                vec = embedder.encode_many(texts)
            else:
                raise TypeError(f"embedder.encode() does not accept list, and encode_many() not available")
    elif hasattr(embedder, "embed"):
        vec = embedder.embed(texts)
    elif callable(embedder):
        vec = embedder(texts)
    else:
        raise TypeError("Unsupported embedder: expected .encode_many/.encode/.embed or callable.")
    vec = np.asarray(vec, dtype=np.float32)
    if vec.ndim == 1:
        vec = vec[None, :]
    return vec

@dataclass
class PredictResult:
    label: str
    score: float
    dist: Dict[str, float]

class PrototypeClassifier:
    def __init__(self, embedder: Any, prototypes: Dict[str, List[str]]):
        self.embedder = embedder
        self.prototypes = prototypes
        self.labels = list(prototypes.keys())
        self.proto_texts = [prototypes[k] for k in self.labels]

        # Build prototype vectors (mean of sentence embeddings)
        proto_vecs = []
        for sents in self.proto_texts:
            embs = _embed_texts(self.embedder, sents)
            embs = _l2_normalize(embs)
            proto = embs.mean(axis=0, keepdims=True)
            proto = _l2_normalize(proto)[0]
            proto_vecs.append(proto)
        self.proto_mat = np.stack(proto_vecs, axis=0).astype(np.float32)  # [K, D]

    def predict(self, text: str) -> PredictResult:
        q = _embed_texts(self.embedder, [text])
        q = _l2_normalize(q)[0]  # [D]
        sims = self.proto_mat @ q  # cosine since normalized
        best_idx = int(np.argmax(sims))
        best_label = self.labels[best_idx]
        best_score = float(sims[best_idx])

        temp = 12.0
        logits = sims * temp
        logits = logits - np.max(logits)
        probs = np.exp(logits)
        probs = probs / np.sum(probs)
        dist = {self.labels[i]: float(probs[i]) for i in range(len(self.labels))}

        return PredictResult(label=best_label, score=best_score, dist=dist)

class TaskScopeClassifier:
    def __init__(
        self,
        embedder: Any,
        task_prototypes: Dict[str, List[str]] = None,
        prototypes_jsonl: Optional[str] = None,
    ):
        if task_prototypes is None:
            path = prototypes_jsonl or DEFAULT_PROTOTYPES_JSONL_PATH
            loaded_task, _ = load_prototypes_from_jsonl(path)
            task_prototypes = loaded_task
        
        self.task_clf = PrototypeClassifier(embedder, task_prototypes)

    def predict_task(self, text: str) -> PredictResult:
        return self.task_clf.predict(text)

def format_topk(dist: Dict[str, float], name_map: Dict[str, str], k: int = 2) -> str:
    items = sorted(dist.items(), key=lambda x: x[1], reverse=True)[:k]
    return ", ".join([f"{name_map.get(lbl, lbl)}={p:.2f}" for lbl, p in items])
