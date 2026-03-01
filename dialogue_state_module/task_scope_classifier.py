# task_scope_classifier.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional
import os
import numpy as np
import json

# 預設範例檔路徑（與本模組同目錄下的 prototypes/task_scope_prototypes.jsonl）
_PROTOTYPES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prototypes")
DEFAULT_PROTOTYPES_JSONL_PATH = os.path.join(_PROTOTYPES_DIR, "task_scope_prototypes.jsonl")


def load_prototypes_from_jsonl(jsonl_path: Optional[str] = None) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Load task and scope prototypes from JSONL file.
    If jsonl_path is None or file doesn't exist, returns default prototypes.
    """
    task_prototypes: Dict[str, List[str]] = {}
    scope_prototypes: Dict[str, List[str]] = {}
    
    if jsonl_path is None:
        return DEFAULT_TASK_PROTOTYPES, DEFAULT_SCOPE_PROTOTYPES
    
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
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
        
        # If file was empty or had no valid entries, use defaults
        if not task_prototypes:
            task_prototypes = DEFAULT_TASK_PROTOTYPES
        if not scope_prototypes:
            scope_prototypes = DEFAULT_SCOPE_PROTOTYPES
            
    except FileNotFoundError:
        # File doesn't exist, use defaults
        task_prototypes = DEFAULT_TASK_PROTOTYPES
        scope_prototypes = DEFAULT_SCOPE_PROTOTYPES
    except Exception:
        # Any other error, use defaults
        task_prototypes = DEFAULT_TASK_PROTOTYPES
        scope_prototypes = DEFAULT_SCOPE_PROTOTYPES
    
    return task_prototypes, scope_prototypes


# Task A–M：依使用資料優先順序與使用者目標定義（方案1）
# 範例句涵蓋多領域，不偏單一領域（粗大動作／精細動作／感覺統合／口語／認知／情緒等）
DEFAULT_TASK_PROTOTYPES = {
    "A": [  # 報告總覽與閱讀順序
        "這份報告主要在評估什麼？可以幫我抓重點嗎？",
        "我應該先看報告的哪一段？",
        "請用三句話說一下目前狀況與下一步建議。",
        "整體來看孩子的結論是什麼？",
        "這份聯評報告各領域的重點與建議可以幫我整理嗎？",
        "精細動作／口語理解／認知這幾塊我該先看哪一部分？",
        "評估結果與訓練方向的重點可以摘要嗎？",
    ],
    "B": [  # 分數/量表/百分位解讀
        "標準分、百分位要怎麼看？代表孩子落在哪裡？",
        "報告寫落後同齡，到底差多少？",
        "分數的意義是什麼？怎麼解讀？",
        "口語理解／精細動作的分數要怎麼看？",
        "發展商數、T分數各代表什麼？",
        "感覺統合／認知功能的分數落點怎麼理解？",
        "同一個領域不同分項分數差很多，代表什麼？",
    ],
    "C": [  # 臨床觀察與表現解讀
        "報告說核心肌力不足，日常會有哪些表現？",
        "平衡較弱是指站不穩嗎？有哪些常見情境？",
        "報告提到觸覺敏感，生活中會怎樣表現？",
        "口語理解落後，日常會有哪些具體狀況？",
        "精細動作協調不佳，怎麼觀察？和動作笨拙一樣嗎？",
        "認知評估說工作記憶弱，平常會怎麼顯現？",
        "情緒行為那欄寫的調節困難，實際上是什麼樣子？",
    ],
    "D": [  # 能力剖面（優勢/需求/優先順序）
        "孩子在各領域的優勢是什麼？我該怎麼善用？",
        "如果只能先挑兩個重點練，建議先練哪兩個？",
        "這些問題會影響上學或日常哪些活動？",
        "口語理解與表達哪一個要先加強？",
        "精細動作和感覺統合，優先順序怎麼排？",
        "認知與情緒方面，目前最需要介入的是什麼？",
        "粗大動作／吞嚥／口腔動作的優先順序怎麼看？",
    ],
    "E": [  # 在家訓練怎麼做
        "在家可以怎麼練？一天大概需要多久？",
        "可以給我幾個居家活動建議嗎？要注意哪些安全？",
        "單腳站不穩／手部操作不好，怎麼循序漸進練？",
        "口語理解或表達在家可以怎麼引導？",
        "感覺統合／觸覺相關的居家活動有哪些？",
        "精細動作或認知類的在家練習怎麼做？",
        "吞嚥或口腔動作在家要注意什麼、怎麼練？",
    ],
    "F": [  # 融入日常作息的練習
        "怎麼把練習融入生活、不增加負擔？",
        "早上很趕，怎麼塞進幾分鐘練習？",
        "日常作息裡可以怎麼順便練？",
        "吃飯、洗澡、出門前可以搭配什麼練習？",
        "玩遊戲或親子時間怎麼自然帶入訓練？",
        "不同領域的練習可以一起融入同一個時段嗎？",
    ],
    "G": [  # 是否需要介入/早療/成效追蹤
        "孩子需要做治療嗎？什麼情況建議開始？",
        "如果要做，一週幾次比較常見？只在家練可以嗎？",
        "怎麼判斷有效？要看哪些生活指標？",
        "口語／認知／情緒方面需要早療嗎？",
        "精細動作或感覺統合什麼時候要尋求專業？",
        "建議多久追蹤一次比較合理？",
    ],
    "H": [  # 轉介與在地資源
        "附近哪裡可以做兒童復健或早療？",
        "建議轉介到哪一類機構？下一步去哪裡？",
        "語言治療／職能治療／物理治療要去哪裡找？",
        "認知或情緒相關的資源哪裡有？",
        "有推薦的早療中心或診所嗎？",
    ],
    "I": [  # 報告分享給誰/隱私與安全
        "我可以把報告給老師看嗎？要給整份嗎？",
        "怎麼分享才不會洩漏太多隱私？",
        "給幼兒園或學校要看哪幾頁？",
        "不同專業（治療師、老師）要給的內容一樣嗎？",
    ],
    "J": [  # 與學校合作
        "要怎麼跟老師說孩子的狀況？",
        "學校體能課或活動可以怎麼調整？需要注意安全嗎？",
        "教室裡可以怎麼配合孩子的需求？",
        "口語或認知方面要怎麼跟老師溝通？",
    ],
    "K": [  # 補助/福利/申請
        "早療或復健有補助嗎？要怎麼申請？",
        "需要準備哪些文件？資格怎麼認定？",
        "身心障礙證明與早療補助的關係？",
    ],
    "L": [  # 後續追蹤/再評估/進步怎麼看
        "建議多久追蹤一次比較合理？",
        "怎麼從生活功能看有沒有進步？",
        "什麼時候需要再評估？",
        "各領域（動作／口語／認知）追蹤頻率一樣嗎？",
        "在家要紀錄什麼才能看出進步？",
    ],
    "M": [  # 家長情緒支持與家庭協作
        "家人覺得不用做治療，我該怎麼溝通？",
        "要怎麼跟孩子說我們要練習而不讓他覺得被罵？",
        "我很焦慮，有什麼資源可以支持？",
        "夫妻或長輩對教養方式不同調怎麼辦？",
        "怎麼建立全家一致可行的計畫？",
    ],
}

# Scope 三類：決定「要檢索多少主題分支」（實際由 DST 規則分類，此處供顯示/備用）
DEFAULT_SCOPE_PROTOTYPES = {
    "S_overview": [
        "這份聯評報告主要在評估什麼？",
        "整體來看，孩子有哪些需要加強的地方？",
        "請幫我整理整份評估的重點與下一步建議。",
        "有哪些領域、評估了哪些？",
    ],
    "S_domain": [
        "請問粗大動作狀況如何？",
        "粗大動作方面孩子的優勢是什麼？我該怎麼善用？",
        "建議多久追蹤一次粗大動作比較合理？",
        "標準分、百分位要怎麼看？代表孩子落在哪裡？",
    ],
    "S_multi_domain": [
        "認知功能和口語溝通有沒有問題？",
        "粗大動作與精細動作的狀況如何？",
        "口語理解和表達方面需要加強嗎？",
    ],
}


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
    Try common embedding APIs:
    - TextEncoder: model.encode_many(list[str]) -> np.ndarray
    - FlagEmbedding BGEM3FlagModel: model.encode(list[str]) -> np.ndarray
    - sentence-transformers style: model.encode(...)
    - callable: model(texts)
    """
    # 優先使用 encode_many（TextEncoder 的 API）
    if hasattr(embedder, "encode_many"):
        vec = embedder.encode_many(texts)
    elif hasattr(embedder, "encode"):
        # 檢查 encode 是否接受列表（如 FlagEmbedding）
        try:
            vec = embedder.encode(texts)
        except (AttributeError, TypeError):
            # 如果不接受列表，嘗試 encode_many
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

        # Turn into a soft distribution (stable, easy to inspect)
        # Note: temperature can be tuned; 12 is a decent starting point.
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
        scope_prototypes: Dict[str, List[str]] = None,
        prototypes_jsonl: Optional[str] = None,
    ):
        """
        Args:
            embedder: Embedding model
            task_prototypes: Task prototypes dict (optional, overrides jsonl)
            scope_prototypes: Scope prototypes dict (optional, overrides jsonl)
            prototypes_jsonl: Path to JSONL file with prototypes (used if task/scope_prototypes are None)
        """
        if task_prototypes is None or scope_prototypes is None:
            path = prototypes_jsonl or DEFAULT_PROTOTYPES_JSONL_PATH
            loaded_task, loaded_scope = load_prototypes_from_jsonl(path)
            if task_prototypes is None:
                task_prototypes = loaded_task
            if scope_prototypes is None:
                scope_prototypes = loaded_scope
        
        self.task_clf = PrototypeClassifier(embedder, task_prototypes)
        self.scope_clf = PrototypeClassifier(embedder, scope_prototypes)

    def predict_task(self, text: str) -> PredictResult:
        return self.task_clf.predict(text)

    def predict_scope(self, text: str) -> PredictResult:
        return self.scope_clf.predict(text)


def format_topk(dist: Dict[str, float], name_map: Dict[str, str], k: int = 2) -> str:
    items = sorted(dist.items(), key=lambda x: x[1], reverse=True)[:k]
    return ", ".join([f"{name_map.get(lbl, lbl)}={p:.2f}" for lbl, p in items])
