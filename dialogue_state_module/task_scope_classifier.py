# task_scope_classifier.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import json


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


DEFAULT_TASK_PROTOTYPES = {
    "T1_report_overview": [
        "這份聯評報告的『粗大動作/物理治療』主要在評估什麼？",
        "可以用三句話幫我抓出粗大動作的重點結論嗎？",
        "我應該先看粗大動作報告的哪一段？",
        "『身體穩定度/移動能力/物品操控』各是什麼意思？",
        "測驗當天孩子精神或配合度會影響結果嗎？",
        "這次的結果可靠嗎？需要多久後再追蹤一次比較好？",
    ],
    "T2_score_interpretation": [
        "標準分、百分位要怎麼看？代表孩子落在哪裡？",
        "報告寫『落後/低於同齡』，到底差多少？",
        "為什麼身體穩定度還可以，但移動能力比較弱？",
        "同樣是跑跳，今天表現不好會讓分數變低嗎？",
        "如果我有上一次的粗大動作分數，可以幫我看是不是進步嗎？",
        "分數進步一點點，生活上會有差嗎？怎麼判斷？",
    ],
    "T3_clinical_to_daily": [
        "報告說『核心肌力不足』，日常會有哪些表現？",
        "『平衡較弱』是指站不穩嗎？有哪些常見情境？",
        "『協調不佳』和『動作笨拙』一樣嗎？",
        "報告提到走路時身體晃、腳步不穩，可能代表什麼？",
        "上/下樓梯需要扶欄或一階一腳，代表哪裡需要加強？",
        "跳不遠或落地不穩，是力量不足還是平衡問題？",
    ],
    "T4_prioritization": [
        "粗大動作方面孩子的優勢是什麼？我該怎麼善用？",
        "如果只能先挑2個重點練，建議先練哪2個？",
        "這些粗大動作問題會影響上學或日常哪些活動？",
        "我希望孩子『走路更穩/上樓更快/跑步不跌倒』，目標怎麼寫？",
        "以孩子現在年齡，哪些粗大動作能力是合理期待？",
        "粗大動作要先從哪個能力開始練，才最有效率？",
    ],
    "T5_coaching": [
        "在家練粗大動作，一天大概需要多久比較可行？",
        "可以給我3個提升核心穩定的居家活動嗎？",
        "單腳站不穩，怎麼循序漸進練到更穩？",
        "要怎麼練『一腳一階上下樓梯』？",
        "早上準備上學很趕，怎麼塞進3分鐘動作練習？",
        "在家訓練要注意哪些安全原則？",
    ],
    "T6_decision_monitoring": [
        "孩子需要做物理治療嗎？什麼情況建議開始？",
        "如果要做PT，一週幾次比較常見？",
        "只在家練可以嗎？什麼情況一定要到院所？",
        "怎麼判斷PT有效？我要看哪些生活指標？",
        "建議多久追蹤一次粗大動作比較合理？",
        "如果練了一段時間沒有明顯進步，下一步怎麼調整？",
    ],
    "T_meta": [
        "我住在___，附近哪裡可以做兒童物理治療/早療？",
        "我可以把粗大動作這一段給老師看嗎？要給整份嗎？",
        "早療或復健有補助嗎？通常要怎麼申請？",
        "健保PT怎麼算？自費PT大概會怎麼收？",
        "家人覺得不用做PT，我該怎麼溝通？",
        "我要怎麼跟孩子說『我們要練習變更穩』而不讓他覺得被罵？",
    ],
}

DEFAULT_SCOPE_PROTOTYPES = {
    "S1_overview": [
        "這份聯評報告主要在評估什麼？",
        "可以用三句話幫我抓出這份報告的重點結論嗎？",
        "我應該先看這份報告的哪一段？",
        "整體來看，孩子有哪些需要加強的地方？",
        "請幫我整理整份評估的重點與下一步建議。",
        "這次的結果可靠嗎？後續要怎麼追蹤？",
    ],
    "S2_domain": [
        "請問粗大動作狀況如何？",
        "粗大動作方面孩子的優勢是什麼？我該怎麼善用？",
        "報告寫『落後/低於同齡』，到底差多少？",
        "建議多久追蹤一次粗大動作比較合理？",
        "孩子需要做物理治療嗎？什麼情況建議開始？",
        "標準分、百分位要怎麼看？代表孩子落在哪裡？",
    ],
    "S3_subskill_context": [
        "單腳站不穩，怎麼循序漸進練到更穩？",
        "要怎麼練『一腳一階上下樓梯』？",
        "跑步容易跌倒，如何練方向改變與停止？",
        "穿褲子時常站不穩，這可以當作平衡練習嗎？",
        "去公園要怎麼安排練習（爬、跳、跑）又不會像在上課？",
        "在家訓練要注意哪些安全原則？",
    ],
    "S4_bridging": [
        "為什麼身體穩定度還可以，但移動能力比較弱？",
        "跳不遠或落地不穩，是力量不足還是平衡問題？",
        "上/下樓梯需要扶欄或一階一腳，代表哪裡需要加強？",
        "分數進步一點點，生活上會有差嗎？怎麼判斷？",
        "這些粗大動作問題會影響上學或日常哪些活動？",
        "報告提到走路時身體晃、腳步不穩，可能代表什麼？",
    ],
    "S5_meta": [
        "我住在___，附近哪裡可以做兒童物理治療/早療？",
        "要怎麼掛兒童復健（PT）？需要轉診或診斷嗎？",
        "早療或復健有補助嗎？通常要怎麼申請？",
        "我可以把粗大動作這一段給老師看嗎？要給整份嗎？",
        "在教室或走廊移動容易撞到/跌倒，學校可以怎麼協助？",
        "家人覺得不用做PT，我該怎麼溝通？",
    ],
}


TASK_NAME_ZH = {
    "T1_report_overview": "報告導覽/摘要",
    "T2_score_interpretation": "分數/量表解讀",
    "T3_clinical_to_daily": "臨床描述轉日常",
    "T4_prioritization": "能力剖面/優先順序",
    "T5_coaching": "訓練教練/在家怎麼做",
    "T6_decision_monitoring": "決策/追蹤與成效",
    "T_meta": "行政/資源/隱私/溝通",
}

SCOPE_NAME_ZH = {
    "S1_overview": "Overview(整體)",
    "S2_domain": "Domain(單領域)",
    "S3_subskill_context": "Subskill/Context(具體能力/情境)",
    "S4_bridging": "Bridging/Attribution(關聯/歸因)",
    "S5_meta": "Meta(非臨床/行政)",
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
            loaded_task, loaded_scope = load_prototypes_from_jsonl(prototypes_jsonl)
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
