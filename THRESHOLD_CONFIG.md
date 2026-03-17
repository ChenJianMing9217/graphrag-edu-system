# 系統門檻值配置 (THRESHOLD_CONFIG)

本文件定義了早療系統全模組的超參數與門檻值，並提供調整指南。

---

## 1. 對話狀態追蹤 (DST) 門檻

### 四象限決策門檻 (`dst_policy.py`)
決定對話流 (Flow) 與檢索動作 (Action) 的核心指標。
- **`C_high_th`**: `0.55` - 上下文相似度 (Context) 高低門檻。
- **`MT_high_th`**: `0.50` - 多主題延續度 (Topic) 高低門檻。
- **`C_soft_th`**: `0.45` - 進入 `shift_soft` 狀態的邊界值。
- **`MT_soft_th`**: `0.30` - 進入 `shift_soft` 狀態的邊界值。

### 模糊度與延續控制
- **`entropy_high_th`**: `0.80` - 正規化熵門檻。高於此值視為意圖模糊，傾向觸發澄清或反問。
- **`ambiguous_continuation_entropy_th`**: `0.80` - 觸發「模糊延續」邏輯的門檻。
- **`extreme_ambiguous_entropy_th`**: `0.90` - 判定為極度模糊，通常會觸發 `DUAL_OR_CLARIFY`。

---

## 2. 檢索與排序門檻 (Retrieval V2)

### 檢索權重與配額 (`topic_ontology.py`)
- **`MAX_TOPICS`**: `4` - 一輪檢索中最多處理的領域 (Domain) 數量。
- **`MIN_PER_TOPIC`**: `1` - 每個領域至少檢索的片段數。
- **`MAIN_TOPIC_RATIO`**: `0.7` - 弱延續時，分配給主要主題的配額比例。

### 排序權重 (`reranker.py`)
- **`semantic_weight`**: `0.6` - BGE-M3 餘弦相似度的原始權重。
- **`structural_weight`**: `0.2` - 節點標籤 (Label) 符合任務需求時的加成權重。
- **`context_weight`**: `0.2` - 與上下文路徑一致時的加成權重。

---

## 3. 領域路由與分類 (Domain Router)

### 激活門檻 (`domain_anchors.py`)
- **`temperature`**: `0.05` - 領域分佈的平滑度，越小越集中。
- **`active_prob_th`**: `0.30` - 單一領域機率需超過此值才被視為「活躍」。
- **`active_ratio_th`**: `0.60` - 領域機率需達到 Top 1 的此比例才被視為「活躍」。

---

## 💡 快速調整指南

### 想要系統更「靈敏」地捕捉主題切換？
- **作法**：調高 `MT_high_th` (如 `0.60`) 或調低 `hard_shift_tv_threshold` (如 `0.50`)。
- **效果**：用戶稍微偏離原話題時，系統會更快判定為新主題。

### 想要系統更頻繁地「反問」地區或意圖？
- **作法**：調低 `entropy_high_th` (如 `0.70`)。
- **效果**：當輸入稍顯含糊時，系統會更傾向觸發 `LOCAL_RESOURCE_CLARIFY`。

### 想要提升檢索结果的「專業精準度」？
- **作法**：在 `reranker.py` 中調高 `structural_weight`。
- **效果**：系統會更重視節點標籤（如：優先看「訓練方向」而非「臨床觀察」）。

---
*註：調整門檻後請務必執行 `python .\app.py` 觀察日誌中的 C、MT 與 Entropy 實際數值變化。*
