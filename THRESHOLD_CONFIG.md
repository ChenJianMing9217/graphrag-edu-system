# 所有分數門檻配置清單

本文檔列出系統中所有可調整的分數門檻，方便統一調整。

---

## 1. DST Policy 門檻 (`dialogue_state_module/dst_policy.py`)

### 主決策門檻（四象限決策）
- **`C_high_th`**: `0.55` - Context similarity 高/低門檻
- **`MT_high_th`**: `0.50` - Multi-topic continuity 高/低門檻

### Flow 輸出門檻（shift_soft 判斷）
- **`MT_soft_th`**: `0.30` - MT 介於 soft~high 時可視為「弱延續」
- **`C_soft_th`**: `0.45` - C 介於 soft~high 時可視為「弱延續」

### 模糊度判斷門檻
- **`entropy_high_th`**: `0.8` - Entropy 高門檻（用於判斷模糊度）
- **`ambiguous_continuation_entropy_th`**: `0.8` - 模糊判斷的熵值門檻（一般模糊）
- **`extreme_ambiguous_entropy_th`**: `0.90` - 極度模糊的熵值門檻（極度模糊）

### 模糊延續配置
- **`ambiguous_continuation_prob_ratio`**: `0.7` - 上一輪領域機率需達到當前top的此比例
- **`ambiguous_continuation_min_overlap`**: `0.5` - 調整後的最小overlap分數

---

## 2. Domain Router 門檻 (`dialogue_state_module/domain_router.py`)

### Softmax 溫度
- **`temperature`**: `0.05` - Softmax 溫度（越小越集中）

### Active Domains 選擇門檻
- **`active_prob_th`**: `0.30` - 絕對機率門檻（>= 此值才算 active）
- **`active_ratio_th`**: `0.60` - 相對機率門檻（>= top1 × 此比例才算 active）
- **`min_active_domains`**: `1` - 至少保留幾個 active domains（保底 topK）
- **`max_active_domains`**: `4` - 最多保留幾個 active domains（避免爆炸）

---

## 3. Context Similarity 門檻 (`dialogue_state_module/context_similarity.py`)

### 第一輪中性值
- **`neutral_first_turn`**: `0.5` - 第一輪沒有 prev 時回傳的中性值

### Bot 回覆處理
- **`bot_max_chars`**: `1200` - 用 bot 回覆計算相似度時，最多使用多少字元
- **`bot_keep_tail_chars`**: `400` - 若 bot 回覆很長，保留尾端多少字元

---

## 4. Multi-Topic Tracker 門檻 (`dialogue_state_module/multi_topic_tracker.py`)

### 記憶更新
- **`decay_factor`**: `0.7` - 主題記憶更新時，舊記憶的保留比例（70%）

### 主題延續判斷
- **`overlap_th`**: `0.5` - 多少 overlap 以上算延續
- **`similarity_th`**: `0.97` - 分布 cosine 相似度門檻（弱 backup）
- **`min_conf_for_similarity`**: `0.10` - 信心太低時不要用 similarity rule

### Overlap 組合權重
- **`w_jaccard`**: `0.5` - Jaccard 權重
- **`w_weighted`**: `0.5` - Weighted 權重

### 強切換判斷
- **`hard_shift_tv_threshold`**: `0.6` - TV 距離超過此值視為強切換

---

## 5. Retrieval Planner 門檻 (`retrieval_module/dst_based_planner.py`)

### 檢索數量
- **`TOTAL_K`**: `10` - 總片段數

### Task/Subdomain 機率門檻
- **`TASK_THRESHOLD`**: `0.1` - Task 機率門檻
- **`SUBDOMAIN_PROB_THRESHOLD`**: `0.03` - Subdomain 機率門檻（用於過濾極小值）

### Entropy 門檻（與 DST 一致）
- **`ENTROPY_THRESHOLD`**: `0.8` - 與 DST 的 `ambiguous_continuation_entropy_th` 一致

### DUAL_OR_CLARIFY 門檻
- **`TOP_DOMAIN_PROB_THRESHOLD`**: `0.3` - 如果 top domain 機率 >= 此值，優先使用 top domain

---

## 快速調整指南

### 調整模糊度敏感度
- **更敏感（更容易觸發模糊延續）**：降低 `entropy_high_th` 和 `ambiguous_continuation_entropy_th`（例如：`0.70`）
- **更不敏感（更難觸發模糊延續）**：提高 `entropy_high_th` 和 `ambiguous_continuation_entropy_th`（例如：`0.85`）
- **當前值**：`0.8`（統一使用）

### 調整領域分布集中度
- **更集中**：降低 `temperature`（例如：`0.03`）
- **更分散**：提高 `temperature`（例如：`0.1`）

### 調整 Active Domains 數量
- **更多 active domains**：降低 `active_prob_th` 或 `active_ratio_th`，提高 `max_active_domains`
- **更少 active domains**：提高 `active_prob_th` 或 `active_ratio_th`，降低 `max_active_domains`

### 調整主題延續敏感度
- **更容易延續**：降低 `overlap_th`（例如：`0.4`）
- **更難延續**：提高 `overlap_th`（例如：`0.6`）

### 調整記憶保留比例
- **保留更多歷史**：提高 `decay_factor`（例如：`0.8`）
- **保留更少歷史**：降低 `decay_factor`（例如：`0.5`）

---

## 注意事項

1. **一致性**：`ENTROPY_THRESHOLD` 應與 `ambiguous_continuation_entropy_th` 保持一致（當前統一為 `0.8`）
2. **範圍**：大部分門檻值應在 `[0, 1]` 範圍內
3. **測試**：調整後建議進行完整測試，確保系統行為符合預期

## 已歸檔的文件

以下文件已移至 `archive/` 目錄：
- `dialogue_state_module/duplicated/` - 舊版本和測試文件
- `duplicated/` - 舊版本的 PDF 處理文件
- `retrieval_module/generic_planner.py` - 未使用的通用規劃器

