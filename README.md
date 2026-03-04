## 對話狀態與策略 README（Dialogue State & Policy）

本文件說明本專案中「對話狀態追蹤（DST）」與「策略決策（Policy）」的設計與運作方式，主要涵蓋：

- 對話狀態是如何被建模與更新的
- 如何根據狀態決定 semantic_flow（延續／切換）
- 如何選擇檢索策略（retrieval_action）
- Task / Scope 對 LLM 回應風格與檢索範圍的影響
- 重要門檻值與調參建議

---

## 架構總覽

### 主要模組與檔案

- `app.py`
  - 初始化 DST 共用元件：`init_dst_components()`
  - 依 user + child 取得 `SemanticFlowClassifier`：`get_dialogue_classifier()`
  - `/api/chat`：一輪對話的完整 pipeline（DST → 檢索策略 → 檢索 → LLM）

- `dialogue_state_module/semantic_flow_module_v2.py`
  - `SemanticFlowClassifier`：DST 主類別
  - `FlowResult`：封裝整輪分析結果（Domain / Context / Topic / Policy / Task / Scope）

- `dialogue_state_module/dst_policy.py`
  - `DSTPolicyConfig`：決策門檻
  - `compute_MT()`：多主題延續度
  - `predicted_flow_from_C_MT()`：由 C + MT 決定 flow（continue / shift_soft / shift_hard）
  - `decide_policy()`：產生 `retrieval_action` 與 `policy_case`

- `dialogue_state_module/domain_anchors.py`
  - `DOMAINS`：所有領域（粗大動作、精細動作、感覺統合、口語、認知等）
  - `DOMAIN_ANCHORS`：每個領域的語意錨點句
  - `OVERVIEW_ANCHORS`：整體查詢錨點句
  - `DomainConfig`：active domain 門檻（prob / ratio / max count）

- `dialogue_state_module/task_scope_classifier.py`
  - `DEFAULT_TASK_PROTOTYPES`：Task A–M 原型句
  - `DEFAULT_SCOPE_PROTOTYPES`：Scope S_overview / S_domain / S_multi_domain 原型句
  - `TaskScopeClassifier`：Task / Scope 的 prototype-based 分類器

- `THRESHOLD_CONFIG.md`
  - 集中說明所有重要門檻（DST、Domain Router、Context Similarity、Multi-Topic Tracker、Retrieval Planner）

---

## 一輪對話的狀態與策略流程

從 `/api/chat` 角度看，一輪對話會經過以下步驟：

1. 使用者輸入訊息（+ 指定 child_id）
2. 取得 (user, child) 對應的 `SemanticFlowClassifier`
3. 執行 `classifier.predict(user_message)` → 得到 `FlowResult`
4. 根據 `FlowResult` 組成 `turn_state`，傳給檢索規劃器
5. 檢索規劃器輸出 `RetrievalPlan`，交給執行器實際從 Neo4j 檢索
6. LLM 根據 user query + candidates + 對話歷史 + Task/Scope 產生回應
7. 更新 DST 狀態與資料庫對話紀錄、對話快取檔

後續章節說明 `FlowResult` 內各部分的意義與關係。

---

## FlowResult 結構

`FlowResult` 由 `SemanticFlowClassifier.predict()` 產生，包含：

- DomainAnalysis：領域分析
- ContextAnalysis：上下文相似度分析
- TopicAnalysis：主題延續分析
- PolicyDecision：策略決策（semantic_flow + retrieval_action）
- Task / Scope：任務與檢索範圍分類
- turn_index：第幾輪對話（從 0 開始）

---

## 領域分析（DomainAnalysis）

對應檔案：

- `dialogue_state_module/domain_anchors.py`
- `dialogue_state_module/semantic_flow_module_v2.py` → `_analyze_domain()`

### 領域與錨點

- `DOMAINS` 定義所有評估領域，例如：
  - 粗大動作、精細動作、感覺統合、口腔動作、吞嚥功能
  - 情緒行為與社會適應功能
  - 口語理解、口語表達、說話
  - 認知功能
- `DOMAIN_ANCHORS` 為每個領域定義多句描述與常見問句，做為語意錨點。
- `OVERVIEW_ANCHORS` 用於偵測整體查詢（Overview），例如「整體狀況」、「整份報告重點」。

### DomainAnalysis 欄位

- `top_domain`：目前最可能的領域
- `distribution`：各領域機率分布（由 TextEncoder + anchors 計算）
- `active_domains`：根據門檻篩選後的「活躍領域」
- `active_domain_probs`：active 領域的機率
- `entropy`：分布的正規化熵（0 = 非常確定，1 = 非常模糊）
- `is_multi_domain`：是否多領域（active_domains 數量 >= 2）
- `fused_distribution`：模糊延續時融合後的分布（可沿用上一輪）
- `is_overview_query` / `overview_distribution`：整體查詢旗標與對應分布

### 重要門檻（摘要）

見 `THRESHOLD_CONFIG.md`：

- Softmax 溫度（domain router）：
  - `temperature = 0.05`
- active domain 門檻：
  - `active_prob_th = 0.30`
  - `active_ratio_th = 0.60`
  - `min_active_domains = 1`
  - `max_active_domains = 4`（或 5，依 DomainConfig）

---

## 上下文相似度分析（ContextAnalysis）

對應檔案：

- `dialogue_state_module/context_similarity.py`
- `dialogue_state_module/semantic_flow_module_v2.py` → `_analyze_context()`

### 指標 C（Context similarity）

- C 反映「當前輸入」與「上一輪 user 或 bot」的語義相似度（0–1）。
- 第 0 輪：沒有歷史，使用中性值：
  - `neutral_first_turn = 0.5`
- 後續輪數：
  - 使用共用 `TextEncoder` 比較當前文字與歷史文字向量。

C 的角色：

- C 高 → 比較像是「接著上一輪的內容繼續問」。
- C 低 → 比較像是「換了一個新話題」。

---

## 主題延續與多領域分析（TopicAnalysis）

對應檔案：

- `dialogue_state_module/multi_topic_tracker.py`
- `dialogue_state_module/semantic_flow_module_v2.py` → `_analyze_topic()`

### Memory 與 TV distance

Multi-Topic Tracker 維護一個跨輪的 `memory_dist`：

- 每一輪領域分布會用 `decay_factor` 做 EMA：
  - 新記憶 = 舊記憶 × decay_factor + 本輪分布 × (1 - decay_factor)
  - 預設 `decay_factor = 0.7`（保留約 70% 歷史，30% 新分布）

同時計算：

- `overlap_score`：根據 Jaccard + 加權 overlap 計算主題重疊程度
- `tv_distance`：memory_dist 與本輪分布的 TV distance
  - `hard_shift_tv_threshold = 0.6`
  - 大於此值視為「強切換」，會重置記憶

### TopicAnalysis 欄位

- `is_continuing`：主題是否延續（布林值）
- `overlap_score`：多主題延續指標（0–1），之後會合成 MT
- `reason`：判斷原因（含 debug 訊息與是否 mem_reset 等）
- `prev_top_domain` / `cur_top_domain`：前一輪／本輪頂級領域
- `prev_dist` / `prev_active_domains`：更新前的上一輪領域分布與活躍領域（供模糊延續使用）
- `tv_distance`：TV 距離，用來決定是否強切換

---

## Task / Scope 分類

對應檔案：

- `dialogue_state_module/task_scope_classifier.py`
- `dialogue_state_module/semantic_flow_module_v2.py` → `_classify_task_only()` / `_classify_scope()`

### Task A–M：任務類型

預設定義在 `DEFAULT_TASK_PROTOTYPES`，例如：

- A：報告總覽與閱讀順序
- B：分數／量表／百分位解讀
- C：臨床觀察與表現解讀
- D：能力剖面（優勢／需求／優先順序）
- E：在家訓練怎麼做
- F：融入日常作息的練習
- G：是否需要早療／成效追蹤
- H：轉介與在地資源
- I：報告分享／隱私與安全
- J：與學校合作
- K：補助／福利／申請
- L：後續追蹤／再評估
- M：家長情緒支持與家庭協作

分類方式：

- 將每個 Task 的原型句嵌入為向量，取平均為 prototype 向量。
- 對輸入文字嵌入為向量，與所有 prototype 計算 cosine similarity。
- 使用溫度 `temp = 12.0` 作 softmax，得到機率分布與最佳 label。

### Scope：檢索範圍

範圍類型（`SCOPE_NAME_ZH`）：

- `S_overview`：Overview（整體）
- `S_domain`：Domain（單領域）
- `S_multi_domain`：Multi-Domain（多領域）

Scope 決策規則（`_classify_scope()`）：

1. 若 `domain_analysis.is_overview_query == True`
   - 直接標為 `S_overview`。
2. 若 `domain_analysis.fused_distribution` 不為空（模糊延續已啟動）
   - Scope 沿用上一輪（`_prev_scope`，預設 `S_overview`）。
3. 若 `turn_index == 0` 且 entropy >= ambiguous_continuation_entropy_th（0.8）
   - 視為首輪模糊查詢，預設 `S_overview`。
4. 否則：
   - active_domains 數量 >= 2 → `S_multi_domain`
   - 否則 → `S_domain`

Scope 決定了「要檢索多少主題分支」，會影響 Retrieval Planner 如何選 topics 與分配 quota。

---

## 策略決策（PolicyDecision）

對應檔案：

- `dialogue_state_module/dst_policy.py`
- `dialogue_state_module/semantic_flow_module_v2.py` → `_decide_policy()`

### 三個核心指標

- C：Context similarity（上下文相似度，0–1）
- MT：Multi-topic continuity（多主題延續度，0–1）
  - `MT = max(overlap_score, 1.0 if topic_continue else 0.0)`
- normalized_entropy：領域分布熵，代表「模糊程度」

### 決策門檻（DSTPolicyConfig）

- 主決策門檻：
  - `C_high_th = 0.55`
  - `MT_high_th = 0.50`
- soft 區間：
  - `C_soft_th = 0.45`
  - `MT_soft_th = 0.30`
- 模糊度：
  - `entropy_high_th = 0.8`
  - `ambiguous_continuation_entropy_th = 0.8`（與 THRESHOLD_CONFIG 保持一致）
- 模糊延續：
  - `enable_ambiguous_continuation = True`
  - `ambiguous_continuation_min_overlap = 0.5`

### Flow 類型：continue / shift_soft / shift_hard

由 `predicted_flow_from_C_MT(C, MT)` 決定：

- `continue`
  - `C >= C_high_th` 且 `MT >= MT_high_th`
  - 代表內容與主題都高度延續。
- `shift_soft`
  - flow_soft_when_one_high = True 時：
    - C 或 MT 其中一個 high，或兩者都在 soft 範圍。
  - 代表仍在同主題池中，但承接較弱。
- `shift_hard`
  - 其他情況（C、MT 都低）
  - 代表可能是新主題或明顯跳題。

### 檢索策略：retrieval_action

由 `decide_policy()` 輸出，主要依 C_level / MT_level + 是否模糊決定：

- `NARROW_GRAPH`
  - C、MT 均高，且非模糊 → 在既有主題範圍內「縮小」檢索。
- `CONTEXT_FIRST`
  - 文字延續但主題分布不明確，或模糊情況下：
    - 優先依上下文處理，不強行縮小檢索。
- `WIDE_IN_DOMAIN`
  - 主題池延續（MT 高），但語義相似度較低，或 C、MT 都低但不模糊：
    - 在同領域（或主題池）內做「較廣」的檢索。
- `DUAL_OR_CLARIFY`
  - C、MT 都低且模糊：
    - 代表「不知道該從哪個主題開始」，規劃器可以選擇雙路檢索或反問澄清。

`PolicyDecision` 欄位：

- `context_level`：C 高或低
- `is_ambiguous`：是否高熵模糊
- `policy_case`：例如 `CH_MTH_NARROW_AMBIG_MD`（便於 debug）
- `retrieval_action`：如上四種
- `semantic_flow`：根據 `action_to_predicted_flow()` 反推 continue / shift_soft / shift_hard

---

## 模糊延續與整體查詢

對應檔案：

- `dialogue_state_module/semantic_flow_module_v2.py` → `_decide_policy()` 與 `_get_overview_distribution()`

### 模糊延續（Ambiguous Continuation）

當條件滿足：

- entropy >= ambiguous_continuation_entropy_th（0.8）
- `turn_index > 0`
- 有有效的 `prev_dist` 與 `prev_active_domains`

系統會：

1. 調整 MT 相關值：
   - 將 `topic_overlap` 至少提升到 0.5
   - 若原本 `topic_continue=False`，改為 True
2. 更新 TopicAnalysis 的 `reason`，標記模糊延續。
3. 回退記憶與分布：
   - `fused_distribution = prev_dist`
   - 更新 `topic_tracker.state.memory_dist`、`prev_dist`、`prev_active_domains` 為上一輪狀態。

效果：

- 使用者講得模糊但大致延續同一大主題時，系統會「沿用上一輪主題與分布」，避免每輪都重新判斷。

### 整體查詢（Overview Query）

利用 `OVERVIEW_ANCHORS` 與向量相似度判斷：

- 使用 TextEncoder 計算當前輸入與 overview anchors 的相似度。
- 若在「模糊」情境下，且：
  - 向量相似度 >= `overview_sim_threshold`，或
  - 上一輪本來就是整體查詢
- 則標記為 `is_overview_query=True`，並建立 `overview_distribution`：
  - 可能策略：
    - 所有領域均勻分布
    - 使用記憶中的領域分布
    - 混合策略（記憶 + 當前 active）

整體查詢會強烈影響 Scope 與 Retrieval Planner：

- Scope 直接設為 `S_overview`
- Retrieval Planner 傾向採用 `GLOBAL_OVERVIEW` 類型的計畫

---

## 檢索與 LLM 的整合

DST 與 Policy 的輸出，被整合成 `turn_state` 丟給檢索規劃器，影響：

- 檢索模式（TOPIC_FOCUSED / GLOBAL_OVERVIEW / REPORT_OVERVIEW / SCORE / META）
- 選擇哪些 topics（單領域、多領域或全域）
- topic / section 的配額分配（`topic_alloc`、`two_level_quota`）
- section 類型的權重（依 Task / Scope 調整）
- 是否走 `CLARIFY` 路徑（直接產生反問，不進 Neo4j）

LLM 端透過：

- `llm_generate_module/prompt_manager.py`
  - 根據 semantic_flow、retrieval_action、task_label、scope_label 等，產生 `LLMGenerationConfig`
  - 動態設定：
    - temperature、max_tokens、max_context_items
    - context_format_style（detailed / concise / structured）
    - response_style（professional / step_by_step / explanatory / comprehensive）
    - system_prompt（繁體中文、格式限制、與家長溝通口吻）
- `llm_generate_module/llm_generator.py`
  - 實際呼叫 LM Studio OpenAI 相容 API
  - 組合 system + history + user prompt（含檢索 context）
  - 回傳回應後移除 markdown `#` 標題，並由 `app.py` 在尾端附上引用頁碼區塊

---

## 門檻調整建議（精簡版）

詳細說明請見 `THRESHOLD_CONFIG.md`，以下是和對話狀態／策略最相關的幾項：

1. 模糊度敏感度
   - 更容易判為模糊：
     - 降低 `entropy_high_th` 與 `ambiguous_continuation_entropy_th`（例如 0.7）
   - 更不容易判為模糊：
     - 提高到 0.85 以上

2. 主題延續敏感度
   - 更容易延續：
     - 降低 `overlap_th`（例如 0.4）
   - 更容易判為切換：
     - 提高到 0.6

3. 領域集中度與多領域
   - 讓 top_domain 更明確：
     - 降低 `Domain Router` 的 `temperature` 或提高 `active_prob_th` / `active_ratio_th`
   - 讓系統更願意考慮多領域：
     - 提高 `temperature` 或放寬 active 門檻，並適度提高 `max_active_domains`

4. 記憶保留
   - 更重視歷史分布：
     - 提高 `decay_factor`（例如 0.8）
   - 更快適應新輸入：
     - 降低到 0.5

調整方式建議：

- 每次只動 1–2 個門檻，搭配 `/api/chat` log 觀察：
  - `entropy`、`C`、`MT`、`tv_distance`
  - `semantic_flow`、`policy_case`、`retrieval_action`
- 特別測試：
  - 清楚單一領域問題
  - 多領域混合問題
  - 模糊問句與整體概覽型問題
  - 強切換情境（從一個領域突然問完全不同領域）

---