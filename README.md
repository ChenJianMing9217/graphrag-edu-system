## 專案對話系統總覽

本專案的核心是一套「以對話狀態追蹤（DST, Dialogue State Tracking）為中心」的問答系統，用來協助家長與治療師根據兒童的早療評估報告進行互動詢問。  
對話流程大致可分為四層：

1. **對話入口與請求驗證**：Flask API `/api/chat`
2. **對話狀態追蹤（DST）**：領域判斷、上下文相似度、主題延續、策略決策
3. **檢索規劃與執行**：基於 DST 的檢索策略從 Neo4j 知識圖譜抓取相關內容
4. **LLM 回應生成**：依 DST＋檢索結果動態調整提示詞與生成參數，並附上引用頁碼

---

## 一、整體對話資料流

1. **前端呼叫 `/api/chat`**
   - 傳入：
     - `message`: 使用者輸入文字
     - `child_id`: 該對話對應的兒童
   - 後端會檢查：
     - 使用者是否登入
     - `child_id` 是否存在且屬於目前登入者（照護者或治療師）

2. **取得 / 建立對話分類器**
   - 透過 `get_dialogue_classifier(current_user.id, child_id)`：
     - 共用全域的：
       - `TextEncoder`（BGE-M3）
       - `DomainRouter`
       - `TaskScopeClassifier`
     - 每一組 `(user_id, child_id)` 會有自己的 `SemanticFlowClassifier` 實例：
       - 內含 `ContextSimilarity`（上下文相似度）
       - 內含 `MultiTopicTracker`（主題記憶）
     - 啟動時會嘗試從檔案載入既有對話狀態（如存在）。

3. **DST 分析（`SemanticFlowClassifier.predict`）**
   - 對目前這一輪 user message 執行：
     - **領域分析（DomainAnalysis）**
     - **上下文相似度分析（ContextAnalysis）**
     - **主題延續分析（TopicAnalysis）**
     - **策略決策（PolicyDecision）**
     - **Task / Scope 分類（TaskScopeClassifier）**
   - 結果封裝在 `FlowResult`，同時更新內部記憶與 `turn_index`。

4. **根據 DST 構建 `turn_state`**
   - 自 `FlowResult` 抽取：
     - `retrieval_action`
     - `domain_distribution`（可能是原始分布、模糊融合分布或整體分布）
     - `task_dist`、`task_pred`
     - `scope_pred`
     - `semantic_flow`
     - `top_domain`、`top_domain_prob`
     - `topic_overlap`
     - `is_ambiguous`
     - `normalized_entropy`
     - DST 提供的 `active_domains`、`prev_active_domains`

5. **檢索規劃與執行**
   - 從最新成功處理的報告中取得 `doc_id`。
   - 取得／建立對應 `(user, child)` 的 `RetrievalState`。
   - 呼叫 `DSTBasedRetrievalPlanner.plan(user_query, turn_state, retrieval_state, doc_id)`：
     - 決定要檢索的主題（subdomains）、每個主題的 quota
     - 決定要使用的 section 類型（assessment / observation / training / suggestion）
     - 決定 rerank 權重與是否需要澄清
   - 呼叫 `RetrievalExecutor.execute` 依計畫實際查詢 Neo4j，得到 `candidates` 等結果。

6. **建構 LLM 輸入與生成回答**
   - 決定是否帶入對話歷史（依 `semantic_flow`、`tv_distance` 等條件，有時會「跳過歷史」）。
   - 呼叫 `generate_llm_response(...)`：
     - 內部利用 `LLMPromptManager.get_config(...)` 產生 `LLMGenerationConfig`
     - 根據 `semantic_flow`、`retrieval_action`、`task_label`、`scope_label` 等調整溫度、長度、回應風格與上下文格式
     - 將檢索到的 `candidates` 以及（可能存在的）`conversation_history` 一起送入 LLM
   - 回應完成後，使用 `add_citation_boxes` 依 `candidates.path.page_start / page_end` 在尾端加入「【引用來源】第 X–Y 頁」。

7. **狀態與記錄更新**
   - DST：`context_similarity.update_bot_only(bot_response)` 並 `save_state(user_id, child_id)`
   - 資料庫：在 `ChatMessage` 中新增一條 user 訊息、一條 bot 訊息
   - 對話緩存檔案：在 `dialogue_cache/user_{user}_child_{child}_dialogue.json` 追加完整 `dst_analysis`
   - 檢索狀態：更新該 `(user, child)` 對應 `RetrievalState.active_topics`

---

## 二、對話狀態追蹤（DST）

### 1. 組成元件

- **文本編碼器 `TextEncoder`**
  - 使用 BGE-M3 模型進行句向量編碼。
  - 全專案共用一個實例（在 `init_dst_components()` 初始化）。

- **領域路由器 `DomainRouter`**
  - 使用 `DOMAINS` 與 `DOMAIN_ANCHORS` 域錨描述來計算「輸入句子對每個領域的相似度分布」。
  - 主要輸出：
    - `top_domain`
    - `dist`（各領域機率）
    - `entropy`
    - `active_domains`（依 `DomainConfig` 門檻決定）

- **上下文相似度 `ContextSimilarity`**
  - 比較當前 user 輸入與：
    - 前一輪 user 輸入
    - 或前一輪 bot 回應
  - 輸出：
    - `C`（0–1）
    - `source`：`"first_turn"` / `"prev_user"` / `"prev_bot"`

- **多主題記憶追蹤 `MultiTopicTracker`**
  - 維護一個跨輪的主題機率分布（memory_dist）。
  - 計算：
    - `topic_continue`（是否延續）
    - `topic_overlap`（主題重疊度 MT）
    - `tv_distance`（前後主題分布的總變異距離，用來偵測強切換）。

- **Task / Scope 分類器 `TaskScopeClassifier`**
  - 以「原型文本」為中心：
    - Task：`T1`–`T6`、`T_meta`
    - Scope：`S1`–`S5`
  - 每個 label 會有多個代表句，使用同一個 embedding 模型做 cosine similarity＋softmax 得到分布。

- **策略決策 `DSTPolicy`**
  - 只依賴：
    - `C`（上下文相似度）
    - `MT`（Multi-topic continuity 分數）
    - `normalized_entropy`（領域分布熵）
    - `is_multi_domain`
  - 輸出：
    - `semantic_flow`
    - `retrieval_action`
    - `context_level`
    - `is_ambiguous`
    - `policy_case`

### 2. FlowResult 結構

DST 每一輪的完整分析包在 `FlowResult` 中，主要欄位如下：

- **DomainAnalysis**
  - `top_domain`: 當前輪最主要領域
  - `top_prob`: 該領域機率
  - `entropy`: 正規化熵（0=很確定、1=很模糊）
  - `distribution`: 所有領域機率分布
  - `active_domains`: 依門檻選出的活躍領域
  - `active_domain_probs`: 活躍領域的機率子集
  - `is_multi_domain`: 是否多領域（活躍領域數≥2）
  - `fused_distribution`: 若觸發「模糊延續」，會回退／融合上一輪分布
  - `is_overview_query`: 是否為「整體查詢」
  - `overview_distribution`: 整體查詢使用的特別分布（目前策略為所有領域均勻）

- **ContextAnalysis**
  - `similarity_score` (`C`): 0–1 的上下文相似度
  - `source`: `"first_turn"` / `"prev_user"` / `"prev_bot"`
  - `is_first_turn`: 是否第一輪

- **TopicAnalysis**
  - `is_continuing`: 是否延續上一輪主題
  - `overlap_score` (`MT`): 主題重疊分數（0–1）
  - `reason`: 判斷原因（包含是否記憶重置等）
  - `prev_top_domain` / `cur_top_domain`
  - `prev_dist`: 更新前上一輪的領域分布（供模糊延續回退）
  - `prev_active_domains`: 更新前上一輪的活躍領域
  - `tv_distance`: 前後主題分布的總變異距離（0=完全相同、1=完全不同）

- **PolicyDecision**
  - `context_level`: `"high"` / `"low"`（由 C 決定）
  - `is_ambiguous`: 是否模糊（由 entropy 判斷）
  - `policy_case`: 如 `"CH_MTH_NARROW_MD"` 的代碼字串
  - `retrieval_action`: 檢索動作（詳見後文）
  - `semantic_flow`: `"continue"` / `"shift_soft"` / `"shift_hard"`

- **任務與範圍**
  - `task_label`: `T1`–`T6` 或 `T_meta`
  - `task_dist`: Task 機率分布
  - `scope_label`: `S1`–`S5`
  - `scope_dist`: Scope 機率分布

---

## 三、已定義的領域（Domains）

由 `dialogue_state_module.domain_anchors.DOMAINS` 定義，目前共有 10 個領域：

1. **粗大動作**
2. **精細動作**
3. **感覺統合**
4. **口腔動作**
5. **情緒行為與社會適應功能**
6. **吞嚥功能**
7. **口語理解**
8. **口語表達**
9. **說話**
10. **認知功能**

每個領域在 `DOMAIN_ANCHORS` 中有多句專業描述＋常見問句作為語意錨點，用來做領域路由與主題記憶。

---

## 四、任務與範圍：Task / Scope 定義

### 1. Task 類型（DEFAULT_TASK_PROTOTYPES ＋ TASK_NAME_ZH）

- **T1_report_overview**：報告導覽／摘要  
  - 幫使用者抓重點、整理主要結論與建議。

- **T2_score_interpretation**：分數／量表解讀  
  - 說明標準分、百分位的意義與發展位置。

- **T3_clinical_to_daily**：臨床描述轉日常  
  - 把臨床用語換算成日常生活表現與情境。

- **T4_prioritization**：能力剖面／優先順序  
  - 找出優勢與弱勢，協助決定先練什麼。

- **T5_coaching**：訓練教練／在家怎麼做  
  - 提供具體可操作的居家訓練步驟與活動設計。

- **T6_decision_monitoring**：決策／追蹤與成效  
  - 是否需要治療？如何安排頻率？怎麼看是否有效？

- **T_meta**：行政／資源／隱私／溝通  
  - 包含醫療資源、補助、與學校／家人溝通等非純臨床問題。

### 2. Scope 類型（DEFAULT_SCOPE_PROTOTYPES ＋ SCOPE_NAME_ZH）

- **S1_overview**：Overview（整體）  
  - 問整份報告或整體發展情況。

- **S2_domain**：Domain（單領域）  
  - 聚焦在某一個領域（例如只問粗大動作）。

- **S3_subskill_context**：Subskill / Context（具體能力／情境）  
  - 問特定技能或情境（如單腳站、上下樓梯）。

- **S4_bridging**：Bridging / Attribution（關聯／歸因）  
  - 問「這個能力和那個狀況有什麼關係？」或「原因是什麼？」。

- **S5_meta**：Meta（非臨床／行政）  
  - 問掛號、補助、資源、如何跟老師／家人說明等。

---

## 五、Semantic Flow 與策略決策

### 1. Semantic Flow 狀態

由 `DSTPolicy` 決定，主要取決於：

- `C`：上下文相似度
- `MT`：主題連續性（由 topic_overlap 和 topic_continue 組合）

三種狀態：

1. **continue**
   - C 高且 MT 高
   - 強烈延續前一輪主題
2. **shift_soft**
   - 一高一低或兩者都在「soft 區間」
   - 主題池仍相關，但有輕微轉向
3. **shift_hard**
   - C 低且 MT 低
   - 視為強切換，接近全新話題

### 2. 檢索動作（retrieval_action）

由 `decide_policy` 輸出：

- **NARROW_GRAPH**
  - 代表「延續性強，可以縮小檢索範圍」。
  - 規劃器會偏向「上一輪與本輪領域交集」。

- **CONTEXT_FIRST**
  - 代表「看起來像延續，但主題池不完全對齊」。
  - 先依當前上下文延續，使用本輪的 active_domains。

- **WIDE_IN_DOMAIN**
  - 代表「同一大領域內的新角度或新子題」。
  - 在 top domain 所屬的 macro-domain 下做廣泛檢索。

- **DUAL_OR_CLARIFY**
  - 代表「C 與 MT 都低或情況模糊」，需要：
    - 同時檢索多個可能領域，或
    - 提示 LLM 以澄清式的回答回問使用者。

`action_to_predicted_flow` 中也定義了 action → semantic_flow 的預測關係：

- `NARROW_GRAPH` / `CONTEXT_FIRST` → **continue**
- `WIDE_IN_DOMAIN` → **shift_soft**
- `DUAL_OR_CLARIFY` → **shift_hard**

### 3. Policy Case 編碼（policy_case）

`policy_case` 的格式大致為：

- `C{H/L}_MT{H/L}_{ACTION}_{修飾}`，例如：
  - `CH_MTH_NARROW_MD`
  - `CL_MTH_WIDE_AMBIG`

其中：

- `CH` / `CL`：Context level（High / Low）
- `MTH` / `MTL`：Multi-topic continuity level（High / Low）
- `_NARROW` / `_CTX` / `_WIDE` / `_DUAL`：對應檢索動作
- `_AMBIG`：當前輪被判定為模糊（entropy 高）
- `_MD`：多領域（`is_multi_domain=True`）

### 4. 模糊與模糊延續（Ambiguity & Ambiguous Continuation）

`DSTPolicyConfig` 中主要門檻：

- `entropy_high_th = 0.8`：entropy ≥ 0.8 視為「高模糊度」
- `enable_ambiguous_continuation = True`：啟用模糊延續
- `ambiguous_continuation_entropy_th = 0.8`
- `ambiguous_continuation_min_overlap = 0.5`

在 `SemanticFlowClassifier._decide_policy` 中：

- 若：
  - entropy 過高，且
  - 有上一輪的分布 `prev_dist` 與 `prev_active_domains`
- 則會：
  - 將 topic_overlap 至少拉到 0.5
  - 把 `topic_continue` 調整為 True
  - **直接使用上一輪的分布作為 fused_distribution**
  - 回退 `memory_dist` 與 `prev_active_domains` 到上一輪狀態

另外，對整體查詢（overview query）：

- 會標記 `is_overview_query=True`
- 並建立 `overview_distribution`（目前策略是所有領域均勻）
- 之後檢索與 LLM-config 都會優先使用這個整體分布。

---

## 六、記憶設計與重置策略

### 1. DST 內部記憶

- **ContextSimilarity**
  - 儲存前一輪 user / bot 向量，用來計算 C。
  - 在 `reset()` 時會清空。

- **MultiTopicTracker**
  - 維護：
    - `memory_dist`：累積的主題分布記憶
    - `prev_dist`：上一輪分布
    - `prev_active_domains`：上一輪活躍領域
  - 若 TV 距離（`tv_distance`）大於某門檻（在 `MultiTopicTracker` 內部），會視為「強切換」，記憶被重置。

### 2. LLM 對話歷史的使用／跳過

在 `chat()` 中會依下列邏輯決定是否帶入 `conversation_history`：

- **跳過歷史（should_skip_history = True）的條件：**
  1. 記憶重置：
     - `tv_distance >= 0.6`，或
     - `semantic_flow == "shift_hard"`，或
     - `topic_analysis.reason` 含有 `"mem_reset"`
  2. 主題延續狀態由延續轉為切換：
     - 前一輪 `semantic_flow == "continue"`，本輪變為 `shift_soft` / `shift_hard`
     - 或 `shift_soft → shift_hard`

- **未跳過歷史時：**
  - 從 `ChatMessage` 查詢最近 10 則屬於該 `(user, child)` 的訊息（含 user/bot），依時間排序，轉成 `conversation_history` 傳入 LLM。

### 3. 永久儲存與重置 API

- **資料庫 `ChatMessage`**
  - 每條 user 訊息包含：
    - `message`
    - `flow_state`（簡化版 DST：domain、entropy、context_similarity、semantic_flow、retrieval_action）
    - `retrieval_info`（flow_type、retrieval_action、context_level）
  - bot 訊息則只存文字內容。

- **檔案系統**
  - `dialogue_states/`：`SemanticFlowClassifier.save_state()` 用來儲存 DST 內部狀態。
  - `dialogue_cache/user_{user}_child_{child}_dialogue.json`：
    - 每一輪追加：
      - `timestamp`
      - `turn_index`
      - `user_message`
      - `bot_response`
      - `dst_analysis`（完整 FlowResult dict）

- **重置 API：`POST /api/chat/reset`**
  - 驗證 `child_id` 權限後，執行：
    1. `SemanticFlowClassifier.reset()`（包含 context & topic 記憶）
    2. `delete_dialogue_state(user_id, child_id)`（刪除 DST 狀態檔）
    3. 刪除該 `(user, child)` 的全部 `ChatMessage`
    4. 刪除對應的 `dialogue_cache` JSON 檔
    5. 重置 `RetrievalState`（若存在）

---

## 七、檢索規劃與執行（DSTBasedRetrievalPlanner）

### 1. 前置條件與基礎元件

- 每個兒童上傳 PDF 報告時：
  - 經 `process_pdf_to_graph` 處理：
    - 解析內容、切分為 grouped units
    - 依子領域／section 類型寫入 Neo4j
  - 在 `KnowledgeGraphProcessing` 紀錄：
    - `doc_id`
    - `status`（成功才會被後續檢索使用）

- 檢索相關全域單例：
  - `GraphClient`：包 Neo4j 連線
  - `DSTBasedRetrievalPlanner`：主要規劃器
  - `RetrievalExecutor`：
    - 若 DST 的 `TextEncoder` 可用，會共用底層 BGE-M3 sentence transformer 作為 rerank embedding
    - 否則回退到 bigram Jaccard 相似度

### 2. 規劃器輸入（turn_state 主要欄位）

- 來自 DST 的：
  - `retrieval_action`
  - `domain_distribution`
  - `task_dist` / `task_pred`
  - `scope_pred`
  - `semantic_flow`
  - `top_domain` / `top_domain_prob`
  - `topic_overlap`
  - `is_ambiguous`
  - `normalized_entropy`
  - `active_domains` / `prev_active_domains`
  - `fused_distribution_used`
  - `turn_index`

### 3. 主流程（簡化版）

1. **根據 entropy 調整 active_domains**
   - entropy 門檻：`ENTROPY_THRESHOLD = 0.8`
   - 若 entropy 高且有 `prev_active_domains`：
     - 若 DST 已觸發模糊延續（`fused_distribution_used=True`）：
       - 直接使用上一輪的 `prev_active_domains`
     - 否則將 `prev_active_domains` 與本輪 `top_domain` 合併成新的 active_domains。

2. **決定 active_tasks**
   - 門檻：`TASK_THRESHOLD = 0.1`
   - 從 `task_dist` 中挑出 ≥ 0.1 的 labels
   - 若沒有分布資料，則 fallback 用 `task_pred`。

3. **決定要用的 section 類型與權重**
   - `use_sections`：
     - 依 active_tasks＋scope 決定要用哪些 section（assessment / observation / training / suggestion）。
   - `section_type_weights`：
     - 依 active_tasks 的機率分布與 `TopicOntology.TASK_TO_SECTION_WEIGHTS` 做機率加權合併。

4. **依 Scope 決定 subdomain 範圍與 quota**
   - 總配額 `TOTAL_K = 10`
   - **先依 `scope_pred`**：
     - `S_overview`：topics = 全部領域（ontology.TOPIC_LABELS），配額依分布或均分
     - `S_domain`：單一領域，全部配額給該領域
     - `S_multi_domain`（或未知）：過濾極小機率 `SUBDOMAIN_PROB_THRESHOLD = 0.03`，**再依 `retrieval_action`** 呼叫不同策略：
     - `NARROW_GRAPH`：
       - 優先使用 DST 的 `prev_active_domains ∩ active_domains`
       - 若無交集，使用本輪 active_domains
     - `CONTEXT_FIRST`：
       - 優先使用本輪 active_domains
       - 若無，使用本輪高機率 subdomains（≥ 0.1）
     - `WIDE_IN_DOMAIN`：
       - 優先使用本輪 active_domains
       - 若無，則以 top_domain 的 macro-domain 為範圍，擴展所有同 macro-domain 且機率≥門檻的 subdomains
     - `DUAL_OR_CLARIFY`：
       - 優先使用本輪 active_domains
       - 若 top_domain 機率≥ 0.3，則只用 top_domain
       - 否則使用所有過門檻 subdomains

   - 最後在選定的 subdomains 之間：
     - 依其機率分布＋「每個至少 1 個」原則分配 10 個 quota。

5. **兩層分配：subdomain → section type**
   - 將每個 subdomain 的 quota，再依 `section_type_weights` 與 `use_sections` 分配到不同 section type。
   - 輸出 `two_level_quota`：`{"粗大動作": {"training": 3, "suggestion": 2, ...}, ...}`。

6. **設定其他計畫欄位**
   - `mode`: 一律為 `RetrievalMode.TOPIC_FOCUSED`
   - `topic_policy`: 一律為 `TopicPolicy.MIX_TOPK`
   - `k_items = TOTAL_K`
   - `ask_clarify`: 當 `retrieval_action == "DUAL_OR_CLARIFY"` 且 DST 標記模糊時為 True
   - `rerank_weights`：
     - `semantic_flow == "continue"`：
       - `sim_q`: 0.4, `sim_anchor`: 0.5, `graph_prox`: 0.1
     - `shift_soft`：
       - `sim_q`: 0.5, `sim_anchor`: 0.3, `graph_prox`: 0.2
     - `shift_hard`：
       - `sim_q`: 0.6, `sim_anchor`: 0.2, `graph_prox`: 0.2

---

## 八、LLM 回應生成與提示策略

### 1. LLMGenerationConfig 主要欄位

- 生成超參數：
  - `temperature`
  - `max_tokens`
  - `top_p`
  - `frequency_penalty`
  - `presence_penalty`
- 提示詞：
  - `system_prompt_template`
  - `user_prompt_template`（實際是在 `build_user_prompt` 裡直接構造字串）
- 上下文處理：
  - `max_context_items`
  - `context_format_style`: `"detailed"` / `"concise"` / `"structured"`
- 回應風格：
  - `response_style`: `"professional"` / `"friendly"` / `"concise"` / `"detailed"` / `"step_by_step"` / `"explanatory"` / `"comprehensive"`
  - `include_examples`
  - `include_caution`

### 2. 配置合併策略

`LLMPromptManager.get_config(...)` 會依下列來源產生多組 config，最後合併：

1. **base_config（依 semantic_flow）**
   - `continue`：偏向較低溫度、較短回答、專業風格。
   - `shift_soft`：略提高溫度，偏向 friendly。
   - `shift_hard`：允許稍長回答與較高溫度，偏向 detailed。

2. **retrieval_config（依 retrieval_action）**
   - `NARROW_GRAPH`：context 數量較少、但每條較詳細。
   - `CONTEXT_FIRST`：偏向 structured。
   - `WIDE_IN_DOMAIN`：允許更多 context，改用 concise 格式。
   - `DUAL_OR_CLARIFY`：加入 caution，提醒答案可能需要澄清。

3. **task_config（依 task_label / scope_label）**
   - 不同 Task／Scope 對應不同：
     - `max_tokens`
     - `response_style`
     - `max_context_items`

4. **special_config（依 is_overview_query / is_multi_domain / is_ambiguous）**
   - overview：允許最多 tokens 與 context，並使用 structured＋comprehensive。
   - multi_domain：強制使用 structured context。
   - ambiguous：降低 temperature、打開 include_caution。

最後由 `_merge_configs` 依優先順序（base < retrieval < task < special）合併為一個 `LLMGenerationConfig`。

### 3. System Prompt 與 User Prompt

- **System Prompt**
  - 強制 LLM 使用繁體中文。
  - 定義整體角色（專業早療助手）。
  - 規範回應格式（避免使用 `#` 標題、`-` 列表等），以符合產品 UI 需求。
  - 若有 Task／Scope，會在提示詞中加入對應的「專長說明」與「回答範圍」。
  - 若 `is_ambiguous=True`，會加入「如何自然地反問與引導」的指示。

- **User Prompt**
  - 若無檢索 context：
    - 若非模糊：直接使用 user query。
    - 若模糊：在 query 後附加一段「如何用自然語氣反問與引導」的指示文字。
  - 若有檢索 context：
    - 先依 `context_format_style` 將 `candidates` 格式化成一段文字。
    - 再附上「回答要求」與原始 user query。
    - 若 `is_ambiguous`，同樣會再加一段「反問與引導」的指示。

### 4. 引用標註（Citation）

- `add_citation_boxes(response, candidates)`：
  - 從每個 candidate 的 `path` 中抓 `page_start` / `page_end`。
  - 去重後依頁碼排序。
  - 在回應尾端附上：
    - `【引用來源】第 X 頁、第 X–Y 頁、...`

---

## 九、現有狀態與策略總表（摘要）

為方便快速查閱，以下是目前系統中「已定義的狀態與策略」列表：

- **領域（Domains）**：10 個  
  - 粗大動作、精細動作、感覺統合、口腔動作、情緒行為與社會適應功能、吞嚥功能、口語理解、口語表達、說話、認知功能

- **Task 類型**：7 個  
  - T1_report_overview（報告導覽／摘要）  
  - T2_score_interpretation（分數／量表解讀）  
  - T3_clinical_to_daily（臨床描述轉日常）  
  - T4_prioritization（能力剖面／優先順序）  
  - T5_coaching（訓練教練／在家怎麼做）  
  - T6_decision_monitoring（決策／追蹤與成效）  
  - T_meta（行政／資源／隱私／溝通）

- **Scope 類型**：3 個（依規則在 DST 偵測，決定要檢索多少主題分支）  
  - S_overview（整體）→ 查整張圖  
  - S_domain（單領域）→ 查單一領域  
  - S_multi_domain（多領域）→ 查多個特定領域  
  - 改版後流程詳見：[docs/DIALOGUE_FLOW_AFTER_SCOPE_UPDATE.md](docs/DIALOGUE_FLOW_AFTER_SCOPE_UPDATE.md)

- **Semantic Flow 狀態**：3 個  
  - `continue`：C 高且 MT 高，強延續  
  - `shift_soft`：中度切換，同主題池內新問題  
  - `shift_hard`：強切換，新主題

- **檢索動作（retrieval_action）**：4 個  
  - `NARROW_GRAPH`：縮小檢索範圍  
  - `CONTEXT_FIRST`：優先使用上下文  
  - `WIDE_IN_DOMAIN`：在同一 macro-domain 內廣泛檢索  
  - `DUAL_OR_CLARIFY`：多路檢索或澄清

- **檢索模式（RetrievalMode）**：定義 5 種，目前實際使用：  
  - 使用中：`TOPIC_FOCUSED`  
  - 保留兼容：`GLOBAL_OVERVIEW`、`REPORT_OVERVIEW`、`DOMAIN`、`SCORE`、`META`

- **主題策略（TopicPolicy / DomainPolicy）**：定義多種，目前實際使用：  
  - 使用中：`MIX_TOPK`  
  - 保留兼容：`LOCK_TOP1`、`TOP2_UNION`、`UNLOCKED`、`SOFT_TOP1`、`UNLOCK_GLOBAL`

- **重要門檻與超參數（摘錄）**  
  - `DSTPolicyConfig.C_high_th = 0.55`  
  - `DSTPolicyConfig.C_soft_th = 0.45`  
  - `DSTPolicyConfig.MT_high_th = 0.50`  
  - `DSTPolicyConfig.MT_soft_th = 0.30`  
  - `DSTPolicyConfig.entropy_high_th = 0.8`（模糊判定與模糊延續門檻）  
  - `ambiguous_continuation_min_overlap = 0.5`（模糊延續時強制提升主題重疊度）  
  - 檢索總配額 `TOTAL_K = 10`  
  - Task 臨界值 `TASK_THRESHOLD = 0.1`  
  - Subdomain 機率下限 `SUBDOMAIN_PROB_THRESHOLD = 0.03`

以上說明涵蓋了目前專案中對話運作的完整管線與所有主要狀態／策略定義，可作為後續開發、調參與除錯的參考文件。

