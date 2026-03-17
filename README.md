# 早療智能助教系統 - 架構與詳細對話流程

本文件詳述系統設計、模組化架構以及每一輪對話的完整運作流程。

---

## 🏗 系統模組架構

### 1. 對話狀態追蹤 (Dialogue State Module - DST)
負責分析用戶意圖、上下文關聯性及主題變遷。
- **`semantic_flow_module_v2.py`**: 核心控制器，執行領域路由與狀態決策。
- **`dst_policy.py`**: 定義決策矩陣與檢索動作 (Action)。
- **`utils/region_extractor.py`**: [NEW] 獨立的台灣地區識別模組，支援全台 22 縣市正規化提取。

### 2. 檢索模組 V2 (Retrieval Module V2)
負責精準提取所需資料。
- **`strategy_mapper.py`**: 將 DST 信號映射為具體的檢索策略。
- **`execution_engine.py`**: 解耦的執行引擎，支援 Neo4j 與 MySQL 平行檢索。
- **`mysql_client.py`**: [NEW] 專用 MySQL 客戶端，負責 `sfaa_units` (社政機構) 與 `community_intervention_units` (社區據點) 雙表在地資源查詢。
- **`reranker.py`**: 基於語義、結構及上下文進行二次排序。

### 3. 生成模組 (LLM Generate Module)
- **`prompt_manager.py`**: 根據 DST 指標動態生成系統提示詞，控製口吻與檢索配額。
- **`llm_generator.py`**: 對接 vLLM 伺服器 (google/gemma-3-4b-it) 生成最終回覆。

---

## 🔄 詳細對話運作流程 (Dialogue Step-by-Step)

當使用者發送一則訊息至 `/api/chat` 時，系統遵循以下 6 個關鍵步驟：

### 步驟 1：輸入攔截與地區識別
- **程序**：透過 `region_extractor` 掃描用戶語句。
- **結果**：若發現「台中」、「台北」等關鍵字，會進行名稱正規化並標記為 `detected_region`。

### 步驟 2：對話狀態分析 (DST Core)
系統執行多層次語義分析：
1. **領域分析 (Domain)**：由 `DomainRouter` 計算各治療領域機率與分佈熵 (Entropy)。
2. **上下文相似度 (Context - C)**：計算當前輸入與上一輪對話的語義距離。
3. **主題追蹤 (Topic - MT)**：追蹤多個活躍領域的延續度。
4. **任務分類 (Task)**：識別 13 種任務類型 (Task A-M)，如解讀報告、尋找資源等。

### 步驟 3：策略決策 (Policy Decision)
根據 C、MT 及 Entropy 決定對話流向與檢索行為：
- **Semantic Flow**：`continue` (強延續), `shift_soft` (弱切換), `shift_hard` (硬切換)。
- **Retrieval Action**：
    - `LOCAL_RESOURCE_CLARIFY`: [NEW] 缺地區資訊時，觸發 LLM 引導反問。
    - `LOCAL_RESOURCE_SEARCH`: [NEW] 配合 MySQL 客戶端執行地區檢索。
    - `NARROW_GRAPH` / `WIDE_IN_DOMAIN`: 調整圖資料庫檢索寬度。

### 步驟 4：檢索策略映射與執行
- `StrategyMapper` 根據 Task 與 Action 組合 `SearchStrategy`（如：摘要檢索 + 單領域細節）。
- `ExecutionEngine` 判斷操作：
    - **Neo4j**: 執行 Cypher 查詢提取報告細節。
    - **MySQL**: 獨立開啟連線查詢 `sfaa_units` 地區機構。

### 步驟 5：結果排序 (Reranking)
- 將所有候選節點交由 `Reranker`。
- 結合 BGE-M3 餘弦相似度與「結構化權重」(如 Task A 優先看 Summary 標籤) 計算最終得分。

### 6. 生成與渲染 (Generation)
- `PromptManager` 根據 **Semantic Flow** 調整提示詞密度（如切換時加入轉場語句）。
- LLM 組合 Context 生成繁體中文回覆。
- 系統移除 Markdown 多餘標題，並由 `app.py` 附加數據源引用頁碼。