"""
LLM 提示詞和參數管理器
根據 DST 和 Task 類型動態選擇提示詞和生成參數
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any


@dataclass
class LLMGenerationConfig:
    """針對不同 DST + Task 的 LLM 生成配置（可覆蓋欄位使用 Optional，None 代表「不覆蓋 / 使用預設」）"""

    # 生成參數（可覆蓋欄位）
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None

    # 提示詞模板（直接儲存組裝後字串，不參與可覆蓋合併）
    system_prompt_template: str = ""
    user_prompt_template: str = ""

    # 上下文處理（可覆蓋欄位）
    max_context_items: Optional[int] = None
    context_format_style: Optional[str] = None  # "detailed" | "concise" | "structured"

    # 回應風格（可覆蓋欄位）
    response_style: Optional[str] = None  # "professional" | "friendly" | "concise" | "detailed" | "step_by_step" | "explanatory" | "comprehensive" ...
    include_examples: Optional[bool] = None
    include_caution: Optional[bool] = None  # 是否包含模糊度警告

    # 模糊相關資訊（用於 build_user_prompt，不作為合併權重）
    is_ambiguous: bool = False
    active_domains: List[str] = field(default_factory=list)
    task_options: List[str] = field(default_factory=list)

    @classmethod
    def default_values(cls) -> "LLMGenerationConfig":
        """取得一份帶有系統預設值的新 Config，用於回填 None 欄位"""
        return cls(
            temperature=0.2,
            max_tokens=2000,
            top_p=0.9,
            frequency_penalty=0.3,   # 減少重複詞彙（範圍：-2.0 到 2.0，正值減少重複）
            presence_penalty=0.1,    # 減少重複主題（範圍：-2.0 到 2.0，正值減少重複）
            max_context_items=10,
            context_format_style="detailed",
            response_style="professional",
            include_examples=False,
            include_caution=False,
            is_ambiguous=False,
            active_domains=[],
            task_options=[],
        )

    def with_defaults(self) -> "LLMGenerationConfig":
        """
        回填所有可覆蓋欄位的 None 為預設值。
        - 外部使用時建議使用本方法，確保不會取得 None（行為與原本固定預設值版本相容）。
        """
        base = self.default_values()
        return LLMGenerationConfig(
            # 可覆蓋欄位：None 則回退到 base
            temperature=self.temperature if self.temperature is not None else base.temperature,
            max_tokens=self.max_tokens if self.max_tokens is not None else base.max_tokens,
            top_p=self.top_p if self.top_p is not None else base.top_p,
            frequency_penalty=(
                self.frequency_penalty
                if self.frequency_penalty is not None
                else base.frequency_penalty
            ),
            presence_penalty=(
                self.presence_penalty
                if self.presence_penalty is not None
                else base.presence_penalty
            ),
            max_context_items=(
                self.max_context_items
                if self.max_context_items is not None
                else base.max_context_items
            ),
            context_format_style=(
                self.context_format_style
                if self.context_format_style is not None
                else base.context_format_style
            ),
            response_style=(
                self.response_style
                if self.response_style is not None
                else base.response_style
            ),
            include_examples=(
                self.include_examples
                if self.include_examples is not None
                else base.include_examples
            ),
            include_caution=(
                self.include_caution
                if self.include_caution is not None
                else base.include_caution
            ),
            # 非可覆蓋欄位：維持原本行為
            system_prompt_template=self.system_prompt_template,
            user_prompt_template=self.user_prompt_template,
            is_ambiguous=self.is_ambiguous,
            active_domains=list(self.active_domains) if self.active_domains else [],
            task_options=list(self.task_options) if self.task_options else [],
        )


class LLMPromptManager:
    """管理不同 DST + Task 組合的提示詞和參數"""

    # Task A–M 名稱映射（中文，領域中性）——需保留
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

    # Scope 名稱映射（中文）：Scope 管「要檢索多少主題分支」——需保留
    SCOPE_NAME_ZH = {
        "S_overview": "Overview(整體)",
        "S_domain": "Domain(單領域)",
        "S_multi_domain": "Multi-Domain(多領域)",
    }

    # semantic_flow 對應的基礎 LLM 配置表
    _SEMANTIC_FLOW_CONFIG: Dict[str, Dict] = {
        "continue": {
            "temperature": 0.2,
            "max_tokens": 1500,
            "response_style": "professional",
            "frequency_penalty": 0.4,  # 延續對話時更容易重複，提高懲罰
            "presence_penalty": 0.15,
        },
        "shift_soft": {
            "temperature": 0.25,
            "max_tokens": 2000,
            "response_style": "friendly",
            "frequency_penalty": 0.3,
            "presence_penalty": 0.1,
        },
        "shift_hard": {
            "temperature": 0.3,
            "max_tokens": 2500,
            "response_style": "detailed",
            "frequency_penalty": 0.25,  # 新主題時重複風險較低，降低懲罰
            "presence_penalty": 0.05,
        },
    }

    # retrieval_action 對應的上下文配置表
    _RETRIEVAL_ACTION_CONFIG: Dict[str, Dict] = {
        "NARROW_GRAPH": {
            "max_context_items": 8,
            "context_format_style": "detailed",
        },
        "CONTEXT_FIRST": {
            "max_context_items": 10,
            "context_format_style": "structured",
        },
        "WIDE_IN_DOMAIN": {
            "max_context_items": 15,
            "context_format_style": "concise",
        },
        "DUAL_OR_CLARIFY": {
            "max_context_items": 8,
            "context_format_style": "structured",
            "include_caution": True,
        },
        "LOCAL_RESOURCE_CLARIFY": {
            "max_context_items": 0,
            "response_style": "friendly",
        },
        "LOCAL_RESOURCE_SEARCH": {
            "max_context_items": 15,
            "response_style": "detailed",
        },
    }

    # Task A–M 專屬配置表（不含 scope）
    _TASK_CONFIG: Dict[str, Dict] = {
        "A": {
            "temperature": 0.25,
            "max_tokens": 2500,
            "response_style": "comprehensive",
            "max_context_items": 15,
        },
        "B": {
            "temperature": 0.2,
            "max_tokens": 2000,
            "response_style": "explanatory",
            "include_caution": True,
        },
        "C": {
            "temperature": 0.2,
            "max_tokens": 2000,
            "response_style": "explanatory",
            "include_caution": True,
        },
        "D": {
            "temperature": 0.2,
            "max_tokens": 2000,
            "response_style": "structured",
        },
        "E": {
            "temperature": 0.15,
            "max_tokens": 2000,
            "response_style": "step_by_step",
            "max_context_items": 8,
        },
        "F": {
            "temperature": 0.15,
            "max_tokens": 2000,
            "response_style": "step_by_step",
            "max_context_items": 8,
        },
        "G": {
            "temperature": 0.2,
            "max_tokens": 2000,
            "response_style": "professional",
        },
        "L": {
            "temperature": 0.2,
            "max_tokens": 2000,
            "response_style": "professional",
        },
        "H": {
            "temperature": 0.2,
            "max_tokens": 1500,
            "response_style": "concise",
        },
        "I": {
            "temperature": 0.2,
            "max_tokens": 1500,
            "response_style": "concise",
        },
        "J": {
            "temperature": 0.2,
            "max_tokens": 1500,
            "response_style": "concise",
        },
        "K": {
            "temperature": 0.2,
            "max_tokens": 1500,
            "response_style": "concise",
        },
        "M": {
            "temperature": 0.2,
            "max_tokens": 1500,
            "response_style": "concise",
        },
        # 預設 Task 設定
        "_default": {
            "temperature": 0.2,
            "max_tokens": 2000,
            "response_style": "professional",
        },
    }

    # Scope 專屬配置表
    _SCOPE_CONFIG: Dict[str, Dict] = {
        "S_overview": {
            "max_tokens": 2500,
            "response_style": "comprehensive",
        },
        "S_domain": {
            "max_tokens": 2000,
        },
        "S_multi_domain": {
            "max_tokens": 2200,
            "context_format_style": "structured",
        },
    }

    # build_user_prompt 中，各種 response_style 對應的提示模板
    _USER_PROMPT_TEMPLATES: Dict[str, str] = {
        "step_by_step": """以下是相關的評估報告資訊：

{context}

基於以上資訊，請以「步驟式」的方式回答以下問題，提供具體、可操作的建議。請：
1. 先抓出 2–4 個最重要的重點，再用清楚的步驟說明家長可以怎麼做。
2. 使用白話、具體的描述，避免過多專業術語，必要時加上簡短解釋。
3. 回答內容需與使用者問題密切相關，避免延伸到不必要的細節。
4. 若有幾組建議可以條列或使用簡單表格整理，幫助家長快速理解。
{query}""",
        "explanatory": """以下是相關的評估報告資訊：

{context}

基於以上資訊，請詳細解釋以下問題，說明分數或觀察結果的意義與所在位置。請：
1. 先用 1–2 句給出整體結論，再分段或分點說明細節。
2. 解釋分數代表的水準（例如落在同齡大約哪個範圍）、可能的優勢與需要關注的地方。
3. 盡量用家長聽得懂的語言，必要時可舉例說明在日常生活中可能看到的表現。
4. 如有需要比較不同量表或不同時間點，可以用條列或表格整理。
{query}""",
        "comprehensive": """以下是相關的評估報告資訊：

{context}

基於以上資訊，請全面但精簡地回答以下問題，整合多個相關面向。請：
1. 先給出一段簡短總結，說明整體狀況與關鍵發現。
2. 依照重要性分段或分點整理不同領域的重點（例如：粗大動作、精細動作、語言等）。
3. 若有建議或後續方向，請明確指出優先順序與可能的下一步。
4. 若有需要比較的資訊，可以用簡單清單或表格協助整理，但不強制特定格式。
{query}""",
        # 預設模板：給未特別指定或其他風格共用
        "_default": """以下是相關的評估報告資訊：

{context}

基於以上資訊，請回答以下問題。請：
1. 先抓出與問題最相關的 2–3 個重點，清楚說明給家長聽。
2. 使用白話、自然的語氣回覆，避免過度制式或過於學術的表達。
3. 回答時緊扣使用者問題與目前對話脈絡，避免無關延伸。
4. 如有多個面向需要說明，可以視情況使用條列或表格整理重點。
{query}""",
    }

    def __init__(self):
        """初始化提示詞管理器（目前僅使用類別層級設定，不需額外初始化狀態）"""
        pass

    def get_config(
        self,
        semantic_flow: str,
        retrieval_action: str,
        task_label: Optional[str],
        scope_label: Optional[str],
        is_ambiguous: bool,
        is_overview_query: bool,
        is_multi_domain: bool,
        top_domain: str,
        active_domains: Optional[List[str]] = None,
        domain_distribution: Optional[Dict[str, float]] = None,
    ) -> LLMGenerationConfig:
        """
        根據 DST 和 Task 獲取配置

        Args:
            semantic_flow: "continue" | "shift_soft" | "shift_hard"
            retrieval_action: "NARROW_GRAPH" | "CONTEXT_FIRST" | "WIDE_IN_DOMAIN" | "DUAL_OR_CLARIFY"
            task_label: Task 類型（可選）
            scope_label: Scope 類型（可選）
            is_ambiguous: 是否模糊
            is_overview_query: 是否為整體查詢
            is_multi_domain: 是否多領域
            top_domain: 頂級領域
            active_domains: 活躍領域列表（可選，用於模糊引導）
            domain_distribution: 領域分布（可選，用於模糊引導）

        Returns:
            LLMGenerationConfig（已回填預設值）
        """
        # 基礎配置（根據 semantic_flow）
        base_config = self._get_base_config_by_flow(semantic_flow, is_ambiguous)

        # 根據 retrieval_action 調整
        retrieval_config = self._get_config_by_retrieval_action(retrieval_action)

        # 根據 task 調整
        task_config = (
            self._get_config_by_task(task_label, scope_label)
            if task_label
            else LLMGenerationConfig()
        )

        # 根據特殊情況調整
        special_config = self._get_special_config(
            is_overview_query=is_overview_query,
            is_multi_domain=is_multi_domain,
            is_ambiguous=is_ambiguous,
        )

        # 合併配置（優先級：task > special > retrieval > base）
        merged_config = self._merge_configs(
            base_config,
            retrieval_config,
            task_config,
            special_config,
        )

        # 在對外輸出前回填預設值，保留原先「欄位一定有值」的使用習慣
        final_config = merged_config.with_defaults()

        # 構建系統提示詞
        final_config.system_prompt_template = self.build_system_prompt(
            task_label,
            scope_label,
            top_domain,
            is_ambiguous,
            is_overview_query,
            active_domains or [],
            domain_distribution or {},
            retrieval_action=retrieval_action,
        )

        # 將模糊相關資訊添加到 config 中（用於 build_user_prompt）
        final_config.is_ambiguous = is_ambiguous
        final_config.active_domains = active_domains or []

        return final_config

    def _get_base_config_by_flow(
        self, semantic_flow: str, is_ambiguous: bool
    ) -> LLMGenerationConfig:
        """根據 semantic_flow 取得基礎配置，主體規則改為查表"""
        config_dict = self._SEMANTIC_FLOW_CONFIG.get(
            semantic_flow, self._SEMANTIC_FLOW_CONFIG["shift_hard"]
        )
        config = LLMGenerationConfig(**config_dict)

        # 「continue + 模糊」時，略微降低溫度以提高穩定性
        if semantic_flow == "continue" and is_ambiguous:
            config.temperature = 0.15

        return config

    def _get_config_by_retrieval_action(
        self, retrieval_action: str
    ) -> LLMGenerationConfig:
        """根據 retrieval_action 取得配置，使用查表避免多層 if-elif"""
        # 預設視為 DUAL_OR_CLARIFY 行為
        config_dict = self._RETRIEVAL_ACTION_CONFIG.get(
            retrieval_action, self._RETRIEVAL_ACTION_CONFIG["DUAL_OR_CLARIFY"]
        )
        return LLMGenerationConfig(**config_dict)

    def _get_config_by_task(
        self, task_label: str, scope_label: Optional[str]
    ) -> LLMGenerationConfig:
        """根據 task A–M 和 scope 取得配置，改為由 Task / Scope 兩個表組合"""
        # 1. 先從 Task 表取得設定
        task_dict = self._TASK_CONFIG.get(
            task_label, self._TASK_CONFIG["_default"]
        )
        task_config = LLMGenerationConfig(**task_dict)

        # 2. 再根據 Scope 表進一步疊加（若有）
        if scope_label:
            scope_dict = self._SCOPE_CONFIG.get(scope_label)
            if scope_dict:
                scope_config = LLMGenerationConfig(**scope_dict)
                task_config = self._merge_configs(task_config, scope_config)

        return task_config

    def _get_special_config(
        self,
        is_overview_query: bool,
        is_multi_domain: bool,
        is_ambiguous: bool,
    ) -> LLMGenerationConfig:
        """根據特殊情況（整體查詢、多領域、模糊）取得額外配置"""
        config = LLMGenerationConfig()

        # 整體查詢：偏向長輸出與結構化
        if is_overview_query:
            config.max_tokens = 3000
            config.max_context_items = 20
            config.context_format_style = "structured"
            config.response_style = "comprehensive"

        # 多領域：強制使用結構化輸出，並增加上下文數量下限
        if is_multi_domain:
            config.context_format_style = "structured"
            # 使用值本身作為「下限」，後續合併順序會依優先度覆蓋
            config.max_context_items = 12

        # 模糊查詢：鼓勵更穩定的輸出，並附帶警示
        if is_ambiguous:
            config.temperature = 0.15
            config.include_caution = True

        return config

    def _merge_configs(self, *configs: LLMGenerationConfig) -> LLMGenerationConfig:
        """
        合併多個配置（後面的優先級更高）

        - 使用 None 代表「不設定 / 不覆蓋」，因此：
          * 0、False、空字串 "" 都會被視為「有效的覆蓋值」並被保留。
        """
        merged = LLMGenerationConfig()

        # 定義會參與「可覆蓋」合併的欄位名稱
        overridable_fields = (
            "temperature",
            "max_tokens",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
            "max_context_items",
            "context_format_style",
            "response_style",
            "include_examples",
            "include_caution",
        )

        for config in configs:
            if config is None:
                continue
            for field_name in overridable_fields:
                value = getattr(config, field_name)
                # 僅當 value 不是 None 時才覆蓋（允許 0 / False / ""）
                if value is not None:
                    setattr(merged, field_name, value)

        return merged

    # === System Prompt 組裝 ===

    def build_system_prompt(
        self,
        task_label: Optional[str],
        scope_label: Optional[str],
        top_domain: str,
        is_ambiguous: bool,
        is_overview_query: bool,
        active_domains: Optional[List[str]] = None,
        domain_distribution: Optional[Dict[str, float]] = None,
        retrieval_action: Optional[str] = None,
    ) -> str:
        """
        構建系統提示詞

        - 改為以多個常數片段與條件片段組裝，而不是大量 base_prompt +=
        """
        # 核心固定片段
        intro = (
            "你是一位專業的早療系統助手，能夠根據評估報告和檢索到的相關資訊，"
            "為家長和治療師提供專業的建議和回答。\n\n"
        )

        language_requirement = (
            "【重要】請務必使用繁體中文回答，不要使用簡體中文或其他語言。\n\n"
        )

        style_guidance = (
            "【回應建議】\n"
            "1. 請讓回答結構清楚，可以視需要使用小標題、條列或表格來整理重點，方便家長閱讀。\n"
            "2. 使用友善、溫暖、親切的語氣，讓家長感到被理解和支持。\n"
            "3. 避免使用過多專業術語，必要時請用簡單易懂的方式解釋。\n"
            "4. 如果問題牽涉多個面向，可以先簡短總結，再分點說明細節。\n\n"
        )

        content_guidance = (
            "【回答內容建議】\n"
            "1. 先摘要與整合重點：不要直接複製檢索到的原始內容，請先理解核心意義，再用自己的話重新組織與表達。\n"
            "2. 以白話清楚說明：用簡單易懂的語句，必要時舉例或比喻，幫助家長快速掌握重點。\n"
            "3. 回答內容需緊扣使用者問題與對話脈絡，不要偏離主題。\n"
            "4. 若檢索結果涉及多個領域，請說明彼此關聯，並標示出優先關注的重點。\n\n"
        )

        parts: List[str] = [
            intro,
            language_requirement,
            style_guidance,
            content_guidance,
        ]

        # Task A–M 角色片段
        if task_label:
            task_name = self.TASK_NAME_ZH.get(task_label, task_label)
            parts.append(
                f"\n\n你的專長是「{task_name}」，請根據使用者需求提供相應的專業建議與說明。"
            )

        # Scope 範圍片段
        if scope_label:
            scope_name = self.SCOPE_NAME_ZH.get(scope_label, scope_label)
            if scope_label == "S_overview":
                parts.append(
                    f"\n\n本次查詢是「{scope_name}」，請整合多個領域的資訊，提供全面但結構化的回答。"
                )
            elif scope_label == "S_domain":
                parts.append(
                    f"\n\n本次查詢聚焦「{scope_name}」（{top_domain}），請針對該領域深入回答。"
                )
            elif scope_label == "S_multi_domain":
                parts.append(
                    f"\n\n本次查詢是「{scope_name}」，請分別針對各相關領域回答，必要時說明彼此關聯。"
                )

        # 模糊查詢處理片段
        active_domains = active_domains or []
        if is_ambiguous:
            ambiguity_parts = [
                "\n\n【模糊查詢處理】本次查詢可能較為模糊，請按照以下方式處理：\n",
                "1. 首先用親切、自然的語氣說明您理解查詢可能涉及多個面向，不要用制式化的開場白。\n",
                "2. **使用親切自然的反問**，避免制式化的問法。例如：\n",
                "   - 好的：「您是想了解 [領域] 的 [內容] 嗎？」或「關於 [領域]，您想了解哪個部分呢？」\n",
                "   - 好的：「您還想了解什麼呢？」或「還有其他想知道的嗎？」\n",
                "   - 避免：「請問您是問 [領域名稱] 相關的 [任務類型] 嗎？」（太制式化）\n",
                "3. 如果有多個可能的領域，用自然的方式逐一詢問，讓家長選擇。例如：\n",
                "   - 「您是想了解粗大動作的評估結果嗎？」\n",
                "   - 「還是想問精細動作的訓練建議？」\n",
                "4. 提供可能的任務類型選項時，用親切的語氣，例如：「您是想了解評估結果，還是想獲得訓練建議呢？」\n",
                "5. 可以使用表格呈現選項，但不要使用 - 或 * 作為列表符號，改用數字編號或表格。\n",
                "6. 語氣要親切、自然、溫暖，就像朋友在聊天一樣，不要讓家長感到被質疑或困惑。\n",
            ]

            if active_domains and len(active_domains) > 1:
                domains_text = "、".join(active_domains[:5])  # 最多顯示 5 個
                ambiguity_parts.append(f"\n可能的相關領域包括：{domains_text}。\n")
                ambiguity_parts.append(
                    "請用親切自然的語氣反問這些領域，例如：「您是想了解 [領域] 的 [內容] 嗎？」或「關於 [領域]，您想問什麼呢？」讓家長選擇。\n"
                )

            parts.extend(ambiguity_parts)

        # 在地資源反問處理
        if retrieval_action == "LOCAL_RESOURCE_CLARIFY":
            parts.append(
                "\n\n【在地資源反問】目前的查詢涉及在地醫療或早療資源，但缺乏地區資訊。\n"
                "請用親切、溫暖且自然的語氣詢問用戶所在的縣市（例如：台北市、台中市等）。\n"
                "不要進行醫療建議或解釋報告，只需單純詢問地區即可。\n"
            )

        # 整體查詢補充說明
        if is_overview_query:
            parts.append(
                "\n\n本次查詢是整體性查詢，請整合多個領域的資訊，提供全面但結構化的回答。"
            )

        # 結尾語氣提醒
        parts.append(
            "\n\n請用友善、專業、溫暖的語氣回答問題，讓家長感到被理解和支持。"
        )

        return "".join(parts)

    # === User Prompt 組裝 ===

    def build_user_prompt(
        self,
        user_query: str,
        retrieved_context: List[Dict],
        config: LLMGenerationConfig,
        is_ambiguous: bool = False,
        active_domains: Optional[List[str]] = None,
        task_options: Optional[List[str]] = None,
    ) -> str:
        """
        構建用戶提示詞（包含上下文）

        - 先依 score 排序 retrieved_context，再截取 max_context_items
        - response_style 對應的提示改由模板字典管理
        - 模糊查詢引導文字抽成共用 helper
        """
        # 確保 config 欄位已回填預設值，外部行為維持原本「一定有值」的假設
        resolved_config = config.with_defaults()

        # 預先產生模糊查詢引導文字（可能為空字串）
        ambiguity_guidance = self._build_ambiguity_guidance(
            is_ambiguous=is_ambiguous,
            active_domains=active_domains or [],
            task_options=task_options or [],
        )

        # 沒有檢索上下文的情況
        if not retrieved_context:
            if is_ambiguous:
                return user_query + ambiguity_guidance
            return user_query

        # 先依 score 由高到低排序，再取前 max_context_items 筆
        def get_score(x):
            if isinstance(x, dict):
                return x.get("score", 0.0)
            return getattr(x, "score", 0.0)

        sorted_context = sorted(
            retrieved_context,
            key=get_score,
            reverse=True,
        )
        max_items = resolved_config.max_context_items or len(sorted_context)
        top_context = sorted_context[:max_items]

        # 根據格式風格格式化上下文（使用 formatter strategy）
        context_text = self._format_context_by_style(
            top_context,
            resolved_config.context_format_style or "structured",
        )

        # 根據 response_style 從模板表取得對應模板
        style_key = resolved_config.response_style or "_default"
        template = self._USER_PROMPT_TEMPLATES.get(
            style_key,
            self._USER_PROMPT_TEMPLATES["_default"],
        )

        user_prompt = template.format(context=context_text, query=user_query)

        # 若為模糊查詢，附加引導提示
        if is_ambiguous and ambiguity_guidance:
            user_prompt += ambiguity_guidance

        return user_prompt

    def _build_ambiguity_guidance(
        self,
        is_ambiguous: bool,
        active_domains: List[str],
        task_options: List[str],
    ) -> str:
        """
        建立「模糊查詢」時附加在 user prompt 後方的引導說明文字。

        - 抽成共用 helper，供有無上下文兩支流程共用。
        """
        if not is_ambiguous:
            return ""

        lines: List[str] = []
        lines.append("\n\n【引導提示】由於查詢較為模糊，請在回答中：\n")
        lines.append(
            "1. **使用親切自然的反問**，避免制式化問法。例如：「您是想了解 [領域] 的 [內容] 嗎？」或「關於 [領域]，您想問什麼呢？」\n"
        )

        if active_domains and len(active_domains) > 1:
            lines.append(
                f"2. 可能的領域包括：{', '.join(active_domains[:5])}，請用親切的語氣逐一詢問這些領域。\n"
            )
            lines.append(
                "   例如：「您是想了解 [領域1] 的評估結果嗎？」或「還是想問 [領域2] 的訓練建議？」\n"
            )

        # 若有 task_options，可鼓勵模型自然地拋出幾個任務型選項（不強制使用）
        if task_options:
            lines.append(
                "3. 可以自然地提供幾種可能的需求方向，協助家長聚焦，例如：\n"
            )
            preview = "、".join(task_options[:5])
            lines.append(f"   「目前看起來可能與：{preview} 有關，您比較想先了解哪一部分呢？」\n")
            extra_index_base = 4
        else:
            extra_index_base = 3

        lines.append(
            f"{extra_index_base}. 提供任務類型選項時，用自然親切的語氣引導家長更清楚地表達需求。\n"
        )
        lines.append(
            f"{extra_index_base + 1}. 請用自己的話摘要和重新組織內容，不要直接引用原始文字。\n"
        )
        lines.append(
            f"{extra_index_base + 2}. 用白話文詳細解釋，篇幅以家長容易閱讀為主，不必刻意拉長。\n"
        )
        lines.append(
            f"{extra_index_base + 3}. 請用親切、自然、溫暖的語氣，就像朋友在聊天一樣，引導家長更清楚地表達需求。\n"
        )

        return "".join(lines)

    # === Context Formatter（Strategy Mapping） ===

    def _format_context_by_style(
        self, retrieved_context: List[Dict], style: str
    ) -> str:
        """
        根據格式風格格式化上下文

        - 透過 mapping 對應到不同 formatter 策略，而非 if-elif 鏈。
        """
        formatter_mapping = {
            "detailed": self._format_detailed_context,
            "concise": self._format_concise_context,
            "structured": self._format_structured_context,
        }
        formatter = formatter_mapping.get(style, self._format_structured_context)
        return formatter(retrieved_context)

    def _get_val(self, item, key, default=None):
        if isinstance(item, dict):
            return item.get(key, default)
        return getattr(item, key, default)

    def _format_detailed_context(self, retrieved_context: List[Any]) -> str:
        """詳細格式：包含完整資訊"""
        formatted_parts = []
        for i, item in enumerate(retrieved_context, 1):
            # 支援 dict 或 CandidateNode
            path = self._get_val(item, "path", {})
            text = self._get_val(item, "text", "")
            score = self._get_val(item, "score", 0.0)

            subdomain = path.get("subdomain", "N/A") if isinstance(path, dict) else getattr(path, "subdomain", "N/A")
            section_type = path.get("section_type", "N/A") if isinstance(path, dict) else getattr(path, "section_type", "N/A")
            section_name = path.get("section_name", "N/A") if isinstance(path, dict) else getattr(path, "section_name", "N/A")

            formatted_parts.append(
                f"[資料 {i}] 領域：{subdomain} | 類型：{section_type} | 名稱：{section_name} | 相關度：{score:.3f}\n"
                f"內容：{text[:600]}{'...' if len(text) > 600 else ''}\n"
            )
        return "\n".join(formatted_parts)

    def _format_concise_context(self, retrieved_context: List[Any]) -> str:
        """簡潔格式：只包含關鍵資訊"""
        formatted_parts = []
        for i, item in enumerate(retrieved_context, 1):
            path = self._get_val(item, "path", {})
            text = self._get_val(item, "text", "")

            subdomain = path.get("subdomain", "N/A") if isinstance(path, dict) else getattr(path, "subdomain", "N/A")
            section_type = path.get("section_type", "N/A") if isinstance(path, dict) else getattr(path, "section_type", "N/A")

            formatted_parts.append(
                f"[{i}] {subdomain} - {section_type}: {text[:300]}{'...' if len(text) > 300 else ''}"
            )
        return "\n".join(formatted_parts)

    def _format_structured_context(self, retrieved_context: List[Any]) -> str:
        """結構化格式：按領域分組"""
        # 按領域分組
        by_domain: Dict[str, List[Any]] = {}
        for item in retrieved_context:
            path = self._get_val(item, "path", {})
            subdomain = self._get_val(path, "subdomain", "N/A")
            by_domain.setdefault(subdomain, []).append(item)

        formatted_parts: List[str] = []
        for domain, items in by_domain.items():
            formatted_parts.append(f"\n【{domain}】")
            for i, item in enumerate(items, 1):
                path = self._get_val(item, "path", {})
                text = self._get_val(item, "text", "")
                section_type = self._get_val(path, "section_type", "N/A")
                section_name = self._get_val(path, "section_name", "N/A")

                formatted_parts.append(
                    f"  {i}. {section_type} - {section_name}:\n"
                    f"     {text[:400]}{'...' if len(text) > 400 else ''}\n"
                )

        return "\n".join(formatted_parts)