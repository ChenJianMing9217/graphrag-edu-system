"""
LLM 提示詞和參數管理器
根據 DST 和 Task 類型動態選擇提示詞和生成參數
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class LLMGenerationConfig:
    """針對不同 DST + Task 的 LLM 生成配置"""
    # 生成參數
    temperature: float = 0.2
    max_tokens: int = 2000
    top_p: float = 0.9
    frequency_penalty: float = 0.3  # 減少重複詞彙（範圍：-2.0 到 2.0，正值減少重複）
    presence_penalty: float = 0.1   # 減少重複主題（範圍：-2.0 到 2.0，正值減少重複）
    
    # 提示詞模板
    system_prompt_template: str = ""
    user_prompt_template: str = ""
    
    # 上下文處理
    max_context_items: int = 10
    context_format_style: str = "detailed"  # "detailed" | "concise" | "structured"
    
    # 回應風格
    response_style: str = "professional"  # "professional" | "friendly" | "concise" | "detailed" | "step_by_step" | "explanatory" | "comprehensive"
    include_examples: bool = False
    include_caution: bool = False  # 是否包含模糊度警告
    
    # 模糊相關資訊（用於 build_user_prompt）
    is_ambiguous: bool = False
    active_domains: List[str] = field(default_factory=list)
    task_options: List[str] = field(default_factory=list)


class LLMPromptManager:
    """管理不同 DST + Task 組合的提示詞和參數"""
    
    # Task A–M 名稱映射（中文，領域中性）
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
    
    # Scope 名稱映射（中文）：Scope 管「要檢索多少主題分支」
    SCOPE_NAME_ZH = {
        "S_overview": "Overview(整體)",
        "S_domain": "Domain(單領域)",
        "S_multi_domain": "Multi-Domain(多領域)",
    }
    
    def __init__(self):
        """初始化提示詞管理器"""
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
        domain_distribution: Optional[Dict[str, float]] = None
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
            LLMGenerationConfig
        """
        # 基礎配置（根據 semantic_flow）
        base_config = self._get_base_config_by_flow(semantic_flow, is_ambiguous)
        
        # 根據 retrieval_action 調整
        retrieval_config = self._get_config_by_retrieval_action(retrieval_action)
        
        # 根據 task 調整
        task_config = self._get_config_by_task(task_label, scope_label) if task_label else LLMGenerationConfig()
        
        # 根據特殊情況調整
        special_config = self._get_special_config(is_overview_query, is_multi_domain, is_ambiguous)
        
        # 合併配置（優先級：task > special > retrieval > base）
        final_config = self._merge_configs(
            base_config,
            retrieval_config,
            task_config,
            special_config
        )
        
        # 構建提示詞
        final_config.system_prompt_template = self.build_system_prompt(
            task_label, scope_label, top_domain, is_ambiguous, is_overview_query,
            active_domains or [], domain_distribution or {}
        )
        
        # 將模糊相關資訊添加到 config 中（用於 build_user_prompt）
        final_config.is_ambiguous = is_ambiguous
        final_config.active_domains = active_domains or []
        
        return final_config
    
    def _get_base_config_by_flow(self, semantic_flow: str, is_ambiguous: bool) -> LLMGenerationConfig:
        """根據 semantic_flow 獲取基礎配置"""
        if semantic_flow == "continue":
            return LLMGenerationConfig(
                temperature=0.15 if is_ambiguous else 0.2,
                max_tokens=1500,
                response_style="professional",
                frequency_penalty=0.4,  # 延續對話時更容易重複，提高懲罰
                presence_penalty=0.15
            )
        elif semantic_flow == "shift_soft":
            return LLMGenerationConfig(
                temperature=0.25,
                max_tokens=2000,
                response_style="friendly",
                frequency_penalty=0.3,  # 默認值
                presence_penalty=0.1
            )
        else:  # shift_hard
            return LLMGenerationConfig(
                temperature=0.3,
                max_tokens=2500,
                response_style="detailed",
                frequency_penalty=0.25,  # 新主題時重複風險較低，降低懲罰
                presence_penalty=0.05
            )
    
    def _get_config_by_retrieval_action(self, retrieval_action: str) -> LLMGenerationConfig:
        """根據 retrieval_action 獲取配置"""
        if retrieval_action == "NARROW_GRAPH":
            return LLMGenerationConfig(
                max_context_items=8,
                context_format_style="detailed"
            )
        elif retrieval_action == "CONTEXT_FIRST":
            return LLMGenerationConfig(
                max_context_items=10,
                context_format_style="structured"
            )
        elif retrieval_action == "WIDE_IN_DOMAIN":
            return LLMGenerationConfig(
                max_context_items=15,
                context_format_style="concise"
            )
        else:  # DUAL_OR_CLARIFY
            return LLMGenerationConfig(
                max_context_items=8,
                context_format_style="structured",
                include_caution=True
            )
    
    def _get_config_by_task(self, task_label: str, scope_label: Optional[str]) -> LLMGenerationConfig:
        """根據 task A–M 和 scope 獲取配置"""
        config = LLMGenerationConfig()
        # A–M 依類型分組給 LLM 參數
        if task_label == "A":
            config.temperature = 0.25
            config.max_tokens = 2500
            config.response_style = "comprehensive"
            config.max_context_items = 15
        elif task_label in ("B", "C"):
            config.temperature = 0.2
            config.max_tokens = 2000
            config.response_style = "explanatory"
            config.include_caution = True
        elif task_label == "D":
            config.temperature = 0.2
            config.max_tokens = 2000
            config.response_style = "structured"
        elif task_label in ("E", "F"):
            config.temperature = 0.15
            config.max_tokens = 2000
            config.response_style = "step_by_step"
            config.max_context_items = 8
        elif task_label in ("G", "L"):
            config.temperature = 0.2
            config.max_tokens = 2000
            config.response_style = "professional"
        elif task_label in ("H", "I", "J", "K", "M"):
            config.temperature = 0.2
            config.max_tokens = 1500
            config.response_style = "concise"
        else:
            config.temperature = 0.2
            config.max_tokens = 2000
            config.response_style = "professional"
        
        # 根據 scope 調整（三類：整體 / 單領域 / 多領域）
        if scope_label == "S_overview":
            config.max_tokens = max(config.max_tokens, 2500)
            config.response_style = "comprehensive"
        elif scope_label == "S_domain":
            config.max_tokens = max(config.max_tokens, 2000)
        elif scope_label == "S_multi_domain":
            config.max_tokens = max(config.max_tokens, 2200)
            config.context_format_style = "structured"
        
        return config
    
    def _get_special_config(self, is_overview_query: bool, is_multi_domain: bool, is_ambiguous: bool) -> LLMGenerationConfig:
        """根據特殊情況獲取配置"""
        config = LLMGenerationConfig()
        
        if is_overview_query:
            config.max_tokens = 3000
            config.max_context_items = 20
            config.context_format_style = "structured"
            config.response_style = "comprehensive"
        
        if is_multi_domain:
            config.context_format_style = "structured"
            config.max_context_items = max(config.max_context_items, 12)
        
        if is_ambiguous:
            config.temperature = min(config.temperature, 0.15)
            config.include_caution = True
        
        return config
    
    def _merge_configs(self, *configs: LLMGenerationConfig) -> LLMGenerationConfig:
        """合併多個配置（後面的優先級更高）"""
        merged = LLMGenerationConfig()
        
        for config in configs:
            if config.temperature > 0:
                merged.temperature = config.temperature
            if config.max_tokens > 0:
                merged.max_tokens = config.max_tokens
            if config.top_p > 0:
                merged.top_p = config.top_p
            if config.frequency_penalty != 0:  # 允許負值，所以用 != 0 判斷
                merged.frequency_penalty = config.frequency_penalty
            if config.presence_penalty != 0:  # 允許負值，所以用 != 0 判斷
                merged.presence_penalty = config.presence_penalty
            if config.max_context_items > 0:
                merged.max_context_items = config.max_context_items
            if config.context_format_style:
                merged.context_format_style = config.context_format_style
            if config.response_style:
                merged.response_style = config.response_style
            if config.include_examples:
                merged.include_examples = config.include_examples
            if config.include_caution:
                merged.include_caution = config.include_caution
        
        return merged
    
    def build_system_prompt(
        self,
        task_label: Optional[str],
        scope_label: Optional[str],
        top_domain: str,
        is_ambiguous: bool,
        is_overview_query: bool,
        active_domains: Optional[List[str]] = None,
        domain_distribution: Optional[Dict[str, float]] = None
    ) -> str:
        """構建系統提示詞"""
        base_prompt = "你是一位專業的早療系統助手，能夠根據評估報告和檢索到的相關資訊，為家長和治療師提供專業的建議和回答。\n\n"
        
        # 語言要求
        base_prompt += "【重要】請務必使用繁體中文回答，不要使用簡體中文或其他語言。\n\n"
        
        # 回應風格與可讀性建議（不強制特定 Markdown 格式）
        base_prompt += "【回應建議】\n"
        base_prompt += "1. 請讓回答結構清楚，可以視需要使用小標題、條列或表格來整理重點，方便家長閱讀。\n"
        base_prompt += "2. 使用友善、溫暖、親切的語氣，讓家長感到被理解和支持。\n"
        base_prompt += "3. 避免使用過多專業術語，必要時請用簡單易懂的方式解釋。\n"
        base_prompt += "4. 如果問題牽涉多個面向，可以先簡短總結，再分點說明細節。\n\n"
        
        # 內容處理建議
        base_prompt += "【回答內容建議】\n"
        base_prompt += "1. 先摘要與整合重點：不要直接複製檢索到的原始內容，請先理解核心意義，再用自己的話重新組織與表達。\n"
        base_prompt += "2. 以白話清楚說明：用簡單易懂的語句，必要時舉例或比喻，幫助家長快速掌握重點。\n"
        base_prompt += "3. 回答內容需緊扣使用者問題與對話脈絡，不要偏離主題。\n"
        base_prompt += "4. 若檢索結果涉及多個領域，請說明彼此關聯，並標示出優先關注的重點。\n\n"
        
        # 根據 task A–M 添加特定角色
        if task_label:
            task_name = self.TASK_NAME_ZH.get(task_label, task_label)
            base_prompt += f"\n\n你的專長是「{task_name}」，請根據使用者需求提供相應的專業建議與說明。"
        
        # 根據 scope 添加範圍說明（三類：整體 / 單領域 / 多領域）
        if scope_label:
            scope_name = self.SCOPE_NAME_ZH.get(scope_label, scope_label)
            if scope_label == "S_overview":
                base_prompt += f"\n\n本次查詢是「{scope_name}」，請整合多個領域的資訊，提供全面但結構化的回答。"
            elif scope_label == "S_domain":
                base_prompt += f"\n\n本次查詢聚焦「{scope_name}」（{top_domain}），請針對該領域深入回答。"
            elif scope_label == "S_multi_domain":
                base_prompt += f"\n\n本次查詢是「{scope_name}」，請分別針對各相關領域回答，必要時說明彼此關聯。"
        
        # 添加模糊度處理（重要：需要反問並引導）
        if is_ambiguous:
            base_prompt += "\n\n【模糊查詢處理】本次查詢可能較為模糊，請按照以下方式處理：\n"
            base_prompt += "1. 首先用親切、自然的語氣說明您理解查詢可能涉及多個面向，不要用制式化的開場白。\n"
            base_prompt += "2. **使用親切自然的反問**，避免制式化的問法。例如：\n"
            base_prompt += "   - 好的：「您是想了解 [領域] 的 [內容] 嗎？」或「關於 [領域]，您想了解哪個部分呢？」\n"
            base_prompt += "   - 好的：「您還想了解什麼呢？」或「還有其他想知道的嗎？」\n"
            base_prompt += "   - 避免：「請問您是問 [領域名稱] 相關的 [任務類型] 嗎？」（太制式化）\n"
            base_prompt += "3. 如果有多個可能的領域，用自然的方式逐一詢問，讓家長選擇。例如：\n"
            base_prompt += "   - 「您是想了解粗大動作的評估結果嗎？」\n"
            base_prompt += "   - 「還是想問精細動作的訓練建議？」\n"
            base_prompt += "4. 提供可能的任務類型選項時，用親切的語氣，例如：「您是想了解評估結果，還是想獲得訓練建議呢？」\n"
            base_prompt += "5. 可以使用表格呈現選項，但不要使用 - 或 * 作為列表符號，改用數字編號或表格。\n"
            base_prompt += "6. 語氣要親切、自然、溫暖，就像朋友在聊天一樣，不要讓家長感到被質疑或困惑。\n"
            
            # 如果有 active_domains，提供具體的領域選項和反問範例
            if active_domains and len(active_domains) > 1:
                domains_text = "、".join(active_domains[:5])  # 最多顯示5個
                base_prompt += f"\n可能的相關領域包括：{domains_text}。\n"
                base_prompt += "請用親切自然的語氣反問這些領域，例如：「您是想了解 [領域] 的 [內容] 嗎？」或「關於 [領域]，您想問什麼呢？」讓家長選擇。\n"
        
        # 添加整體查詢說明
        if is_overview_query:
            base_prompt += "\n\n本次查詢是整體性查詢，請整合多個領域的資訊，提供全面但結構化的回答。"
        
        base_prompt += "\n\n請用友善、專業、溫暖的語氣回答問題，讓家長感到被理解和支持。"
        
        return base_prompt
    
    def build_user_prompt(
        self,
        user_query: str,
        retrieved_context: List[Dict],
        config: LLMGenerationConfig,
        is_ambiguous: bool = False,
        active_domains: Optional[List[str]] = None,
        task_options: Optional[List[str]] = None
    ) -> str:
        """構建用戶提示詞（包含上下文）"""
        if not retrieved_context:
            # 如果沒有上下文且模糊，添加引導提示
            if is_ambiguous:
                guidance = "\n\n【引導提示】由於查詢較為模糊，請在回答中：\n"
                guidance += "1. **使用親切自然的反問**，避免制式化問法。例如：「您是想了解 [領域] 的 [內容] 嗎？」或「關於 [領域]，您想問什麼呢？」\n"
                if active_domains and len(active_domains) > 1:
                    guidance += f"2. 可能的領域包括：{', '.join(active_domains[:5])}，請用親切的語氣逐一詢問這些領域。\n"
                    guidance += "   例如：「您是想了解 [領域1] 的評估結果嗎？」或「還是想問 [領域2] 的訓練建議？」\n"
                guidance += "3. 提供任務類型選項時，用自然親切的語氣：\n"
                guidance += "   「您是想了解評估結果和分數嗎？」\n"
                guidance += "   「還是想獲得訓練建議？」\n"
                guidance += "   「或是想了解日常表現？」\n"
                guidance += "4. 回答時，請先簡短回應目前能確定的部分，再用親切語氣請家長補充更具體的需求。\n"
                guidance += "5. 請用自己的話摘要和重新組織內容，不要直接引用原始文字。\n"
                guidance += "6. 用白話文詳細解釋，篇幅以家長易於閱讀為主，避免過長或過於簡略。\n"
                guidance += "7. 請用親切、自然、溫暖的語氣，就像朋友在聊天一樣，引導家長更清楚地表達需求。\n"
                return user_query + guidance
            return user_query
        
        # 根據格式風格格式化上下文
        context_text = self._format_context_by_style(
            retrieved_context[:config.max_context_items],
            config.context_format_style
        )
        
        # 根據 response_style 調整提示詞格式
        if config.response_style == "step_by_step":
            prompt_template = """以下是相關的評估報告資訊：

{context}

基於以上資訊，請以「步驟式」的方式回答以下問題，提供具體、可操作的建議。請：
1. 先抓出 2–4 個最重要的重點，再用清楚的步驟說明家長可以怎麼做。
2. 使用白話、具體的描述，避免過多專業術語，必要時加上簡短解釋。
3. 回答內容需與使用者問題密切相關，避免延伸到不必要的細節。
4. 若有幾組建議可以條列或使用簡單表格整理，幫助家長快速理解。
{query}"""
        elif config.response_style == "explanatory":
            prompt_template = """以下是相關的評估報告資訊：

{context}

基於以上資訊，請詳細解釋以下問題，說明分數或觀察結果的意義與所在位置。請：
1. 先用 1–2 句給出整體結論，再分段或分點說明細節。
2. 解釋分數代表的水準（例如落在同齡大約哪個範圍）、可能的優勢與需要關注的地方。
3. 盡量用家長聽得懂的語言，必要時可舉例說明在日常生活中可能看到的表現。
4. 如有需要比較不同量表或不同時間點，可以用條列或表格整理。
{query}"""
        elif config.response_style == "comprehensive":
            prompt_template = """以下是相關的評估報告資訊：

{context}

基於以上資訊，請全面但精簡地回答以下問題，整合多個相關面向。請：
1. 先給出一段簡短總結，說明整體狀況與關鍵發現。
2. 依照重要性分段或分點整理不同領域的重點（例如：粗大動作、精細動作、語言等）。
3. 若有建議或後續方向，請明確指出優先順序與可能的下一步。
4. 若有需要比較的資訊，可以用簡單清單或表格協助整理，但不強制特定格式。
{query}"""
        else:
            prompt_template = """以下是相關的評估報告資訊：

{context}

基於以上資訊，請回答以下問題。請：
1. 先抓出與問題最相關的 2–3 個重點，清楚說明給家長聽。
2. 使用白話、自然的語氣回覆，避免過度制式或過於學術的表達。
3. 回答時緊扣使用者問題與目前對話脈絡，避免無關延伸。
4. 如有多個面向需要說明，可以視情況使用條列或表格整理重點。
{query}"""
        
        user_prompt = prompt_template.format(context=context_text, query=user_query)
        
        # 如果模糊，添加引導提示
        if is_ambiguous:
            guidance = "\n\n【引導提示】由於查詢較為模糊，請在回答中：\n"
            guidance += "1. **使用親切自然的反問**，避免制式化問法。例如：「您是想了解 [領域] 的 [內容] 嗎？」或「關於 [領域]，您想問什麼呢？」\n"
            if active_domains and len(active_domains) > 1:
                guidance += f"2. 可能的領域包括：{', '.join(active_domains[:5])}，請用親切的語氣逐一詢問這些領域。\n"
                guidance += "   例如：「您是想了解 [領域1] 的評估結果嗎？」或「還是想問 [領域2] 的訓練建議？」\n"
            guidance += "3. 提供任務類型選項時，用自然親切的語氣引導家長更清楚地表達需求。\n"
            guidance += "4. 請用自己的話摘要和重新組織內容，不要直接引用原始文字。\n"
            guidance += "5. 用白話文詳細解釋，篇幅以家長容易閱讀為主，不必刻意拉長。\n"
            user_prompt += guidance
        
        return user_prompt
    
    def _format_context_by_style(self, retrieved_context: List[Dict], style: str) -> str:
        """根據格式風格格式化上下文"""
        if style == "detailed":
            return self._format_detailed_context(retrieved_context)
        elif style == "concise":
            return self._format_concise_context(retrieved_context)
        else:  # structured
            return self._format_structured_context(retrieved_context)
    
    def _format_detailed_context(self, retrieved_context: List[Dict]) -> str:
        """詳細格式：包含完整資訊"""
        formatted_parts = []
        for i, item in enumerate(retrieved_context, 1):
            path = item.get('path', {})
            text = item.get('text', '')
            score = item.get('score', 0.0)
            
            subdomain = path.get('subdomain', 'N/A')
            section_type = path.get('section_type', 'N/A')
            section_name = path.get('section_name', 'N/A')
            
            formatted_parts.append(
                f"[資料 {i}] 領域：{subdomain} | 類型：{section_type} | 名稱：{section_name} | 相關度：{score:.3f}\n"
                f"內容：{text[:600]}{'...' if len(text) > 600 else ''}\n"
            )
        return "\n".join(formatted_parts)
    
    def _format_concise_context(self, retrieved_context: List[Dict]) -> str:
        """簡潔格式：只包含關鍵資訊"""
        formatted_parts = []
        for i, item in enumerate(retrieved_context, 1):
            path = item.get('path', {})
            text = item.get('text', '')
            
            subdomain = path.get('subdomain', 'N/A')
            section_type = path.get('section_type', 'N/A')
            
            # 提取前 300 字
            formatted_parts.append(
                f"[{i}] {subdomain} - {section_type}: {text[:300]}{'...' if len(text) > 300 else ''}"
            )
        return "\n".join(formatted_parts)
    
    def _format_structured_context(self, retrieved_context: List[Dict]) -> str:
        """結構化格式：按領域分組"""
        # 按領域分組
        by_domain = {}
        for item in retrieved_context:
            path = item.get('path', {})
            subdomain = path.get('subdomain', 'N/A')
            if subdomain not in by_domain:
                by_domain[subdomain] = []
            by_domain[subdomain].append(item)
        
        formatted_parts = []
        for domain, items in by_domain.items():
            formatted_parts.append(f"\n【{domain}】")
            for i, item in enumerate(items, 1):
                path = item.get('path', {})
                text = item.get('text', '')
                section_type = path.get('section_type', 'N/A')
                section_name = path.get('section_name', 'N/A')
                
                formatted_parts.append(
                    f"  {i}. {section_type} - {section_name}:\n"
                    f"     {text[:400]}{'...' if len(text) > 400 else ''}\n"
                )
        
        return "\n".join(formatted_parts)

