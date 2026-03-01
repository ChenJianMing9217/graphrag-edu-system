"""
LLM 生成模組
使用 LM Studio 的 OpenAI 兼容 API 生成回應
"""
from dataclasses import dataclass
from typing import List, Dict, Optional
from openai import OpenAI
from .prompt_manager import LLMPromptManager, LLMGenerationConfig


@dataclass
class LLMConfig:
    """LLM 基礎配置（API 相關）"""
    base_url: str = "https://task-wise-medieval-generated.trycloudflare.com/v1" #暫用cloudflare的代理
    api_key: str = "lm-studio"
    model: str = "qwen2.5-14b-instruct"


class LLMGenerator:
    """LLM 生成器"""
    
    def __init__(self, config: Optional[LLMConfig] = None):
        """
        初始化 LLM 生成器
        
        Args:
            config: LLM 基礎配置，如果為 None 則使用默認配置
        """
        self.config = config or LLMConfig()
        self.client = OpenAI(
            base_url=self.config.base_url,
            api_key=self.config.api_key
        )
        self.prompt_manager = LLMPromptManager()
    
    def generate_response(
        self,
        user_query: str,
        retrieved_context: List[Dict] = None,
        conversation_history: List[Dict] = None,
        system_prompt: Optional[str] = None,
        generation_config: Optional[LLMGenerationConfig] = None
    ) -> str:
        """
        生成 LLM 回應
        
        Args:
            user_query: 使用者查詢
            retrieved_context: 檢索到的上下文資料（可選）
            conversation_history: 對話歷史（可選）
            system_prompt: 系統提示詞（可選，如果提供則覆蓋 generation_config 中的提示詞）
            generation_config: 生成配置（可選，如果提供則使用此配置的參數和提示詞）
        
        Returns:
            LLM 生成的回應
        """
        # 使用 generation_config 或默認配置
        if generation_config is None:
            generation_config = LLMGenerationConfig()
        
        # 構建消息列表
        messages = []
        
        # 添加系統提示詞（優先使用傳入的 system_prompt，否則使用 generation_config 中的）
        final_system_prompt = system_prompt or generation_config.system_prompt_template
        if not final_system_prompt:
            final_system_prompt = "你是一位專業的早療系統助手，能夠根據評估報告和檢索到的相關資訊，為家長和治療師提供專業的建議和回答。請用友善、專業的語氣回答問題。"
        
        messages.append({"role": "system", "content": final_system_prompt})
        
        # 添加對話歷史
        if conversation_history:
            messages.extend(conversation_history)
        
        # 構建用戶查詢（包含檢索到的上下文）
        if generation_config.user_prompt_template:
            # 使用配置中的模板
            user_content = generation_config.user_prompt_template.format(
                query=user_query,
                context=self._format_retrieved_context(retrieved_context or [], generation_config) if retrieved_context else ""
            )
        else:
            # 使用 prompt_manager 構建
            # 從 generation_config 中提取模糊相關資訊
            is_ambiguous = generation_config.is_ambiguous if hasattr(generation_config, 'is_ambiguous') else False
            active_domains = generation_config.active_domains if hasattr(generation_config, 'active_domains') else None
            task_options = generation_config.task_options if hasattr(generation_config, 'task_options') else None
            
            user_content = self.prompt_manager.build_user_prompt(
                user_query,
                retrieved_context or [],
                generation_config,
                is_ambiguous=is_ambiguous,
                active_domains=active_domains or [],
                task_options=task_options or []
            )
        
        messages.append({"role": "user", "content": user_content})
        
        try:
            # 調用 LLM API（使用 generation_config 中的參數）
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=generation_config.temperature,
                max_tokens=generation_config.max_tokens,
                top_p=generation_config.top_p,
                frequency_penalty=generation_config.frequency_penalty,
                presence_penalty=generation_config.presence_penalty
            )
            
            # 提取回應內容
            generated_text = response.choices[0].message.content.strip()
            
            # 後處理：移除井號標題符號（#、##、###、#### 等）
            generated_text = self._remove_markdown_headers(generated_text)
            
            return generated_text
        
        except Exception as e:
            print(f"[LLM 生成錯誤] {e}")
            import traceback
            traceback.print_exc()
            # 返回錯誤提示
            return f"抱歉，生成回應時發生錯誤：{str(e)}"
    
    def _format_retrieved_context(self, retrieved_context: List[Dict], config: LLMGenerationConfig) -> str:
        """
        格式化檢索到的上下文（使用 prompt_manager）
        
        Args:
            retrieved_context: 檢索結果列表
            config: 生成配置
        
        Returns:
            格式化後的上下文文字
        """
        return self.prompt_manager._format_context_by_style(
            retrieved_context[:config.max_context_items],
            config.context_format_style
        )
    
    def _remove_markdown_headers(self, text: str) -> str:
        """
        移除 Markdown 標題符號（#、##、### 等）
        
        Args:
            text: 原始文字
            
        Returns:
            處理後的文字
        """
        import re
        
        if not text:
            return text
        
        # 移除行首的井號標題符號（#、##、###、####、#####、######）
        # 匹配模式：行首的 1-6 個井號，後面可能跟著空格
        pattern = r'^#{1,6}\s*'
        
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # 移除行首的井號標題符號
            cleaned_line = re.sub(pattern, '', line)
            cleaned_lines.append(cleaned_line)
        
        return '\n'.join(cleaned_lines)

