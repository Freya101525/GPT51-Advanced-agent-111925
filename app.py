import os
import io
import time
import base64
import tempfile
from typing import List, Dict, Any, Tuple
from datetime import datetime
import streamlit as st
import yaml
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import pandas as pd
# Embedded modules (combined)
import pdfplumber
from pdf2image import convert_from_bytes
import pytesseract
from openai import OpenAI
import google.generativeai as genai
from xai_sdk import Client as XAIClient
from xai_sdk.chat import user as xai_user, system as xai_system
from anthropic import Anthropic
# ==================== THEME SYSTEM ====================
FLOWER_THEMES = {
    "櫻花 Cherry Blossom": {
        "primary": "#FFB7C5",
        "secondary": "#FFC0CB",
        "accent": "#FF69B4",
        "bg_light": "linear-gradient(135deg, #ffe6f0 0%, #fff5f8 50%, #ffe6f0 100%)",
        "bg_dark": "linear-gradient(135deg, #2d1b2e 0%, #3d2533 50%, #2d1b2e 100%)",
        "icon": "🌸"
    },
    "玫瑰 Rose": {
        "primary": "#E91E63",
        "secondary": "#F06292",
        "accent": "#C2185B",
        "bg_light": "linear-gradient(135deg, #fce4ec 0%, #fff 50%, #fce4ec 100%)",
        "bg_dark": "linear-gradient(135deg, #1a0e13 0%, #2d1420 50%, #1a0e13 100%)",
        "icon": "🌹"
    },
    "薰衣草 Lavender": {
        "primary": "#9C27B0",
        "secondary": "#BA68C8",
        "accent": "#7B1FA2",
        "bg_light": "linear-gradient(135deg, #f3e5f5 0%, #fff 50%, #f3e5f5 100%)",
        "bg_dark": "linear-gradient(135deg, #1a0d1f 0%, #2d1a33 50%, #1a0d1f 100%)",
        "icon": "💜"
    },
    "鬱金香 Tulip": {
        "primary": "#FF5722",
        "secondary": "#FF8A65",
        "accent": "#E64A19",
        "bg_light": "linear-gradient(135deg, #fbe9e7 0%, #fff 50%, #fbe9e7 100%)",
        "bg_dark": "linear-gradient(135deg, #1f0e0a 0%, #331814 50%, #1f0e0a 100%)",
        "icon": "🌷"
    },
    "向日葵 Sunflower": {
        "primary": "#FFC107",
        "secondary": "#FFD54F",
        "accent": "#FFA000",
        "bg_light": "linear-gradient(135deg, #fff9e6 0%, #fffef5 50%, #fff9e6 100%)",
        "bg_dark": "linear-gradient(135deg, #1f1a0a 0%, #332814 50%, #1f1a0a 100%)",
        "icon": "🌻"
    },
    "蓮花 Lotus": {
        "primary": "#E91E8C",
        "secondary": "#F48FB1",
        "accent": "#AD1457",
        "bg_light": "linear-gradient(135deg, #fce4f0 0%, #fff 50%, #fce4f0 100%)",
        "bg_dark": "linear-gradient(135deg, #1f0e1a 0%, #331826 50%, #1f0e1a 100%)",
        "icon": "🪷"
    },
    "蘭花 Orchid": {
        "primary": "#9C27B0",
        "secondary": "#CE93D8",
        "accent": "#6A1B9A",
        "bg_light": "linear-gradient(135deg, #f3e5f5 0%, #faf5ff 50%, #f3e5f5 100%)",
        "bg_dark": "linear-gradient(135deg, #1a0d1f 0%, #2d1a33 50%, #1a0d1f 100%)",
        "icon": "🌺"
    },
    "茉莉 Jasmine": {
        "primary": "#4CAF50",
        "secondary": "#81C784",
        "accent": "#388E3C",
        "bg_light": "linear-gradient(135deg, #e8f5e9 0%, #f1f8f1 50%, #e8f5e9 100%)",
        "bg_dark": "linear-gradient(135deg, #0a1f0d 0%, #14331a 50%, #0a1f0d 100%)",
        "icon": "🤍"
    },
    "牡丹 Peony": {
        "primary": "#E91E63",
        "secondary": "#F06292",
        "accent": "#C2185B",
        "bg_light": "linear-gradient(135deg, #fce4ec 0%, #fff 50%, #fce4ec 100%)",
        "bg_dark": "linear-gradient(135deg, #1f0e13 0%, #331826 50%, #1f0e13 100%)",
        "icon": "🌺"
    },
    "百合 Lily": {
        "primary": "#FFFFFF",
        "secondary": "#F5F5F5",
        "accent": "#E0E0E0",
        "bg_light": "linear-gradient(135deg, #fafafa 0%, #fff 50%, #fafafa 100%)",
        "bg_dark": "linear-gradient(135deg, #0d0d0d 0%, #1a1a1a 50%, #0d0d0d 100%)",
        "icon": "⚪"
    },
    "紫羅蘭 Violet": {
        "primary": "#673AB7",
        "secondary": "#9575CD",
        "accent": "#512DA8",
        "bg_light": "linear-gradient(135deg, #ede7f6 0%, #f8f5ff 50%, #ede7f6 100%)",
        "bg_dark": "linear-gradient(135deg, #0d0a1f 0%, #1a1433 50%, #0d0a1f 100%)",
        "icon": "💜"
    },
    "梅花 Plum Blossom": {
        "primary": "#E91E63",
        "secondary": "#F48FB1",
        "accent": "#C2185B",
        "bg_light": "linear-gradient(135deg, #fce4ec 0%, #fff5f8 50%, #fce4ec 100%)",
        "bg_dark": "linear-gradient(135deg, #1f0e13 0%, #2d1a20 50%, #1f0e13 100%)",
        "icon": "🌸"
    },
    "茶花 Camellia": {
        "primary": "#D32F2F",
        "secondary": "#EF5350",
        "accent": "#B71C1C",
        "bg_light": "linear-gradient(135deg, #ffebee 0%, #fff 50%, #ffebee 100%)",
        "bg_dark": "linear-gradient(135deg, #1f0a0a 0%, #330d0d 50%, #1f0a0a 100%)",
        "icon": "🌹"
    },
    "康乃馨 Carnation": {
        "primary": "#F06292",
        "secondary": "#F8BBD0",
        "accent": "#E91E63",
        "bg_light": "linear-gradient(135deg, #fce4ec 0%, #fff5f8 50%, #fce4ec 100%)",
        "bg_dark": "linear-gradient(135deg, #1f0e13 0%, #2d1a20 50%, #1f0e13 100%)",
        "icon": "💐"
    },
    "海棠 Begonia": {
        "primary": "#FF5252",
        "secondary": "#FF8A80",
        "accent": "#D50000",
        "bg_light": "linear-gradient(135deg, #ffebee 0%, #fff 50%, #ffebee 100%)",
        "bg_dark": "linear-gradient(135deg, #1f0a0a 0%, #330d0d 50%, #1f0a0a 100%)",
        "icon": "🌺"
    },
    "桂花 Osmanthus": {
        "primary": "#FF9800",
        "secondary": "#FFB74D",
        "accent": "#F57C00",
        "bg_light": "linear-gradient(135deg, #fff3e0 0%, #fffaf5 50%, #fff3e0 100%)",
        "bg_dark": "linear-gradient(135deg, #1f140a 0%, #332014 50%, #1f140a 100%)",
        "icon": "🟡"
    },
    "紫藤 Wisteria": {
        "primary": "#9C27B0",
        "secondary": "#BA68C8",
        "accent": "#7B1FA2",
        "bg_light": "linear-gradient(135deg, #f3e5f5 0%, #faf5ff 50%, #f3e5f5 100%)",
        "bg_dark": "linear-gradient(135deg, #1a0d1f 0%, #2d1a33 50%, #1a0d1f 100%)",
        "icon": "💜"
    },
    "水仙 Narcissus": {
        "primary": "#FFEB3B",
        "secondary": "#FFF59D",
        "accent": "#F9A825",
        "bg_light": "linear-gradient(135deg, #fffde7 0%, #fffff5 50%, #fffde7 100%)",
        "bg_dark": "linear-gradient(135deg, #1f1f0a 0%, #33330d 50%, #1f1f0a 100%)",
        "icon": "🌼"
    },
    "杜鵑 Azalea": {
        "primary": "#E91E63",
        "secondary": "#F06292",
        "accent": "#C2185B",
        "bg_light": "linear-gradient(135deg, #fce4ec 0%, #fff 50%, #fce4ec 100%)",
        "bg_dark": "linear-gradient(135deg, #1f0e13 0%, #2d1a20 50%, #1f0e13 100%)",
        "icon": "🌸"
    },
    "芙蓉 Hibiscus": {
        "primary": "#FF5722",
        "secondary": "#FF8A65",
        "accent": "#E64A19",
        "bg_light": "linear-gradient(135deg, #fbe9e7 0%, #fff 50%, #fbe9e7 100%)",
        "bg_dark": "linear-gradient(135deg, #1f0e0a 0%, #331814 50%, #1f0e0a 100%)",
        "icon": "🌺"
    }
}
TRANSLATIONS = {
    "zh_TW": {
        "title": "🌸 TFDA Agentic AI代理人輔助審查系統",
        "subtitle": "智慧文件分析與資料提取 AI 代理人平台",
        "theme_selector": "選擇花卉主題",
        "language": "語言",
        "dark_mode": "深色模式",
        "upload_tab": "1) 上傳與OCR",
        "preview_tab": "2) 預覽與編輯",
        "config_tab": "3) 代理設定",
        "execute_tab": "4) 執行",
        "dashboard_tab": "5) 儀表板",
        "notes_tab": "6) 審查筆記",
        "upload_pdf": "上傳 PDF 檔案",
        "ocr_mode": "OCR 模式",
        "ocr_lang": "OCR 語言",
        "page_range": "頁碼範圍",
        "start_ocr": "開始 OCR",
        "save_agents": "儲存 agents.yaml",
        "download_agents": "下載 agents.yaml",
        "reset_agents": "重置為預設",
        "providers": "API 供應商",
        "connected": "已連線",
        "not_connected": "未連線"
    },
    "en": {
        "title": "🌸 TFDA Agentic AI Assistance Review System",
        "subtitle": "Intelligent Document Analysis & Data Extraction AI Agent Platform",
        "theme_selector": "Select Floral Theme",
        "language": "Language",
        "dark_mode": "Dark Mode",
        "upload_tab": "1) Upload & OCR",
        "preview_tab": "2) Preview & Edit",
        "config_tab": "3) Agent Config",
        "execute_tab": "4) Execute",
        "dashboard_tab": "5) Dashboard",
        "notes_tab": "6) Review Notes",
        "upload_pdf": "Upload PDF File",
        "ocr_mode": "OCR Mode",
        "ocr_lang": "OCR Language",
        "page_range": "Page Range",
        "start_ocr": "Start OCR",
        "save_agents": "Save agents.yaml",
        "download_agents": "Download agents.yaml",
        "reset_agents": "Reset to Default",
        "providers": "API Providers",
        "connected": "Connected",
        "not_connected": "Not Connected"
    }
}
# ==================== LLM ROUTER ====================
ModelChoice = {
    "gpt-5-nano": "openai",
    "gpt-4o-mini": "openai",
    "gpt-4.1-mini": "openai",
    "gemini-2.5-flash": "gemini",
    "gemini-2.5-flash-lite": "gemini",
    "grok-4-fast-reasoning": "grok",
    "grok-3-mini": "grok",
    "claude-sonnet-4.5": "anthropic",
    "claude-sonnet-4-20250514": "anthropic",
    "claude-haiku-4.5": "anthropic",
}
class LLMRouter:
    def __init__(self):
        self._openai_client = None
        self._gemini_ready = False
        self._xai_client = None
        self._anthropic_client = None
        self._init_clients()
    def _init_clients(self):
        if os.getenv("OPENAI_API_KEY"):
            self._openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        if os.getenv("GEMINI_API_KEY"):
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            self._gemini_ready = True
        if os.getenv("XAI_API_KEY"):
            self._xai_client = XAIClient(api_key=os.getenv("XAI_API_KEY"), timeout=3600)
        if os.getenv("ANTHROPIC_API_KEY"):
            self._anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    def generate_text(self, model_name: str, messages: List[Dict], params: Dict) -> Tuple[str, Dict, str]:
        provider = ModelChoice.get(model_name, "openai")        
        try:
            if provider == "openai":
                if not self._openai_client:
                    raise Exception("OpenAI API not configured")
                return self._openai_chat(model_name, messages, params), {"total_tokens": self._estimate_tokens(messages)}, "OpenAI"
            elif provider == "gemini":
                if not self._gemini_ready:
                    raise Exception("Gemini API not configured")
                return self._gemini_chat(model_name, messages, params), {"total_tokens": self._estimate_tokens(messages)}, "Gemini"
            elif provider == "grok":
                if not self._xai_client:
                    raise Exception("Grok API not configured")
                return self._grok_chat(model_name, messages, params), {"total_tokens": self._estimate_tokens(messages)}, "Grok"
            elif provider == "anthropic":
                if not self._anthropic_client:
                    raise Exception("Anthropic API not configured")
                return self._anthropic_chat(model_name, messages, params), {"total_tokens": self._estimate_tokens(messages)}, "Anthropic"
            else:
                raise Exception(f"Unknown provider: {provider}")
        except Exception as e:
            # Return error message instead of crashing
            return f"Error: {str(e)}", {"total_tokens": 0}, provider.capitalize() if provider else "Unknown"
    def generate_vision(self, model_name: str, prompt: str, images: List) -> str:
        provider = ModelChoice.get(model_name, "openai")
        if provider == "gemini":
            return self._gemini_vision(model_name, prompt, images)
        elif provider == "openai":
            return self._openai_vision(model_name, prompt, images)
        elif provider == "anthropic":
            return self._anthropic_vision(model_name, prompt, images)
        return "Vision not supported"
    def _openai_chat(self, model: str, messages: List, params: Dict) -> str:
        resp = self._openai_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=params.get("temperature", 0.4),
            top_p=params.get("top_p", 0.95),
            max_tokens=params.get("max_tokens", 800)
        )
        return resp.choices[0].message.content
    def _gemini_chat(self, model: str, messages: List, params: Dict) -> str:
        mm = genai.GenerativeModel(model)
        sys = "\n".join([m["content"] for m in messages if m["role"] == "system"]).strip()
        usr = "\n".join([m["content"] for m in messages if m["role"] == "user"]).strip()
        final = (sys + "\n\n" + usr).strip() if sys else usr
        resp = mm.generate_content(final, generation_config=genai.types.GenerationConfig(
            temperature=params.get("temperature", 0.4),
            top_p=params.get("top_p", 0.95),
            max_output_tokens=params.get("max_tokens", 800)
        ))
        return resp.text
    def _grok_chat(self, model: str, messages: List, params: Dict) -> str:
        chat = self._xai_client.chat.create(model=model)
        for m in messages:
            if m["role"] == "system":
                chat.append(xai_system(m["content"]))
            elif m["role"] == "user":
                chat.append(xai_user(m["content"]))
        return chat.sample().content
    def _gemini_vision(self, model: str, prompt: str, images: List) -> str:
        mm = genai.GenerativeModel(model)
        parts = [prompt] + [genai.Image.from_pil(img) for img in images]
        return mm.generate_content(parts).text
    def _openai_vision(self, model: str, prompt: str, images: List) -> str:
        contents = [{"type": "text", "text": prompt}]
        for img in images:
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            contents.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
        resp = self._openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": contents}]
        )
        return resp.choices[0].message.content
    def _estimate_tokens(self, messages: List) -> int:
        return max(1, sum(len(m.get("content", "")) for m in messages) // 4)
    def _anthropic_chat(self, model: str, messages: List, params: Dict) -> str:
    # Check if client is initialized
        if not self._anthropic_client:
            raise Exception("Anthropic API not configured. Please add ANTHROPIC_API_KEY to environment variables.")
    
    # Convert messages to Anthropic format
        system_msgs = [m["content"] for m in messages if m["role"] == "system"]
        system_prompt = "\n\n".join(system_msgs) if system_msgs else ""
        
        anthropic_messages = []
        for m in messages:
            if m["role"] == "user":
                anthropic_messages.append({"role": "user", "content": m["content"]})
            elif m["role"] == "assistant":
                anthropic_messages.append({"role": "assistant", "content": m["content"]})
        
        # If no user messages, add the system content as user message
        if not anthropic_messages:
            anthropic_messages.append({"role": "user", "content": system_prompt})
            system_prompt = ""
        
        kwargs = {
            "model": model,
            "messages": anthropic_messages,
            "temperature": params.get("temperature", 0.4),
            "top_p": params.get("top_p", 0.95),
            "max_tokens": params.get("max_tokens", 800)
        }
        
        if system_prompt:
            kwargs["system"] = system_prompt
        
        response = self._anthropic_client.messages.create(**kwargs)
        return response.content[0].text

def _anthropic_vision(self, model: str, prompt: str, images: List) -> str:
    # Check if client is initialized
    if not self._anthropic_client:
        return "Anthropic API not configured. Please add ANTHROPIC_API_KEY."
    
    # Claude Haiku doesn't support vision
    if "haiku" in model.lower():
        return "Claude Haiku doesn't support vision. Please use Sonnet models for vision tasks."
    
    content = [{"type": "text", "text": prompt}]
    
    for img in images:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": b64
            }
        })
    
    try:
        response = self._anthropic_client.messages.create(
            model=model,
            messages=[{"role": "user", "content": content}],
            max_tokens=1024
        )
        return response.content[0].text
    except Exception as e:
        return f"Error in Anthropic vision processing: {str(e)}"        
# ==================== OCR FUNCTIONS ====================
def render_pdf_pages(pdf_bytes: bytes, dpi: int = 150, max_pages: int = 30) -> List[Tuple[int, Image.Image]]:
    pages = convert_from_bytes(pdf_bytes, dpi=dpi, first_page=1, last_page=None)
    return [(idx, im) for idx, im in enumerate(pages[:max_pages])]
def extract_text_python(pdf_bytes: bytes, selected_pages: List[int], ocr_language: str = "english") -> str:
    text_parts = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for i in selected_pages:
            if i < len(pdf.pages):
                txt = pdf.pages[i].extract_text() or ""
                if txt.strip():
                    text_parts.append(f"[PAGE {i+1} - TEXT]\n{txt.strip()}\n")
    lang = "eng" if ocr_language == "english" else "chi_tra"
    for p in selected_pages:
        ims = convert_from_bytes(pdf_bytes, dpi=220, first_page=p+1, last_page=p+1)
        if ims:
            t = pytesseract.image_to_string(ims[0], lang=lang)
            if t.strip():
                text_parts.append(f"[PAGE {p+1} - OCR]\n{t.strip()}\n")
    return "\n".join(text_parts).strip()
def extract_text_llm(page_images: List[Image.Image], model_name: str, router) -> str:
    prompt = "請將圖片中的文字完整轉錄（保持原文、段落與標點）。若有表格，請以Markdown表格呈現。"
    text_blocks = []
    for idx, im in enumerate(page_images):
        out = router.generate_vision(model_name, f"{prompt}\n頁面 {idx+1}：", [im])
        text_blocks.append(f"[PAGE {idx+1} - LLM OCR]\n{out}\n")
    return "\n".join(text_blocks).strip()
# ==================== APP CONFIG ====================
st.set_page_config(
    page_title="🌸 TFDA Agentic AI Assistance Review System",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)
# ==================== SESSION STATE ====================
if "theme" not in st.session_state:
    st.session_state.theme = "櫻花 Cherry Blossom"
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False
if "language" not in st.session_state:
    st.session_state.language = "zh_TW"
if "agents_config" not in st.session_state:
    st.session_state.agents_config = []
if "ocr_text" not in st.session_state:
    st.session_state.ocr_text = ""
if "page_images" not in st.session_state:
    st.session_state.page_images = []
if "agent_outputs" not in st.session_state:
    st.session_state.agent_outputs = []
if "selected_agent_count" not in st.session_state:
    st.session_state.selected_agent_count = 5
if "run_metrics" not in st.session_state:
    st.session_state.run_metrics = []
if "review_notes" not in st.session_state:
    st.session_state.review_notes = "# 審查筆記\n\n在這裡記錄您的審查筆記。支援 Markdown 格式。\n\n使用 HTML 標籤改變文字顏色，例如：<span style='color:red'>紅色文字</span>\n\n## 後續問題\n- 問題1？\n- 問題2？"
# ==================== DEFAULT FDA AGENTS ====================
DEFAULT_FDA_AGENTS = """agents: 
  - name: 申請資料提取器 
    description: 進行繁體中文摘要 
    system_prompt: | 
      你是一位醫療器材法規專家。根據提供的文件，進行繁體中文摘要in markdown in traditional chinese with keywords in coral color. Please also create a table include 20 key items。
      - 識別：廠商名稱、地址、品名、類別、證書編號、日期、機構 
      - 標註不確定項目，保留原文引用 
      - 以結構化格式輸出（表格或JSON） 
    user_prompt: "你是一位醫療器材法規專家。根據提供的文件，進行繁體中文摘要in markdown in traditional chinese with keywords in coral color. Please also create a table include 20 key items。" 
    model: claude-sonnet-4.5 
    temperature: 0 
    top_p: 0.9 
    max_tokens: 6000 
  - name: 合約資料分析師 
    description: 合約資料分析師
    system_prompt: | 
      合約資料分析師，請確認合約中包含以下內容，請摘要合約內容。 
      - 委託者及受託者之名稱及地址： 委託者(甲方)名稱、地址，受託者(乙方)名稱、地址
      - 託製造之合意：委託者義務、受託者義務。 
      - 委託製造之醫療器材分類分級品項：品項名稱：(舉例 M.5925 軟式隱形眼鏡)、管理等級：(舉例第二等級) 
      - 委託製造之製程：委託製程範圍：(舉例：全部製程委託製造、滅菌、原料準備、模具成型、鏡片加工、包裝、品質檢驗等全部製程。 
      - 委託者及受託者之權利義務：委託者權利義務：舉例：有權查核製造紀錄及品質管理文件。應提供必要之技術文件(MDF/DMR)及產品規格。應依約定支付製造費用。乙方所有生產製程應符合醫療器材品質管理系統準則(QMS)及相關法令要求。 
    user_prompt: "請確認合約中包含以下內容，請摘要合約內容 in markdown in traditional chinese with keywords in coral color" 
    model: gpt-4o-mini 
    temperature: 0.3 
    top_p: 0.9 
    max_tokens: 1200 
  - name: 醫療器材查驗登記形式審查分析師 
    description: 醫療器材查驗登記形式審查 
    system_prompt: | 
      你是醫療器材審查專家，請確認申請資料包含以下內容：。 
      - 類似品：是否檢附本部核准類似品之相關資料
      - 申請書：加蓋醫療器材商及負責人印鑑、載明產品中文及英文名稱、型號、規格、須與製售證明及授權書相符、載明申請醫療器材商名稱、地址、須與醫療器材商許可執照相符、載明製造業者之名稱、地址
    user_prompt: "請評估以下文件中的不良反應資訊：" 
    model: gpt-4o-mini 
    temperature: 0.3 
    top_p: 0.9 
    max_tokens: 1500 
  - name: 藥物交互作用分析器 
    description: 識別藥物-藥物、藥物-食物交互作用 
    system_prompt: | 
      你是臨床藥學專家，專注於交互作用分析。 
      - 識別：藥物-藥物、藥物-食物、藥物-疾病交互作用 
      - 評估臨床意義與處置建議 
      - 標註禁止併用與謹慎併用項目 
    user_prompt: "請分析以下文件的藥物交互作用：" 
    model: gpt-4o-mini 
    temperature: 0.3 
    top_p: 0.9 
    max_tokens: 1200 
  - name: 禁忌症與警語提取器 
    description: 提取禁忌症、警語、注意事項 
    system_prompt: | 
      你是藥品安全管理專家。 
      - 提取：絕對禁忌、相對禁忌、特殊警語 
      - 區分不同嚴重程度 
      - 標註特殊族群注意事項（孕婦、哺乳、兒童、老年） 
    user_prompt: "請提取以下文件的禁忌症與警語：" 
    model: gpt-4o-mini 
    temperature: 0.2 
    top_p: 0.9 
    max_tokens: 1000 
  - name: 藥動學參數提取器 
    description: 提取吸收、分布、代謝、排泄（ADME）資訊 
    system_prompt: | 
      你是臨床藥理學專家。 
      - 提取：生體可用率、半衰期、清除率、分布體積 
      - 識別代謝酵素（CYP450等）、排泄途徑 
      - 以表格呈現藥動學參數 
    user_prompt: "請提取以下文件的藥動學參數：" 
    model: gpt-4o-mini 
    temperature: 0.2 
    top_p: 0.9 
    max_tokens: 1000 
  - name: 臨床試驗資料分析器 
    description: 分析臨床試驗設計、結果、統計顯著性 
    system_prompt: | 
      你是臨床試驗專家。 
      - 提取：試驗設計（Phase I/II/III/IV）、受試者數、主要終點 
      - 分析：療效指標、安全性數據、統計顯著性 
      - 標註研究限制與偏差風險 
    user_prompt: "請分析以下臨床試驗資料：" 
    model: gpt-4o-mini 
    temperature: 0.3 
    top_p: 0.9 
    max_tokens: 1500 
  - name: 藥品許可證資訊提取器 
    description: 提取許可證字號、核准日期、廠商資訊 
    system_prompt: | 
      你是藥政法規專家。 
      - 提取：許可證字號、核准日期、有效期限 
      - 識別：製造商、進口商、國內代理商資訊 
      - 標註許可變更歷史 
    user_prompt: "請提取以下文件的許可證資訊：" 
    model: gpt-4o-mini 
    temperature: 0.2 
    top_p: 0.9 
    max_tokens: 800 
  - name: 仿單變更比對器 
    description: 比對仿單版本差異，識別重要變更 
    system_prompt: | 
      你是法規文件比對專家。 
      - 識別新舊版本差異（新增、刪除、修改） 
      - 標註重要安全性變更 
      - 以對照表呈現差異 
    user_prompt: "請比對以下文件的版本差異：" 
    model: gpt-4o-mini 
    temperature: 0.2 
    top_p: 0.9 
    max_tokens: 1200 
  - name: 特殊族群用藥分析器 
    description: 分析孕婦、哺乳、兒童、老年用藥安全性 
    system_prompt: | 
      你是特殊族群用藥專家。 
      - 評估：孕婦安全等級、哺乳期安全性 
      - 分析：兒童用藥、老年人劑量調整 
      - 標註肝腎功能不全用藥建議 
    user_prompt: "請分析以下特殊族群用藥資訊：" 
    model: gpt-4o-mini 
    temperature: 0.3 
    top_p: 0.9 
    max_tokens: 1200 
  - name: 藥品儲存與安定性分析器 
    description: 提取儲存條件、有效期限、安定性資料 
    system_prompt: | 
      你是藥品品質管理專家。 
      - 提取：儲存溫度、濕度、光線要求 
      - 識別：有效期限、開封後效期 
      - 標註特殊儲存注意事項 
    user_prompt: "請分析以下儲存與安定性資訊：" 
    model: gpt-4o-mini 
    temperature: 0.2 
    top_p: 0.9 
    max_tokens: 800 
  - name: 過量與中毒處置分析器 
    description: 分析藥品過量症狀與處置方式 
    system_prompt: | 
      你是臨床毒理學專家。 
      - 識別：過量症狀、中毒機轉、致死劑量 
      - 提取：解毒劑、緊急處置、支持療法 
      - 標註需監測的生理指標 
    user_prompt: "請分析以下過量與中毒處置資訊：" 
    model: gpt-4o-mini 
    temperature: 0.3 
    top_p: 0.9 
    max_tokens: 1000 
  - name: 藥品外觀辨識器 
    description: 提取藥品外觀特徵、辨識碼 
    system_prompt: | 
      你是藥品鑑別專家。 
      - 描述：形狀、顏色、大小、刻痕 
      - 提取：藥品辨識碼、包裝特徵 
      - 協助防偽辨識 
    user_prompt: "請提取以下藥品外觀資訊：" 
    model: gpt-4o-mini 
    temperature: 0.2 
    top_p: 0.9 
    max_tokens: 800 
  - name: 賦形劑分析器 
    description: 識別賦形劑成分與過敏原 
    system_prompt: | 
      你是藥劑學專家。 
      - 列出所有賦形劑成分 
      - 標註常見過敏原（乳糖、麩質等） 
      - 識別著色劑、防腐劑 
    user_prompt: "請分析以下賦形劑資訊：" 
    model: gpt-4o-mini 
    temperature: 0.2 
    top_p: 0.9 
    max_tokens: 800 
  - name: 用藥指導建議生成器 
    description: 生成病人用藥指導衛教資料 
    system_prompt: | 
      你是藥師衛教專家。 
      - 以淺顯易懂語言說明用法 
      - 提供服藥時間、飲食注意 
      - 標註應就醫的警訊症狀 
    user_prompt: "請生成以下藥品的病人用藥指導：" 
    model: gpt-4o-mini 
    temperature: 0.4 
    top_p: 0.9 
    max_tokens: 1000 
  - name: 法規符合性檢查器 
    description: 檢查文件是否符合FDA法規要求 
    system_prompt: | 
      你是藥政法規稽核專家。 
      - 檢查必要項目完整性 
      - 識別缺漏或不符合規定處 
      - 提供改善建議 
    user_prompt: "請檢查以下文件的法規符合性：" 
    model: gpt-4o-mini 
    temperature: 0.3 
    top_p: 0.9 
    max_tokens: 1200 
  - name: 風險效益評估器 
    description: 綜合評估藥品風險與效益 
    system_prompt: | 
      你是藥品風險管理專家。 
      - 量化：療效證據強度、不良反應風險 
      - 評估：風險效益比、適用族群 
      - 提供決策建議 
    user_prompt: "請評估以下藥品的風險效益：" 
    model: gpt-4o-mini 
    temperature: 0.4 
    top_p: 0.9 
    max_tokens: 1500 
  - name: 學名藥生體相等性分析器 
    description: 分析學名藥與原廠藥生體相等性 
    system_prompt: | 
      你是生體相等性評估專家。 
      - 提取：BE試驗設計、AUC、Cmax數據 
      - 評估：90%信賴區間、符合性 
      - 標註溶離曲線比對結果 
    user_prompt: "請分析以下生體相等性資料：" 
    model: gpt-4o-mini 
    temperature: 0.2 
    top_p: 0.9 
    max_tokens: 1000 
  - name: 藥品經濟學分析器 
    description: 分析藥品成本效益與健保給付 
    system_prompt: | 
      你是藥品經濟學專家。 
      - 評估：成本效益比、QALY、ICER 
      - 分析：健保給付條件、支付價格 
      - 比較同類藥品經濟性 
    user_prompt: "請分析以下藥品經濟學資料：" 
    model: gpt-4o-mini 
    temperature: 0.3 
    top_p: 0.9 
    max_tokens: 1200 
  - name: 藥品回收與下架分析器 
    description: 分析藥品回收原因與影響範圍 
    system_prompt: | 
      你是藥品安全監控專家。 
      - 識別：回收等級、原因、批號 
      - 評估：影響範圍、替代方案 
      - 提供處置建議 
    user_prompt: "請分析以下藥品回收資訊：" 
    model: gpt-4o-mini 
    temperature: 0.3 
    top_p: 0.9 
    max_tokens: 1000 
  - name: 上市後監測資料分析器 
    description: 分析真實世界數據與上市後安全性 
    system_prompt: | 
      你是藥物流行病學專家。 
      - 分析：不良事件通報、信號偵測 
      - 評估：長期安全性、罕見風險 
      - 識別需進一步研究的議題 
    user_prompt: "請分析以下上市後監測資料：" 
    model: gpt-4o-mini 
    temperature: 0.3 
    top_p: 0.9 
    max_tokens: 1200 
  - name: 藥品品質檢驗標準提取器 
    description: 提取品質規格與檢驗方法 
    system_prompt: | 
      你是藥品品管專家。 
      - 提取：含量規格、純度標準 
      - 識別：檢驗方法、接受標準 
      - 標註關鍵品質屬性 
    user_prompt: "請提取以下品質檢驗標準：" 
    model: gpt-4o-mini 
    temperature: 0.2 
    top_p: 0.9 
    max_tokens: 1000 
  - name: 製程與製造資訊分析器 
    description: 分析製造流程與GMP符合性 
    system_prompt: | 
      你是藥品製造專家。 
      - 描述：製程步驟、關鍵參數 
      - 評估：GMP符合性、品質控制 
      - 識別關鍵製程步驟 
    user_prompt: "請分析以下製程資訊：" 
    model: gpt-4o-mini 
    temperature: 0.3 
    top_p: 0.9 
    max_tokens: 1000 
  - name: 藥品分類與管制級別分析器 
    description: 判定藥品分類與管制等級 
    system_prompt: | 
      你是藥事法規分類專家。 
      - 判定：處方/指示/成藥分類 
      - 識別：管制藥品級別（1-4級） 
      - 說明管制原因與規定 
    user_prompt: "請分析以下藥品分類資訊：" 
    model: gpt-4o-mini 
    temperature: 0.2 
    top_p: 0.9 
    max_tokens: 800 
  - name: 國際藥典比對器 
    description: 比對各國藥典標準差異 
    system_prompt: | 
      你是國際藥典專家。 
      - 比對：USP、BP、EP、JP標準差異 
      - 識別：各國特殊要求 
      - 提供符合性建議 
    user_prompt: "請比對以下國際藥典標準：" 
    model: gpt-4o-mini 
    temperature: 0.3 
    top_p: 0.9 
    max_tokens: 1200 
  - name: 藥品標籤與說明書檢查器 
    description: 檢查標籤說明書格式與完整性 
    system_prompt: | 
      你是藥品標示審查專家。 
      - 檢查：必要資訊完整性、格式規範 
      - 識別：字體大小、警語標示 
      - 提供修改建議 
    user_prompt: "請檢查以下標籤說明書：" 
    model: gpt-4o-mini 
    temperature: 0.2 
    top_p: 0.9 
    max_tokens: 1000 
  - name: 藥品專利分析器 
    description: 分析藥品專利狀態與到期時間 
    system_prompt: | 
      你是藥品專利分析專家。 
      - 識別：成分專利、製程專利、用途專利 
      - 分析：專利到期時間、延長狀況 
      - 評估學名藥上市時機 
    user_prompt: "請分析以下藥品專利資訊：" 
    model: gpt-4o-mini 
    temperature: 0.3 
    top_p: 0.9 
    max_tokens: 1000 
  - name: 藥品命名規範檢查器 
    description: 檢查藥品命名是否符合規範 
    system_prompt: | 
      你是藥品命名審查專家。 
      - 檢查：與既有藥品相似度 
      - 評估：混淆風險、誤用可能 
      - 提供命名建議 
    user_prompt: "請檢查以下藥品命名：" 
    model: gpt-4o-mini 
    temperature: 0.3 
    top_p: 0.9 
    max_tokens: 800 
  - name: 臨床指引比對器 
    description: 比對藥品使用與臨床指引符合性 
    system_prompt: | 
      你是實證醫學專家。 
      - 比對：適應症與指引建議 
      - 評估：證據等級、建議強度 
      - 識別超適應症使用 
    user_prompt: "請比對以下臨床指引：" 
    model: gpt-4o-mini 
    temperature: 0.3 
    top_p: 0.9 
    max_tokens: 1200 
  - name: 綜合報告生成器 
    description: 整合所有分析結果生成完整報告 
    system_prompt: | 
      你是FDA文件整合專家。 
      - 彙整：前述所有代理的分析結果 
      - 生成：結構化完整報告 
      - 標註：重點發現、風險警示、建議事項 
      - 以專業格式輸出（含目錄、章節） 
    user_prompt: "請整合以下所有分析結果生成綜合報告：" 
    model: gpt-4o-mini 
    temperature: 0.4 
    top_p: 0.95 
    max_tokens: 2000"""
# ==================== LOAD/SAVE AGENTS ====================
def load_agents_yaml(yaml_text: str):
    try:
        data = yaml.safe_load(yaml_text)
        st.session_state.agents_config = data.get("agents", [])
        st.session_state.selected_agent_count = min(5, len(st.session_state.agents_config))
        st.session_state.agent_outputs = [
            {"input": "", "output": "", "time": 0.0, "tokens": 0, "provider": "", "model": ""}
            for _ in st.session_state.agents_config
        ]
        return True
    except Exception as e:
        st.error(f"YAML 載入失敗: {e}")
        return False
# ==================== THEME GENERATOR ====================
def generate_theme_css(theme_name: str, dark_mode: bool):
    theme = FLOWER_THEMES[theme_name]
    bg = theme["bg_dark"] if dark_mode else theme["bg_light"]
    text_color = "#FFFFFF" if dark_mode else "#1a1a1a"
    card_bg = "rgba(30, 30, 30, 0.85)" if dark_mode else "rgba(255, 255, 255, 0.85)"
    border_color = theme["accent"] if dark_mode else theme["primary"]
    return f""" 
    <style> 
        @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+TC:wght@300;400;500;700&display=swap'); 
        [data-testid="stAppViewContainer"] > .main {{ 
            background: {bg}; 
            font-family: 'Noto Sans TC', sans-serif; 
            color: {text_color}; 
        }} 
        .block-container {{ 
            padding-top: 2rem; 
            padding-bottom: 3rem; 
            max-width: 1400px; 
        }} 
        .wow-card {{ 
            background: {card_bg}; 
            backdrop-filter: blur(15px); 
            border: 2px solid {border_color}40; 
            border-radius: 20px; 
            padding: 1.5rem; 
            margin: 1rem 0; 
            box-shadow: 0 8px 32px rgba(0,0,0,0.1); 
            transition: all 0.3s ease; 
        }} 
        .wow-card:hover {{ 
            transform: translateY(-2px); 
            box-shadow: 0 12px 48px rgba(0,0,0,0.15); 
            border-color: {border_color}80; 
        }} 
        .pill {{ 
            display: inline-flex; 
            align-items: center; 
            gap: 8px; 
            background: {theme['primary']}20; 
            color: {theme['accent']}; 
            border: 2px solid {theme['primary']}40; 
            padding: 8px 16px; 
            border-radius: 999px; 
            font-weight: 600; 
            font-size: 0.95rem; 
            transition: all 0.3s ease; 
        }} 
        .pill:hover {{ 
            background: {theme['primary']}40; 
            transform: scale(1.05); 
        }} 
        .badge-ok {{ 
            background: rgba(0, 200, 83, 0.15); 
            border-color: #00C85380; 
            color: #00C853; 
        }} 
        .badge-warn {{ 
            background: rgba(255, 193, 7, 0.15); 
            border-color: #FFC10780; 
            color: #F9A825; 
        }} 
        .badge-err {{ 
            background: rgba(244, 67, 54, 0.15); 
            border-color: #F4433680; 
            color: #D32F2F; 
        }} 
        .agent-step {{ 
            border-left: 6px solid {theme['accent']}; 
            background: {card_bg}; 
            border-radius: 16px; 
            padding: 1.5rem; 
            margin: 1rem 0; 
            box-shadow: 0 4px 16px rgba(0,0,0,0.08); 
        }} 
        h1, h2, h3 {{ 
            color: {theme['accent']} !important; 
            font-weight: 700; 
        }} 
        .stButton > button {{ 
            background: linear-gradient(135deg, {theme['primary']}, {theme['secondary']}); 
            color: white; 
            border: none; 
            border-radius: 12px; 
            padding: 0.75rem 2rem; 
            font-weight: 600; 
            transition: all 0.3s ease; 
            box-shadow: 0 4px 16px {theme['primary']}40; 
        }} 
        .stButton > button:hover {{ 
            transform: translateY(-2px); 
            box-shadow: 0 8px 24px {theme['primary']}60; 
        }} 
        .stTextInput > div > div > input, .stTextArea > div > div > textarea, .stSelectbox > div > div {{ 
            background: {card_bg}; 
            border: 2px solid {border_color}40; 
            border-radius: 12px; 
            color: {text_color}; 
        }} 
        .stTabs [data-baseweb="tab-list"] {{ 
            gap: 8px; 
            background: {card_bg}; 
            border-radius: 16px; 
            padding: 0.5rem; 
        }} 
        .stTabs [data-baseweb="tab"] {{ 
            border-radius: 12px; 
            color: {text_color}; 
            font-weight: 500; 
        }} 
        .stTabs [aria-selected="true"] {{ 
            background: linear-gradient(135deg, {theme['primary']}, {theme['secondary']}); 
            color: white; 
        }} 
        .metric-card {{ 
            background: {card_bg}; 
            border: 2px solid {theme['primary']}40; 
            border-radius: 16px; 
            padding: 1.5rem; 
            text-align: center; 
            transition: all 0.3s ease; 
        }} 
        .metric-card:hover {{ 
            transform: scale(1.05); 
            border-color: {theme['accent']}; 
        }} 
        .metric-value {{ 
            font-size: 2.5rem; 
            font-weight: 700; 
            color: {theme['accent']}; 
            margin: 0.5rem 0; 
        }} 
        .metric-label {{ 
            font-size: 0.9rem; 
            color: {text_color}80; 
            font-weight: 500; 
        }} 
    </style> 
    """
# ==================== INITIALIZE ====================
router = LLMRouter()
# Load default agents if empty
if not st.session_state.agents_config:
    load_agents_yaml(DEFAULT_FDA_AGENTS)
# ==================== SIDEBAR ====================
with st.sidebar:
    t = TRANSLATIONS[st.session_state.language]
    st.markdown(f"### {t['theme_selector']}")
    new_theme = st.selectbox(
        "Theme",
        list(FLOWER_THEMES.keys()),
        index=list(FLOWER_THEMES.keys()).index(st.session_state.theme),
        format_func=lambda x: f"{FLOWER_THEMES[x]['icon']} {x}",
        label_visibility="collapsed"
    )
    if new_theme != st.session_state.theme:
        st.session_state.theme = new_theme
        st.rerun()
    col1, col2 = st.columns(2)
    with col1:
        new_dark = st.checkbox(t["dark_mode"], value=st.session_state.dark_mode)
        if new_dark != st.session_state.dark_mode:
            st.session_state.dark_mode = new_dark
            st.rerun()
    with col2:
        new_lang = st.selectbox(
            t["language"],
            ["zh_TW", "en"],
            index=0 if st.session_state.language == "zh_TW" else 1,
            format_func=lambda x: "繁體中文" if x == "zh_TW" else "English"
        )
        if new_lang != st.session_state.language:
            st.session_state.language = new_lang
            st.rerun()
    st.markdown("---")
    st.markdown(f"### 🔐 {t['providers']}")
    def show_provider_status(name: str, env_var: str):
        connected = bool(os.getenv(env_var))
        status = t["connected"] if connected else t["not_connected"]
        badge = "badge-ok" if connected else "badge-warn"
        st.markdown(f'<div class="pill {badge}">{name}: {status}</div>', unsafe_allow_html=True)
        if not connected:
            key = st.text_input(f"{name} Key", type="password", key=f"key_{env_var}")
            if key:
                os.environ[env_var] = key
                st.success(f"{name} {t['connected']}")
    show_provider_status("OpenAI", "OPENAI_API_KEY")
    show_provider_status("Gemini", "GEMINI_API_KEY")
    show_provider_status("Grok", "XAI_API_KEY")
    show_provider_status("Anthropic", "ANTHROPIC_API_KEY")
    st.markdown("---")
    st.markdown("### 🤖 Agents YAML")
    agents_text = st.text_area(
        "agents.yaml",
        value=yaml.dump({"agents": st.session_state.agents_config}, allow_unicode=True, sort_keys=False),
        height=400,
        label_visibility="collapsed"
    )
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        if st.button(t["save_agents"], use_container_width=True):
            if load_agents_yaml(agents_text):
                st.success("✅ Saved!")
    with col_b:
        st.download_button(
            t["download_agents"],
            data=agents_text,
            file_name=f"agents_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml",
            mime="text/yaml",
            use_container_width=True
        )
    with col_c:
        if st.button(t["reset_agents"], use_container_width=True):
            load_agents_yaml(DEFAULT_FDA_AGENTS)
            st.success("✅ Reset!")
            st.rerun()
# Apply theme
st.markdown(generate_theme_css(st.session_state.theme, st.session_state.dark_mode), unsafe_allow_html=True)
# ==================== HEADER ====================
t = TRANSLATIONS[st.session_state.language]
theme_icon = FLOWER_THEMES[st.session_state.theme]["icon"]
col1, col2, col3 = st.columns([1, 3, 1])
with col1:
    st.markdown(f'<div class="pill">{theme_icon} TFDA AI</div>', unsafe_allow_html=True)
with col2:
    st.title(t["title"])
    st.caption(t["subtitle"])
with col3:
    providers_ok = sum([
        bool(os.getenv("OPENAI_API_KEY")),
        bool(os.getenv("GEMINI_API_KEY")),
        bool(os.getenv("XAI_API_KEY")),
        bool(os.getenv("ANTHROPIC_API_KEY"))
    ])
    st.markdown(f""" 
        <div class="wow-card"> 
            <div class="metric-value">{providers_ok}/4</div> 
            <div class="metric-label">Active Providers</div> 
        </div> 
        """, unsafe_allow_html=True)
st.markdown("---")
# ==================== TABS ====================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    t["upload_tab"],
    t["preview_tab"],
    t["config_tab"],
    t["execute_tab"],
    t["dashboard_tab"],
    t["notes_tab"]
])
# Tab 1: Upload & OCR
with tab1:
    st.markdown('<div class="wow-card">', unsafe_allow_html=True)
    st.subheader(f"{theme_icon} {t['upload_pdf']}")
    uploaded = st.file_uploader(t["upload_pdf"], type=["pdf"], label_visibility="collapsed")
    col1, col2, col3 = st.columns(3)
    with col1:
        ocr_mode = st.selectbox(
            t["ocr_mode"],
            ["Python OCR (pdfplumber + Tesseract)", "LLM OCR (Vision model)"]
        )
    with col2:
        ocr_lang = st.selectbox(t["ocr_lang"], ["english", "traditional-chinese"])
    with col3:
        page_range_input = st.text_input(t["page_range"], value="1-5")
    if ocr_mode.startswith("LLM"):
        llm_ocr_model = st.selectbox("LLM Model", [
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
            "gpt-4o-mini",
            "claude-sonnet-4.5",
            "claude-haiku-4.5"
        ])
    if uploaded:
        pdf_bytes = uploaded.read()
        with st.spinner("Rendering pages..."):
            page_imgs = render_pdf_pages(pdf_bytes, dpi=140, max_pages=12)
        st.session_state.page_images = page_imgs
        st.caption(f"Preview (showing {len(page_imgs)} pages)")
        cols = st.columns(4)
        for i, (idx, im) in enumerate(page_imgs):
            cols[i % 4].image(im, caption=f"Page {idx+1}", use_column_width=True)
    if st.button(t["start_ocr"], type="primary", use_container_width=True):
        def parse_range(s: str, total: int) -> List[int]:
            pages = set()
            for part in s.replace("，", ",").split(","):
                if "-" in part:
                    a, b = map(int, part.split("-"))
                    pages.update(range(max(0, a-1), min(total, b)))
                else:
                    p = int(part) - 1
                    if 0 <= p < total:
                        pages.add(p)
            return sorted(list(pages))
        selected = parse_range(page_range_input, len(page_imgs))
        if selected:
            with st.spinner("Processing OCR..."):
                if ocr_mode.startswith("Python"):
                    text = extract_text_python(pdf_bytes, selected, ocr_lang)
                else:
                    text = extract_text_llm(
                        [page_imgs[i][1] for i in selected],
                        llm_ocr_model,
                        router
                    )
            st.session_state.ocr_text = text
            st.balloons()
            st.success("✅ OCR Complete!")
    st.markdown('</div>', unsafe_allow_html=True)
# Tab 2: Preview & Edit
with tab2:
    st.markdown('<div class="wow-card">', unsafe_allow_html=True)
    st.subheader(f"{theme_icon} Document Text")
    st.session_state.ocr_text = st.text_area(
        "Edit OCR output",
        value=st.session_state.ocr_text,
        height=500,
        label_visibility="collapsed"
    )
    with st.expander("🔍 Keyword Highlighter"):
        keywords = st.text_input("Keywords (comma-separated)", value="藥品,適應症,不良反應")
        if st.button("Highlight"):
            out = st.session_state.ocr_text
            for kw in keywords.split(","):
                kw = kw.strip()
                if kw:
                    out = out.replace(kw, f"**:blue[{kw}]**")
            st.markdown(out)
    st.markdown('</div>', unsafe_allow_html=True)
# Tab 3: Agent Config
with tab3:
    st.markdown('<div class="wow-card">', unsafe_allow_html=True)
    st.subheader(f"{theme_icon} Agent Configuration")
    st.session_state.selected_agent_count = st.slider(
        "Number of agents to use",
        1,
        len(st.session_state.agents_config),
        min(5, len(st.session_state.agents_config))
    )
    global_prompt = st.text_area(
        "Global System Prompt",
        height=150,
        value="""你是FDA文件分析專家，請遵循：1) 保持資訊準確性，引用原文時必須精確2) 結構化輸出（表格、JSON、清單）3) 標註不確定項目並說明理由4) 識別潛在風險與需注意事項"""
    )
    st.markdown("---")
    for i in range(st.session_state.selected_agent_count):
        agent = st.session_state.agents_config[i]
        with st.expander(f"### Agent {i+1}: {agent.get('name', 'Unnamed')}", expanded=(i==0)):
            st.markdown('<div class="agent-step">', unsafe_allow_html=True)
            col1, col2 = st.columns([2, 1])
            with col1:
                agent["system_prompt"] = st.text_area(
                    "System Prompt",
                    value=agent.get("system_prompt", ""),
                    height=150,
                    key=f"sys_{i}"
                )
            with col2:
                agent["model"] = st.selectbox(
                    "Model",
                    ["gpt-4o-mini", "gpt-5-nano", "gemini-2.5-flash", "gemini-2.5-flash-lite",
                     "grok-3-mini", "claude-sonnet-4.5", "claude-sonnet-4-20250514", "claude-haiku-4.5"],
                    index=0,
                    key=f"model_{i}"
                )
                agent["temperature"] = st.slider("Temp", 0.0, 2.0, float(agent.get("temperature", 0.3)), 0.1, key=f"temp_{i}")
                agent["max_tokens"] = st.number_input("Max tokens", 64, 8192, int(agent.get("max_tokens", 1000)), 64, key=f"max_{i}")
            st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
# Tab 4: Execute
with tab4:
    st.markdown('<div class="wow-card">', unsafe_allow_html=True)
    st.subheader(f"{theme_icon} Execute Agent Pipeline")
    if not st.session_state.ocr_text.strip():
        st.warning("⚠️ Please complete OCR first (Tab 1)")
    else:
        # Initialize outputs if needed
        if len(st.session_state.agent_outputs) < len(st.session_state.agents_config):
            st.session_state.agent_outputs = [
                {"input": "", "output": "", "time": 0.0, "tokens": 0, "provider": "", "model": ""}
                for _ in st.session_state.agents_config
            ]
        # Reset first agent input
        if st.button("🔄 Reset Agent 1 Input to OCR Text"):
            st.session_state.agent_outputs[0]["input"] = st.session_state.ocr_text
            st.success("✅ Reset!")
        st.markdown("---")
        # Agent pipeline
        for i in range(st.session_state.selected_agent_count):
            agent = st.session_state.agents_config[i]
            st.markdown(f'<div class="agent-step">', unsafe_allow_html=True)
            st.markdown(f"#### 🤖 Agent {i+1}: {agent.get('name', '')}")
            st.caption(agent.get('description', ''))
            with st.expander("📥 Input (editable)", expanded=(i==0)):
                default_input = st.session_state.ocr_text if i == 0 and not st.session_state.agent_outputs[i]["input"] else st.session_state.agent_outputs[i]["input"]
                st.session_state.agent_outputs[i]["input"] = st.text_area(
                    f"Agent {i+1} Input",
                    value=default_input,
                    height=200,
                    key=f"in_{i}",
                    label_visibility="collapsed"
                )
            col_run, col_pass = st.columns([1, 2])
            with col_run:
                if st.button(f"▶️ Execute Agent {i+1}", key=f"run_{i}", type="primary"):
                    with st.spinner(f"Agent {i+1} processing..."):
                        t0 = time.time()
                        messages = [
                            {"role": "system", "content": global_prompt},
                            {"role": "system", "content": agent.get("system_prompt", "")},
                            {"role": "user", "content": f"{agent.get('user_prompt', '')}\n\n{st.session_state.agent_outputs[i]['input']}"}
                        ]
                        params = {
                            "temperature": float(agent.get("temperature", 0.3)),
                            "top_p": float(agent.get("top_p", 0.95)),
                            "max_tokens": int(agent.get("max_tokens", 1000))
                        }
                        try:
                            output, usage, provider = router.generate_text(
                                agent.get("model", "gpt-4o-mini"),
                                messages,
                                params
                            )
                            elapsed = time.time() - t0
                            st.session_state.agent_outputs[i]["output"] = output
                            st.session_state.agent_outputs[i]["time"] = elapsed
                            st.session_state.agent_outputs[i]["tokens"] = usage.get("total_tokens", 0)
                            st.session_state.agent_outputs[i]["provider"] = provider
                            st.session_state.agent_outputs[i]["model"] = agent.get("model", "")
                            st.session_state.run_metrics.append({
                                "agent": agent.get("name", ""),
                                "latency": elapsed,
                                "tokens": usage.get("total_tokens", 0),
                                "provider": provider
                            })
                            st.success(f"✅ Completed in {elapsed:.2f}s | {usage.get('total_tokens', 0)} tokens")
                            st.balloons()
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
            with col_pass:
                if i < st.session_state.selected_agent_count - 1:
                    if st.button(f"➡️ Pass to Agent {i+2}", key=f"pass_{i}"):
                        st.session_state.agent_outputs[i+1]["input"] = st.session_state.agent_outputs[i]["output"]
                        st.success(f"✅ Passed to Agent {i+2}")
                        st.rerun()
            # Show output
            st.markdown("##### 📤 Output")
            output_text = st.session_state.agent_outputs[i]["output"]
            if output_text:
                # Metrics
                col_m1, col_m2, col_m3 = st.columns(3)
                with col_m1:
                    st.markdown(f'<div class="metric-card"><div class="metric-value">{st.session_state.agent_outputs[i]["time"]:.2f}s</div><div class="metric-label">Latency</div></div>', unsafe_allow_html=True)
                with col_m2:
                    st.markdown(f'<div class="metric-card"><div class="metric-value">{st.session_state.agent_outputs[i]["tokens"]}</div><div class="metric-label">Tokens</div></div>', unsafe_allow_html=True)
                with col_m3:
                    st.markdown(f'<div class="metric-card"><div class="metric-value">{st.session_state.agent_outputs[i]["provider"]}</div><div class="metric-label">Provider</div></div>', unsafe_allow_html=True)
                st.text_area(
                    f"Agent {i+1} Output",
                    value=output_text,
                    height=300,
                    key=f"out_{i}",
                    label_visibility="collapsed"
                )
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown("---")
        # Export options
        st.markdown("### 💾 Export Results")
        col_j, col_m, col_r = st.columns(3)
        with col_j:
            if st.button("📥 Download JSON", use_container_width=True):
                import json
                payload = {
                    "timestamp": datetime.now().isoformat(),
                    "theme": st.session_state.theme,
                    "ocr_text": st.session_state.ocr_text,
                    "agents": st.session_state.agents_config[:st.session_state.selected_agent_count],
                    "outputs": st.session_state.agent_outputs[:st.session_state.selected_agent_count]
                }
                st.download_button(
                    "Download JSON",
                    data=json.dumps(payload, ensure_ascii=False, indent=2),
                    file_name=f"fda_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
        with col_m:
            if st.button("📄 Download Markdown Report", use_container_width=True):
                report = f"# FDA Document Analysis Report\n\n"
                report += f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                report += f"**Theme:** {st.session_state.theme}\n\n"
                report += f"## OCR Text\n\n{st.session_state.ocr_text}\n\n"
                report += "---\n\n"
                for i in range(st.session_state.selected_agent_count):
                    agent = st.session_state.agents_config[i]
                    report += f"## Agent {i+1}: {agent.get('name', '')}\n\n"
                    report += f"**Description:** {agent.get('description', '')}\n\n"
                    report += f"**Model:** {st.session_state.agent_outputs[i]['model']}\n\n"
                    report += f"**Provider:** {st.session_state.agent_outputs[i]['provider']}\n\n"
                    report += f"**Processing Time:** {st.session_state.agent_outputs[i]['time']:.2f}s\n\n"
                    report += f"### Output\n\n{st.session_state.agent_outputs[i]['output']}\n\n"
                    report += "---\n\n"
                st.download_button(
                    "Download Markdown",
                    data=report,
                    file_name=f"fda_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown",
                    use_container_width=True
                )
        with col_r:
            restore_file = st.file_uploader("📤 Restore Session JSON", type=["json"], key="restore")
            if restore_file:
                import json
                data = json.loads(restore_file.read())
                st.session_state.ocr_text = data.get("ocr_text", "")
                st.session_state.agents_config = data.get("agents", [])
                st.session_state.agent_outputs = data.get("outputs", [])
                st.session_state.selected_agent_count = len(st.session_state.agents_config)
                st.success("✅ Session restored!")
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
# Tab 5: Dashboard
with tab5:
    st.markdown('<div class="wow-card">', unsafe_allow_html=True)
    st.subheader(f"{theme_icon} Analytics Dashboard")
    if not st.session_state.run_metrics:
        st.info("📊 No data yet. Execute agents in Tab 4 to see analytics.")
    else:
        df = pd.DataFrame(st.session_state.run_metrics)
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            total_time = df['latency'].sum()
            st.markdown(f'<div class="metric-card"><div class="metric-value">{total_time:.2f}s</div><div class="metric-label">Total Time</div></div>', unsafe_allow_html=True)
        with col2:
            total_tokens = df['tokens'].sum()
            st.markdown(f'<div class="metric-card"><div class="metric-value">{total_tokens:,}</div><div class="metric-label">Total Tokens</div></div>', unsafe_allow_html=True)
        with col3:
            avg_latency = df['latency'].mean()
            st.markdown(f'<div class="metric-card"><div class="metric-value">{avg_latency:.2f}s</div><div class="metric-label">Avg Latency</div></div>', unsafe_allow_html=True)
        with col4:
            agents_run = len(df)
            st.markdown(f'<div class="metric-card"><div class="metric-value">{agents_run}</div><div class="metric-label">Agents Run</div></div>', unsafe_allow_html=True)
        st.markdown("---")
        # Charts
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            fig1 = px.bar(
                df,
                x="agent",
                y="latency",
                color="provider",
                title="Agent Latency (seconds)",
                color_discrete_map={
                    "OpenAI": "#10a37f",
                    "Gemini": "#4285f4",
                    "Grok": "#ff6b6b",
                    "Anthropic": "#d97757"
                }
            )
            fig1.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color=FLOWER_THEMES[st.session_state.theme]["accent"])
            )
            st.plotly_chart(fig1, use_container_width=True)
        with col_c2:
            fig2 = px.bar(
                df,
                x="agent",
                y="tokens",
                color="provider",
                title="Token Usage by Agent",
                color_discrete_map={
                    "OpenAI": "#10a37f",
                    "Gemini": "#4285f4",
                    "Grok": "#ff6b6b",
                    "Anthropic": "#d97757"
                }
            )
            fig2.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color=FLOWER_THEMES[st.session_state.theme]["accent"])
            )
            st.plotly_chart(fig2, use_container_width=True)
        # Provider distribution
        st.markdown("### Provider Distribution")
        provider_counts = df['provider'].value_counts()
        fig3 = px.pie(
            values=provider_counts.values,
            names=provider_counts.index,
            title="API Calls by Provider",
            color_discrete_map={
                "OpenAI": "#10a37f",
                "Gemini": "#4285f4",
                "Grok": "#ff6b6b",
                "Anthropic": "#d97757"
            }
        )
        fig3.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color=FLOWER_THEMES[st.session_state.theme]["accent"])
        )
        st.plotly_chart(fig3, use_container_width=True)
        # Pipeline flow visualization
        st.markdown("### Pipeline Flow")
        try:
            import graphviz
            dot = graphviz.Digraph()
            dot.attr(bgcolor='transparent')
            dot.attr('node', shape='box', style='filled,rounded', fillcolor=FLOWER_THEMES[st.session_state.theme]["primary"]+'40', color=FLOWER_THEMES[st.session_state.theme]["accent"])
            for i, rec in enumerate(df.to_dict('records')):
                label = f"{i+1}. {rec['agent']}\\n{rec['provider']}\\n{rec['latency']:.2f}s | {rec['tokens']} tok"
                dot.node(f"a{i}", label)
                if i > 0:
                    dot.edge(f"a{i-1}", f"a{i}", color=FLOWER_THEMES[st.session_state.theme]["accent"])
            st.graphviz_chart(dot)
        except Exception as e:
            st.info(f"Graphviz visualization unavailable: {str(e)}")
        # Detailed table
        st.markdown("### Detailed Metrics")
        st.dataframe(
            df[['agent', 'provider', 'latency', 'tokens']].style.format({
                'latency': '{:.3f}s',
                'tokens': '{:,}'
            }),
            use_container_width=True
        )
    st.markdown('</div>', unsafe_allow_html=True)
# Tab 6: Review Notes
with tab6:
    st.markdown('<div class="wow-card">', unsafe_allow_html=True)
    st.subheader(f"{theme_icon} 審查筆記")
    st.info("在這裡編輯您的審查筆記。支援 Markdown 和 HTML 顏色標籤，例如 <span style='color:blue'>藍色文字</span>。筆記會自動儲存於會話中。")
    st.session_state.review_notes = st.text_area(
        "編輯筆記",
        value=st.session_state.review_notes,
        height=500,
        label_visibility="collapsed"
    )
    st.markdown("### 預覽筆記")
    st.markdown(st.session_state.review_notes, unsafe_allow_html=True)
    if st.button("產生後續問題建議"):
        with st.spinner("產生中..."):
            messages = [
                {"role": "system", "content": "你是審查專家，請根據提供的筆記生成 3-5 個後續問題，以 Markdown 清單格式輸出。"},
                {"role": "user", "content": st.session_state.review_notes}
            ]
            params = {"temperature": 0.5, "max_tokens": 500}
            output, _, _ = router.generate_text("gpt-4o-mini", messages, params)
            st.session_state.review_notes += f"\n\n## 後續問題建議（自動生成）\n{output}"
        st.success("✅ 已新增後續問題至筆記末尾！")
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
# ==================== FOOTER ====================
st.markdown("---")
st.markdown(f"""<div style="text-align: center; padding: 2rem; opacity: 0.7;"> 
    <p>{theme_icon} <strong>TFDA Agentic AI Assistance Review System</strong></p> 
    <p>Powered by OpenAI, Google Gemini, xAI Grok & Anthropic Claude • Built with Streamlit</p> 
    <p style="font-size: 0.8rem;">© 2024 • Theme: {st.session_state.theme}</p></div>""", unsafe_allow_html=True)