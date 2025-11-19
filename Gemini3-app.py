import os
import io
import time
import base64
import json
from typing import List, Dict, Any, Tuple
from datetime import datetime

import streamlit as st
import yaml
import plotly.express as px
import pandas as pd
from PIL import Image

# Embedded modules (combined)
import pdfplumber
from pdf2image import convert_from_bytes
import pytesseract

# AI Clients
from openai import OpenAI
import google.generativeai as genai
from xai_sdk import Client as XAIClient
from xai_sdk.chat import user as xai_user, system as xai_system
from anthropic import Anthropic

# ==================== CONFIGURATION & CONSTANTS ====================

# Updated Model IDs to current production versions
ModelChoice = {
    "gpt-4o-mini": "openai",
    "gpt-4o": "openai",
    "gemini-1.5-flash": "gemini",
    "gemini-1.5-pro": "gemini",
    "grok-beta": "grok",
    "claude-3-5-sonnet-20241022": "anthropic",
    "claude-3-5-haiku-20241022": "anthropic",
}

FLOWER_THEMES = {
    "櫻花 Cherry Blossom": {
        "primary": "#FFB7C5", "secondary": "#FFC0CB", "accent": "#FF69B4",
        "bg_light": "linear-gradient(135deg, #fff0f5 0%, #fff 100%)",
        "bg_dark": "linear-gradient(135deg, #2d1b2e 0%, #1a0f11 100%)",
        "icon": "🌸"
    },
    "玫瑰 Rose": {
        "primary": "#E91E63", "secondary": "#F06292", "accent": "#C2185B",
        "bg_light": "linear-gradient(135deg, #fff0f3 0%, #fff 100%)",
        "bg_dark": "linear-gradient(135deg, #2d0f16 0%, #1a0508 100%)",
        "icon": "🌹"
    },
    "薰衣草 Lavender": {
        "primary": "#9C27B0", "secondary": "#BA68C8", "accent": "#7B1FA2",
        "bg_light": "linear-gradient(135deg, #f8f0ff 0%, #fff 100%)",
        "bg_dark": "linear-gradient(135deg, #1f0f2d 0%, #0f051a 100%)",
        "icon": "💜"
    },
    "向日葵 Sunflower": {
        "primary": "#FFC107", "secondary": "#FFD54F", "accent": "#FFA000",
        "bg_light": "linear-gradient(135deg, #fffbf0 0%, #fff 100%)",
        "bg_dark": "linear-gradient(135deg, #2d260f 0%, #1a1505 100%)",
        "icon": "🌻"
    },
    "茉莉 Jasmine": {
        "primary": "#4CAF50", "secondary": "#81C784", "accent": "#388E3C",
        "bg_light": "linear-gradient(135deg, #f0fff2 0%, #fff 100%)",
        "bg_dark": "linear-gradient(135deg, #0f2d13 0%, #051a07 100%)",
        "icon": "🤍"
    },
    "海洋 Ocean": {
         "primary": "#03A9F4", "secondary": "#4FC3F7", "accent": "#0288D1",
         "bg_light": "linear-gradient(135deg, #f0faff 0%, #fff 100%)",
         "bg_dark": "linear-gradient(135deg, #0f1e2d 0%, #050e1a 100%)",
         "icon": "🌊"
    }
}

TRANSLATIONS = {
    "zh_TW": {
        "title": "TFDA Agentic AI 輔助審查系統",
        "subtitle": "智慧文件分析與資料提取自動化平台",
        "theme_selector": "介面主題",
        "language": "語言",
        "dark_mode": "深色模式",
        "upload_tab": "1. 上傳與辨識",
        "preview_tab": "2. 預覽與編輯",
        "config_tab": "3. 代理設定",
        "execute_tab": "4. 執行審查",
        "dashboard_tab": "5. 分析儀表板",
        "notes_tab": "6. 審查筆記",
        "upload_pdf": "上傳 PDF 文件",
        "ocr_mode": "OCR 模式",
        "ocr_lang": "語言",
        "page_range": "頁碼範圍",
        "start_ocr": "開始辨識",
        "save_agents": "儲存設定",
        "download_agents": "下載 YAML",
        "reset_agents": "重置預設",
        "providers": "API 金鑰設定",
        "connected": "已連線",
        "not_connected": "未設定",
        "run_all": "⚡ 自動執行所有代理人"
    },
    "en": {
        "title": "TFDA Agentic AI Review System",
        "subtitle": "Intelligent Document Analysis & Data Extraction Platform",
        "theme_selector": "Theme",
        "language": "Language",
        "dark_mode": "Dark Mode",
        "upload_tab": "1. Upload & OCR",
        "preview_tab": "2. Preview & Edit",
        "config_tab": "3. Agent Config",
        "execute_tab": "4. Execute",
        "dashboard_tab": "5. Dashboard",
        "notes_tab": "6. Notes",
        "upload_pdf": "Upload PDF",
        "ocr_mode": "OCR Mode",
        "ocr_lang": "Language",
        "page_range": "Page Range",
        "start_ocr": "Start OCR",
        "save_agents": "Save Config",
        "download_agents": "Download YAML",
        "reset_agents": "Reset Default",
        "providers": "API Keys",
        "connected": "Connected",
        "not_connected": "Not Set",
        "run_all": "⚡ Auto-Run All Agents"
    }
}

# ==================== LLM ROUTER ====================

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
            self._xai_client = XAIClient(api_key=os.getenv("XAI_API_KEY"))
        if os.getenv("ANTHROPIC_API_KEY"):
            self._anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def generate_text(self, model_name: str, messages: List[Dict], params: Dict) -> Tuple[str, Dict, str]:
        # Fallback mapping for generic names
        if model_name not in ModelChoice:
            if "gpt" in model_name: provider = "openai"
            elif "gemini" in model_name: provider = "gemini"
            elif "claude" in model_name: provider = "anthropic"
            elif "grok" in model_name: provider = "grok"
            else: provider = "openai"
        else:
            provider = ModelChoice[model_name]

        try:
            if provider == "openai":
                if not self._openai_client: raise Exception("OpenAI Key missing")
                return self._openai_chat(model_name, messages, params), {"total_tokens": self._estimate_tokens(messages)}, "OpenAI"
            elif provider == "gemini":
                if not self._gemini_ready: raise Exception("Gemini Key missing")
                return self._gemini_chat(model_name, messages, params), {"total_tokens": self._estimate_tokens(messages)}, "Gemini"
            elif provider == "grok":
                if not self._xai_client: raise Exception("xAI Key missing")
                return self._grok_chat(model_name, messages, params), {"total_tokens": self._estimate_tokens(messages)}, "Grok"
            elif provider == "anthropic":
                if not self._anthropic_client: raise Exception("Anthropic Key missing")
                return self._anthropic_chat(model_name, messages, params), {"total_tokens": self._estimate_tokens(messages)}, "Anthropic"
            else:
                raise Exception(f"Unknown provider")
        except Exception as e:
            return f"System Error: {str(e)}", {"total_tokens": 0}, "Error"

    def generate_vision(self, model_name: str, prompt: str, images: List) -> str:
        provider = ModelChoice.get(model_name, "openai")
        try:
            if provider == "gemini" and self._gemini_ready:
                return self._gemini_vision(model_name, prompt, images)
            elif provider == "openai" and self._openai_client:
                return self._openai_vision(model_name, prompt, images)
            elif provider == "anthropic" and self._anthropic_client:
                return self._anthropic_vision(model_name, prompt, images)
            return "Vision Provider not configured or supported."
        except Exception as e:
            return f"Vision Error: {str(e)}"

    def _openai_chat(self, model: str, messages: List, params: Dict) -> str:
        resp = self._openai_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=params.get("temperature", 0.3),
            top_p=params.get("top_p", 0.95),
            max_tokens=params.get("max_tokens", 1000)
        )
        return resp.choices[0].message.content

    def _gemini_chat(self, model: str, messages: List, params: Dict) -> str:
        mm = genai.GenerativeModel(model)
        sys_prompt = next((m["content"] for m in messages if m["role"] == "system"), "")
        user_msgs = [m["content"] for m in messages if m["role"] == "user"]
        full_prompt = (sys_prompt + "\n\n" + "\n".join(user_msgs)).strip()
        
        resp = mm.generate_content(full_prompt, generation_config=genai.types.GenerationConfig(
            temperature=params.get("temperature", 0.3),
            top_p=params.get("top_p", 0.95),
            max_output_tokens=params.get("max_tokens", 1000)
        ))
        return resp.text

    def _grok_chat(self, model: str, messages: List, params: Dict) -> str:
        chat = self._xai_client.chat.create(model=model)
        for m in messages:
            if m["role"] == "system": chat.append(xai_system(m["content"]))
            elif m["role"] == "user": chat.append(xai_user(m["content"]))
        return chat.sample().content

    def _anthropic_chat(self, model: str, messages: List, params: Dict) -> str:
        sys_prompt = next((m["content"] for m in messages if m["role"] == "system"), "")
        anth_msgs = [{"role": m["role"] if m["role"] != "system" else "user", "content": m["content"]} for m in messages if m["role"] != "system"]
        
        # Anthropic requires first message to be user. Handle edge case.
        if not anth_msgs: anth_msgs = [{"role": "user", "content": sys_prompt}]
        
        kwargs = {
            "model": model,
            "messages": anth_msgs,
            "temperature": params.get("temperature", 0.3),
            "max_tokens": params.get("max_tokens", 1000)
        }
        if sys_prompt: kwargs["system"] = sys_prompt
        
        resp = self._anthropic_client.messages.create(**kwargs)
        return resp.content[0].text

    def _openai_vision(self, model: str, prompt: str, images: List) -> str:
        content = [{"type": "text", "text": prompt}]
        for img in images:
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
        resp = self._openai_client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": content}], max_tokens=1500
        )
        return resp.choices[0].message.content

    def _gemini_vision(self, model: str, prompt: str, images: List) -> str:
        mm = genai.GenerativeModel(model)
        return mm.generate_content([prompt] + images).text

    def _anthropic_vision(self, model: str, prompt: str, images: List) -> str:
        if "haiku" in model and "3-5" not in model: return "Old Haiku no vision."
        content = [{"type": "text", "text": prompt}]
        for img in images:
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": b64}
            })
        resp = self._anthropic_client.messages.create(
            model=model, messages=[{"role": "user", "content": content}], max_tokens=1500
        )
        return resp.content[0].text

    def _estimate_tokens(self, messages: List) -> int:
        return sum(len(m.get("content", "")) for m in messages) // 4

# ==================== UTILS & CACHING ====================

@st.cache_data(show_spinner=False)
def render_pdf_pages(pdf_bytes: bytes, dpi: int = 150, max_pages: int = 30) -> List[Tuple[int, Image.Image]]:
    """Cache PDF rendering to avoid re-processing on re-runs."""
    try:
        pages = convert_from_bytes(pdf_bytes, dpi=dpi, first_page=1, last_page=None)
        return [(idx, im) for idx, im in enumerate(pages[:max_pages])]
    except Exception as e:
        st.error(f"Error rendering PDF: {e}. Please ensure poppler-utils is installed.")
        return []

@st.cache_data(show_spinner=False)
def extract_text_python(pdf_bytes: bytes, selected_pages: List[int], lang: str) -> str:
    """Cache Python-based OCR."""
    text_parts = []
    # 1. Text Extraction
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for i in selected_pages:
            if i < len(pdf.pages):
                txt = pdf.pages[i].extract_text() or ""
                if txt.strip():
                    text_parts.append(f"--- PAGE {i+1} (Text Layer) ---\n{txt.strip()}\n")
    
    # 2. OCR Fallback if needed (simplified logic)
    tess_lang = "eng" if lang == "english" else "chi_tra"
    page_imgs = render_pdf_pages(pdf_bytes, dpi=200) # Re-render high qual for OCR
    for i in selected_pages:
        if i < len(page_imgs):
            # Simple check: if text layer was small, do OCR
            ocr_txt = pytesseract.image_to_string(page_imgs[i][1], lang=tess_lang)
            if len(ocr_txt) > 50:
                 text_parts.append(f"--- PAGE {i+1} (OCR Layer) ---\n{ocr_txt.strip()}\n")
                 
    return "\n".join(text_parts).strip()

def extract_text_llm(page_images: List[Tuple[int, Image.Image]], model_name: str, router) -> str:
    """LLM Vision extraction (not cached by default as it costs money/tokens)."""
    prompt = "請將圖片中的文字完整轉錄（保持原文、段落與標點）。表格請轉為Markdown格式。忽略頁眉頁腳。"
    text_blocks = []
    progress_bar = st.progress(0)
    for i, (idx, im) in enumerate(page_images):
        out = router.generate_vision(model_name, f"{prompt}\n頁面 {idx+1}：", [im])
        text_blocks.append(f"--- PAGE {idx+1} (LLM Vision) ---\n{out}\n")
        progress_bar.progress((i + 1) / len(page_images))
    progress_bar.empty()
    return "\n".join(text_blocks).strip()

# ==================== THEME & STYLING ====================

def generate_theme_css(theme_name: str, dark_mode: bool):
    t = FLOWER_THEMES[theme_name]
    bg = t["bg_dark"] if dark_mode else t["bg_light"]
    text = "#FAFAFA" if dark_mode else "#222222"
    card_bg = "rgba(20, 20, 25, 0.85)" if dark_mode else "rgba(255, 255, 255, 0.90)"
    accent = t["accent"]
    
    return f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+TC:wght@400;500;700&display=swap');
        
        [data-testid="stAppViewContainer"] > .main {{
            background: {bg} !important;
            background-attachment: fixed !important;
            color: {text};
            font-family: 'Noto Sans TC', sans-serif;
        }}
        
        h1, h2, h3, h4 {{ font-weight: 700; color: {text} !important; }}
        
        .stButton > button {{
            background: linear-gradient(135deg, {t['primary']}, {t['secondary']});
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.5rem 1rem;
            font-weight: 600;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
        }}
        .stButton > button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 16px rgba(0,0,0,0.2);
        }}
        
        .wow-card {{
            background: {card_bg};
            backdrop-filter: blur(10px);
            border: 1px solid {accent}40;
            border-radius: 16px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            box-shadow: 0 8px 32px rgba(0,0,0,0.05);
        }}
        
        /* Tabs styling */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 10px;
            background: transparent;
        }}
        .stTabs [data-baseweb="tab"] {{
            background: {card_bg};
            border-radius: 8px 8px 0 0;
            border: 1px solid {accent}20;
            padding: 8px 16px;
        }}
        .stTabs [aria-selected="true"] {{
            background: {accent} !important;
            color: white !important;
        }}
        
        /* Metric Cards */
        .metric-box {{
            text-align: center;
            padding: 10px;
            background: {t['primary']}15;
            border-radius: 10px;
            border: 1px solid {t['primary']}30;
        }}
        .metric-val {{ font-size: 1.8rem; font-weight: bold; color: {accent}; }}
        .metric-lbl {{ font-size: 0.85rem; opacity: 0.8; }}
        
        /* Highlight pill */
        .pill {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            background: {t['primary']}30;
            color: {accent};
            font-weight: bold;
            font-size: 0.9rem;
            margin-bottom: 10px;
        }}
    </style>
    """

# ==================== DEFAULT CONFIGURATION ====================

DEFAULT_AGENTS = """agents:
  - name: "1. 申請資料提取器 (Extraction)"
    description: "提取基本行政資料、廠商資訊、證書細節。"
    system_prompt: "你是一位專業的醫療器材法規專家。請從文件中提取關鍵行政資訊：廠商名稱、地址、產品名稱、類別、證書編號、日期。若有不確定資訊請標註。輸出為Markdown表格。"
    user_prompt: "分析文件並提取申請基本資料："
    model: claude-3-5-sonnet-20241022
    temperature: 0
    max_tokens: 2000

  - name: "2. 適應症與禁忌症分析 (Clinical)"
    description: "分析產品適應症、禁忌症及副作用。"
    system_prompt: "你是臨床醫學專家。請分析文件的：1. 適應症 (Indications) 2. 禁忌症 (Contraindications) 3. 副作用與警語。請用列點方式呈現，並標註風險等級。"
    user_prompt: "請分析以下內容的臨床相關資訊："
    model: gpt-4o-mini
    temperature: 0.3
    max_tokens: 1500

  - name: "3. 技術規格與檢驗摘要 (Technical)"
    description: "摘要產品技術規格、檢驗標準與測試結果。"
    system_prompt: "你是生醫工程專家。請摘要：1. 產品技術規格 2. 已進行的測試項目 (如生物相容性、電性安全) 3. 檢驗結果摘要。忽略過於瑣碎的數據，只抓重點。"
    user_prompt: "請摘要技術規格與檢驗結果："
    model: claude-3-5-haiku-20241022
    temperature: 0.2
    max_tokens: 1500

  - name: "4. 法規符合性檢查 (Compliance)"
    description: "根據TFDA要求檢查文件完整性與合規性。"
    system_prompt: "你是資深法規稽核員。根據前述資訊與原文，檢查：1. 是否符合醫療器材分類分級規定？ 2. 標示是否包含必要警語？ 3. 是否有明顯缺漏文件？提供審查建議。"
    user_prompt: "請進行法規符合性檢查並提供建議："
    model: gpt-4o
    temperature: 0.4
    max_tokens: 1500

  - name: "5. 綜合審查報告生成 (Reporting)"
    description: "整合所有分析，生成最終審查報告。"
    system_prompt: "你是審查報告主筆。請根據上下文提供的所有分析結果，撰寫一份結構完整的「醫療器材查驗登記審查報告」。包含：摘要、產品描述、臨床評估、技術評估、結論與建議。"
    user_prompt: "請撰寫綜合審查報告："
    model: claude-3-5-sonnet-20241022
    temperature: 0.5
    max_tokens: 3000
"""

# ==================== APP LOGIC ====================

def main():
    st.set_page_config(page_title="TFDA AI Review", page_icon="🌸", layout="wide")
    
    # Session State Init
    if "theme" not in st.session_state: st.session_state.theme = "櫻花 Cherry Blossom"
    if "dark_mode" not in st.session_state: st.session_state.dark_mode = False
    if "language" not in st.session_state: st.session_state.language = "zh_TW"
    if "ocr_text" not in st.session_state: st.session_state.ocr_text = ""
    if "page_images" not in st.session_state: st.session_state.page_images = [] # List of (idx, image)
    if "agents_config" not in st.session_state:
        data = yaml.safe_load(DEFAULT_AGENTS)
        st.session_state.agents_config = data.get("agents", [])
    if "agent_outputs" not in st.session_state: st.session_state.agent_outputs = []
    if "run_metrics" not in st.session_state: st.session_state.run_metrics = []
    if "review_notes" not in st.session_state: st.session_state.review_notes = "## 審查筆記\n"

    # Apply Theme
    st.markdown(generate_theme_css(st.session_state.theme, st.session_state.dark_mode), unsafe_allow_html=True)
    
    # Localization
    t = TRANSLATIONS[st.session_state.language]
    router = LLMRouter()

    # Sidebar
    with st.sidebar:
        st.markdown(f"### {FLOWER_THEMES[st.session_state.theme]['icon']} Settings")
        
        # Theme Controls
        c1, c2 = st.columns(2)
        with c1:
            st.session_state.theme = st.selectbox(t["theme_selector"], list(FLOWER_THEMES.keys()), 
                                                 index=list(FLOWER_THEMES.keys()).index(st.session_state.theme), label_visibility="collapsed")
        with c2:
            st.session_state.language = st.selectbox(t["language"], ["zh_TW", "en"], 
                                                    index=0 if st.session_state.language == "zh_TW" else 1, label_visibility="collapsed")
        st.toggle(t["dark_mode"], key="dark_mode")
        
        st.markdown("---")
        
        # API Keys (Collapsed)
        with st.expander(f"🔐 {t['providers']}", expanded=False):
            def key_input(label, env_key):
                val = os.getenv(env_key, "")
                new_val = st.text_input(label, value=val, type="password")
                if new_val: os.environ[env_key] = new_val
                status = "✅" if os.getenv(env_key) else "❌"
                st.caption(f"Status: {status}")

            key_input("OpenAI API Key", "OPENAI_API_KEY")
            key_input("Anthropic API Key", "ANTHROPIC_API_KEY")
            key_input("Gemini API Key", "GEMINI_API_KEY")
            key_input("xAI/Grok API Key", "XAI_API_KEY")
            
        # YAML Config
        with st.expander(f"⚙️ {t['config_tab']}", expanded=False):
            yaml_txt = st.text_area("Edit YAML", value=yaml.dump({"agents": st.session_state.agents_config}, sort_keys=False, allow_unicode=True), height=300)
            if st.button(t["save_agents"], use_container_width=True):
                try:
                    parsed = yaml.safe_load(yaml_txt)
                    st.session_state.agents_config = parsed["agents"]
                    st.success("Saved!")
                except Exception as e:
                    st.error(f"YAML Error: {e}")
            
            if st.button(t["reset_agents"], use_container_width=True):
                data = yaml.safe_load(DEFAULT_AGENTS)
                st.session_state.agents_config = data.get("agents", [])
                st.rerun()

    # Main Header
    col_h1, col_h2 = st.columns([3, 1])
    with col_h1:
        st.title(t["title"])
        st.markdown(f"<div style='opacity:0.7'>{t['subtitle']}</div>", unsafe_allow_html=True)
    with col_h2:
        # Quick Stats
        active_keys = sum([1 for k in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "XAI_API_KEY"] if os.getenv(k)])
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-val">{active_keys} / 4</div>
            <div class="metric-lbl">Active Models</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Tabs
    tabs = st.tabs([t["upload_tab"], t["preview_tab"], t["config_tab"], t["execute_tab"], t["dashboard_tab"], t["notes_tab"]])

    # TAB 1: UPLOAD & OCR
    with tabs[0]:
        st.markdown('<div class="wow-card">', unsafe_allow_html=True)
        uploaded = st.file_uploader(t["upload_pdf"], type="pdf")
        
        if uploaded:
            pdf_bytes = uploaded.read()
            st.info(f"File loaded: {uploaded.name} ({len(pdf_bytes)/1024:.1f} KB)")
            
            # Preview
            if not st.session_state.page_images:
                with st.spinner("Rendering PDF pages..."):
                    st.session_state.page_images = render_pdf_pages(pdf_bytes)
            
            with st.expander("📄 Page Preview", expanded=True):
                cols = st.columns(5)
                for i, (idx, img) in enumerate(st.session_state.page_images[:10]):
                    cols[i%5].image(img, caption=f"Page {idx+1}", use_container_width=True)
            
            st.markdown("#### OCR Settings")
            c1, c2, c3 = st.columns(3)
            with c1:
                mode = st.selectbox(t["ocr_mode"], ["Python Native (Fast)", "LLM Vision (High Accuracy)"])
            with c2:
                lang = st.selectbox(t["ocr_lang"], ["traditional-chinese", "english"])
            with c3:
                model_ocr = st.selectbox("Vision Model", ["claude-3-5-sonnet-20241022", "gpt-4o", "gemini-1.5-flash"]) if "LLM" in mode else None
            
            if st.button(t["start_ocr"], type="primary", use_container_width=True):
                # Simple range parser logic
                pages_to_proc = [p[0] for p in st.session_state.page_images[:10]] # Limit to first 10 for demo
                
                with st.status("Processing Document...", expanded=True) as status:
                    start_t = time.time()
                    if "LLM" in mode:
                        status.write("Sending images to Vision Model...")
                        txt = extract_text_llm([st.session_state.page_images[i] for i in range(len(pages_to_proc))], model_ocr, router)
                    else:
                        status.write("Running Text Extraction & Tesseract...")
                        txt = extract_text_python(pdf_bytes, pages_to_proc, lang)
                    
                    st.session_state.ocr_text = txt
                    status.update(label=f"OCR Complete in {time.time()-start_t:.2f}s", state="complete")
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # TAB 2: PREVIEW & EDIT
    with tabs[1]:
        st.markdown('<div class="wow-card">', unsafe_allow_html=True)
        c1, c2 = st.columns([3, 1])
        with c1:
            st.subheader("Document Content")
        with c2:
             if st.button("Clear Text"):
                 st.session_state.ocr_text = ""
                 st.rerun()
        
        st.session_state.ocr_text = st.text_area("Full Text Context", st.session_state.ocr_text, height=600)
        st.markdown('</div>', unsafe_allow_html=True)

    # TAB 3: AGENT CONFIG
    with tabs[2]:
        st.markdown('<div class="wow-card">', unsafe_allow_html=True)
        st.subheader("Agent Pipeline Configuration")
        st.info("Modify the prompt or model for each step in the chain.")
        
        for i, agent in enumerate(st.session_state.agents_config):
            with st.expander(f"🤖 Agent {i+1}: {agent['name']}", expanded=False):
                c1, c2 = st.columns([2, 1])
                with c1:
                    agent['system_prompt'] = st.text_area("System Prompt", agent['system_prompt'], key=f"sys_{i}")
                    agent['user_prompt'] = st.text_area("User Prompt", agent['user_prompt'], key=f"usr_{i}")
                with c2:
                    agent['model'] = st.selectbox("Model", list(ModelChoice.keys()), index=list(ModelChoice.keys()).index(agent.get('model', 'gpt-4o-mini')) if agent.get('model') in ModelChoice else 0, key=f"mod_{i}")
                    agent['temperature'] = st.slider("Temperature", 0.0, 1.0, float(agent.get('temperature', 0.3)), key=f"tmp_{i}")
                    agent['max_tokens'] = st.number_input("Max Tokens", 100, 10000, int(agent.get('max_tokens', 1000)), key=f"tok_{i}")
        st.markdown('</div>', unsafe_allow_html=True)

    # TAB 4: EXECUTE
    with tabs[3]:
        st.markdown('<div class="wow-card">', unsafe_allow_html=True)
        if not st.session_state.ocr_text:
            st.warning("Please complete OCR in Tab 1 first.")
        else:
            # Initialize outputs
            if len(st.session_state.agent_outputs) != len(st.session_state.agents_config):
                st.session_state.agent_outputs = [None] * len(st.session_state.agents_config)

            # AUTO RUN BUTTON
            if st.button(t["run_all"], type="primary", use_container_width=True):
                current_context = st.session_state.ocr_text
                st.session_state.run_metrics = []
                
                with st.status("Executing Agent Pipeline...", expanded=True) as status:
                    for i, agent in enumerate(st.session_state.agents_config):
                        status.write(f"**Agent {i+1} ({agent['name']})** is thinking...")
                        
                        # Construct messages
                        msgs = [
                            {"role": "system", "content": agent['system_prompt']},
                            {"role": "user", "content": f"{agent['user_prompt']}\n\n---\nCONTEXT:\n{current_context[:50000]}"} # Limit context window slightly
                        ]
                        
                        # Call LLM
                        t0 = time.time()
                        out_text, meta, provider = router.generate_text(
                            agent['model'], msgs, 
                            {"temperature": agent['temperature'], "max_tokens": agent['max_tokens']}
                        )
                        t1 = time.time()
                        
                        # Store result
                        st.session_state.agent_outputs[i] = out_text
                        st.session_state.run_metrics.append({
                            "agent": agent['name'],
                            "latency": t1-t0,
                            "tokens": meta['total_tokens'],
                            "provider": provider
                        })
                        
                        # Update context for next agent? 
                        # Strategy: Append output to context or Keep original context? 
                        # For this workflow, usually we keep original context + previous analysis
                        current_context += f"\n\n=== Analysis from {agent['name']} ===\n{out_text}"
                        
                    status.update(label="Pipeline Completed Successfully! 🎉", state="complete")

            st.markdown("---")
            
            # DISPLAY RESULTS
            for i, agent in enumerate(st.session_state.agents_config):
                result = st.session_state.agent_outputs[i]
                if result:
                    with st.expander(f"✅ Output: {agent['name']}", expanded=True):
                        st.markdown(result)
                        # Tools
                        c1, c2 = st.columns([1, 4])
                        if c1.button(f"Add to Notes #{i+1}"):
                            st.session_state.review_notes += f"\n\n### {agent['name']} Result\n{result}"
                            st.toast("Added to notes")
        st.markdown('</div>', unsafe_allow_html=True)

    # TAB 5: DASHBOARD
    with tabs[4]:
        st.markdown('<div class="wow-card">', unsafe_allow_html=True)
        if not st.session_state.run_metrics:
            st.info("No run data available yet. Execute the pipeline first.")
        else:
            df = pd.DataFrame(st.session_state.run_metrics)
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Total Latency", f"{df['latency'].sum():.2f}s")
            m2.metric("Avg Tokens", f"{int(df['tokens'].mean())}")
            m3.metric("Total Cost Est.", f"${df['tokens'].sum() * 0.000005:.4f}") # Dummy cost math
            
            c1, c2 = st.columns(2)
            with c1:
                fig = px.bar(df, x='agent', y='latency', color='provider', title="Latency by Agent")
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color=FLOWER_THEMES[st.session_state.theme]['accent'])
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig2 = px.pie(df, names='provider', values='tokens', title="Token Distribution")
                fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color=FLOWER_THEMES[st.session_state.theme]['accent'])
                st.plotly_chart(fig2, use_container_width=True)
                
            st.dataframe(df, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # TAB 6: NOTES
    with tabs[5]:
        st.markdown('<div class="wow-card">', unsafe_allow_html=True)
        c1, c2 = st.columns([2, 1])
        with c1:
            st.subheader("Review Notes")
        with c2:
            if st.button("📥 Download Report (MD)"):
                st.download_button("Click to Download", st.session_state.review_notes, "review_report.md")
        
        st.session_state.review_notes = st.text_area("Markdown Editor", st.session_state.review_notes, height=600)
        
        st.markdown("### Preview")
        st.markdown("---")
        st.markdown(st.session_state.review_notes)
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
