import os
import io
import time
import base64
from pathlib import Path
from typing import List, Dict, Any, Optional

import streamlit as st
import yaml
from PIL import Image
from fpdf import FPDF
from pdf2image import convert_from_bytes

# Optional: external LLM clients (install via requirements.txt)
try:
    import google.generativeai as genai
except Exception:
    genai = None

try:
    from openai import OpenAI as OpenAIClient
except Exception:
    OpenAIClient = None

try:
    import anthropic
except Exception:
    anthropic = None

try:
    from openai import OpenAI as XAIClient  # xAI OpenAI-compatible client
except Exception:
    XAIClient = None


# ========== CONSTANTS & THEME SYSTEM ========================================

FLOWER_THEMES: Dict[str, Dict[str, str]] = {
    "Cherry Blossom": {
        "icon": "🌸",
        "primary": "#ff7eb3",
        "secondary": "#ffb3c6",
        "accent": "#ff4f81",
        "bg_light": "radial-gradient(circle at top, #ffe4f0 0, #ffffff 40%, #ffeef6 100%)",
        "bg_dark": "radial-gradient(circle at top, #331522 0, #0f0712 50%, #050308 100%)",
    },
    "Rose": {
        "icon": "🌹",
        "primary": "#e11d48",
        "secondary": "#fb7185",
        "accent": "#be123c",
        "bg_light": "radial-gradient(circle at top, #ffe4e6 0, #ffffff 40%, #fee2e2 100%)",
        "bg_dark": "radial-gradient(circle at top, #3b0715 0, #111827 50%, #020617 100%)",
    },
    "Sunflower": {
        "icon": "🌻",
        "primary": "#facc15",
        "secondary": "#f97316",
        "accent": "#d97706",
        "bg_light": "radial-gradient(circle at top, #fef9c3 0, #ffffff 40%, #fffbeb 100%)",
        "bg_dark": "radial-gradient(circle at top, #422006 0, #111827 50%, #020617 100%)",
    },
    "Lotus": {
        "icon": "🪷",
        "primary": "#a855f7",
        "secondary": "#ec4899",
        "accent": "#d946ef",
        "bg_light": "radial-gradient(circle at top, #f3e8ff 0, #ffffff 40%, #fce7f3 100%)",
        "bg_dark": "radial-gradient(circle at top, #312e81 0, #020617 60%, #020617 100%)",
    },
    "Lavender": {
        "icon": "💜",
        "primary": "#a78bfa",
        "secondary": "#c4b5fd",
        "accent": "#8b5cf6",
        "bg_light": "radial-gradient(circle at top, #ede9fe 0, #ffffff 40%, #f3e8ff 100%)",
        "bg_dark": "radial-gradient(circle at top, #312e81 0, #020617 50%, #020617 100%)",
    },
    "Peony": {
        "icon": "🌺",
        "primary": "#f472b6",
        "secondary": "#fb7185",
        "accent": "#db2777",
        "bg_light": "radial-gradient(circle at top, #ffe4f6 0, #ffffff 40%, #ffe4e6 100%)",
        "bg_dark": "radial-gradient(circle at top, #4a044e 0, #020617 50%, #020617 100%)",
    },
    "Orchid": {
        "icon": "🪻",
        "primary": "#c084fc",
        "secondary": "#a855f7",
        "accent": "#7c3aed",
        "bg_light": "radial-gradient(circle at top, #f5f3ff 0, #ffffff 40%, #ede9fe 100%)",
        "bg_dark": "radial-gradient(circle at top, #312e81 0, #020617 50%, #020617 100%)",
    },
    "Camellia": {
        "icon": "🌺",
        "primary": "#f97373",
        "secondary": "#fb923c",
        "accent": "#ef4444",
        "bg_light": "radial-gradient(circle at top, #fee2e2 0, #ffffff 40%, #ffedd5 100%)",
        "bg_dark": "radial-gradient(circle at top, #450a0a 0, #020617 50%, #020617 100%)",
    },
    "Hydrangea": {
        "icon": "🌼",
        "primary": "#60a5fa",
        "secondary": "#a5b4fc",
        "accent": "#3b82f6",
        "bg_light": "radial-gradient(circle at top, #dbeafe 0, #ffffff 40%, #e0f2fe 100%)",
        "bg_dark": "radial-gradient(circle at top, #1e3a8a 0, #020617 50%, #020617 100%)",
    },
    "Magnolia": {
        "icon": "🤍",
        "primary": "#f9a8d4",
        "secondary": "#fecdd3",
        "accent": "#f472b6",
        "bg_light": "radial-gradient(circle at top, #fff1f2 0, #ffffff 40%, #fee2e2 100%)",
        "bg_dark": "radial-gradient(circle at top, #4a044e 0, #020617 50%, #020617 100%)",
    },
    "Plum Blossom": {
        "icon": "🌸",
        "primary": "#fb7185",
        "secondary": "#f97373",
        "accent": "#e11d48",
        "bg_light": "radial-gradient(circle at top, #ffe4e6 0, #ffffff 40%, #fee2e2 100%)",
        "bg_dark": "radial-gradient(circle at top, #4c0519 0, #020617 50%, #020617 100%)",
    },
    "Tulip": {
        "icon": "🌷",
        "primary": "#f97316",
        "secondary": "#facc15",
        "accent": "#ea580c",
        "bg_light": "radial-gradient(circle at top, #ffedd5 0, #ffffff 40%, #fef9c3 100%)",
        "bg_dark": "radial-gradient(circle at top, #3b0764 0, #020617 50%, #020617 100%)",
    },
    "Daisy": {
        "icon": "🌼",
        "primary": "#facc15",
        "secondary": "#22c55e",
        "accent": "#eab308",
        "bg_light": "radial-gradient(circle at top, #fef9c3 0, #ffffff 40%, #dcfce7 100%)",
        "bg_dark": "radial-gradient(circle at top, #14532d 0, #020617 50%, #020617 100%)",
    },
    "Iris": {
        "icon": "🌸",
        "primary": "#6366f1",
        "secondary": "#a855f7",
        "accent": "#4f46e5",
        "bg_light": "radial-gradient(circle at top, #e0e7ff 0, #ffffff 40%, #f3e8ff 100%)",
        "bg_dark": "radial-gradient(circle at top, #1e3a8a 0, #020617 50%, #020617 100%)",
    },
    "Poppy": {
        "icon": "🌺",
        "primary": "#f97373",
        "secondary": "#fb923c",
        "accent": "#b91c1c",
        "bg_light": "radial-gradient(circle at top, #fee2e2 0, #ffffff 40%, #ffedd5 100%)",
        "bg_dark": "radial-gradient(circle at top, #450a0a 0, #020617 50%, #020617 100%)",
    },
    "Gardenia": {
        "icon": "🤍",
        "primary": "#e5e5e5",
        "secondary": "#a3a3a3",
        "accent": "#737373",
        "bg_light": "radial-gradient(circle at top, #f5f5f5 0, #ffffff 40%, #e5e5e5 100%)",
        "bg_dark": "radial-gradient(circle at top, #171717 0, #020617 50%, #020617 100%)",
    },
    "Bluebell": {
        "icon": "🔵",
        "primary": "#3b82f6",
        "secondary": "#22c55e",
        "accent": "#2563eb",
        "bg_light": "radial-gradient(circle at top, #dbeafe 0, #ffffff 40%, #dcfce7 100%)",
        "bg_dark": "radial-gradient(circle at top, #1e3a8a 0, #020617 50%, #020617 100%)",
    },
    "Wisteria": {
        "icon": "🌸",
        "primary": "#c4b5fd",
        "secondary": "#a855f7",
        "accent": "#7e22ce",
        "bg_light": "radial-gradient(circle at top, #ede9fe 0, #ffffff 40%, #f3e8ff 100%)",
        "bg_dark": "radial-gradient(circle at top, #312e81 0, #020617 50%, #020617 100%)",
    },
    "Chrysanthemum": {
        "icon": "🌼",
        "primary": "#fbbf24",
        "secondary": "#fb7185",
        "accent": "#f59e0b",
        "bg_light": "radial-gradient(circle at top, #fef3c7 0, #ffffff 40%, #ffe4e6 100%)",
        "bg_dark": "radial-gradient(circle at top, #78350f 0, #020617 50%, #020617 100%)",
    },
    "Jasmine": {
        "icon": "🤍",
        "primary": "#fde68a",
        "secondary": "#a7f3d0",
        "accent": "#fbbf24",
        "bg_light": "radial-gradient(circle at top, #fef9c3 0, #ffffff 40%, #d1fae5 100%)",
        "bg_dark": "radial-gradient(circle at top, #365314 0, #020617 50%, #020617 100%)",
    },
}


LANG_LABELS = {
    "en": {
        "title": "TFDA AI Review System",
        "upload": "1. Upload & OCR",
        "preview": "2. Preview & Edit",
        "config": "3. Agent Config",
        "execute": "4. Execute Pipeline",
        "dashboard": "5. Analytics",
        "notes": "Quick Notes",
        "ocr_settings": "OCR Settings",
        "ocr_model": "OCR Model",
        "page_range": "Page Range",
        "run_ocr": "Start Vision OCR",
        "document_content": "Document Content",
        "next_pipeline": "Next: Pipeline →",
        "clear": "Clear",
        "agent_config": "Agent Configuration",
        "reset_defaults": "Reset to defaults (from YAML)",
        "run_swarm": "Execute Selected",
        "agent_selection": "Agent Selection",
        "select_all": "Select All",
        "deselect_all": "Deselect All",
        "run_pipeline": "Run Pipeline",
        "analytics": "Analytics Dashboard",
        "download_notes_pdf": "Download Notes as PDF",
        "ai_auto_format": "AI Auto-Format",
        "run_format": "Run Format",
        "markdown": "To Markdown",
        "fix_grammar": "Fix Grammar",
        "checklist": "Checklist",
        "instruction_prompt": "Instruction Prompt",
        "model": "Model",
        "max_tokens": "Max Tokens",
        "save_settings": "Save Settings",
        "using_env_key": "Using API key from environment.",
        "need_key": "API key not found in environment, please enter below.",
        "provider": "Provider",
        "wow_timeline": "Execution Timeline",
        "status_waiting": "Waiting",
        "status_running": "Running",
        "status_done": "Done",
        "status_error": "Error",
        "language_label": "Language",
        "theme_label": "Floral Theme",
        "dark_mode": "Dark Mode",
    },
    "zh-TW": {
        "title": "TFDA AI 審查系統",
        "upload": "1. 上傳檔案與 OCR",
        "preview": "2. 預覽與編輯",
        "config": "3. 智能代理設定",
        "execute": "4. 執行分析流程",
        "dashboard": "5. 分析儀表板",
        "notes": "快速筆記",
        "ocr_settings": "OCR 設定",
        "ocr_model": "OCR 模型",
        "page_range": "頁碼範圍",
        "run_ocr": "開始視覺 OCR",
        "document_content": "文件內容",
        "next_pipeline": "下一步：分析流程 →",
        "clear": "清除",
        "agent_config": "代理設定",
        "reset_defaults": "從 YAML 還原預設",
        "run_swarm": "執行已選代理",
        "agent_selection": "代理選擇",
        "select_all": "全選",
        "deselect_all": "全不選",
        "run_pipeline": "執行分析流程",
        "analytics": "分析儀表板",
        "download_notes_pdf": "下載筆記 PDF",
        "ai_auto_format": "AI 自動排版",
        "run_format": "執行排版",
        "markdown": "轉為 Markdown",
        "fix_grammar": "修正文法",
        "checklist": "轉為清單",
        "instruction_prompt": "指令提示",
        "model": "模型",
        "max_tokens": "最大 Token 數",
        "save_settings": "儲存設定",
        "using_env_key": "已從環境變數載入 API 金鑰。",
        "need_key": "環境中未找到 API 金鑰，請在下方輸入。",
        "provider": "服務供應商",
        "wow_timeline": "執行時間軸",
        "status_waiting": "等待中",
        "status_running": "執行中",
        "status_done": "完成",
        "status_error": "錯誤",
        "language_label": "語言",
        "theme_label": "花卉主題",
        "dark_mode": "深色模式",
    },
}


PROVIDER_ENV_VARS = {
    "gemini": "GEMINI_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "xai": "XAI_API_KEY",
}

# Default model lists per provider
PROVIDER_MODELS = {
    "gemini": ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.0-pro-exp"],
    "openai": ["gpt-4o-mini", "gpt-4.1-mini", "gpt-5-nano"],
    "anthropic": ["claude-3-5-sonnet-latest", "claude-3-opus-latest", "claude-3-haiku-latest"],
    "xai": ["grok-beta", "grok-2-mini"],
}

# Vision-capable models by provider (for OCR)
VISION_MODELS = {
    "gemini": ["gemini-2.5-flash", "gemini-2.5-flash-lite"],
    "openai": ["gpt-4o-mini", "gpt-4.1-mini"],
    # Anthropic / xAI vision omitted for OCR to keep implementation simpler
}


AGENTS_FILE = Path("agents.yaml")


# ========== HELPER FUNCTIONS =================================================


def inject_global_css(theme: Dict[str, str], dark_mode: bool):
    bg = theme["bg_dark"] if dark_mode else theme["bg_light"]
    primary = theme["primary"]
    secondary = theme["secondary"]
    accent = theme["accent"]

    st.markdown(
        f"""
        <style>
            body {{
                background: {bg} !important;
            }}
            .stApp {{
                background: transparent;
                color: {"#f9fafb" if dark_mode else "#111827"};
            }}
            .main-card {{
                border-radius: 32px;
                padding: 1.5rem 2rem;
                background: rgba(255,255,255,{0.04 if dark_mode else 0.88});
                box-shadow: 0 30px 60px rgba(15,23,42,{0.65 if dark_mode else 0.12});
                border: 1px solid rgba(255,255,255,{0.08 if dark_mode else 0.6});
            }}
            .glass-panel {{
                background: rgba(255,255,255,{0.06 if dark_mode else 0.65});
                border-radius: 24px;
                padding: 1rem;
                box-shadow: 0 18px 45px rgba(15,23,42,{0.55 if dark_mode else 0.16});
                border: 1px solid rgba(255,255,255,{0.14 if dark_mode else 0.55});
            }}
            .wow-pill {{
                border-radius: 999px;
                padding: 0.35rem 0.9rem;
                display: inline-flex;
                align-items: center;
                gap: 0.45rem;
                font-size: 0.75rem;
                font-weight: 600;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                border: 1px solid rgba(255,255,255,0.4);
                background: linear-gradient(135deg, {primary}22, {secondary}10);
                backdrop-filter: blur(16px);
            }}
            .wow-status-dot {{
                width: 8px;
                height: 8px;
                border-radius: 999px;
                background: {accent};
                box-shadow: 0 0 0 6px {accent}33;
            }}
            .timeline-bar {{
                height: 4px;
                border-radius: 999px;
                background: rgba(148,163,184,0.35);
                overflow: hidden;
            }}
            .timeline-fill {{
                height: 100%;
                border-radius: inherit;
                background: linear-gradient(90deg, {primary}, {accent});
                transition: width 350ms ease-out;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def load_agents_from_yaml() -> List[Dict[str, Any]]:
    if AGENTS_FILE.exists():
        with open(AGENTS_FILE, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        agents = data.get("agents", [])
    else:
        agents = []
    # Enforce required keys with minimal defaults
    normalized = []
    for i, ag in enumerate(agents):
        normalized.append(
            {
                "id": ag.get("id", f"agent_{i+1}"),
                "name": ag.get("name", f"Agent {i+1}"),
                "description": ag.get("description", ""),
                "provider": ag.get("provider", "gemini"),
                "model": ag.get("model", "gemini-2.5-flash"),
                "system_prompt": ag.get("system_prompt", ""),
                "user_prompt": ag.get("user_prompt", ""),
                "temperature": float(ag.get("temperature", 0.2)),
                "max_tokens": int(ag.get("max_tokens", 2048)),
            }
        )
    return normalized


def parse_page_range(range_str: str, max_pages: int) -> List[int]:
    pages = set()
    for part in range_str.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            try:
                start = int(start_s.strip())
                end = int(end_s.strip())
            except ValueError:
                continue
            for p in range(start, end + 1):
                if 1 <= p <= max_pages:
                    pages.add(p)
        else:
            try:
                p = int(part)
            except ValueError:
                continue
            if 1 <= p <= max_pages:
                pages.add(p)
    return sorted(pages)


def get_api_key_ui(provider: str, lang_key: str) -> Optional[str]:
    env_var = PROVIDER_ENV_VARS.get(provider)
    env_value = os.getenv(env_var) if env_var else None
    labels = LANG_LABELS[lang_key]
    label_map = {
        "gemini": "Gemini API Key",
        "openai": "OpenAI API Key",
        "anthropic": "Anthropic API Key",
        "xai": "xAI API Key",
    }
    st.write(f"**{label_map.get(provider, provider)}**")
    if env_value:
        st.caption(f"✅ {labels['using_env_key']}")
        return env_value
    else:
        st.caption(f"⚠️ {labels['need_key']}")
        key = st.text_input(
            "API Key",
            type="password",
            key=f"{provider}_manual_key",
        )
        return key or None


# ========== LLM / VISION CALLS ==============================================


def call_gemini_text(
    api_key: str,
    model: str,
    system_prompt: str,
    user_content: str,
    temperature: float,
    max_tokens: int,
) -> Dict[str, Any]:
    if genai is None:
        raise RuntimeError("google-generativeai is not installed.")
    genai.configure(api_key=api_key)
    generation_config = {
        "temperature": temperature,
        "max_output_tokens": max_tokens,
    }
    model_obj = genai.GenerativeModel(model)
    contents = []
    if system_prompt:
        contents.append({"role": "system", "parts": [system_prompt]})
    contents.append({"role": "user", "parts": [user_content]})
    t0 = time.time()
    resp = model_obj.generate_content(contents, generation_config=generation_config)
    latency = time.time() - t0
    usage = getattr(resp, "usage_metadata", None)
    total_tokens = usage.total_token_count if usage else 0
    return {"text": resp.text or "", "latency": latency, "tokens": total_tokens}


def call_gemini_vision_ocr(
    api_key: str,
    model: str,
    images: List[Image.Image],
    temperature: float = 0.0,
    max_tokens: int = 8192,
) -> Dict[str, Any]:
    if genai is None:
        raise RuntimeError("google-generativeai is not installed.")
    genai.configure(api_key=api_key)
    generation_config = {
        "temperature": temperature,
        "max_output_tokens": max_tokens,
    }
    model_obj = genai.GenerativeModel(model)
    parts = [
        "You are a high-accuracy OCR system. Extract all legible text from these pages and preserve structure when possible."
    ]
    for img in images:
        parts.append(img)
    t0 = time.time()
    resp = model_obj.generate_content(parts, generation_config=generation_config)
    latency = time.time() - t0
    usage = getattr(resp, "usage_metadata", None)
    total_tokens = usage.total_token_count if usage else 0
    return {"text": resp.text or "", "latency": latency, "tokens": total_tokens}


def call_openai_text(
    api_key: str,
    model: str,
    system_prompt: str,
    user_content: str,
    temperature: float,
    max_tokens: int,
    base_url: Optional[str] = None,
) -> Dict[str, Any]:
    if OpenAIClient is None:
        raise RuntimeError("openai client is not installed.")
    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = OpenAIClient(**client_kwargs)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_content})
    t0 = time.time()
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    latency = time.time() - t0
    choice = resp.choices[0]
    text = choice.message.content or ""
    usage = getattr(resp, "usage", None)
    total_tokens = (usage.total_tokens if usage else 0) or 0
    return {"text": text, "latency": latency, "tokens": total_tokens}


def call_openai_vision_ocr(
    api_key: str,
    model: str,
    images: List[Image.Image],
    temperature: float = 0.0,
    max_tokens: int = 8192,
    base_url: Optional[str] = None,
) -> Dict[str, Any]:
    if OpenAIClient is None:
        raise RuntimeError("openai client is not installed.")
    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = OpenAIClient(**client_kwargs)

    contents = [
        {
            "type": "text",
            "text": "You are a high-accuracy OCR system. Extract and return all visible text from these images.",
        }
    ]
    for img in images:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        contents.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            }
        )

    messages = [{"role": "user", "content": contents}]
    t0 = time.time()
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    latency = time.time() - t0
    choice = resp.choices[0]
    text = choice.message.content or ""
    usage = getattr(resp, "usage", None)
    total_tokens = (usage.total_tokens if usage else 0) or 0
    return {"text": text, "latency": latency, "tokens": total_tokens}


def call_anthropic_text(
    api_key: str,
    model: str,
    system_prompt: str,
    user_content: str,
    temperature: float,
    max_tokens: int,
) -> Dict[str, Any]:
    if anthropic is None:
        raise RuntimeError("anthropic client is not installed.")
    client = anthropic.Anthropic(api_key=api_key)
    t0 = time.time()
    resp = client.messages.create(
        model=model,
        system=system_prompt or None,
        messages=[{"role": "user", "content": user_content}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    latency = time.time() - t0
    text = "".join([c.text for c in resp.content if getattr(c, "type", "") == "text"])
    usage = getattr(resp, "usage", None)
    total_tokens = getattr(usage, "output_tokens", 0) + getattr(usage, "input_tokens", 0)
    return {"text": text, "latency": latency, "tokens": total_tokens}


def call_xai_text(
    api_key: str,
    model: str,
    system_prompt: str,
    user_content: str,
    temperature: float,
    max_tokens: int,
) -> Dict[str, Any]:
    if XAIClient is None:
        raise RuntimeError("openai client (for xAI) is not installed.")
    client = XAIClient(api_key=api_key, base_url="https://api.x.ai/v1")
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_content})
    t0 = time.time()
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    latency = time.time() - t0
    choice = resp.choices[0]
    text = choice.message.content or ""
    usage = getattr(resp, "usage", None)
    total_tokens = (usage.total_tokens if usage else 0) or 0
    return {"text": text, "latency": latency, "tokens": total_tokens}


def unified_call_text(
    provider: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_content: str,
    temperature: float,
    max_tokens: int,
) -> Dict[str, Any]:
    if provider == "gemini":
        return call_gemini_text(api_key, model, system_prompt, user_content, temperature, max_tokens)
    elif provider == "openai":
        return call_openai_text(api_key, model, system_prompt, user_content, temperature, max_tokens)
    elif provider == "anthropic":
        return call_anthropic_text(api_key, model, system_prompt, user_content, temperature, max_tokens)
    elif provider == "xai":
        return call_xai_text(api_key, model, system_prompt, user_content, temperature, max_tokens)
    else:
        raise ValueError(f"Unknown provider: {provider}")


def unified_vision_ocr(
    provider: str,
    api_key: str,
    model: str,
    images: List[Image.Image],
) -> Dict[str, Any]:
    if provider == "gemini":
        return call_gemini_vision_ocr(api_key, model, images)
    elif provider == "openai":
        return call_openai_vision_ocr(api_key, model, images)
    else:
        raise ValueError(f"Vision OCR is not implemented for provider: {provider}")


# ========== DASHBOARD ========================================================

def render_dashboard(logs: List[Dict[str, Any]], lang_key: str, theme: Dict[str, str]):
    import pandas as pd
    import altair as alt

    labels = LANG_LABELS[lang_key]
    st.subheader(labels["analytics"])
    if not logs:
        st.info("No runs yet. Execute the pipeline first.")
        return

    df = pd.DataFrame(logs)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Runs", len(df))
    col2.metric("Total Agents Run", df["agent_id"].nunique())
    col3.metric("Total Tokens", int(df["tokens"].fillna(0).sum()))
    col4.metric("Avg Latency (s)", round(df["latency"].mean(), 2))

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Latency by Agent**")
        chart = (
            alt.Chart(df)
            .mark_bar()
            .encode(
                x=alt.X("agent_name:N", sort="-y"),
                y="latency:Q",
                color=alt.Color("provider:N", scale=alt.Scale(range=[theme["primary"], theme["secondary"], theme["accent"]])),
                tooltip=["agent_name", "provider", "latency", "tokens"],
            )
            .properties(height=360)
        )
        st.altair_chart(chart, use_container_width=True)

    with c2:
        st.markdown("**Tokens by Agent**")
        chart2 = (
            alt.Chart(df)
            .mark_bar()
            .encode(
                x=alt.X("agent_name:N", sort="-y"),
                y="tokens:Q",
                color=alt.Color("provider:N", scale=alt.Scale(range=[theme["secondary"], theme["primary"], theme["accent"]])),
                tooltip=["agent_name", "provider", "latency", "tokens"],
            )
            .properties(height=360)
        )
        st.altair_chart(chart2, use_container_width=True)

    with st.expander("Show Raw Logs"):
        st.dataframe(df)


# ========== QUICK NOTES PDF ==================================================

def notes_to_pdf_bytes(text: str) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.add_font("Helvetica", "", fname="", uni=True)
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Helvetica", size=11)
    for line in text.split("\n"):
        pdf.multi_cell(0, 6, line)
    return pdf.output(dest="S").encode("latin-1")


# ========== STREAMLIT APP ====================================================

st.set_page_config(
    page_title="TFDA AI Review System",
    page_icon="🌸",
    layout="wide",
)

# Session State init
if "agents" not in st.session_state:
    st.session_state.agents = load_agents_from_yaml()
if "selected_agents" not in st.session_state:
    st.session_state.selected_agents = {ag["id"] for ag in st.session_state.agents}
if "logs" not in st.session_state:
    st.session_state.logs = []
if "ocr_text" not in st.session_state:
    st.session_state.ocr_text = ""
if "notes" not in st.session_state:
    st.session_state.notes = "# Quick Notes\n\n- [ ] Check contraindications\n- [ ] Verify dosage"
if "language" not in st.session_state:
    st.session_state.language = "en"
if "theme_name" not in st.session_state:
    st.session_state.theme_name = "Cherry Blossom"
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False
if "active_section" not in st.session_state:
    st.session_state.active_section = "upload"


theme = FLOWER_THEMES[st.session_state.theme_name]
lang_key = st.session_state.language
labels = LANG_LABELS[lang_key]
inject_global_css(theme, st.session_state.dark_mode)

# ========== SIDEBAR (THEME & GLOBAL CONFIG) =================================

with st.sidebar:
    icon = theme["icon"]
    st.markdown(
        f"""
        <div class="glass-panel" style="text-align:left;">
            <div style="display:flex;align-items:center;gap:0.75rem;">
                <div style="font-size:2.4rem;">{icon}</div>
                <div>
                    <div style="font-weight:800;font-size:1.3rem;
                                background:linear-gradient(90deg,{theme['primary']},{theme['accent']});
                                -webkit-background-clip:text;color:transparent;">
                        {labels['title']}
                    </div>
                    <div style="font-size:0.68rem;letter-spacing:0.22em;
                                text-transform:uppercase;opacity:0.6;">
                        Multi-Agent Review Console
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    # Language & theme toggles
    c1, c2 = st.columns(2)
    with c1:
        st.session_state.language = st.selectbox(
            labels["language_label"],
            options=["en", "zh-TW"],
            format_func=lambda v: "English" if v == "en" else "繁體中文",
            index=0 if st.session_state.language == "en" else 1,
        )
        lang_key = st.session_state.language
        labels = LANG_LABELS[lang_key]
    with c2:
        st.session_state.dark_mode = st.toggle(labels["dark_mode"], value=st.session_state.dark_mode)

    st.session_state.theme_name = st.selectbox(labels["theme_label"], list(FLOWER_THEMES.keys()), index=list(FLOWER_THEMES.keys()).index(st.session_state.theme_name))

    st.markdown("---")

    st.markdown("#### API Keys & Providers")

    # OCR provider
    st.caption("OCR Provider (Vision)")
    ocr_provider = st.radio(
        "OCR Provider",
        options=list(VISION_MODELS.keys()),
        format_func=lambda p: p.capitalize(),
        key="ocr_provider_radio",
    )
    ocr_model = st.selectbox(
        labels["ocr_model"],
        options=VISION_MODELS[ocr_provider],
        key="ocr_model_select",
    )
    st.markdown("---")

    st.markdown(f"#### {labels['notes']}")
    st.session_state.notes = st.text_area(
        "",
        value=st.session_state.notes,
        height=180,
        key="notes_area",
    )

    with st.expander(labels["ai_auto_format"], expanded=False):
        note_provider = st.selectbox(
            labels["provider"],
            options=list(PROVIDER_ENV_VARS.keys()),
            format_func=lambda p: p.capitalize(),
            key="note_provider",
        )
        note_model = st.selectbox(
            labels["model"],
            options=PROVIDER_MODELS[note_provider],
            key="note_model",
        )
        note_prompt = st.text_area(
            labels["instruction_prompt"],
            value="Clean up grammar and formatting. Make it concise.",
            key="note_prompt",
        )
        note_max_tokens = st.number_input(
            labels["max_tokens"],
            min_value=256,
            max_value=8192,
            value=2000,
            step=256,
            key="note_max_tokens",
        )

        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            if st.button(labels["fix_grammar"]):
                st.session_state.note_preset = "Fix grammar and improve readability."
                st.session_state["note_prompt"] = st.session_state.note_preset
        with c2:
            if st.button(labels["checklist"]):
                st.session_state.note_preset = "Convert to a Markdown checklist."
                st.session_state["note_prompt"] = st.session_state.note_preset
        with c3:
            if st.button(labels["markdown"]):
                st.session_state.note_preset = (
                    "Convert the following text into clean, well-structured Markdown. "
                    "Use headers, bullet points, and bold text to improve readability."
                )
                st.session_state["note_prompt"] = st.session_state.note_preset

        if st.button(labels["run_format"], use_container_width=True):
            api_key = get_api_key_ui(note_provider, lang_key)
            if api_key and st.session_state.notes.strip():
                with st.spinner("Refining notes with AI..."):
                    res = unified_call_text(
                        provider=note_provider,
                        api_key=api_key,
                        model=note_model,
                        system_prompt="You are a helpful assistant that reformats notes.",
                        user_content=f"{st.session_state['note_prompt']}\n\nTEXT:\n{st.session_state.notes}",
                        temperature=0.2,
                        max_tokens=note_max_tokens,
                    )
                    st.session_state.notes = res["text"]
                st.toast("Notes updated by AI.")
            else:
                st.warning("Missing API key or empty notes.")

        pdf_bytes = notes_to_pdf_bytes(st.session_state.notes)
        st.download_button(
            labels["download_notes_pdf"],
            data=pdf_bytes,
            file_name="notes.pdf",
            mime="application/pdf",
            use_container_width=True,
        )


# ========== MAIN LAYOUT ======================================================

labels = LANG_LABELS[st.session_state.language]
theme = FLOWER_THEMES[st.session_state.theme_name]

# Top nav
nav_cols = st.columns(5)
sections = ["upload", "preview", "config", "execute", "dashboard"]
section_labels = [
    labels["upload"],
    labels["preview"],
    labels["config"],
    labels["execute"],
    labels["dashboard"],
]
for i, (col, sec, lab) in enumerate(zip(nav_cols, sections, section_labels)):
    with col:
        if st.button(lab, key=f"nav_{sec}"):
            st.session_state.active_section = sec

st.write("")

with st.container():
    st.markdown('<div class="main-card">', unsafe_allow_html=True)

    # Section header with wow pill
    if st.session_state.active_section == "upload":
        section_title = labels["upload"]
        subtitle = "Upload PDF or image, select pages and run OCR."
    elif st.session_state.active_section == "preview":
        section_title = labels["preview"]
        subtitle = "Review & refine extracted text before analysis."
    elif st.session_state.active_section == "config":
        section_title = labels["config"]
        subtitle = "Configure multi-agent swarm, prompts & models."
    elif st.session_state.active_section == "execute":
        section_title = labels["execute"]
        subtitle = "Run the pipeline with wow status indicators."
    else:
        section_title = labels["dashboard"]
        subtitle = "Interactive analytics dashboard for agent performance."

    left, right = st.columns([3, 2])
    with left:
        st.markdown(
            f"""
            <div style="display:flex;flex-direction:column;gap:0.3rem;">
                <div class="wow-pill">
                    <div class="wow-status-dot"></div>
                    <span>{labels['wow_timeline']}</span>
                </div>
                <h1 style="margin:0;font-size:1.8rem;font-weight:800;
                           background:linear-gradient(90deg,{theme['primary']},{theme['accent']});
                           -webkit-background-clip:text;color:transparent;">
                    {section_title}
                </h1>
                <div style="opacity:0.7;font-size:0.9rem;">{subtitle}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        # Simple cumulative progress indicator based on logs
        total_agents = len(st.session_state.agents)
        finished_agents = len({log["agent_id"] for log in st.session_state.logs})
        pct = 0.0
        if total_agents > 0:
            pct = finished_agents / total_agents * 100.0
        st.markdown(
            f"""
            <div style="display:flex;flex-direction:column;gap:0.4rem;align-items:flex-end;">
                <div style="font-size:0.75rem;opacity:0.7;">Pipeline progress</div>
                <div class="timeline-bar" style="width:100%;">
                    <div class="timeline-fill" style="width:{pct}%;"></div>
                </div>
                <div style="font-size:0.8rem;opacity:0.75;">
                    {finished_agents}/{total_agents} agents completed
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ========== SECTION: UPLOAD & OCR =======================================

    if st.session_state.active_section == "upload":
        upload_col, cfg_col = st.columns([2, 1])

        with upload_col:
            uploaded = st.file_uploader(
                "Upload PDF or image",
                type=["pdf", "png", "jpg", "jpeg"],
            )
            if uploaded is not None:
                file_bytes = uploaded.read()
                st.session_state.uploaded_bytes = file_bytes
                st.session_state.uploaded_name = uploaded.name
                st.info(f"File loaded: {uploaded.name}")
                if uploaded.type == "application/pdf":
                    try:
                        # Only to detect page count
                        pages = convert_from_bytes(file_bytes, dpi=100)
                        st.session_state.pdf_page_count = len(pages)
                    except Exception as e:
                        st.error(f"Error reading PDF pages: {e}")
                else:
                    st.session_state.pdf_page_count = 1

        with cfg_col:
            st.subheader(labels["ocr_settings"])
            if "pdf_page_count" in st.session_state:
                total_pages = st.session_state.pdf_page_count
                st.caption(f"{total_pages} pages detected.")
                default_range = f"1-{min(5, total_pages)}"
            else:
                total_pages = 0
                default_range = "1-3"

            page_range = st.text_input(labels["page_range"], value=default_range, key="page_range_input")
            api_key = get_api_key_ui(ocr_provider, lang_key)

            if st.button(labels["run_ocr"], type="primary", use_container_width=True):
                if not uploaded:
                    st.warning("Please upload a file first.")
                elif not api_key:
                    st.warning("Please provide an API key.")
                elif ocr_provider not in VISION_MODELS:
                    st.warning("Selected provider has no vision OCR implementation.")
                else:
                    if uploaded.type == "application/pdf":
                        if "pdf_page_count" not in st.session_state:
                            st.error("Could not determine PDF pages.")
                        else:
                            pages = convert_from_bytes(st.session_state.uploaded_bytes, dpi=200)
                            total = len(pages)
                            selected_pages = parse_page_range(page_range, total)
                            if not selected_pages:
                                st.warning("Invalid page range.")
                            else:
                                images = [pages[i - 1] for i in selected_pages]
                                with st.spinner("Running OCR on selected pages..."):
                                    res = unified_vision_ocr(ocr_provider, api_key, ocr_model, images)
                                    st.session_state.ocr_text = res["text"]
                                    st.success(f"OCR completed in {res['latency']:.2f}s")
                    else:
                        try:
                            img = Image.open(io.BytesIO(st.session_state.uploaded_bytes))
                            with st.spinner("Running OCR on image..."):
                                res = unified_vision_ocr(ocr_provider, api_key, ocr_model, [img])
                                st.session_state.ocr_text = res["text"]
                                st.success(f"OCR completed in {res['latency']:.2f}s")
                        except Exception as e:
                            st.error(f"Failed to process image: {e}")

    # ========== SECTION: PREVIEW ============================================

    elif st.session_state.active_section == "preview":
        top_bar = st.columns([3, 1, 1])
        with top_bar[0]:
            st.subheader(labels["document_content"])
        with top_bar[1]:
            if st.button(labels["clear"]):
                st.session_state.ocr_text = ""
        with top_bar[2]:
            if st.button(labels["next_pipeline"], type="primary"):
                st.session_state.active_section = "execute"

        st.session_state.ocr_text = st.text_area(
            "",
            value=st.session_state.ocr_text,
            height=450,
            key="ocr_preview_textarea",
        )

    # ========== SECTION: AGENT CONFIG =======================================

    elif st.session_state.active_section == "config":
        st.subheader(labels["agent_config"])

        if st.button(labels["reset_defaults"]):
            st.session_state.agents = load_agents_from_yaml()
            st.session_state.selected_agents = {ag["id"] for ag in st.session_state.agents}
            st.success("Agents reset from agents.yaml")

        agents = st.session_state.agents
        if not agents:
            st.warning("No agents defined. Add agents in agents.yaml.")
        else:
            for idx, ag in enumerate(agents):
                st.markdown(f"### {idx + 1}. {ag['name']}")
                st.markdown(f"*{ag['description']}*")
                st.checkbox(
                    "Selected for pipeline",
                    value=ag["id"] in st.session_state.selected_agents,
                    key=f"agent_select_{ag['id']}",
                )
                c1, c2, c3 = st.columns([2, 2, 1])
                with c1:
                    ag["name"] = st.text_input("Name", value=ag["name"], key=f"name_{ag['id']}")
                with c2:
                    ag["description"] = st.text_input(
                        "Description", value=ag["description"], key=f"desc_{ag['id']}"
                    )
                with c3:
                    ag["provider"] = st.selectbox(
                        labels["provider"],
                        options=list(PROVIDER_ENV_VARS.keys()),
                        index=list(PROVIDER_ENV_VARS.keys()).index(ag["provider"])
                        if ag["provider"] in PROVIDER_ENV_VARS
                        else 0,
                        key=f"prov_{ag['id']}",
                    )
                c4, c5 = st.columns([2, 1])
                with c4:
                    ag["model"] = st.selectbox(
                        labels["model"],
                        options=PROVIDER_MODELS[ag["provider"]],
                        index=PROVIDER_MODELS[ag["provider"]].index(ag["model"])
                        if ag["model"] in PROVIDER_MODELS[ag["provider"]]
                        else 0,
                        key=f"model_{ag['id']}",
                    )
                with c5:
                    ag["temperature"] = st.slider(
                        "Temperature",
                        min_value=0.0,
                        max_value=1.0,
                        value=float(ag["temperature"]),
                        step=0.05,
                        key=f"temp_{ag['id']}",
                    )
                c6, c7 = st.columns([3, 1])
                with c6:
                    ag["system_prompt"] = st.text_area(
                        "System Prompt", value=ag["system_prompt"], height=80, key=f"sys_{ag['id']}"
                    )
                with c7:
                    ag["max_tokens"] = st.number_input(
                        labels["max_tokens"],
                        min_value=128,
                        max_value=8192,
                        value=int(ag["max_tokens"]),
                        step=128,
                        key=f"max_{ag['id']}",
                    )
                ag["user_prompt"] = st.text_area(
                    "User Prompt (prefix)", value=ag["user_prompt"], height=80, key=f"user_{ag['id']}"
                )
                st.markdown("---")

            # Update selected_agents set
            selected = set()
            for ag in agents:
                if st.session_state.get(f"agent_select_{ag['id']}", True):
                    selected.add(ag["id"])
            st.session_state.selected_agents = selected

            yaml_data = {"agents": agents}
            yaml_str = yaml.dump(yaml_data, allow_unicode=True, sort_keys=False)
            st.download_button(
                "Download updated agents.yaml",
                data=yaml_str.encode("utf-8"),
                file_name="agents_updated.yaml",
                mime="text/yaml",
            )

    # ========== SECTION: EXECUTE PIPELINE ===================================

    elif st.session_state.active_section == "execute":
        st.subheader(labels["execute"])
        agents = st.session_state.agents
        selected_ids = st.session_state.selected_agents

        if not st.session_state.ocr_text.strip():
            st.warning("OCR text is empty. Please run OCR or paste text in Preview section.")
        elif not agents or not selected_ids:
            st.warning("No agents selected. Configure in the Agent Config section.")
        else:
            # Show summary
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Selected Agents", len(selected_ids))
            with col_b:
                st.metric("Chars in Input", len(st.session_state.ocr_text))
            with col_c:
                st.metric("Previous Runs", len(st.session_state.logs))

            if st.button(labels["run_pipeline"], type="primary", use_container_width=True):
                st.session_state.logs = []
                text_input = st.session_state.ocr_text
                agents_to_run = [ag for ag in agents if ag["id"] in selected_ids]

                progress_placeholder = st.empty()
                log_placeholder = st.container()

                for idx, ag in enumerate(agents_to_run, start=1):
                    provider = ag["provider"]
                    api_key = get_api_key_ui(provider, lang_key)
                    if not api_key:
                        st.warning(f"Missing API key for provider {provider}. Aborting.")
                        break
                    step_pct = idx / len(agents_to_run) * 100.0
                    progress_placeholder.markdown(
                        f"""
                        <div style="margin-bottom:0.6rem;">
                            <div style="font-size:0.8rem;opacity:0.75;">
                                Running agent {idx}/{len(agents_to_run)}: <b>{ag['name']}</b>
                            </div>
                            <div class="timeline-bar">
                                <div class="timeline-fill" style="width:{step_pct}%;"></div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    user_content = f"{ag['user_prompt']}\n\n{('Document text:\n' + text_input).strip()}"
                    try:
                        with st.spinner(f"Agent {ag['name']} in progress..."):
                            res = unified_call_text(
                                provider=provider,
                                api_key=api_key,
                                model=ag["model"],
                                system_prompt=ag["system_prompt"],
                                user_content=user_content,
                                temperature=float(ag["temperature"]),
                                max_tokens=int(ag["max_tokens"]),
                            )
                    except Exception as e:
                        st.error(f"Agent {ag['name']} failed: {e}")
                        break

                    log_entry = {
                        "agent_id": ag["id"],
                        "agent_name": ag["name"],
                        "provider": provider,
                        "model": ag["model"],
                        "input": text_input,
                        "output": res["text"],
                        "latency": float(res["latency"]),
                        "tokens": int(res["tokens"]),
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    }
                    st.session_state.logs.append(log_entry)
                    text_input = res["output"]  # chaining

                    with log_placeholder:
                        with st.expander(f"Output: {ag['name']} ({provider}/{ag['model']})", expanded=False):
                            st.write(res["text"])

                progress_placeholder.empty()
                st.success("Pipeline finished. Check the Analytics tab for insights.")

    # ========== SECTION: DASHBOARD ==========================================

    else:
        render_dashboard(st.session_state.logs, lang_key, theme)

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================================
