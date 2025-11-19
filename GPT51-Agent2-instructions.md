Below is a full Streamlit implementation that ports your React design, adds the requested “wow” UI features, and uses `agents.yaml` plus multi‑provider LLM support.

It includes:

- Streamlit app (`app.py`) with:
  - Tabs: Source PDF / Agents Config / Results / Dashboard
  - PDF upload + per‑page selection
  - Agent editing (prompt, provider, model, temperature, max tokens, enabled)
  - Pipeline execution with progress, per‑agent results, cost estimation
  - Notes area with markdown preview, AI “optimize notes”, and download
  - Wow status indicators + interactive dashboard
  - Wow UI: light/dark themes, English / Traditional Chinese, 20 flower styles
  - API key handling using env vars or UI input (no env key ever shown)
  - Model selection for:
    - Gemini: `gemini-2.5-flash`, `gemini-2.5-flash-lite`
    - OpenAI: `gpt-4o-mini`, `gpt-4.1-mini`, `gpt-5-nano`
    - Anthropic: example `claude-3-5-sonnet-latest`
    - xAI: example `grok-beta`
- Sample `agents.yaml` for initial agents

You can drop these directly into a Hugging Face Space using Streamlit.

---

## 1. `agents.yaml`

```yaml
# agents.yaml
agents:
  - id: "exec_summary"
    name: "Executive Summary"
    description: "Create a concise executive summary of the selected PDF pages."
    provider: "gemini"
    model: "gemini-2.5-flash"
    system_prompt: |
      You are a senior financial analyst. Read the provided PDF content
      and produce a clear, concise executive summary. Focus on key findings,
      notable trends, and any risks or opportunities.
    temperature: 0.2
    max_output_tokens: 2048
    enabled: true

  - id: "detailed_analysis"
    name: "Detailed Analysis"
    description: "Provide a structured, detailed analysis with bullet points."
    provider: "openai"
    model: "gpt-4o-mini"
    system_prompt: |
      You are a meticulous financial analyst. Given the PDF content,
      create a detailed, structured analysis using headings, bullet points,
      and clear reasoning.
    temperature: 0.3
    max_output_tokens: 4096
    enabled: true

  - id: "risk_assessment"
    name: "Risk Assessment"
    description: "Identify and explain key risks from the document."
    provider: "anthropic"
    model: "claude-3-5-sonnet-latest"
    system_prompt: |
      You are a risk officer. From the provided PDF content, identify key
      financial, operational, and strategic risks, and explain why they matter.
    temperature: 0.2
    max_output_tokens: 2048
    enabled: true

  - id: "insight_extractor"
    name: "Key Insights"
    description: "Extract key insights, anomalies, and action items."
    provider: "xai"
    model: "grok-beta"
    system_prompt: |
      You are an insights analyst. Extract key insights, anomalies, and
      recommended next actions from the PDF content.
    temperature: 0.4
    max_output_tokens: 2048
    enabled: true
```

---

## 2. `app.py` (Streamlit app)

```python
import os
import io
import time
from dataclasses import dataclass, asdict
from enum import Enum
from typing import List, Optional, Dict, Any

import streamlit as st
from PIL import Image
import yaml
import pdfplumber
import pandas as pd

# LLM SDKs (install in requirements.txt):
# google-generativeai, openai, anthropic, requests (for xAI if needed)
import google.generativeai as genai
from openai import OpenAI
import anthropic
import requests

# -----------------------------
# 0. CONFIG & CONSTANTS
# -----------------------------

APP_VERSION = "v2.5.0-beta"

SUPPORTED_GEMINI_MODELS = ["gemini-2.5-flash", "gemini-2.5-flash-lite"]
SUPPORTED_OPENAI_MODELS = ["gpt-4o-mini", "gpt-4.1-mini", "gpt-5-nano"]
SUPPORTED_ANTHROPIC_MODELS = ["claude-3-5-sonnet-latest"]
SUPPORTED_XAI_MODELS = ["grok-beta"]

# Simple (placeholder) cost table per 1K tokens (input+output combined)
COST_PER_1K = {
    "gemini": 0.0005,
    "openai": 0.0010,
    "anthropic": 0.0012,
    "xai": 0.0008,
}

# 20 flower styles
FLOWER_STYLES = [
    "Rose", "Lotus", "Sakura", "Lavender", "Sunflower",
    "Peony", "Tulip", "Orchid", "Magnolia", "Camellia",
    "Iris", "Dahlia", "Hydrangea", "Marigold", "Jasmine",
    "Lilac", "Poppy", "Chrysanthemum", "Gardenia", "Carnation",
]

# Map flower styles to accent colors & optional background patterns
FLOWER_THEME_MAP = {
    "Rose":        {"accent": "#e11d48", "bg": "#fff1f2"},
    "Lotus":       {"accent": "#0ea5e9", "bg": "#ecfeff"},
    "Sakura":      {"accent": "#fb7185", "bg": "#fff1f2"},
    "Lavender":    {"accent": "#a855f7", "bg": "#faf5ff"},
    "Sunflower":   {"accent": "#f59e0b", "bg": "#fffbeb"},
    "Peony":       {"accent": "#ec4899", "bg": "#fdf2f8"},
    "Tulip":       {"accent": "#f97316", "bg": "#fff7ed"},
    "Orchid":      {"accent": "#8b5cf6", "bg": "#f5f3ff"},
    "Magnolia":    {"accent": "#22c55e", "bg": "#f0fdf4"},
    "Camellia":    {"accent": "#db2777", "bg": "#fdf2f8"},
    "Iris":        {"accent": "#6366f1", "bg": "#eef2ff"},
    "Dahlia":      {"accent": "#f97316", "bg": "#fff7ed"},
    "Hydrangea":   {"accent": "#38bdf8", "bg": "#ecfeff"},
    "Marigold":    {"accent": "#facc15", "bg": "#fefce8"},
    "Jasmine":     {"accent": "#22c55e", "bg": "#f0fdf4"},
    "Lilac":       {"accent": "#a855f7", "bg": "#faf5ff"},
    "Poppy":       {"accent": "#ef4444", "bg": "#fef2f2"},
    "Chrysanthemum": {"accent": "#eab308", "bg": "#fffbeb"},
    "Gardenia":    {"accent": "#14b8a6", "bg": "#ecfeff"},
    "Carnation":   {"accent": "#f97316", "bg": "#fff7ed"},
}

# Basic i18n (English / Traditional Chinese)
I18N = {
    "en": {
        "app_title": "AgentFlow PDF Intelligence",
        "source_pdf": "Source PDF",
        "agents_config": "Agents Config",
        "results": "Results",
        "dashboard": "Dashboard",
        "upload_pdf": "Upload PDF",
        "drag_drop_pdf": "Drag & drop a PDF or browse files",
        "pages_detected": "Pages detected",
        "select_pages": "Select pages to analyze",
        "agents": "Agents",
        "agent_list": "Agent List",
        "run_pipeline": "Run Analysis",
        "running_pipeline": "Running Pipeline...",
        "no_results": "No results yet. Configure agents and press Run.",
        "notes": "Quick Notes",
        "notes_placeholder": "Type notes here or quote from results. Use AI to auto-format.",
        "optimize_notes": "AI Format (Magic)",
        "download_md": "Download Markdown",
        "markdown_preview": "Markdown Preview",
        "edit_mode": "Edit",
        "save_version": "Save Version",
        "project_label": "Project",
        "project_name": "Financial Analysis Q3",
        "app_status_idle": "Idle",
        "app_status_processing": "Processing",
        "app_status_completed": "Completed",
        "app_status_error": "Error",
        "theme": "Theme",
        "light": "Light",
        "dark": "Dark",
        "flower_style": "Flower Style",
        "language": "Language",
        "english": "English",
        "traditional_chinese": "Traditional Chinese",
        "api_settings": "API Settings",
        "using_env_key": "Using environment key",
        "enter_api_key": "Enter API key",
        "gemini_key": "Gemini API Key",
        "openai_key": "OpenAI API Key",
        "anthropic_key": "Anthropic API Key",
        "xai_key": "xAI API Key",
        "status_overview": "Status Overview",
        "total_cost": "Total Cost (est.)",
        "total_runs": "Total Runs",
        "agents_used": "Agents Used",
        "pages_selected": "Pages Selected",
        "run_history": "Run History",
        "agent_outputs": "Agent Outputs",
        "quote_to_notes": "Quote to Notes",
        "provider": "Provider",
        "model": "Model",
        "system_prompt": "System Prompt",
        "temperature": "Temperature",
        "max_tokens": "Max Output Tokens",
        "enabled": "Enabled",
        "select_all": "Select All",
        "clear_all": "Clear All",
        "wow_status": "Wow Status",
        "ready_to_run": "Ready to run",
        "missing_pdf_or_pages": "Upload a PDF and select pages first.",
        "missing_agents": "Add or enable at least one agent.",
        "pipeline_error": "Pipeline failed. Check logs and API keys.",
        "optimize_notes_error": "Failed to optimize notes. Check API keys.",
        "no_api_key": "No API key available. Please configure API settings.",
        "config_versions": "Config Versions",
        "restore": "Restore",
        "load_agents_fail": "Could not load agents.yaml; using in-app defaults.",
    },
    "zh-TW": {
        "app_title": "AgentFlow PDF 智能分析",
        "source_pdf": "來源 PDF",
        "agents_config": "智能代理設定",
        "results": "分析結果",
        "dashboard": "總覽儀表板",
        "upload_pdf": "上傳 PDF",
        "drag_drop_pdf": "拖拉 PDF 或選擇檔案",
        "pages_detected": "偵測到的頁面",
        "select_pages": "選擇要分析的頁面",
        "agents": "智能代理",
        "agent_list": "代理列表",
        "run_pipeline": "執行分析",
        "running_pipeline": "分析管線執行中...",
        "no_results": "尚無結果。請先設定代理並按下「執行分析」。",
        "notes": "快速筆記",
        "notes_placeholder": "在此輸入筆記或從結果引用。可使用 AI 自動整理格式。",
        "optimize_notes": "AI 美化 (魔法)",
        "download_md": "下載 Markdown",
        "markdown_preview": "Markdown 預覽",
        "edit_mode": "編輯",
        "save_version": "儲存版本",
        "project_label": "專案",
        "project_name": "第三季財務分析",
        "app_status_idle": "待機中",
        "app_status_processing": "處理中",
        "app_status_completed": "已完成",
        "app_status_error": "錯誤",
        "theme": "主題",
        "light": "亮色",
        "dark": "暗色",
        "flower_style": "花卉風格",
        "language": "語言",
        "english": "英文",
        "traditional_chinese": "繁體中文",
        "api_settings": "API 設定",
        "using_env_key": "使用環境變數",
        "enter_api_key": "輸入 API Key",
        "gemini_key": "Gemini API 金鑰",
        "openai_key": "OpenAI API 金鑰",
        "anthropic_key": "Anthropic API 金鑰",
        "xai_key": "xAI API 金鑰",
        "status_overview": "狀態總覽",
        "total_cost": "預估總成本",
        "total_runs": "執行次數",
        "agents_used": "使用代理數",
        "pages_selected": "選取頁數",
        "run_history": "執行紀錄",
        "agent_outputs": "代理輸出",
        "quote_to_notes": "引用到筆記",
        "provider": "供應商",
        "model": "模型",
        "system_prompt": "系統提示詞",
        "temperature": "溫度",
        "max_tokens": "最大輸出 Token",
        "enabled": "啟用",
        "select_all": "全選",
        "clear_all": "全不選",
        "wow_status": "Wow 狀態",
        "ready_to_run": "已就緒",
        "missing_pdf_or_pages": "請先上傳 PDF 並選擇頁面。",
        "missing_agents": "請新增或啟用至少一個代理。",
        "pipeline_error": "管線執行失敗，請檢查記錄與 API 金鑰。",
        "optimize_notes_error": "筆記優化失敗，請檢查 API 金鑰。",
        "no_api_key": "尚未設定 API 金鑰，請到 API 設定頁面。",
        "config_versions": "設定版本",
        "restore": "還原",
        "load_agents_fail": "無法讀取 agents.yaml，改用內建預設值。",
    },
}

# -----------------------------
# 1. DATA MODELS
# -----------------------------

class AppStatus(str, Enum):
    IDLE = "idle"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class PdfPage:
    page_number: int
    text: str = ""
    selected: bool = True


@dataclass
class AgentConfig:
    id: str
    name: str
    description: str
    provider: str
    model: str
    system_prompt: str
    temperature: float = 0.2
    max_output_tokens: int = 2048
    enabled: bool = True


@dataclass
class ExecutionResult:
    agent_id: str
    agent_name: str
    output: str
    timestamp: float
    cost: float
    status: str
    provider_used: str
    model_used: str
    tokens: int


@dataclass
class ProjectConfig:
    id: str
    name: str
    created_at: float
    agents: List[AgentConfig]


# -----------------------------
# 2. UTILITIES
# -----------------------------

def t(key: str) -> str:
    """Translate a UI string based on current language."""
    lang = st.session_state.get("language", "en")
    return I18N.get(lang, I18N["en"]).get(key, key)


def init_session_state():
    if "initialized" not in st.session_state:
        st.session_state.initialized = True
        st.session_state.active_tab = "upload"
        st.session_state.pages: List[PdfPage] = []
        st.session_state.pdf_name = None
        st.session_state.agents: List[AgentConfig] = []
        st.session_state.execution_results: List[ExecutionResult] = []
        st.session_state.status: AppStatus = AppStatus.IDLE
        st.session_state.total_cost: float = 0.0
        st.session_state.run_count: int = 0
        st.session_state.config_history: List[ProjectConfig] = []
        st.session_state.notes: str = ""
        st.session_state.notes_preview: bool = False
        st.session_state.language: str = "en"
        st.session_state.theme_mode: str = "light"
        st.session_state.flower_style: str = FLOWER_STYLES[0]
        # API keys from user input (fallback when env not provided)
        st.session_state.user_gemini_api_key = ""
        st.session_state.user_openai_api_key = ""
        st.session_state.user_anthropic_api_key = ""
        st.session_state.user_xai_api_key = ""


def load_agents_from_yaml(path: str = "agents.yaml") -> List[AgentConfig]:
    agents: List[AgentConfig] = []
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            for item in data.get("agents", []):
                agents.append(AgentConfig(**item))
        except Exception:
            st.sidebar.warning(t("load_agents_fail"))
    if not agents:
        # Minimal fallback
        agents = [
            AgentConfig(
                id="fallback_summary",
                name="Summary (Fallback)",
                description="Simple summary agent (fallback when YAML not loaded).",
                provider="gemini",
                model=SUPPORTED_GEMINI_MODELS[0],
                system_prompt="You are a helpful summarization assistant.",
            )
        ]
    return agents


def apply_theme():
    """Inject CSS based on theme_mode + flower_style."""
    mode = st.session_state.get("theme_mode", "light")
    flower = st.session_state.get("flower_style", FLOWER_STYLES[0])
    theme = FLOWER_THEME_MAP.get(flower, FLOWER_THEME_MAP[FLOWER_STYLES[0]])

    accent = theme["accent"]
    bg = theme["bg"]
    base_bg = "#0f172a" if mode == "dark" else bg
    base_text = "#e5e7eb" if mode == "dark" else "#0f172a"
    card_bg = "#1f2937" if mode == "dark" else "#ffffff"
    muted = "#9ca3af" if mode == "dark" else "#6b7280"

    css = f"""
    <style>
    :root {{
        --af-accent: {accent};
        --af-bg: {base_bg};
        --af-text: {base_text};
        --af-card-bg: {card_bg};
        --af-muted: {muted};
    }}
    .stApp {{
        background: radial-gradient(circle at top left, {accent}11 0, transparent 40%), {base_bg};
        color: {base_text};
    }}
    .af-card {{
        background-color: {card_bg} !important;
        border-radius: 16px;
        padding: 1.25rem 1.5rem;
        box-shadow: 0 18px 35px rgba(0,0,0,0.08);
        border: 1px solid rgba(148,163,184,0.25);
    }}
    .af-badge {{
        display:inline-flex;
        align-items:center;
        padding:0.1rem 0.5rem;
        border-radius:999px;
        font-size:0.7rem;
        font-weight:600;
        background-color: {accent}22;
        color:{accent};
        border:1px solid {accent}55;
    }}
    .af-status-chip {{
        display:inline-flex;
        align-items:center;
        gap:0.4rem;
        padding:0.2rem 0.7rem;
        border-radius:999px;
        font-size:0.75rem;
        font-weight:600;
        background: linear-gradient(90deg, {accent}33, transparent);
        color:{accent};
        border:1px solid {accent}77;
    }}
    .af-status-dot {{
        width:0.55rem;
        height:0.55rem;
        border-radius:999px;
        background-color:{accent};
        box-shadow:0 0 0 6px {accent}22;
    }}
    .af-tab-title {{
        font-weight:600;
        margin-bottom:0.5rem;
    }}
    textarea, .stTextArea textarea {{
        font-family: "JetBrains Mono", "SF Mono", ui-monospace, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace !important;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def get_effective_api_key(env_name: str, user_key_name: str) -> Optional[str]:
    """Get API key from env or from user input in session_state."""
    env_key = os.getenv(env_name)
    if env_key:
        return env_key
    return st.session_state.get(user_key_name) or None


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def estimate_cost(provider: str, tokens: int) -> float:
    base = COST_PER_1K.get(provider, 0.001)
    return (tokens / 1000.0) * base


# -----------------------------
# 3. PDF HANDLING
# -----------------------------

def load_pdf_pages(file_bytes: bytes) -> List[PdfPage]:
    pages: List[PdfPage] = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""
            pages.append(PdfPage(page_number=i, text=text, selected=True))
    return pages


# -----------------------------
# 4. LLM EXECUTION
# -----------------------------

def call_gemini(
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    temperature: float,
) -> str:
    key = get_effective_api_key("GEMINI_API_KEY", "user_gemini_api_key")
    if not key:
        raise RuntimeError(t("no_api_key") + " [Gemini]")
    genai.configure(api_key=key)
    model = genai.GenerativeModel(
        model_name,
        system_instruction=system_prompt or None,
    )
    response = model.generate_content(
        [user_prompt],
        generation_config={
            "max_output_tokens": max_tokens,
            "temperature": temperature,
        },
    )
    return response.text or ""


def call_openai(
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    temperature: float,
) -> str:
    key = get_effective_api_key("OPENAI_API_KEY", "user_openai_api_key")
    if not key:
        raise RuntimeError(t("no_api_key") + " [OpenAI]")
    client = OpenAI(api_key=key)
    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt or ""},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return resp.choices[0].message.content or ""


def call_anthropic(
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    temperature: float,
) -> str:
    key = get_effective_api_key("ANTHROPIC_API_KEY", "user_anthropic_api_key")
    if not key:
        raise RuntimeError(t("no_api_key") + " [Anthropic]")
    client = anthropic.Anthropic(api_key=key)
    resp = client.messages.create(
        model=model_name,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system_prompt or "",
        messages=[{"role": "user", "content": user_prompt}],
    )
    return "".join([b.text for b in resp.content if getattr(b, "type", "") == "text"])


def call_xai(
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    temperature: float,
) -> str:
    """
    xAI's Grok API is OpenAI-compatible at https://api.x.ai/v1.
    Adjust if your deployment differs.
    """
    key = get_effective_api_key("XAI_API_KEY", "user_xai_api_key")
    if not key:
        raise RuntimeError(t("no_api_key") + " [xAI]")
    url = "https://api.x.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt or ""},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    resp = requests.post(url, json=payload, headers=headers, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def execute_agent_on_text(agent: AgentConfig, context: str) -> ExecutionResult:
    start = time.time()
    provider = agent.provider.lower()
    text = ""

    if provider == "gemini":
        text = call_gemini(
            agent.model,
            agent.system_prompt,
            context,
            agent.max_output_tokens,
            agent.temperature,
        )
    elif provider == "openai":
        text = call_openai(
            agent.model,
            agent.system_prompt,
            context,
            agent.max_output_tokens,
            agent.temperature,
        )
    elif provider == "anthropic":
        text = call_anthropic(
            agent.model,
            agent.system_prompt,
            context,
            agent.max_output_tokens,
            agent.temperature,
        )
    elif provider == "xai":
        text = call_xai(
            agent.model,
            agent.system_prompt,
            context,
            agent.max_output_tokens,
            agent.temperature,
        )
    else:
        raise RuntimeError(f"Unsupported provider: {provider}")

    tokens = estimate_tokens(text)
    cost = estimate_cost(provider, tokens)
    end = time.time()
    return ExecutionResult(
        agent_id=agent.id,
        agent_name=agent.name,
        output=text,
        timestamp=end,
        cost=cost,
        status="success",
        provider_used=provider,
        model_used=agent.model,
        tokens=tokens,
    )


def optimize_notes_with_gemini(notes: str) -> str:
    key = get_effective_api_key("GEMINI_API_KEY", "user_gemini_api_key")
    if not key:
        raise RuntimeError(t("no_api_key") + " [Gemini]")
    genai.configure(api_key=key)
    system_prompt = (
        "You are a meticulous technical editor. Given some rough notes in "
        "Markdown, reformat and lightly edit for clarity, preserving all content."
    )
    model = genai.GenerativeModel(
        "gemini-2.5-flash-lite", system_instruction=system_prompt
    )
    response = model.generate_content(
        [f"Here are the notes to polish:\n\n{notes}"],
        generation_config={"max_output_tokens": 2048, "temperature": 0.2},
    )
    return response.text or notes


# -----------------------------
# 5. PIPELINE EXECUTION
# -----------------------------

def run_pipeline():
    pages: List[PdfPage] = st.session_state.pages
    agents: List[AgentConfig] = st.session_state.agents

    selected_pages = [p for p in pages if p.selected]
    enabled_agents = [a for a in agents if a.enabled]

    if not selected_pages:
        st.error(t("missing_pdf_or_pages"))
        return
    if not enabled_agents:
        st.error(t("missing_agents"))
        return

    st.session_state.status = AppStatus.PROCESSING
    st.session_state.execution_results = []

    # Aggregate text from selected pages
    aggregated = ""
    for p in selected_pages:
        aggregated += f"[Page {p.page_number}]\n{p.text}\n\n"

    progress = st.progress(0.0)
    status_placeholder = st.empty()
    total_agents = len(enabled_agents)

    results: List[ExecutionResult] = []
    running_cost = st.session_state.total_cost

    for idx, agent in enumerate(enabled_agents, start=1):
        status_placeholder.markdown(
            f"**{t('wow_status')}**: {agent.name} ({idx}/{total_agents})"
        )
        try:
            result = execute_agent_on_text(agent, aggregated)
        except Exception as e:
            st.error(f"{t('pipeline_error')}: {e}")
            st.session_state.status = AppStatus.ERROR
            return

        results.append(result)
        running_cost += result.cost
        st.session_state.execution_results = results
        st.session_state.total_cost = running_cost
        progress.progress(idx / total_agents)

    progress.progress(1.0)
    st.session_state.status = AppStatus.COMPLETED
    st.session_state.run_count += 1


# -----------------------------
# 6. UI SECTIONS
# -----------------------------

def render_api_settings():
    st.subheader(t("api_settings"))
    cols = st.columns(2)

    with cols[0]:
        gem_env = os.getenv("GEMINI_API_KEY")
        st.caption(t("gemini_key"))
        if gem_env:
            st.info(t("using_env_key"))
        else:
            st.session_state.user_gemini_api_key = st.text_input(
                t("enter_api_key"),
                type="password",
                key="input_gemini_key",
                value=st.session_state.user_gemini_api_key,
            )

        openai_env = os.getenv("OPENAI_API_KEY")
        st.caption(t("openai_key"))
        if openai_env:
            st.info(t("using_env_key"))
        else:
            st.session_state.user_openai_api_key = st.text_input(
                t("enter_api_key"),
                type="password",
                key="input_openai_key",
                value=st.session_state.user_openai_api_key,
            )

    with cols[1]:
        anth_env = os.getenv("ANTHROPIC_API_KEY")
        st.caption(t("anthropic_key"))
        if anth_env:
            st.info(t("using_env_key"))
        else:
            st.session_state.user_anthropic_api_key = st.text_input(
                t("enter_api_key"),
                type="password",
                key="input_anthropic_key",
                value=st.session_state.user_anthropic_api_key,
            )

        xai_env = os.getenv("XAI_API_KEY")
        st.caption(t("xai_key"))
        if xai_env:
            st.info(t("using_env_key"))
        else:
            st.session_state.user_xai_api_key = st.text_input(
                t("enter_api_key"),
                type="password",
                key="input_xai_key",
                value=st.session_state.user_xai_api_key,
            )


def render_sidebar():
    with st.sidebar:
        # Language & Theme controls
        st.markdown(f"### {t('app_title')}")
        lang = st.radio(
            t("language"),
            ["en", "zh-TW"],
            format_func=lambda v: t("english") if v == "en" else t("traditional_chinese"),
            key="language_radio",
            index=0 if st.session_state.language == "en" else 1,
        )
        st.session_state.language = lang

        col_theme1, col_theme2 = st.columns(2)
        with col_theme1:
            mode = st.selectbox(
                t("theme"),
                ["light", "dark"],
                format_func=lambda v: t("light") if v == "light" else t("dark"),
                key="theme_mode",
                index=0 if st.session_state.theme_mode == "light" else 1,
            )
            st.session_state.theme_mode = mode
        with col_theme2:
            flower_style = st.selectbox(
                t("flower_style"),
                FLOWER_STYLES,
                key="flower_style_select",
                index=FLOWER_STYLES.index(st.session_state.flower_style)
                if st.session_state.flower_style in FLOWER_STYLES
                else 0,
            )
            st.session_state.flower_style = flower_style

        # Navigation
        st.markdown("---")
        tabs = {
            "upload": t("source_pdf"),
            "agents": t("agents_config"),
            "run": t("results"),
            "dashboard": t("dashboard"),
        }
        for key, label in tabs.items():
            if st.button(
                label,
                key=f"nav_{key}",
                use_container_width=True,
                type="primary" if st.session_state.active_tab == key else "secondary",
            ):
                st.session_state.active_tab = key

        st.markdown("---")
        # Wow status indicator
        status_label = {
            AppStatus.IDLE: t("app_status_idle"),
            AppStatus.PROCESSING: t("app_status_processing"),
            AppStatus.COMPLETED: t("app_status_completed"),
            AppStatus.ERROR: t("app_status_error"),
        }[st.session_state.status]

        st.markdown(
            f"""
            <div class="af-card">
              <div style="display:flex;justify-content:space-between;align-items:center;">
                <div>
                  <div style="font-size:0.75rem;color:var(--af-muted);margin-bottom:0.1rem;">{t("wow_status")}</div>
                  <div class="af-status-chip">
                    <div class="af-status-dot"></div>
                    <span>{status_label}</span>
                  </div>
                </div>
                <div style="text-align:right;font-size:0.75rem;color:var(--af-muted);">
                  <div>{t("total_cost")}</div>
                  <div style="font-weight:600;color:var(--af-text);">${st.session_state.total_cost:,.4f}</div>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")
        # Notes / Scratchpad
        st.markdown(f"#### {t('notes')}")
        toggle_col1, toggle_col2 = st.columns([1, 1])
        with toggle_col1:
            if st.button(
                t("markdown_preview") if not st.session_state.notes_preview else t("edit_mode"),
                key="toggle_notes_mode",
                use_container_width=True,
            ):
                st.session_state.notes_preview = not st.session_state.notes_preview
        with toggle_col2:
            if st.button(
                t("optimize_notes"),
                key="btn_optimize_notes",
                use_container_width=True,
                disabled=not st.session_state.notes.strip(),
            ):
                try:
                    with st.spinner(t("running_pipeline")):
                        optimized = optimize_notes_with_gemini(st.session_state.notes)
                    st.session_state.notes = optimized
                    st.session_state.notes_preview = True
                except Exception as e:
                    st.error(f"{t('optimize_notes_error')}: {e}")

        if st.session_state.notes_preview:
            st.markdown(st.session_state.notes or "`(empty)`")
        else:
            st.session_state.notes = st.text_area(
                "",
                value=st.session_state.notes,
                height=160,
                placeholder=t("notes_placeholder"),
                key="notes_textarea",
            )

        st.download_button(
            t("download_md"),
            data=st.session_state.notes.encode("utf-8"),
            file_name=f"AgentFlow-Notes-{time.strftime('%Y-%m-%d')}.md",
            mime="text/markdown",
            disabled=not st.session_state.notes.strip(),
            use_container_width=True,
        )

        st.markdown("---")
        if st.checkbox(t("api_settings"), key="toggle_api_settings"):
            render_api_settings()

        st.caption(APP_VERSION)


def render_topbar():
    cols = st.columns([2, 3, 2])
    with cols[0]:
        st.markdown(
            f"**{t('project_label')}:** {t('project_name')}",
        )
    with cols[1]:
        st.markdown(
            f"<span class='af-badge'>{t('ready_to_run') if st.session_state.status in (AppStatus.IDLE, AppStatus.COMPLETED) else t('app_status_processing')}</span>",
            unsafe_allow_html=True,
        )
    with cols[2]:
        c1, c2 = st.columns([1, 1])
        with c1:
            if st.session_state.active_tab == "agents":
                if st.button(t("save_version"), key="btn_save_version", use_container_width=True):
                    save_configuration()
        with c2:
            disabled = st.session_state.status == AppStatus.PROCESSING
            label = t("running_pipeline") if st.session_state.status == AppStatus.PROCESSING else t("run_pipeline")
            if st.button(
                label,
                key="btn_run_pipeline",
                use_container_width=True,
                disabled=disabled,
            ):
                with st.spinner(t("running_pipeline")):
                    run_pipeline()


def save_configuration():
    config = ProjectConfig(
        id=str(int(time.time() * 1000)),
        name=f"Config {time.strftime('%H:%M:%S')}",
        created_at=time.time(),
        agents=[AgentConfig(**asdict(a)) for a in st.session_state.agents],
    )
    history = st.session_state.config_history
    history.insert(0, config)
    st.session_state.config_history = history[:10]


def render_upload_tab():
    st.markdown(f"### {t('upload_pdf')}")
    uploaded = st.file_uploader(
        t("drag_drop_pdf"),
        type=["pdf"],
        key="uploader_pdf",
    )
    if uploaded is not None:
        st.session_state.pdf_name = uploaded.name
        with st.spinner("Loading PDF..."):
            pages = load_pdf_pages(uploaded.read())
        st.session_state.pages = pages

    pages = st.session_state.pages
    if pages:
        st.markdown(f"**{t('pages_detected')}:** {len(pages)}")
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button(t("select_all"), key="btn_select_all"):
                for p in pages:
                    p.selected = True
            if st.button(t("clear_all"), key="btn_clear_all"):
                for p in pages:
                    p.selected = False
        with col2:
            for page in pages:
                checked = st.checkbox(
                    f"Page {page.page_number}",
                    value=page.selected,
                    key=f"page_sel_{page.page_number}",
                )
                page.selected = checked


def render_agents_tab():
    st.markdown(f"### {t('agents')}")
    agents: List[AgentConfig] = st.session_state.agents
    st.markdown(f"**{t('agent_list')}:** {len(agents)}")

    for idx, agent in enumerate(agents):
        with st.expander(f"{agent.name} ({agent.provider}/{agent.model})", expanded=False):
            enabled = st.checkbox(t("enabled"), value=agent.enabled, key=f"agent_enabled_{idx}")
            provider = st.selectbox(
                t("provider"),
                ["gemini", "openai", "anthropic", "xai"],
                index=["gemini", "openai", "anthropic", "xai"].index(agent.provider)
                if agent.provider in ["gemini", "openai", "anthropic", "xai"]
                else 0,
                key=f"agent_provider_{idx}",
            )
            if provider == "gemini":
                models = SUPPORTED_GEMINI_MODELS
            elif provider == "openai":
                models = SUPPORTED_OPENAI_MODELS
            elif provider == "anthropic":
                models = SUPPORTED_ANTHROPIC_MODELS
            else:
                models = SUPPORTED_XAI_MODELS

            model = st.selectbox(
                t("model"),
                models,
                index=models.index(agent.model) if agent.model in models else 0,
                key=f"agent_model_{idx}",
            )
            system_prompt = st.text_area(
                t("system_prompt"),
                value=agent.system_prompt,
                height=120,
                key=f"agent_prompt_{idx}",
            )
            cols = st.columns(2)
            with cols[0]:
                temperature = st.slider(
                    t("temperature"),
                    0.0,
                    1.0,
                    value=float(agent.temperature),
                    step=0.05,
                    key=f"agent_temp_{idx}",
                )
            with cols[1]:
                max_tokens = st.number_input(
                    t("max_tokens"),
                    min_value=128,
                    max_value=8192,
                    value=int(agent.max_output_tokens),
                    step=128,
                    key=f"agent_max_{idx}",
                )

            # Update agent object
            agent.enabled = enabled
            agent.provider = provider
            agent.model = model
            agent.system_prompt = system_prompt
            agent.temperature = float(temperature)
            agent.max_output_tokens = int(max_tokens)

    st.session_state.agents = agents


def render_results_tab():
    st.markdown(f"### {t('agent_outputs')}")
    results: List[ExecutionResult] = st.session_state.execution_results
    if not results:
        st.info(t("no_results"))
        return

    for res in results:
        with st.expander(f"{res.agent_name} · {res.provider_used}/{res.model_used} · ${res.cost:,.4f}", expanded=True):
            st.markdown(res.output)
            if st.button(
                t("quote_to_notes"),
                key=f"btn_quote_{res.agent_id}",
            ):
                snippet = res.output[:300].replace("\n", " ")
                quote = f"\n> {snippet}...\n(Source: {res.agent_name})\n"
                st.session_state.notes += quote


def render_dashboard_tab():
    st.markdown(f"### {t('dashboard')}")

    pages = st.session_state.pages
    results: List[ExecutionResult] = st.session_state.execution_results

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(t("pages_selected"), f"{sum(1 for p in pages if p.selected)} / {len(pages) or 0}")
    with col2:
        st.metric(t("agents_used"), f"{sum(1 for a in st.session_state.agents if a.enabled)}")
    with col3:
        st.metric(t("total_runs"), str(st.session_state.run_count))
    with col4:
        st.metric(t("total_cost"), f"${st.session_state.total_cost:,.4f}")

    if results:
        df = pd.DataFrame(
            [
                {
                    "agent": r.agent_name,
                    "provider": r.provider_used,
                    "model": r.model_used,
                    "cost": r.cost,
                    "tokens": r.tokens,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r.timestamp)),
                }
                for r in results
            ]
        )
        st.markdown(f"#### {t('run_history')}")
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.markdown("#### Cost by Agent")
        cost_chart = (
            df.groupby("agent")["cost"]
            .sum()
            .reset_index()
            .rename(columns={"cost": "Estimated Cost"})
        )
        st.bar_chart(cost_chart.set_index("agent"))

    # Config history
    if st.session_state.config_history:
        st.markdown(f"#### {t('config_versions')}")
        for cfg in st.session_state.config_history:
            cols = st.columns([3, 2, 1])
            with cols[0]:
                st.write(cfg.name)
            with cols[1]:
                st.caption(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(cfg.created_at)))
            with cols[2]:
                if st.button(t("restore"), key=f"restore_{cfg.id}"):
                    st.session_state.agents = [AgentConfig(**asdict(a)) for a in cfg.agents]
                    st.success("Configuration restored.")


# -----------------------------
# 7. MAIN ENTRYPOINT
# -----------------------------

def main():
    st.set_page_config(
        page_title="AgentFlow PDF Intelligence",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    init_session_state()
    apply_theme()

    # Initial agent load from agents.yaml
    if not st.session_state.agents:
        st.session_state.agents = load_agents_from_yaml()

    render_sidebar()

    with st.container():
        render_topbar()
        st.markdown("---")

        tab = st.session_state.active_tab
        if tab == "upload":
            render_upload_tab()
        elif tab == "agents":
            render_agents_tab()
        elif tab == "run":
            render_results_tab()
        elif tab == "dashboard":
            render_dashboard_tab()


if __name__ == "__main__":
    main()
```

---

## 3. Notes for Hugging Face Space deployment

- `requirements.txt` example:

```text
streamlit
pillow
pdfplumber
pyyaml
google-generativeai
openai
anthropic
requests
pandas
```

- Place `app.py` and `agents.yaml` at repo root.
- Configure environment variables in Space Secrets for:
  - `GEMINI_API_KEY`
  - `OPENAI_API_KEY`
  - `ANTHROPIC_API_KEY`
  - `XAI_API_KEY`
- If any env key is missing, the app will show a password field in the sidebar’s “API Settings” section for that provider and store the key only in `st.session_state`, never displaying environment keys.

---

## 4. Twenty follow‑up questions

1. How large are the typical PDFs you plan to analyze (page count and file size), and do you need special handling for very large documents (e.g., chunking or streaming)?
2. Do you need true image‑based OCR via Gemini Vision for scanned PDFs, or is text extraction via `pdfplumber` sufficient for your use cases?
3. Should each agent see the combined context of all selected pages (current behavior) or operate page‑by‑page with separate outputs and references?
4. How important is strict cost tracking accuracy, and would you like real token accounting via each provider’s usage metadata instead of the current heuristic?
5. Are there additional models from Gemini, OpenAI, Anthropic, or xAI that you want included in the model selectors besides the ones specified?
6. Do you want to support provider‑specific agent features (e.g., tools/functions, JSON mode, structured outputs) or keep agents to plain text in/out?
7. Should agents be able to depend on prior agents’ outputs (multi‑step pipelines with explicit dependencies), or is the current parallel‑over‑common‑context design enough?
8. How would you like to persist data beyond the session (e.g., saving execution history, notes, and configs to a database or Hugging Face Hub dataset)?
9. Do you want authentication or multi‑user isolation for your Space so that different users have separate configs, notes, and history?
10. For the “wow” dashboard, are there any specific charts or KPIs (e.g., average cost per run, latency per provider, agent success/failure rates) you’d like added?
11. Should the language toggle also change the agents’ system prompts to Traditional Chinese variants, or do you prefer prompts remain in English for now?
12. Would you like to localize the content of the analysis (e.g., force agents to answer in English vs Traditional Chinese depending on UI language)?
13. Do you want a per‑run “scenario” name or tag so you can distinguish different experiments in the dashboard history?
14. Should the notes optimization always use Gemini, or would you like a selector to choose which provider/model handles note polishing?
15. Do you need export of results and configurations (e.g., JSON/CSV export of all agent outputs and metadata) for offline analysis?
16. Are there any access or rate‑limit concerns (for example, needing built‑in throttling or queuing when many users run pipelines simultaneously)?
17. Would you like an advanced “prompt playground” panel where you can interact with a single model and prompt before turning it into an agent definition?
18. Should page‑level previews show extracted text, images, or both, and do you want a way to manually correct OCR/extraction before running agents?
19. Do you want to support additional file types beyond PDF (e.g., images, Word documents, PowerPoint) in the same interface?
20. How strongly do you want to customize the “flower styles” (e.g., custom backgrounds, icons, animations per style), and should they be user‑extensible via config files?
