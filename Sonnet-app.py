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

# Embedded modules
import pdfplumber
from pdf2image import convert_from_bytes
import pytesseract
from openai import OpenAI
import google.generativeai as genai
from anthropic import Anthropic

# ==================== CONFIGURATION ====================
PRICING = {
    "openai": {"gpt-4o-mini": {"input": 0.15, "output": 0.6}, "gpt-5-nano": {"input": 0.1, "output": 0.4}},
    "gemini": {"gemini-2.5-flash": {"input": 0.075, "output": 0.3}, "gemini-2.5-flash-lite": {"input": 0.05, "output": 0.2}},
    "anthropic": {"claude-sonnet-4.5": {"input": 3.0, "output": 15.0}, "claude-haiku-4.5": {"input": 0.8, "output": 4.0}},
    "grok": {"grok-3-mini": {"input": 0.1, "output": 0.5}}
}

THEMES = {
    "櫻花 Cherry": {"primary": "#FFB7C5", "accent": "#FF69B4", "icon": "🌸"},
    "薰衣草 Lavender": {"primary": "#9C27B0", "accent": "#7B1FA2", "icon": "💜"},
    "向日葵 Sunflower": {"primary": "#FFC107", "accent": "#FFA000", "icon": "🌻"}
}

DEFAULT_AGENTS = """agents:
  - name: 申請資料提取器
    description: 提取申請文件核心資訊
    system_prompt: "提取：廠商、品名、類別、證書編號，以表格呈現"
    model: claude-sonnet-4.5
    temperature: 0
    max_tokens: 2000
  - name: 合約分析師
    description: 分析委託製造合約
    system_prompt: "確認：委託者/受託者、製程範圍、權利義務"
    model: gpt-4o-mini
    temperature: 0.3
    max_tokens: 1500
"""

ModelChoice = {
    "gpt-4o-mini": "openai", "gpt-5-nano": "openai",
    "gemini-2.5-flash": "gemini", "gemini-2.5-flash-lite": "gemini",
    "claude-sonnet-4.5": "anthropic", "claude-haiku-4.5": "anthropic",
    "grok-3-mini": "grok"
}

# ==================== LLM ROUTER WITH FALLBACK ====================
class LLMRouter:
    def __init__(self):
        self._openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) if os.getenv("OPENAI_API_KEY") else None
        self._anthropic = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY")) if os.getenv("ANTHROPIC_API_KEY") else None
        self._gemini_ready = False
        if os.getenv("GEMINI_API_KEY"):
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            self._gemini_ready = True
    
    def generate_text(self, model: str, messages: List[Dict], params: Dict) -> Tuple[str, Dict, str]:
        provider = ModelChoice.get(model, "openai")
        fallback_order = ["anthropic", "openai", "gemini"]
        
        # Try primary provider
        try:
            return self._call_provider(provider, model, messages, params)
        except Exception as e:
            st.warning(f"⚠️ {provider.upper()} 失敗: {e}")
            
            # Try fallback providers
            for fallback in fallback_order:
                if fallback != provider:
                    try:
                        fallback_model = self._get_fallback_model(fallback)
                        st.info(f"🔄 切換至 {fallback.upper()} ({fallback_model})")
                        return self._call_provider(fallback, fallback_model, messages, params)
                    except:
                        continue
            
            return f"Error: All providers failed", {"total_tokens": 0}, "Failed"
    
    def _call_provider(self, provider: str, model: str, messages: List, params: Dict) -> Tuple[str, Dict, str]:
        if provider == "openai" and self._openai:
            resp = self._openai.chat.completions.create(model=model, messages=messages, **params)
            return resp.choices[0].message.content, {"total_tokens": resp.usage.total_tokens}, "OpenAI"
        
        elif provider == "anthropic" and self._anthropic:
            sys = "\n".join([m["content"] for m in messages if m["role"] == "system"])
            usr_msgs = [{"role": m["role"], "content": m["content"]} for m in messages if m["role"] != "system"]
            resp = self._anthropic.messages.create(model=model, system=sys, messages=usr_msgs, **params)
            return resp.content[0].text, {"total_tokens": resp.usage.input_tokens + resp.usage.output_tokens}, "Anthropic"
        
        elif provider == "gemini" and self._gemini_ready:
            mm = genai.GenerativeModel(model)
            combined = "\n".join([m["content"] for m in messages])
            resp = mm.generate_content(combined)
            return resp.text, {"total_tokens": len(combined)//4}, "Gemini"
        
        raise Exception(f"{provider} not configured")
    
    def _get_fallback_model(self, provider: str) -> str:
        defaults = {"openai": "gpt-4o-mini", "anthropic": "claude-haiku-4.5", "gemini": "gemini-2.5-flash-lite"}
        return defaults.get(provider, "gpt-4o-mini")

# ==================== COST ESTIMATOR ====================
def estimate_cost(model: str, tokens: int) -> float:
    provider = ModelChoice.get(model, "openai")
    if provider in PRICING and model in PRICING[provider]:
        rates = PRICING[provider][model]
        return (tokens * 0.7 * rates["input"] + tokens * 0.3 * rates["output"]) / 1_000_000
    return 0.0

# ==================== OCR FUNCTIONS ====================
def render_pdf_thumbnails(pdf_bytes: bytes) -> List[Tuple[int, Image.Image]]:
    pages = convert_from_bytes(pdf_bytes, dpi=120)
    return [(i, img.resize((200, 280))) for i, img in enumerate(pages[:20])]

def extract_text_python(pdf_bytes: bytes, pages: List[int]) -> str:
    texts = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for i in pages:
            if i < len(pdf.pages):
                txt = pdf.pages[i].extract_text() or ""
                if txt.strip():
                    texts.append(f"[PAGE {i+1}]\n{txt}")
    return "\n\n".join(texts)

# ==================== SESSION STATE ====================
if "theme" not in st.session_state:
    st.session_state.theme = "櫻花 Cherry"
if "agents_config" not in st.session_state:
    st.session_state.agents_config = yaml.safe_load(DEFAULT_AGENTS)["agents"]
if "ocr_text" not in st.session_state:
    st.session_state.ocr_text = ""
if "agent_outputs" not in st.session_state:
    st.session_state.agent_outputs = []
if "review_notes" not in st.session_state:
    st.session_state.review_notes = "# 審查筆記\n\n"
if "agent_versions" not in st.session_state:
    st.session_state.agent_versions = []
if "execution_order" not in st.session_state:
    st.session_state.execution_order = list(range(len(st.session_state.agents_config)))

# ==================== THEME & PAGE CONFIG ====================
st.set_page_config(page_title="🌸 TFDA AI Review System", layout="wide", page_icon="🌸")

theme = THEMES[st.session_state.theme]
st.markdown(f"""<style>
    [data-testid="stAppViewContainer"] {{ background: linear-gradient(135deg, {theme['primary']}20, white); }}
    .stButton>button {{ background: {theme['primary']}; color: white; border-radius: 8px; }}
    .agent-card {{ border-left: 4px solid {theme['accent']}; padding: 1rem; background: white; border-radius: 8px; margin: 1rem 0; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
</style>""", unsafe_allow_html=True)

# ==================== HEADER ====================
col1, col2 = st.columns([3, 1])
with col1:
    st.title(f"{theme['icon']} TFDA Agentic AI 審查系統")
with col2:
    st.session_state.theme = st.selectbox("主題", list(THEMES.keys()), label_visibility="collapsed")

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("### 🔐 API 連線狀態")
    for name, var in [("OpenAI", "OPENAI_API_KEY"), ("Gemini", "GEMINI_API_KEY"), ("Anthropic", "ANTHROPIC_API_KEY")]:
        status = "✅" if os.getenv(var) else "❌"
        st.markdown(f"{status} **{name}**")
    
    st.markdown("---")
    st.markdown("### 📋 agents.yaml")
    yaml_text = st.text_area("YAML編輯器", yaml.dump({"agents": st.session_state.agents_config}, allow_unicode=True), height=300, label_visibility="collapsed")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 儲存", use_container_width=True):
            st.session_state.agents_config = yaml.safe_load(yaml_text)["agents"]
            st.session_state.agent_versions.append({
                "timestamp": datetime.now().isoformat(),
                "config": st.session_state.agents_config.copy()
            })
            st.success("✅ 已儲存")
    with col2:
        st.download_button("📥 下載", yaml_text, f"agents_{datetime.now():%Y%m%d_%H%M%S}.yaml", use_container_width=True)

# ==================== TABS ====================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📄 上傳OCR", "🎨 編輯器", "⚙️ 代理設定", "▶️ 執行", "📊 儀表板", "📝 筆記"])

# ===== TAB 1: UPLOAD & OCR =====
with tab1:
    st.markdown('<div class="agent-card">', unsafe_allow_html=True)
    uploaded = st.file_uploader("上傳 PDF", type=["pdf"])
    
    if uploaded:
        pdf_bytes = uploaded.read()
        thumbnails = render_pdf_thumbnails(pdf_bytes)
        
        st.markdown("#### 📑 選擇頁面 (多選)")
        cols = st.columns(5)
        selected_pages = []
        for i, (idx, thumb) in enumerate(thumbnails):
            with cols[i % 5]:
                if st.checkbox(f"頁 {idx+1}", key=f"page_{idx}"):
                    selected_pages.append(idx)
                st.image(thumb, use_column_width=True)
        
        if st.button("🚀 開始 OCR", type="primary"):
            with st.spinner("處理中..."):
                st.session_state.ocr_text = extract_text_python(pdf_bytes, selected_pages)
            st.success(f"✅ 完成！已處理 {len(selected_pages)} 頁")
    st.markdown('</div>', unsafe_allow_html=True)

# ===== TAB 2: PREVIEW & EDIT =====
with tab2:
    st.session_state.ocr_text = st.text_area("編輯 OCR 文字", st.session_state.ocr_text, height=500)

# ===== TAB 3: AGENT CONFIG (FORM-BASED EDITOR) =====
with tab3:
    st.markdown("### 🤖 圖形化代理編輯器")
    
    if st.button("➕ 新增代理"):
        st.session_state.agents_config.append({
            "name": "新代理", "description": "", "system_prompt": "",
            "model": "gpt-4o-mini", "temperature": 0.3, "max_tokens": 1000
        })
    
    for i, agent in enumerate(st.session_state.agents_config):
        with st.expander(f"### Agent {i+1}: {agent.get('name', 'Unnamed')}", expanded=(i==0)):
            agent["name"] = st.text_input("名稱", agent.get("name", ""), key=f"name_{i}")
            agent["description"] = st.text_input("描述", agent.get("description", ""), key=f"desc_{i}")
            agent["system_prompt"] = st.text_area("System Prompt", agent.get("system_prompt", ""), height=150, key=f"sys_{i}")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                agent["model"] = st.selectbox("模型", list(ModelChoice.keys()), key=f"model_{i}")
            with col2:
                agent["temperature"] = st.slider("Temperature", 0.0, 2.0, float(agent.get("temperature", 0.3)), key=f"temp_{i}")
            with col3:
                agent["max_tokens"] = st.number_input("Max Tokens", 64, 8192, int(agent.get("max_tokens", 1000)), key=f"max_{i}")
            
            if st.button("🗑️ 刪除此代理", key=f"del_{i}"):
                st.session_state.agents_config.pop(i)
                st.rerun()

# ===== TAB 4: EXECUTION =====
with tab4:
    st.markdown("### ▶️ 執行管線")
    
    # Drag-and-drop execution order (simulated with selectbox)
    st.markdown("#### 🔀 自訂執行順序")
    order_labels = [f"{i+1}. {st.session_state.agents_config[i]['name']}" for i in range(len(st.session_state.agents_config))]
    new_order = st.multiselect("拖曳排序 (選擇順序)", order_labels, default=order_labels[:len(st.session_state.agents_config)])
    st.session_state.execution_order = [order_labels.index(label) for label in new_order]
    
    # One-click execution
    if st.button("🚀 一鍵全執行", type="primary"):
        router = LLMRouter()
        st.session_state.agent_outputs = []
        total_cost = 0.0
        
        progress = st.progress(0)
        for step, i in enumerate(st.session_state.execution_order):
            agent = st.session_state.agents_config[i]
            with st.spinner(f"執行 {agent['name']}..."):
                input_text = st.session_state.ocr_text if step == 0 else st.session_state.agent_outputs[-1]["output"]
                messages = [
                    {"role": "system", "content": agent.get("system_prompt", "")},
                    {"role": "user", "content": input_text}
                ]
                params = {"temperature": agent["temperature"], "max_tokens": agent["max_tokens"]}
                
                t0 = time.time()
                output, usage, provider = router.generate_text(agent["model"], messages, params)
                elapsed = time.time() - t0
                cost = estimate_cost(agent["model"], usage["total_tokens"])
                total_cost += cost
                
                st.session_state.agent_outputs.append({
                    "agent": agent["name"], "output": output, "time": elapsed,
                    "tokens": usage["total_tokens"], "provider": provider, "cost": cost
                })
            progress.progress((step + 1) / len(st.session_state.execution_order))
        
        st.success(f"✅ 全部完成！總成本: ${total_cost:.4f} USD")
        st.balloons()
    
    # Individual execution
    st.markdown("---")
    for i, agent in enumerate(st.session_state.agents_config):
        with st.expander(f"Agent {i+1}: {agent['name']}"):
            if st.button(f"▶️ 執行", key=f"run_{i}"):
                router = LLMRouter()
                input_text = st.session_state.ocr_text if i == 0 else st.session_state.agent_outputs[i-1]["output"]
                messages = [{"role": "system", "content": agent["system_prompt"]}, {"role": "user", "content": input_text}]
                params = {"temperature": agent["temperature"], "max_tokens": agent["max_tokens"]}
                
                output, usage, provider = router.generate_text(agent["model"], messages, params)
                cost = estimate_cost(agent["model"], usage["total_tokens"])
                
                st.markdown(f"**輸出** ({provider}, ${cost:.4f}):\n\n{output}")

# ===== TAB 5: DASHBOARD =====
with tab5:
    st.markdown("### 📊 分析儀表板")
    
    if st.session_state.agent_outputs:
        df = pd.DataFrame(st.session_state.agent_outputs)
        
        # Cost estimation
        total_cost = df["cost"].sum()
        st.markdown(f'<div class="agent-card"><h2 style="color:{theme["accent"]}">💰 總成本: ${total_cost:.4f} USD</h2></div>', unsafe_allow_html=True)
        
        # Charts
        col1, col2 = st.columns(2)
        with col1:
            fig1 = px.bar(df, x="agent", y="time", title="執行時間", color="provider")
            st.plotly_chart(fig1, use_container_width=True)
        with col2:
            fig2 = px.bar(df, x="agent", y="tokens", title="Token 用量")
            st.plotly_chart(fig2, use_container_width=True)
        
        # Agent comparison table
        st.markdown("#### 🔍 代理比較")
        st.dataframe(df[["agent", "provider", "time", "tokens", "cost"]].style.format({"time": "{:.2f}s", "cost": "${:.4f}"}))
    else:
        st.info("尚無執行數據")

# ===== TAB 6: REVIEW NOTES =====
with tab6:
    st.markdown("### 📝 審查筆記")
    
    # Note keeping with auto-reference
    st.session_state.review_notes = st.text_area("編輯筆記 (支援 Markdown)", st.session_state.review_notes, height=400)
    
    st.markdown("#### 🔗 引用 Agent 輸出")
    if st.session_state.agent_outputs:
        for i, out in enumerate(st.session_state.agent_outputs):
            if st.button(f"插入 Agent {i+1} 引用", key=f"ref_{i}"):
                ref_text = f"\n\n> **引用自 {out['agent']}:**\n> {out['output'][:200]}...\n\n"
                st.session_state.review_notes += ref_text
                st.rerun()
    
    st.markdown("---")
    st.markdown("### 預覽")
    st.markdown(st.session_state.review_notes, unsafe_allow_html=True)
    
    # Version history
    if st.session_state.agent_versions:
        st.markdown("#### 📜 版本歷史")
        for v in st.session_state.agent_versions[-5:]:
            st.caption(f"版本: {v['timestamp']}")

# ==================== FOOTER ====================
st.markdown("---")
st.markdown(f"<center>{theme['icon']} TFDA AI Review System • Powered by Multi-LLM with Auto-Fallback</center>", unsafe_allow_html=True)
