"""
╔══════════════════════════════════════════════════════════════════════╗
║  AI Document Intelligence & Artwork Navneet Processing System        ║
║  Professional Prepress Automation · Adobe Illustrator · PSD · EPS   ║
║                                                                      ║
║  ARCHITECTURE:                                                       ║
║   • LangGraph    — 5-node stateful pipeline                         ║
║   • LangChain    — RAG · ChatPromptTemplate · StrOutputParser       ║
║   • OpenRouter   — multi-model LLM routing                          ║
║   • Hybrid RAG   — BM25 sparse + TF-IDF cosine dense               ║
║   • CV Layers    — OpenCV deterministic geometry                     ║
║   • PSD Builder  — 7-layer BGRA binary PSD                         ║
║   • AI Exporter  — Adobe Illustrator / EPS / SVG vector output      ║
║   • Batch Mode   — process multiple files simultaneously             ║
╚══════════════════════════════════════════════════════════════════════╝
"""
from __future__ import annotations
import io, json, os, sys, time, zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="AI Document Intelligence & Artwork Navneet Processing System",
    page_icon="🖨️",
    layout="wide",
    initial_sidebar_state="expanded",
)

sys.path.insert(0, str(Path(__file__).parent))

from modules.schemas     import ArtworkSpec, PipelineState, PrepressResult
from modules.spec_parser import parse_spec_from_text, STANDARD_SIZES
from modules.rag_engine  import get_rag, query_rag

@st.cache_resource(show_spinner=False)
def _cached_rag():
    return get_rag()
from modules.pipeline    import (
    run_pipeline_sync, make_psd_bytes, make_ai_bytes,
    make_eps_bytes, make_svg_bytes, make_zip_bundle,
)

# ── Constants ─────────────────────────────────────────────────────────────────
APP_VERSION   = "V35"
APP_NAME      = "AI Document Intelligence & Artwork Navneet Processing System"
APP_SHORT     = "AI Doc Intelligence · Navneet"
APP_ACRONYM   = "AIDA · Navneet Processing System"

# ── Developer / Author ────────────────────────────────────────────────────────
DEV_NAME      = "Snehal Laxman Jadhav"
DEV_TITLE     = "Image AI Engineer"
DEV_COMPANY   = "Navneet Education Limited"
DEV_YEAR      = "2026"
DEV_GMAIL     = "snehaljadhav.ai@gmail.com"                       # ← update to real address
DEV_LINKEDIN  = "https://www.linkedin.com/in/snehal-jadhav"       # ← update to real profile

LAYER_NAMES = ["ARTWORK", "BLEED", "CUTMARKS", "DIELINE", "DIMENSIONS", "MEASURE", "TEXT"]

OPENROUTER_MODELS: Dict[str, str] = {
    "GPT-OSS 120B (Free)":    "openai/gpt-oss-120b:free",
    "Qwen2.5-VL 72B (Free)":  "qwen/qwen-2.5-vl-72b-instruct:free",
    "Qwen2.5-VL 7B (Free)":   "qwen/qwen2.5-vl-7b-instruct:free",
    "Gemma-3 27B (Free)":     "google/gemma-3-27b-it:free",
    "DeepSeek-R1 (Free)":     "deepseek/deepseek-r1:free",
    "Llama 3.3 70B (Free)":   "meta-llama/llama-3.3-70b-instruct:free",
    "Claude Sonnet 4":        "anthropic/claude-sonnet-4",
    "GPT-4o":                 "openai/gpt-4o",
    "Gemini 2.0 Flash":       "google/gemini-2.0-flash-001",
}

STANDARD_SIZES_DISPLAY: Dict[str, Tuple[float, float]] = {
    "A4 — 21 × 29.7 cm":  (21.0, 29.7),
    "A3 — 29.7 × 42 cm":  (29.7, 42.0),
    "A5 — 14.8 × 21 cm":  (14.8, 21.0),
    "A2 — 42 × 59.4 cm":  (42.0, 59.4),
    "A1 — 59.4 × 84.1 cm":(59.4, 84.1),
    "B4 — 25 × 35.3 cm":  (25.0, 35.3),
    "B5 — 17.6 × 25 cm":  (17.6, 25.0),
    "US Letter — 21.59 × 27.94 cm": (21.59, 27.94),
    "Custom": None,
}

DOC_TYPES    = ["Inside Pages (Coloring Book)", "Cover", "Flat Sheet", "Booklet", "Brochure", "Poster"]
BINDING_TYPES= ["Perfect Bound", "Saddle Stitch", "Spiral / Wire-O", "Case Bound", "None / Flat"]
OUTPUT_FORMATS = ["Adobe Illustrator (.ai)", "Photoshop PSD", "EPS", "SVG", "ZIP Bundle (All)"]


# ══════════════════════════════════════════════════════════════════════════════
#  CSS — Pro dark theme
# ══════════════════════════════════════════════════════════════════════════════
def inject_css():
    st.markdown("""
<style>
/* ── Base ──────────────────────────────────────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"] {
    background: #080b18 !important; color: #dde3f8 !important;
}
* { box-sizing: border-box; }

/* ── Scrollbar ─────────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #080b18; }
::-webkit-scrollbar-thumb { background: #1e2448; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #2d3a6a; }

/* ── Sidebar ───────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #080e1f !important;
    border-right: 1px solid #1a2040 !important;
}
[data-testid="stSidebar"] * { color: #b8c4e8 !important; }
[data-testid="stSidebar"] hr { border-color: #1a2040 !important; }

/* ── Header ────────────────────────────────────────────────────────────────── */
.pp-header {
    background: linear-gradient(135deg, #070a16 0%, #0f1630 50%, #080f1e 100%);
    border: 1px solid #1a2040;
    border-top: 1px solid #2a3560;
    border-radius: 16px;
    padding: 28px 32px 22px; margin-bottom: 24px;
    box-shadow: 0 8px 40px rgba(0,50,180,.15), 0 1px 0 rgba(77,159,255,.08) inset;
}
.pp-title { font-size: 2rem; font-weight: 900; color: #f0f4ff; letter-spacing: -.02em; }
.pp-title span { color: #4d9fff; }
.pp-badge-row { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 14px; }
.pp-badge {
    font-size: .58rem; font-weight: 800; letter-spacing: .08em;
    padding: 3px 11px; border-radius: 20px; border: 1px solid;
    text-transform: uppercase; transition: opacity .2s;
}
.pp-badge:hover { opacity: .75; }

/* ── Section cards ─────────────────────────────────────────────────────────── */
.pp-card {
    background: #0b0f22; border: 1px solid #1a2040;
    border-radius: 12px; padding: 18px 20px; margin-bottom: 16px;
    transition: border-color .2s;
}
.pp-card:hover { border-color: #2d3a6a; }
.pp-card-title {
    font-size: .72rem; font-weight: 800; letter-spacing: .08em;
    text-transform: uppercase; color: #dde3f8; margin-bottom: 2px;
}
.pp-card-sub { font-size: .68rem; color: #4e5e88; margin-bottom: 14px; }

/* ── Buttons — base ────────────────────────────────────────────────────────── */
.stButton > button {
    border-radius: 9px !important;
    font-weight: 700 !important;
    font-size: .79rem !important;
    letter-spacing: .04em !important;
    padding: 0.45rem 1.1rem !important;
    transition: all .18s ease !important;
    border: 1px solid rgba(255,255,255,.08) !important;
    box-shadow: 0 2px 8px rgba(0,0,0,.35) !important;
    background: #0f1530 !important;
    color: #b8c4e8 !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(77,159,255,.18) !important;
    border-color: rgba(77,159,255,.3) !important;
    color: #ffffff !important;
    background: #141c3a !important;
}
.stButton > button:active {
    transform: translateY(0px) !important;
    box-shadow: 0 1px 4px rgba(0,0,0,.4) !important;
}

/* ── Buttons — primary (Run Pipeline) ─────────────────────────────────────── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #1741c8 0%, #2563eb 50%, #4d9fff 100%) !important;
    border: none !important;
    color: #ffffff !important;
    box-shadow: 0 4px 18px rgba(37,99,235,.45), 0 1px 0 rgba(255,255,255,.12) inset !important;
    text-shadow: 0 1px 2px rgba(0,0,0,.3) !important;
}
.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #1d4ed8 0%, #3b82f6 50%, #60afff 100%) !important;
    box-shadow: 0 8px 28px rgba(37,99,235,.6), 0 1px 0 rgba(255,255,255,.15) inset !important;
    transform: translateY(-2px) !important;
}
.stButton > button[kind="primary"]:active {
    transform: translateY(0px) !important;
    box-shadow: 0 3px 12px rgba(37,99,235,.4) !important;
}

/* ── Download buttons ──────────────────────────────────────────────────────── */
.stDownloadButton > button {
    border-radius: 9px !important;
    font-weight: 700 !important;
    font-size: .77rem !important;
    letter-spacing: .04em !important;
    padding: 0.45rem 1rem !important;
    transition: all .18s ease !important;
    background: linear-gradient(135deg, #091830, #0d2244) !important;
    border: 1px solid #1a3568 !important;
    color: #6aaeff !important;
    box-shadow: 0 2px 10px rgba(0,0,0,.4) !important;
}
.stDownloadButton > button:hover {
    background: linear-gradient(135deg, #112654, #1741c8) !important;
    border-color: #4d9fff !important;
    color: #ffffff !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(77,159,255,.35) !important;
}
.stDownloadButton > button:active {
    transform: translateY(0px) !important;
}

/* ── Tabs ──────────────────────────────────────────────────────────────────── */
div[data-testid="stTabs"] button[role="tab"] {
    font-weight: 700 !important;
    font-size: .8rem !important;
    letter-spacing: .03em !important;
    transition: color .15s, border-color .15s !important;
    padding: 0.5rem 1rem !important;
}
div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    color: #4d9fff !important;
}
div[data-testid="stTabs"] button[role="tab"]:hover {
    color: #93c5fd !important;
}

/* ── Expanders ─────────────────────────────────────────────────────────────── */
.stExpander {
    border: 1px solid #1a2040 !important;
    border-radius: 12px !important;
    background: #0b0f22 !important;
    transition: border-color .2s !important;
}
.stExpander:hover { border-color: #2d3a6a !important; }
.stExpander summary {
    font-weight: 600 !important;
    font-size: .82rem !important;
    padding: 12px 16px !important;
}

/* ── Input fields ──────────────────────────────────────────────────────────── */
.stTextInput input, .stNumberInput input, .stTextArea textarea {
    background: #08111e !important;
    color: #dde3f8 !important;
    border: 1px solid #1a2040 !important;
    border-radius: 8px !important;
    transition: border-color .15s, box-shadow .15s !important;
}
.stTextInput input:focus, .stNumberInput input:focus, .stTextArea textarea:focus {
    border-color: #4d9fff !important;
    box-shadow: 0 0 0 3px rgba(77,159,255,.12) !important;
}
div[data-baseweb="select"] > div {
    background: #08111e !important;
    border-color: #1a2040 !important;
    border-radius: 8px !important;
    transition: border-color .15s !important;
}
div[data-baseweb="select"] > div:hover { border-color: #2d3a6a !important; }

/* ── Metric cards ──────────────────────────────────────────────────────────── */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #0b0f22, #0d1228) !important;
    border: 1px solid #1a2040 !important;
    border-radius: 12px !important;
    padding: 14px 16px !important;
    transition: border-color .2s !important;
}
[data-testid="metric-container"]:hover { border-color: #2d3a6a !important; }
[data-testid="stMetricValue"] { color: #f0f4ff !important; font-weight: 800 !important; }
[data-testid="stMetricLabel"] { color: #4e5e88 !important; font-size: .72rem !important; }

/* ── Progress bar ──────────────────────────────────────────────────────────── */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #1d4ed8, #4d9fff) !important;
    border-radius: 4px !important;
}
.stProgress > div > div {
    background: #0f1530 !important; border-radius: 4px !important;
}

/* ── File uploader ─────────────────────────────────────────────────────────── */
[data-testid="stFileUploader"] {
    border: 2px dashed #1a2040 !important;
    border-radius: 12px !important;
    padding: 8px !important;
    transition: border-color .2s !important;
}
[data-testid="stFileUploader"]:hover { border-color: #4d9fff !important; }

/* ── Checkboxes ────────────────────────────────────────────────────────────── */
.stCheckbox label span { color: #9aabcc !important; font-size: .79rem !important; }

/* ── Alerts / info boxes ───────────────────────────────────────────────────── */
.stAlert { border-radius: 10px !important; border-left-width: 3px !important; }
.stSuccess { border-color: #22d3ee !important; background: rgba(34,211,238,.06) !important; }
.stInfo    { border-color: #4d9fff !important; background: rgba(77,159,255,.06) !important; }
.stWarning { border-color: #fbbf24 !important; background: rgba(251,191,36,.06) !important; }
.stError   { border-color: #f87171 !important; background: rgba(248,113,113,.06) !important; }

/* ── RAG bar ───────────────────────────────────────────────────────────────── */
.rag-bar {
    background: #08111e; border: 1px solid #1a2040;
    border-radius: 8px; padding: 9px 13px; margin-bottom: 7px;
    transition: border-color .15s;
}
.rag-bar:hover { border-color: #2d3a6a; }

/* ── Pipeline node ─────────────────────────────────────────────────────────── */
.pipe-node {
    background: #08111e; border: 1px solid #1a2040;
    border-radius: 11px; padding: 13px 16px; margin-bottom: 7px;
    display: flex; align-items: flex-start; gap: 14px;
    transition: border-color .2s, box-shadow .2s;
}
.pipe-node:hover {
    border-color: #2d3a6a;
    box-shadow: 0 4px 16px rgba(0,0,0,.25);
}
.pipe-icon { font-size: 1.5rem; flex-shrink: 0; }
.pipe-title { font-size: .78rem; font-weight: 700; color: #dde3f8; }
.pipe-desc  { font-size: .68rem; color: #4e5e88; margin-top: 3px; line-height: 1.5; }

/* ── Monospace prompt box ──────────────────────────────────────────────────── */
.prompt-box {
    background: #04060e; border: 1px solid #1a2040;
    border-radius: 11px; padding: 20px;
    font-family: 'Fira Code', 'Cascadia Code', 'Courier New', monospace;
    font-size: .72rem; line-height: 1.9; color: #8899dd;
    white-space: pre-wrap; word-break: break-word;
    max-height: 68vh; overflow-y: auto;
}

/* ── Spec reference card ───────────────────────────────────────────────────── */
.ref-card {
    background: linear-gradient(135deg, #0b0f22, #080b18);
    border: 1px solid #1a2040; border-radius: 12px;
    padding: 14px 16px; margin-bottom: 14px;
}
.ref-row {
    display: flex; justify-content: space-between;
    font-size: .73rem; padding: 5px 0;
    border-bottom: 1px solid rgba(255,255,255,.035);
}
.ref-key { color: #4e5e88; }
.ref-val { color: #dde3f8; font-weight: 700; }

/* ── Status pill ───────────────────────────────────────────────────────────── */
.status-pill {
    display: inline-flex; align-items: center; gap: 7px;
    background: #0b0f22; border: 1px solid #1a2040;
    border-radius: 20px; padding: 5px 14px;
    font-size: .7rem; color: #4e5e88;
}
.status-dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: #22d3ee; animation: blink 2.5s infinite;
    box-shadow: 0 0 6px rgba(34,211,238,.5);
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:.2} }

/* ── Layer preview ─────────────────────────────────────────────────────────── */
.layer-preview {
    background: #050810; border: 1px solid #1a2040;
    border-radius: 10px; padding: 12px; margin-bottom: 10px;
}

/* ── Spinner / stale element ───────────────────────────────────────────────── */
.stSpinner > div { border-top-color: #4d9fff !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  SESSION INIT
# ══════════════════════════════════════════════════════════════════════════════
def init_session():
    defaults = dict(
        app_stage="landing",        # landing → login → app
        landing_section="home",     # home · about · contact
        authenticated=False,
        user_name="",
        pipeline_state=None,
        generated_prompt="",
        ref_rag_loaded=False,
        active_layers=set(LAYER_NAMES),
        rag_results=[],
        batch_results=[],
        pp_file_groups=[],   # [{filename, results}] — persists across download clicks
        pp_zip=None,         # cached ZIP bytes
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ══════════════════════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════════════════════
def render_header():
    st.markdown(f"""
<div class="pp-header">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;">
    <div style="font-size:.58rem;letter-spacing:.22em;color:#3a5090;text-transform:uppercase;
                font-weight:700;margin-bottom:10px;">
      {APP_NAME} · {APP_VERSION} · Professional Edition
    </div>
    <div style="font-size:.62rem;color:#4e5e88;text-align:right;white-space:nowrap;">
      <span style="color:#22d3ee;font-weight:700;">{DEV_NAME}</span>
      <span style="color:#3a5090;"> · {DEV_TITLE} · {DEV_COMPANY} · {DEV_YEAR}</span>
    </div>
  </div>
  <div class="pp-title">AI Document Intelligence <span>&amp; Artwork Navneet Processing System</span></div>
  <div style="font-size:.8rem;color:#5a6899;margin-top:8px;max-width:760px;line-height:1.6;">
    Production-grade document intelligence &amp; prepress automation — dielines, cutmarks, dimensions,
    multi-format export (AI · PSD · EPS · SVG) with correct ISO 12647 specifications.
    Batch-process entire books.
  </div>
  <div class="pp-badge-row">
    <span class="pp-badge" style="color:#c084fc;border-color:#7c3aed55;background:#7c3aed18;">LangGraph</span>
    <span class="pp-badge" style="color:#22d3ee;border-color:#0891b255;background:#0891b218;">LangChain RAG</span>
    <span class="pp-badge" style="color:#4ade80;border-color:#16a34a55;background:#16a34a18;">OpenRouter LLM</span>
    <span class="pp-badge" style="color:#fbbf24;border-color:#d9770655;background:#d9770618;">OpenCV CV</span>
    <span class="pp-badge" style="color:#f472b6;border-color:#be185d55;background:#be185d18;">7-Layer PSD</span>
    <span class="pp-badge" style="color:#22d3ee;border-color:#0891b255;background:#0891b218;">.ai · EPS · SVG</span>
    <span class="pp-badge" style="color:#c084fc;border-color:#7c3aed55;background:#7c3aed18;">ISO 12647</span>
    <span class="pp-badge" style="color:#4ade80;border-color:#16a34a55;background:#16a34a18;">BM25 + VectorDB</span>
    <span class="pp-badge" style="color:#fbbf24;border-color:#d9770655;background:#d9770618;">Batch Mode</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
def render_sidebar() -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}
    with st.sidebar:
        _user = st.session_state.get("user_name", "")
        if _user:
            st.markdown(
                f'<div style="font-size:.72rem;padding:7px 12px;border-radius:8px;'
                f'background:rgba(34,211,238,.08);border:1px solid rgba(34,211,238,.25);'
                f'color:#22d3ee;margin-bottom:8px;">👤 Signed in as <b>{_user}</b></div>',
                unsafe_allow_html=True,
            )
        if st.button("🚪 Sign Out", use_container_width=True, key="btn_signout"):
            st.session_state.authenticated = False
            st.session_state.user_name = ""
            st.session_state.app_stage = "landing"
            st.session_state.landing_section = "home"
            st.rerun()

        st.markdown("### ⚙️ Configuration")
        st.divider()

        # ── API key from secrets.toml (never exposed in UI) ───────────────────
        _api_key = ""
        try:
            _api_key = st.secrets.get("OPENROUTER_API_KEY", "") or ""
        except Exception:
            pass

        cfg["api_key"] = _api_key

        st.markdown("**🤖 OpenRouter AI**")
        if _api_key:
            st.markdown(
                '<div style="font-size:.7rem;padding:5px 10px;border-radius:6px;'
                'background:rgba(34,211,238,.1);border:1px solid rgba(34,211,238,.25);'
                'color:#22d3ee;margin-bottom:8px;">✓ API Key connected via secrets</div>',
                unsafe_allow_html=True,
            )
        model_label = st.selectbox("LLM Model", list(OPENROUTER_MODELS.keys()))
        cfg["model"] = OPENROUTER_MODELS[model_label]

        st.divider()
        st.markdown("**🔧 Pipeline**")
        cfg["flag_rag"]    = st.checkbox("RAG Context Injection", True)
        cfg["flag_llm"]    = st.checkbox("LLM Spec Parser", True)
        cfg["flag_cv"]     = st.checkbox("CV Layer Builder", True)
        cfg["flag_review"] = st.checkbox("LLM Vision QA Review", bool(_api_key))

        st.divider()
        st.markdown("**🎨 Layer Visibility**")
        cfg["layers"] = []
        cols = st.columns(2)
        for i, lyr in enumerate(LAYER_NAMES):
            with cols[i % 2]:
                if st.checkbox(lyr, True, key=f"lyr_{lyr}"):
                    cfg["layers"].append(lyr)

        st.divider()
        st.markdown("**📐 Export Settings**")
        cfg["dpi"] = st.selectbox("Resolution (DPI)", [300, 150, 72, 600], index=0)
        cfg["jpeg_quality"] = st.slider("JPEG Quality", 70, 100, 92)

        st.divider()
        st.markdown("**📦 Export Formats**")
        cfg["export_psd"] = st.checkbox("PSD (Photoshop Layers)", True)
        cfg["export_ai"]  = st.checkbox("AI (Adobe Illustrator)", True)
        cfg["export_svg"] = st.checkbox("SVG (Web / Open)", True)

        st.divider()
        st.markdown(
            f"<div class='status-pill'><span class='status-dot'></span>"
            f"{APP_VERSION} · Ready</div>",
            unsafe_allow_html=True,
        )
    return cfg


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER — Spec Reference Card
# ══════════════════════════════════════════════════════════════════════════════
def render_spec_card(sd: Dict[str, Any]):
    w, h = sd["die_w"], sd["die_h"]
    bleed = sd.get("bleed_mm", 3.0)
    rows = [
        ("Document Type", sd.get("doc_type", "Inside Pages")),
        ("Binding", sd.get("binding", "Perfect Bound")),
        ("Trim Size", f"{w} × {h} cm"),
        ("Trim Size (mm)", f"{w*10:.1f} × {h*10:.1f} mm"),
        ("Trim Size (pt)", f"{w*72/2.54:.2f} × {h*72/2.54:.2f} pt"),
        ("Bleed", f"{bleed} mm ({bleed/10:.2f} cm)"),
        ("Cutmark Offset", f"{sd['cm_offset']} cm ({sd['cm_offset']*10:.2f} mm)"),
        ("Cutmark Length", f"{sd['cm_length']} cm ({sd['cm_length']*72/2.54:.2f} pt)"),
        ("Cutmark Stroke", f"{sd['cm_stroke']} pt"),
        ("Dieline Colour", "Red — RGB (220, 0, 0)"),
        ("Cutmark Colour", "Black — RGB (0, 0, 0)"),
        ("Dimension Colour", "Blue — RGB (0, 100, 220)"),
        ("Dimension Font", sd.get("dim_font", "Arial")),
        ("Paper Weight", "80 gsm"),
        ("Sheets", "36"),
    ]
    html = '<div class="ref-card">'
    for k, v in rows:
        html += f'<div class="ref-row"><span class="ref-key">{k}</span><span class="ref-val">{v}</span></div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — JOB SETUP
# ══════════════════════════════════════════════════════════════════════════════
def tab_job_setup(cfg: Dict[str, Any]) -> Dict[str, Any]:
    sd: Dict[str, Any] = {}

    st.markdown("""
<div class="pp-card">
  <div class="pp-card-title">💬 Natural Language Spec Input</div>
  <div class="pp-card-sub">Paste client spec. NLP + LLM hybrid parser extracts all parameters automatically.</div>
</div>""", unsafe_allow_html=True)

    default_spec = (
        "Careefour - CT97138 - FA287 - Perfect Bnd Colouring Bk 36S 80G - Inside Coloring Book\n\n"
        "Die-line: 21cm x 29.7cm (Red)\n"
        "Cutmarks: A4 Size\n"
        "Dimensions: 21cm x 29.7cm\n"
        "Cutmarks offset: 0.25 cm from Die-line\n"
        "Cutmarks length: 1.27 cm (0.5 inch)\n"
        "Cutmarks stroke: 0.3 pt\n"
        "Dimension font: Arial\n"
        "Bleed: 3 mm\n"
        "Binding: Perfect Bound\n"
        "Paper: 80 gsm"
    )

    nlp_input = st.text_area("Client Specification", default_spec, height=200)

    if st.button("⚡ Parse Specification", use_container_width=True):
        parsed = parse_spec_from_text(nlp_input)
        st.session_state._parsed_spec = parsed
        st.success(f"✓ Parsed: {parsed.label} — {parsed.width_cm} × {parsed.height_cm} cm")

    st.divider()

    # ── Manual override form ──────────────────────────────────────────────────
    st.markdown("""
<div class="pp-card">
  <div class="pp-card-title">📐 Artwork Specification</div>
  <div class="pp-card-sub">Manual parameters — these override the NLP parser.</div>
</div>""", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Document**")
        doc_type = st.selectbox("Document Type", DOC_TYPES, index=0)
        binding  = st.selectbox("Binding Type", BINDING_TYPES, index=0)

        # FIX: when user changes Standard Size, reset width/height fields
        # so changes (A4 → A5, etc.) actually propagate to the pipeline.
        prev_size = st.session_state.get("_prev_size_sel")
        size_sel = st.selectbox("Standard Size",
                                list(STANDARD_SIZES_DISPLAY.keys()),
                                index=0, key="size_sel")
        if size_sel != prev_size and STANDARD_SIZES_DISPLAY[size_sel] is not None:
            pw, ph = STANDARD_SIZES_DISPLAY[size_sel]
            st.session_state["die_w_val"] = float(pw)
            st.session_state["die_h_val"] = float(ph)
        st.session_state["_prev_size_sel"] = size_sel

        if STANDARD_SIZES_DISPLAY[size_sel]:
            pre_w, pre_h = STANDARD_SIZES_DISPLAY[size_sel]
        else:
            pre_w, pre_h = 21.0, 29.7

        die_w = st.number_input("Width (cm)", 1.0, 200.0,
                                value=st.session_state.get("die_w_val", pre_w),
                                step=0.1, format="%.2f", key="die_w_val")
        die_h = st.number_input("Height (cm)", 1.0, 300.0,
                                value=st.session_state.get("die_h_val", pre_h),
                                step=0.1, format="%.2f", key="die_h_val")
        bleed = st.number_input("Bleed (mm)", 0.0, 10.0, 3.0, 0.5)

    with col2:
        st.markdown("**Cutmarks**")
        cm_offset = st.number_input("Offset from Dieline (cm)", 0.0, 2.0, 0.0, 0.05, format="%.3f",
                                    help="0 = cutmarks start flush with the page/dieline edge (PDF reference style)")
        cm_length = st.number_input("Mark Length (cm)", 0.3, 5.0, 1.27, 0.05, format="%.3f")
        cm_stroke = st.number_input("Stroke Weight (pt)", 0.1, 2.0, 0.3, 0.05, format="%.2f")

        st.markdown("**Colours** (fixed to ISO spec)")
        st.markdown(
            '<div style="font-size:.72rem;margin-top:6px;">'
            '<span style="color:#dc2626;">● Dieline — Red</span><br>'
            '<span style="color:#111;">● Cutmarks — Black</span><br>'
            '<span style="color:#1d4ed8;">● Dimensions — Blue</span>'
            '</div>',
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown("**Dimensions & Output**")
        dim_font = st.text_input("Dimension Font", "Arial")

        st.markdown("**Paper**")
        paper_gsm = st.selectbox("Paper Weight", ["80 gsm", "100 gsm", "115 gsm", "130 gsm", "170 gsm"])
        sheets    = st.number_input("Sheets / Pages", 4, 500, 36, 4)

        st.markdown("**Label**")
        size_label = size_sel.split("—")[0].strip() if "—" in size_sel else "A4"

    sd = {
        "nlp_input": nlp_input,
        "doc_type": doc_type, "binding": binding,
        "size_label": size_label,
        "die_w": die_w, "die_h": die_h,
        "bleed_mm": bleed,
        "cm_offset": cm_offset, "cm_length": cm_length, "cm_stroke": cm_stroke,
        "dim_font": dim_font,
        "paper_gsm": paper_gsm, "sheets": sheets,
    }

    st.divider()

    # ── Spec reference card ───────────────────────────────────────────────────
    col_ref, col_pts = st.columns(2)
    with col_ref:
        st.markdown("**📋 Specification Summary**")
        render_spec_card(sd)
    with col_pts:
        st.markdown("**📏 Unit Conversions**")
        st.markdown(f"""
<div class="ref-card">
  <div class="ref-row"><span class="ref-key">Width (cm)</span><span class="ref-val">{die_w} cm</span></div>
  <div class="ref-row"><span class="ref-key">Width (mm)</span><span class="ref-val">{die_w*10:.2f} mm</span></div>
  <div class="ref-row"><span class="ref-key">Width (pt)</span><span class="ref-val">{die_w*72/2.54:.3f} pt</span></div>
  <div class="ref-row"><span class="ref-key">Width (px @300)</span><span class="ref-val">{int(die_w/2.54*300)} px</span></div>
  <div class="ref-row"><span class="ref-key">Height (cm)</span><span class="ref-val">{die_h} cm</span></div>
  <div class="ref-row"><span class="ref-key">Height (mm)</span><span class="ref-val">{die_h*10:.2f} mm</span></div>
  <div class="ref-row"><span class="ref-key">Height (pt)</span><span class="ref-val">{die_h*72/2.54:.3f} pt</span></div>
  <div class="ref-row"><span class="ref-key">Height (px @300)</span><span class="ref-val">{int(die_h/2.54*300)} px</span></div>
  <div class="ref-row"><span class="ref-key">CM Offset (pt)</span><span class="ref-val">{cm_offset*72/2.54:.3f} pt</span></div>
  <div class="ref-row"><span class="ref-key">CM Length (pt)</span><span class="ref-val">{cm_length*72/2.54:.3f} pt</span></div>
  <div class="ref-row"><span class="ref-key">Total w/ Marks</span><span class="ref-val">{die_w + 2*(cm_offset+cm_length):.3f} cm</span></div>
</div>""", unsafe_allow_html=True)

    return sd


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════════
def tab_architecture():
    nodes = [
        ("🔍", "parse_spec", "Spec Parser Node",
         "Hybrid NLP regex + LLM. Extracts: dimensions, cutmark offset/length/stroke, font, bleed, binding, doc type.",
         ["spec_parser.py", "ArtworkSpec", "STANDARD_SIZES", "OpenRouter fallback"]),
        ("📄", "load_pdf", "PDF Loader Node",
         "PyMuPDF rasterisation at configured DPI. RGB → BGRA. Text block extraction with bbox. Handles multi-page PDFs.",
         ["pymupdf/fitz", "cv2", "numpy", "PdfPage dataclass"]),
        ("🎨", "build_layers", "Layer Builder Node",
         "OpenCV geometry engine. Builds 7 BGRA layers: ARTWORK, BLEED, DIELINE, CUTMARKS, DIMENSIONS, MEASURE, TEXT. Exact ISO 12647 placement.",
         ["layer_builder.py", "cv2", "Pillow", "numpy", "PackBits"]),
        ("🤖", "llm_review", "LLM QA Review Node",
         "Vision-Language Model reviews composite image. Risk scoring 0-100. JSON-structured QA report. Skipped if no API key.",
         ["llm_reviewer.py", "OpenRouter", "AsyncOpenAI", "VLM vision"]),
        ("📦", "package_exports", "Export Packager Node",
         "Generates PSD (7-layer binary), AI (PDF/PostScript), EPS (DSC 3.0), SVG (Inkscape layers). ZIP bundle for batch.",
         ["psd_exporter.py", "ai_exporter.py", "reportlab", "struct"]),
    ]

    st.markdown("""
<div class="pp-card">
  <div class="pp-card-title">🔗 LangGraph Pipeline — 5 Nodes</div>
  <div class="pp-card-sub">parse_spec → load_pdf → build_layers → llm_review → package_exports</div>
</div>""", unsafe_allow_html=True)

    for icon, node_id, name, desc, libs in nodes:
        libs_html = "".join(
            f'<span style="font-size:.58rem;padding:2px 8px;border-radius:20px;'
            f'background:rgba(77,159,255,.12);color:#4d9fff;border:1px solid rgba(77,159,255,.25);'
            f'margin:2px;display:inline-block;">{l}</span>'
            for l in libs
        )
        st.markdown(f"""
<div class="pipe-node">
  <div class="pipe-icon">{icon}</div>
  <div>
    <div class="pipe-title">{name}
      <span style="font-size:.6rem;color:#3a5090;margin-left:8px;">[{node_id}]</span>
    </div>
    <div class="pipe-desc">{desc}</div>
    <div style="margin-top:8px;">{libs_html}</div>
  </div>
</div>
<div style="text-align:center;color:#1e2448;font-size:.9rem;margin:-2px 0 2px;">▼</div>
""", unsafe_allow_html=True)

    st.divider()

    # RAG Engine details
    st.markdown("""
<div class="pp-card">
  <div class="pp-card-title">📚 RAG Engine — Hybrid BM25 + TF-IDF Cosine</div>
  <div class="pp-card-sub">32+ prepress domain knowledge chunks · k1=1.5 · b=0.75 · Top-6 retrieval</div>
</div>""", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.code("""# Hybrid retrieval
combined = 0.5 * norm(bm25_scores)
         + 0.5 * norm(cosine_scores)
top_k    = argsort(combined)[::-1][:6]

# BM25 scoring
idf = log((N - df + 0.5) / (df + 0.5) + 1)
bm25 += idf * tf * (k1+1) / (tf + k1*(1-b+b*dl/avgdl))

# Dense embedding (TF-IDF sparse)
vec[vocab[token]] += 1.0
vec = vec / norm(vec)  # L2-normalise""", language="python")
    with col_b:
        st.code("""@dataclass ArtworkSpec:
  label, width_cm, height_cm
  bleed_mm, cutmark_offset_cm
  cutmark_length_cm, cutmark_stroke_pt
  dimension_font, has_dieline
  has_cutmarks, has_dimensions
  binding, doc_type, notes

@dataclass PrepressResult:
  artwork_layer   # BGRA ndarray
  bleed_layer     # BGRA ndarray
  dieline_layer   # BGRA ndarray
  cutmark_layer   # BGRA ndarray
  dimension_layer # BGRA ndarray
  measure_layer   # BGRA ndarray
  text_layer      # BGRA ndarray
  composite       # merged preview
  dieline_bbox_px # (x0,y0,x1,y1)""", language="python")


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — AI PIPELINE VISUALISER
# ══════════════════════════════════════════════════════════════════════════════
def tab_ai_pipeline(sd: Dict[str, Any], cfg: Dict[str, Any]):
    col_l, col_r = st.columns([1, 1])

    with col_l:
        st.markdown("""
<div class="pp-card">
  <div class="pp-card-title">⚙️ Pipeline Status</div>
</div>""", unsafe_allow_html=True)

        pipeline_nodes = [
            ("parse_spec",       "🔍", "Spec Parser",       "NLP + LLM hybrid",      "#4d9fff", "✓ Ready"),
            ("load_pdf",         "📄", "PDF Loader",         "PyMuPDF + BGRA conv",   "#22d3ee", "✓ Ready"),
            ("build_layers",     "🎨", "Layer Builder",      "OpenCV 7-layer stack",  "#4ade80", "✓ Ready"),
            ("llm_review",       "🤖", "LLM QA",            "VLM risk scoring",      "#fbbf24",
             "✓ Active" if cfg.get("api_key") and cfg.get("flag_review") else "⚠ No API Key"),
            ("package_exports",  "📦", "Export Packager",    "PSD · AI · EPS · SVG", "#c084fc", "✓ Ready"),
        ]

        for node_id, icon, name, desc, color, status in pipeline_nodes:
            st.markdown(f"""
<div class="pipe-node" style="border-color:{color}33;">
  <div class="pipe-icon">{icon}</div>
  <div style="flex:1">
    <div class="pipe-title" style="color:{color};">{name}</div>
    <div class="pipe-desc">{desc}</div>
  </div>
  <span style="font-size:.62rem;font-weight:700;color:{color};white-space:nowrap;">{status}</span>
</div>
<div style="text-align:center;color:#1e2448;margin:-2px 0 2px;">▼</div>
""", unsafe_allow_html=True)

    with col_r:
        st.markdown("""
<div class="pp-card">
  <div class="pp-card-title">📚 RAG Retrieval</div>
  <div class="pp-card-sub">Live BM25 + cosine scores for current job query</div>
</div>""", unsafe_allow_html=True)

        query = f"dieline cutmarks dimensions {sd.get('size_label','A4')} {sd.get('die_w',21)}cm"
        rag = _cached_rag()
        bm25_s  = rag._bm25(query)
        dense_s = rag._cosine(query)
        def _n(a):
            mn, mx = a.min(), a.max()
            return (a - mn) / max(1e-9, mx - mn)
        combined = 0.5 * _n(bm25_s) + 0.5 * _n(dense_s)
        top_idx = sorted(range(len(combined)), key=lambda i: -combined[i])[:8]

        for idx in top_idx:
            chunk = rag._chunks[idx]
            score = float(combined[idx])
            short = chunk[:70] + "…" if len(chunk) > 70 else chunk
            pct   = int(score * 100)
            st.markdown(f"""
<div class="rag-bar">
  <div style="font-size:.67rem;color:#5a6899;margin-bottom:4px;">{short}</div>
  <div style="display:flex;align-items:center;gap:8px;">
    <div style="flex:1;background:#0a0d1e;border-radius:3px;height:4px;">
      <div style="width:{pct}%;height:4px;border-radius:3px;
                  background:linear-gradient(90deg,#4d9fff,#22d3ee);"></div>
    </div>
    <span style="font-size:.66rem;color:#22d3ee;min-width:32px;">{score:.3f}</span>
  </div>
</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 4 — GENERATE MASTER PROMPT
# ══════════════════════════════════════════════════════════════════════════════
def tab_generated_prompt(sd: Dict[str, Any], cfg: Dict[str, Any]):
    w, h = sd["die_w"], sd["die_h"]
    bleed = sd.get("bleed_mm", 3.0)
    off, lng, stk = sd["cm_offset"], sd["cm_length"], sd["cm_stroke"]
    layers_str = " · ".join(cfg.get("layers", LAYER_NAMES))

    prompt = f"""╔══════════════════════════════════════════════════════════════════════════════╗
║  MASTER PREPRESS SYSTEM PROMPT — AI Document Intelligence & Artwork       ║
║  Navneet Processing System {APP_VERSION}                                          ║
╚══════════════════════════════════════════════════════════════════════════════╝

JOB REFERENCE:  {sd.get('doc_type','Inside Pages')} — {sd.get('binding','Perfect Bound')}
ARTWORK SIZE:   {w} cm × {h} cm  ({w*10:.1f} mm × {h*10:.1f} mm)  ({w*72/2.54:.3f} pt × {h*72/2.54:.3f} pt)
BLEED:          {bleed} mm ({bleed/10:.3f} cm) on all 4 sides
PAPER:          {sd.get('paper_gsm','80 gsm')} — {sd.get('sheets', 36)} sheets

━━━ LAYER STACK (PSD top → bottom / AI layer order) ━━━━━━━━━━━━━━━━━━━━━━━━━
{layers_str}

━━━ DIELINE SPECIFICATION ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Size:        {w} cm × {h} cm  (exact trim)
• Colour:      RGB (220, 0, 0)  |  C=0 M=100 Y=100 K=14  |  Pantone 485
• Stroke:      0.5 pt  |  No fill
• Placement:   Centred on page canvas
• Layer:       DIELINE (isolated, no artwork bleed)

━━━ CUTMARK SPECIFICATION (ISO 12647) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Count:       8 marks (2 per corner — 1 horizontal + 1 vertical)
• Colour:      RGB (0, 0, 0)  Black — registration colour
• Offset:      {off} cm ({off*10:.3f} mm / {off*72/2.54:.4f} pt) from dieline edge
• Length:      {lng} cm ({lng*10:.3f} mm / {lng*72/2.54:.4f} pt)
• Stroke:      {stk} pt  (hairline — never thicker than 0.5 pt)
• Layer:       CUTMARKS (must not overlap artwork or dieline)

━━━ DIMENSION ANNOTATION SPECIFICATION ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Width line:  Horizontal, placed BELOW dieline, between x0 and x1 of dieline
• Height line: Vertical, placed to the RIGHT of dieline, between y0 and y1
• Tick marks:  At each endpoint (perpendicular, 4 pt long)
• Arrowheads:  Pointing inward (closed triangle fill)
• Colour:      RGB (0, 100, 220) — Cyan-Blue
• Stroke:      0.5 pt
• Labels:      "{w} cm" (width, centred below horizontal line)
               "{h} cm" (height, rotated 90°, beside vertical line)
• Font:        {sd.get('dim_font','Arial')}, 8 pt, same blue colour
• Layer:       DIMENSIONS (lines) + MEASURE (text labels)

━━━ BLEED BOX SPECIFICATION ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Extend:      {bleed} mm beyond dieline on all 4 sides
• Total size:  {w+2*bleed/10:.3f} cm × {h+2*bleed/10:.3f} cm
• Style:       Dashed Magenta, 0.25 pt
• Layer:       BLEED

━━━ ADOBE ILLUSTRATOR OUTPUT SPEC ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Format:      PDF 1.7 with %%AI preamble (Illustrator CC 2024)
• Layer groups: Optional Content Groups (OCGs) per named layer
• Artwork:     Embedded JPEG, 92% quality, positioned at exact trim size
• Vectors:     All cutmarks, dieline, dimension lines are native vector
• Font:        Helvetica (PDF equivalent of Arial for dimension labels)
• Page size:   {w + 2*(off+lng+0.8):.3f} cm × {h + 2*(off+lng+0.8):.3f} cm (with mark margins)

━━━ PHOTOSHOP PSD OUTPUT SPEC ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Format:      PSD v1, 8 bits/channel, RGB + Alpha (RGBA)
• Layers:      TEXT / MEASURE / DIMENSIONS / DIELINE / CUTMARKS / BLEED / ARTWORK
• Blend mode:  Normal @ 100% opacity on all layers
• Compression: PackBits RLE per channel
• Unicode:     Layer names stored as Unicode (luni extra data)
• Resolution:  {cfg.get('dpi',300)} DPI

━━━ EPS OUTPUT SPEC ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Format:      EPS DSC 3.0 with %%BeginObject / %%EndObject layer DSC comments
• Vectors:     All prepress marks as PostScript paths
• Font:        Helvetica (standard PS Type1)

━━━ SVG OUTPUT SPEC ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Format:      SVG 1.1 with Inkscape layer extensions
• Units:       cm (scalable, resolution-independent)
• Layers:      inkscape:groupmode="layer" groups
• Artwork:     Embedded JPEG data URI

━━━ LANGGRAPH PIPELINE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
parse_spec → load_pdf → build_layers → llm_review → package_exports
  [NLP+LLM]    [PyMuPDF]  [OpenCV×7]    [VLM QA]     [PSD+AI+EPS+SVG+ZIP]

━━━ RAG KNOWLEDGE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Engine:  Hybrid BM25 (k1=1.5, b=0.75) + TF-IDF cosine
Chunks:  32+ ISO/FOGRA prepress domain rules
Retrieval: Top-6, combined score = 0.5×BM25 + 0.5×cosine
"""

    st.markdown("""
<div class="pp-card">
  <div class="pp-card-title">📋 Generated Master Prompt</div>
  <div class="pp-card-sub">Copy-paste into any LLM, Claude Project, or use as AI system prompt.</div>
</div>""", unsafe_allow_html=True)

    st.markdown(f'<div class="prompt-box">{prompt}</div>', unsafe_allow_html=True)
    st.download_button(
        "⬇️ Download Master Prompt (.txt)",
        data=prompt.encode("utf-8"),
        file_name=f"prepress_master_prompt_{int(time.time())}.txt",
        mime="text/plain",
        use_container_width=True,
    )
    st.session_state.generated_prompt = prompt


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS — Result display
# ══════════════════════════════════════════════════════════════════════════════
def _build_spec_from_sd(sd: Dict[str, Any]) -> ArtworkSpec:
    # FIX: pass layer-toggle flags from sidebar dict into ArtworkSpec.
    # Previously has_dieline / has_cutmarks / has_dimensions were never read
    # from sd, so unchecking the sidebar checkboxes had no effect — all layers
    # were always built regardless of user selection.
    return ArtworkSpec(
        label=sd["size_label"],
        width_cm=sd["die_w"],
        height_cm=sd["die_h"],
        bleed_mm=sd.get("bleed_mm", 3.0),
        cutmark_offset_cm=sd["cm_offset"],
        cutmark_length_cm=sd["cm_length"],
        cutmark_stroke_pt=sd["cm_stroke"],
        dimension_font=sd.get("dim_font", "Arial"),
        binding=sd.get("binding", "Perfect Bound"),
        doc_type=sd.get("doc_type", "Inside Pages"),
        has_dieline=sd.get("has_dieline", True),
        has_cutmarks=sd.get("has_cutmarks", True),
        has_dimensions=sd.get("has_dimensions", True),
        has_artwork=sd.get("has_artwork", True),
        has_bleed_box=sd.get("bleed_mm", 3.0) > 0,
        notes=sd.get("nlp_input", ""),
    )


def _display_result(result: PrepressResult, pg: int, cfg: Dict[str, Any]):
    """Render one page result with tabs for layers, preview, downloads."""
    w_cm, h_cm = result.spec.width_cm, result.spec.height_cm
    with st.expander(f"📄  Page {pg}  ·  {w_cm} × {h_cm} cm  ·  {result.dpi} DPI",
                     expanded=(pg == 1)):
        tabs = st.tabs(["🖼 Preview", "🎨 Layers", "🤖 QA Review", "⬇️ Downloads"])

        with tabs[0]:
            comp_rgb = result.composite[:, :, [2, 1, 0, 3]]
            pil = Image.fromarray(comp_rgb, "RGBA")
            st.image(pil, caption=f"Page {pg} — Composite (all layers)", use_container_width=True)
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            col_m1.metric("Width", f"{result.spec.width_cm} cm")
            col_m2.metric("Height", f"{result.spec.height_cm} cm")
            col_m3.metric("Resolution", f"{result.dpi} DPI")
            col_m4.metric("Canvas", f"{result.width_px}×{result.height_px}px")

        with tabs[1]:
            LAYER_MAP = [
                ("ARTWORK",    "artwork_layer",    "Clean raster artwork"),
                ("BLEED",      "bleed_layer",      "Bleed box (magenta dashed)"),
                ("DIELINE",    "dieline_layer",    "Red trim rectangle"),
                ("CUTMARKS",   "cutmark_layer",    "Black hairline marks"),
                ("DIMENSIONS", "dimension_layer",  "Blue dimension lines"),
                ("MEASURE",    "measure_layer",    "Dimension text labels"),
                ("TEXT",       "text_layer",       "Product code / header"),
            ]
            show_layers = cfg.get("layers", LAYER_NAMES)
            for lname, attr, ldesc in LAYER_MAP:
                if lname not in show_layers:
                    continue
                arr = getattr(result, attr, None)
                if arr is None:
                    continue
                rgb = arr[:, :, [2, 1, 0, 3]]
                pil2 = Image.fromarray(rgb, "RGBA")
                st.image(pil2, caption=f"{lname} — {ldesc}", use_container_width=True)

        with tabs[2]:
            review = result.ai_review or {}
            rs = review.get("risk_score", -1)
            has_key = bool(cfg.get("api_key"))
            is_on   = cfg.get("flag_review", False)

            # ── Status banner ─────────────────────────────────────────────────
            if has_key:
                st.markdown(
                    '<div style="font-size:.72rem;padding:7px 12px;border-radius:8px;'
                    'background:rgba(34,211,238,.08);border:1px solid rgba(34,211,238,.3);'
                    'color:#22d3ee;margin-bottom:10px;">'
                    '✓ LLM Vision QA Review is enabled. API Key connected ✓. '
                    + ('Review run for this page.' if rs >= 0
                       else ("Enable 'LLM Vision QA Review' in sidebar to activate."
                             if not is_on else "Awaiting next run — click ▶ Run."))
                    + '</div>', unsafe_allow_html=True,
                )
            else:
                st.warning("⚠ Add OPENROUTER_API_KEY to .streamlit/secrets.toml to enable QA.")

            if rs >= 0:
                col_r1, col_r2, col_r3 = st.columns(3)
                risk_label = "🟢 Low Risk" if rs < 30 else ("🟡 Medium" if rs < 60 else "🔴 High Risk")
                col_r1.metric("Risk Score", f"{rs} / 100", delta=risk_label)
                col_r2.metric("Confidence", f"{review.get('confidence', 0):.0%}")
                col_r3.metric("Status", "Pass" if rs < 50 else "Review")
                if review.get("readiness_summary"):
                    st.info(review["readiness_summary"])
                for field, label in [
                    ("dieline_quality",    "Dieline"),
                    ("cutmarks_quality",   "Cutmarks"),
                    ("dimensions_quality", "Dimensions"),
                ]:
                    if review.get(field):
                        st.markdown(f"**{label}:** {review[field]}")
                if review.get("missing_elements"):
                    st.markdown("**Missing Elements:**")
                    for el in review["missing_elements"]:
                        st.markdown(f"- {el}")
                if review.get("operator_actions"):
                    st.markdown("**Required Actions:**")
                    for action in review["operator_actions"]:
                        st.markdown(f"- {action}")
                with st.expander("Full QA Report (JSON)"):
                    st.json(review)
            elif review.get("error"):
                st.warning(f"QA Review error: {review['error']}")
            else:
                if has_key and not is_on:
                    st.info("Tick 'LLM Vision QA Review' in the sidebar, then click ▶ Run.")
                elif not has_key:
                    pass  # banner already shown above
                else:
                    st.info("No risk score yet — click ▶ Run with QA enabled.")

            if result.pipeline_trace:
                with st.expander("Pipeline Trace"):
                    for t in result.pipeline_trace[-15:]:
                        st.caption(f"• {t}")

        with tabs[3]:
            # Fast downloads — pre-built bytes stored in result._cache_*
            cols = st.columns(3)
            with cols[0]:
                try:
                    if not getattr(result, "_cache_psd", None):
                        result._cache_psd = make_psd_bytes(result)
                    st.download_button(
                        "⬇️ PSD (Photoshop)",
                        data=result._cache_psd,
                        file_name=f"page_{pg:03d}.psd",
                        mime="application/octet-stream",
                        key=f"psd_{pg}",
                        use_container_width=True,
                    )
                except Exception as e:
                    st.error(f"PSD: {e}")
            with cols[1]:
                try:
                    if not getattr(result, "_cache_ai", None):
                        result._cache_ai = make_ai_bytes(result)
                    st.download_button(
                        "⬇️ AI (Illustrator)",
                        data=result._cache_ai,
                        file_name=f"page_{pg:03d}.ai",
                        mime="application/postscript",
                        key=f"ai_{pg}",
                        use_container_width=True,
                    )
                except Exception as e:
                    st.error(f"AI: {e}")
            with cols[2]:
                try:
                    if not getattr(result, "_cache_svg", None):
                        result._cache_svg = make_svg_bytes(result)
                    st.download_button(
                        "⬇️ SVG",
                        data=result._cache_svg,
                        file_name=f"page_{pg:03d}.svg",
                        mime="image/svg+xml",
                        key=f"svg_{pg}",
                        use_container_width=True,
                    )
                except Exception as e:
                    st.error(f"SVG: {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — PROCESS ARTWORK (Single + Batch)
# ══════════════════════════════════════════════════════════════════════════════
def tab_process_artwork(sd: Dict[str, Any], cfg: Dict[str, Any]):
    # Current job size banner so user sees what size will be applied
    w, h = sd.get("die_w", 21.0), sd.get("die_h", 29.7)
    size_label = sd.get("size_label", "A4")
    st.markdown(f"""
<div class="pp-card">
  <div class="pp-card-title">🖨️ Process Artwork</div>
  <div class="pp-card-sub">Upload PDF(s) — single file or batch.
    Output is rendered at the job size you set in <b>Job Setup</b>:
    <span style="color:#4d9fff;font-weight:700;">{size_label} · {w} × {h} cm</span>.
    Change the size there to regenerate at a different size.</div>
</div>""", unsafe_allow_html=True)

    upload_mode = st.radio("Upload Mode", ["Single File", "Batch (Multiple Files)"],
                           horizontal=True, label_visibility="collapsed")

    if upload_mode == "Single File":
        uploaded = st.file_uploader("Upload PDF", type=["pdf"], key="single_upload")
        files = [uploaded] if uploaded else []
    else:
        files = st.file_uploader("Upload PDFs (multiple)", type=["pdf"],
                                  accept_multiple_files=True, key="batch_upload") or []

    if files:
        st.markdown(f"**{len(files)} file(s) loaded.**")

    col_a, col_b = st.columns(2)
    with col_a:
        pages_input = st.text_input("Pages to process (blank = all, e.g. 1,3,5)", "")
    with col_b:
        use_ai = st.checkbox("Enable LLM QA Review",
                             value=bool(cfg.get("api_key")) and cfg.get("flag_review", False),
                             help="Vision-LLM reviews each page for production readiness.")

    col_run, col_clr = st.columns([5, 1])
    with col_run:
        run_clicked = st.button("▶  Run", type="primary",
                                use_container_width=True, disabled=not files,
                                key="btn_run")
    with col_clr:
        has_results = bool(st.session_state.get("pp_file_groups"))
        if st.button("🗑️ Clear", use_container_width=True, disabled=not has_results,
                     key="btn_clear"):
            st.session_state["pp_file_groups"] = []
            st.session_state["pp_zip"]         = None
            st.session_state.pipeline_state    = None
            st.session_state.batch_results     = []
            st.rerun()

    # ── Run pipeline only when button is explicitly clicked ───────────────────
    if run_clicked and files:
        selected_pages: List[int] = []
        if pages_input.strip():
            try:
                selected_pages = [int(x.strip()) - 1 for x in pages_input.split(",")]
            except Exception:
                st.warning("Invalid page input — processing all pages")

        spec      = _build_spec_from_sd(sd)
        spec_text = sd["nlp_input"]
        file_groups: List[Dict] = []
        last_state = None
        overall_progress = st.progress(0, "Initialising…")
        status_text = st.empty()

        for file_idx, uf in enumerate(files):
            prog_start = file_idx / len(files)
            prog_end   = (file_idx + 1) / len(files)
            overall_progress.progress(prog_start,
                f"⚙️  Building layers — {uf.name}  ({file_idx+1}/{len(files)})")
            status_text.caption(f"Processing **{uf.name}** at {cfg['dpi']} DPI …")
            try:
                state = run_pipeline_sync(
                    file_bytes=uf.getvalue(),
                    filename=uf.name,
                    spec_text=spec_text,
                    dpi=cfg["dpi"],
                    selected_pages=selected_pages,
                    ai_enabled=use_ai and bool(cfg.get("api_key")),
                    api_key=cfg.get("api_key", ""),
                    ai_model=cfg["model"],
                    spec=spec,
                )
                if state.error:
                    st.error(f"{uf.name}: {state.error}")
                    continue
                file_groups.append({"filename": uf.name, "results": state.results})
                last_state = state
                overall_progress.progress(prog_end,
                    f"✓  {uf.name}  ·  {len(state.results)} page(s) done")
            except Exception as exc:
                st.error(f"{uf.name}: {exc}")
                import traceback
                st.code(traceback.format_exc())

        overall_progress.progress(1.0, "✅  All files processed successfully!")
        status_text.empty()
        st.session_state["pp_file_groups"] = file_groups
        st.session_state["pp_zip"]         = None          # invalidate cached ZIP
        st.session_state.pipeline_state    = last_state if len(files) == 1 else None
        st.session_state.batch_results     = [
            r for g in file_groups for r in g["results"]
        ]

    # ── Display results from session state (survives download reruns) ─────────
    file_groups = st.session_state.get("pp_file_groups", [])
    if not file_groups:
        return

    all_results = [r for g in file_groups for r in g["results"]]
    st.success(f"✓ {len(all_results)} page(s) from {len(file_groups)} file(s) — "
               f"click 🗑️ Clear to reset")

    for group in file_groups:
        fname, fresults = group["filename"], group["results"]
        st.markdown(f"#### 📁 {fname} — {len(fresults)} page(s)")
        for result in fresults:
            _display_result(result, result.page_index + 1, cfg)

    # ── Bundle download (ZIP cached in session state) ─────────────────────────
    st.markdown("---")
    st.markdown("""
<div class="pp-card">
  <div class="pp-card-title">📦 Bundle Download</div>
  <div class="pp-card-sub">All pages · all formats · single ZIP archive.</div>
</div>""", unsafe_allow_html=True)

    col_z1, col_z2 = st.columns(2)
    with col_z1:
        if st.session_state.get("pp_zip") is None:
            st.session_state["pp_zip"] = make_zip_bundle(
                all_results,
                include_psd=cfg.get("export_psd", True),
                include_ai=cfg.get("export_ai",  True),
                include_eps=False,
                include_svg=cfg.get("export_svg", True),
            )
        zip_bytes = st.session_state["pp_zip"]
        st.download_button(
            f"📦 Download All ({len(all_results)} pages, all formats)",
            data=zip_bytes,
            file_name="prepress_bundle.zip",
            mime="application/zip",
            use_container_width=True,
        )
    with col_z2:
        zip_bytes = st.session_state.get("pp_zip") or b""
        st.metric("Total Pages", len(all_results))
        st.metric("Bundle Size", f"{len(zip_bytes)/1024:.0f} KB")


# ══════════════════════════════════════════════════════════════════════════════
#  LANDING + LOGIN — Creative entry experience
# ══════════════════════════════════════════════════════════════════════════════
def inject_landing_css():
    st.markdown("""
<style>
/* ── Hide Streamlit chrome on landing / login ──────────────────────────────── */
#MainMenu, header[data-testid="stHeader"], footer { visibility: hidden; }

/* ── 3D animated space background (perspective grid + glow orbs) ────────────── */
[data-testid="stAppViewContainer"] { background:#05060f !important; }
/* keep all real content above the animated backdrop */
[data-testid="stMain"], [data-testid="stSidebar"],
[data-testid="stHeader"], .block-container { position:relative; z-index:1; }

.lp-bg { position:fixed; inset:0; z-index:0; overflow:hidden; pointer-events:none; }
.lp-bg .aurora {
    position:absolute; inset:0;
    background:
      radial-gradient(1000px 620px at 12% -8%,  rgba(124,58,237,.32), transparent 60%),
      radial-gradient(900px 620px at 88% 2%,    rgba(34,211,238,.24), transparent 58%),
      radial-gradient(820px 720px at 50% 116%,  rgba(255,60,172,.22), transparent 60%),
      radial-gradient(600px 600px at 75% 80%,   rgba(74,222,128,.12), transparent 60%);
    background-size: 200% 200%;
    animation: auroraShift 22s ease-in-out infinite;
}
@keyframes auroraShift {
    0%,100% { background-position: 0% 0%, 100% 0%, 50% 100%, 70% 70%; }
    50%     { background-position: 100% 60%, 0% 40%, 40% 70%, 30% 40%; }
}
/* twinkling star field */
.lp-bg .stars {
    position:absolute; inset:-60%;
    background-image:
      radial-gradient(1.6px 1.6px at 20% 30%, rgba(255,255,255,.9), transparent),
      radial-gradient(1.4px 1.4px at 60% 70%, rgba(180,220,255,.8), transparent),
      radial-gradient(1.2px 1.2px at 80% 18%, rgba(255,255,255,.7), transparent),
      radial-gradient(1.5px 1.5px at 35% 85%, rgba(200,180,255,.8), transparent),
      radial-gradient(1.3px 1.3px at 90% 55%, rgba(255,255,255,.7), transparent),
      radial-gradient(1.2px 1.2px at 10% 60%, rgba(180,255,230,.7), transparent),
      radial-gradient(1.4px 1.4px at 48% 12%, rgba(255,255,255,.75), transparent);
    background-repeat: repeat; background-size: 300px 300px;
    animation: twinkle 4.5s ease-in-out infinite, drift 70s linear infinite;
}
@keyframes twinkle { 0%,100% { opacity:.45; } 50% { opacity:1; } }
@keyframes drift   { to { transform: translateY(-140px); } }
/* perspective floor grid — the 3D depth, raised so it's visible on screen */
.lp-bg .grid {
    position:absolute; left:50%; bottom:-6%; width:300%; height:92%;
    transform: translateX(-50%) perspective(300px) rotateX(60deg);
    transform-origin: 50% 100%;
    background-image:
      linear-gradient(rgba(124,58,237,.55) 1px, transparent 1px),
      linear-gradient(90deg, rgba(34,211,238,.42) 1px, transparent 1px);
    background-size: 52px 52px;
    -webkit-mask-image: linear-gradient(to top, #000 0%, transparent 80%);
    mask-image: linear-gradient(to top, #000 0%, transparent 80%);
    animation: gridMove 4s linear infinite;
}
@keyframes gridMove { from { background-position: 0 0, 0 0; } to { background-position: 0 52px, 0 52px; } }
/* rotating planet + orbit rings — echoes the reference globe */
.lp-bg .planet {
    position:absolute; top:9%; right:6%; width:220px; height:220px; border-radius:50%;
    background: radial-gradient(circle at 34% 30%, rgba(34,211,238,.55), rgba(124,58,237,.4) 55%, rgba(8,10,24,.15) 76%);
    box-shadow: inset -22px -12px 60px rgba(0,0,0,.55), 0 0 70px rgba(124,58,237,.45);
    opacity:.85;
}
.lp-bg .planet::before, .lp-bg .planet::after {
    content:""; position:absolute; border-radius:50%;
}
.lp-bg .planet::before { inset:-46px; border:1px dashed rgba(34,211,238,.45); animation: spin 18s linear infinite; }
.lp-bg .planet::after  { inset:-80px; border:1px dotted rgba(255,60,172,.35); animation: spin 28s linear infinite reverse; }
/* big soft glow orbs */
.lp-bg .orb { position:absolute; border-radius:50%; filter: blur(70px); }
.lp-bg .orb.a { top:-160px; right:-120px; width:520px; height:520px; opacity:.20;
    background: conic-gradient(from 0deg,#ff3cac,#ffd200,#4ade80,#22d3ee,#4d9fff,#c084fc,#ff3cac);
    animation: spin 30s linear infinite; }
.lp-bg .orb.b { bottom:-200px; left:-160px; width:480px; height:480px; opacity:.16;
    background: conic-gradient(from 180deg,#22d3ee,#4d9fff,#c084fc,#ff3cac,#22d3ee);
    animation: spin 40s linear infinite reverse; }
@keyframes spin { to { transform: rotate(360deg); } }

/* ── Navbar (single row, professional) ──────────────────────────────────────── */
/* pull the whole page up — kill Streamlit's big default top padding */
.block-container { padding-top: 1.1rem !important; max-width: 1180px !important; }
/* the nav row itself */
[data-testid="stHorizontalBlock"]:first-of-type { animation: fadeDown .7s ease both; }
.lp-navline {
    height: 1px; margin: 6px 0 2px; border-radius: 2px;
    background: linear-gradient(90deg, transparent, rgba(255,60,172,.5), rgba(34,211,238,.5), rgba(192,132,252,.5), transparent);
    box-shadow: 0 0 14px rgba(124,58,237,.4);
}
.lp-brand { display:flex; align-items:center; gap:12px; }
.lp-logo {
    width:42px; height:42px; border-radius:12px;
    background: conic-gradient(from 0deg,#ff3cac,#ffd200,#4ade80,#22d3ee,#4d9fff,#c084fc,#ff3cac);
    display:flex; align-items:center; justify-content:center;
    font-size:1.32rem; flex-shrink:0;
    box-shadow:0 4px 20px rgba(124,58,237,.5);
    animation: floaty 4s ease-in-out infinite, spin 14s linear infinite;
}
.lp-logo span {
    width:34px; height:34px; border-radius:9px; background:#0a0c18;
    display:flex; align-items:center; justify-content:center; font-size:1.2rem;
    animation: spin 14s linear infinite reverse;
}
.lp-brand-text { font-weight:900; font-size:1.04rem; color:#f2f5ff; letter-spacing:-.01em; line-height:1.1; }
.lp-brand-sub  {
    font-size:.56rem; letter-spacing:.2em; text-transform:uppercase; font-weight:800;
    background: linear-gradient(90deg,#ff3cac,#22d3ee,#c084fc);
    -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent;
}

/* ── Landing buttons — nav links + rainbow CTA ──────────────────────────────── */
.stButton > button {
    background: rgba(255,255,255,.025) !important;
    border: 1px solid rgba(124,58,237,.20) !important;
    color: #d4ddf7 !important;
    border-radius: 11px !important;
    font-weight: 700 !important; font-size: .82rem !important;
    letter-spacing: .01em !important;
    padding: .5rem .85rem !important;
    transition: all .2s ease !important;
}
.stButton > button:hover {
    color: #ffffff !important;
    transform: translateY(-2px) !important;
    border-color: transparent !important;
    background: linear-gradient(90deg, rgba(124,58,237,.30), rgba(34,211,238,.30)) !important;
    box-shadow: 0 8px 22px rgba(124,58,237,.35) !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(90deg,#ff3cac,#ff8a00,#ffd200,#4ade80,#22d3ee,#4d9fff,#c084fc) !important;
    background-size: 300% auto !important;
    border: none !important; color: #0a0a14 !important; font-weight: 900 !important;
    text-shadow: none !important;
    animation: rainbowMove 5s linear infinite, ctaPulse 2.6s ease-in-out infinite !important;
}
.stButton > button[kind="primary"]:hover { transform: translateY(-2px) !important; filter: brightness(1.08) !important; }
/* form submit buttons (login / sign-up) — same rainbow CTA treatment */
.stFormSubmitButton > button {
    background: linear-gradient(90deg,#ff3cac,#ff8a00,#ffd200,#4ade80,#22d3ee,#4d9fff,#c084fc) !important;
    background-size: 300% auto !important;
    border: none !important; color: #0a0a14 !important; font-weight: 900 !important;
    border-radius: 11px !important; padding: .55rem 1rem !important;
    animation: rainbowMove 5s linear infinite, ctaPulse 2.6s ease-in-out infinite !important;
    transition: filter .2s ease, transform .2s ease !important;
}
.stFormSubmitButton > button:hover { transform: translateY(-2px) !important; filter: brightness(1.08) !important; }
@keyframes rainbowMove { to { background-position: 300% center; } }

/* ── Hero ───────────────────────────────────────────────────────────────────── */
.lp-hero { text-align:center; padding: 38px 16px 8px; animation: fadeUp 1s ease both; }
.lp-eyebrow {
    display:inline-block; font-size:.64rem; letter-spacing:.28em; text-transform:uppercase;
    font-weight:900; padding:6px 18px; border-radius:30px; margin-bottom:22px;
    border:1px solid rgba(124,58,237,.35); background:rgba(124,58,237,.10);
    background-image: linear-gradient(90deg,#ff3cac,#ffd200,#22d3ee,#c084fc);
    -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent;
}
.lp-title {
    font-size: clamp(2.2rem, 5.4vw, 4rem); font-weight:900; line-height:1.04;
    letter-spacing:-.025em; margin:0 auto; max-width: 18ch;
    background: linear-gradient(90deg,#ff3cac,#ff8a00,#ffd200,#4ade80,#22d3ee,#4d9fff,#c084fc,#ff3cac);
    background-size: 220% auto; -webkit-background-clip:text; background-clip:text;
    -webkit-text-fill-color:transparent; animation: shimmer 6s linear infinite;
    filter: drop-shadow(0 6px 30px rgba(124,58,237,.35));
}
.lp-sub {
    margin: 22px auto 0; max-width: 660px; font-size: 1rem; line-height:1.7; color:#9aa6cf;
}
.lp-chips { display:flex; flex-wrap:wrap; gap:8px; justify-content:center; margin:26px auto 4px; max-width:740px; }
.lp-chip {
    font-size:.62rem; font-weight:800; letter-spacing:.06em; text-transform:uppercase;
    padding:5px 13px; border-radius:30px; border:1px solid rgba(255,255,255,.12);
    color:#e7ecff; background:rgba(255,255,255,.04);
}
.lp-chip:nth-child(7n+1){ border-color:rgba(255,60,172,.45); color:#ff8fcf; }
.lp-chip:nth-child(7n+2){ border-color:rgba(255,138,0,.45);  color:#ffc285; }
.lp-chip:nth-child(7n+3){ border-color:rgba(255,210,0,.40);  color:#ffe27a; }
.lp-chip:nth-child(7n+4){ border-color:rgba(74,222,128,.45); color:#8ff0b4; }
.lp-chip:nth-child(7n+5){ border-color:rgba(34,211,238,.45); color:#7fe6f5; }
.lp-chip:nth-child(7n+6){ border-color:rgba(77,159,255,.45); color:#9fc4ff; }
.lp-chip:nth-child(7n+7){ border-color:rgba(192,132,252,.45);color:#d3b3fb; }

/* ── Developer card ─────────────────────────────────────────────────────────── */
.lp-dev {
    margin: 30px auto 0; max-width: 560px;
    background: linear-gradient(135deg, rgba(11,15,34,.85), rgba(8,11,24,.85));
    border:1px solid rgba(77,159,255,.18); border-radius:18px; padding:20px 24px;
    display:flex; align-items:center; gap:18px; text-align:left;
    box-shadow:0 10px 40px rgba(0,20,80,.3); animation: fadeUp 1.2s ease both;
}
.lp-dev-avatar {
    width:58px; height:58px; flex-shrink:0; border-radius:16px;
    background: conic-gradient(from 0deg,#ff3cac,#ffd200,#4ade80,#22d3ee,#4d9fff,#c084fc,#ff3cac);
    color:#0a0a14; display:flex; align-items:center; justify-content:center;
    font-weight:900; font-size:1.4rem; box-shadow:0 6px 22px rgba(124,58,237,.45);
}
.lp-dev-name { font-weight:900; font-size:1.04rem; color:#eaf1ff; }
.lp-dev-role { font-size:.72rem; color:#7da0d8; margin-top:2px; }
.lp-dev-links { display:flex; gap:10px; margin-top:9px; }
.lp-dev-links a {
    font-size:.68rem; font-weight:700; text-decoration:none; color:#9fc4ff;
    padding:4px 11px; border-radius:20px; border:1px solid rgba(77,159,255,.28);
    background:rgba(77,159,255,.06); transition:all .18s ease;
}
.lp-dev-links a:hover { color:#fff; border-color:#22d3ee; background:rgba(34,211,238,.14); }

/* ── About / Contact panels ─────────────────────────────────────────────────── */
.lp-panel {
    margin: 20px auto 0; max-width: 760px;
    background: rgba(11,15,34,.7); border:1px solid rgba(77,159,255,.16);
    border-radius:18px; padding:26px 30px; animation: fadeUp .8s ease both;
}
.lp-panel h3 { color:#eaf1ff; font-size:1.2rem; font-weight:900; margin:0 0 12px; }
.lp-panel p  { color:#90a0cc; font-size:.9rem; line-height:1.75; }
.lp-feature-row { display:flex; gap:14px; align-items:flex-start; margin-top:14px; }
.lp-feature-ico { font-size:1.3rem; }
.lp-feature-t   { color:#dde3f8; font-weight:700; font-size:.86rem; }
.lp-feature-d   { color:#7d8cb5; font-size:.76rem; line-height:1.55; }

/* ── Start button (the big animated CTA) ────────────────────────────────────── */
@keyframes ctaPulse {
    0%,100% { box-shadow: 0 6px 26px rgba(124,58,237,.5), 0 0 0 0 rgba(34,211,238,.45); }
    50%     { box-shadow: 0 12px 36px rgba(255,60,172,.55), 0 0 0 13px rgba(34,211,238,0); }
}

/* ── Developer profile page ─────────────────────────────────────────────────── */
.lp-profile {
    margin: 16px auto 0; max-width: 720px; text-align:center;
    background: linear-gradient(135deg, rgba(13,16,38,.85), rgba(8,11,24,.9));
    border:1px solid rgba(124,58,237,.22); border-radius:22px; padding:36px 32px 30px;
    box-shadow:0 16px 60px rgba(0,10,60,.4); animation: fadeUp .9s ease both;
}
.lp-profile-avatar {
    width:108px; height:108px; margin:0 auto 16px; border-radius:28px;
    background: conic-gradient(from 0deg,#ff3cac,#ffd200,#4ade80,#22d3ee,#4d9fff,#c084fc,#ff3cac);
    color:#0a0a14; display:flex; align-items:center; justify-content:center;
    font-weight:900; font-size:2.4rem; box-shadow:0 10px 36px rgba(124,58,237,.55);
    animation: floaty 4s ease-in-out infinite;
}
.lp-profile-name { font-size:1.7rem; font-weight:900; color:#f2f5ff; letter-spacing:-.01em; }
.lp-profile-role {
    font-size:.84rem; font-weight:800; margin-top:6px;
    background: linear-gradient(90deg,#ff3cac,#22d3ee,#c084fc);
    -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent;
}
.lp-profile-bio { color:#9aa6cf; font-size:.9rem; line-height:1.75; max-width:560px; margin:16px auto 0; }
.lp-skill-row { display:flex; flex-wrap:wrap; gap:8px; justify-content:center; margin:20px auto 6px; }
.lp-contact-row { display:flex; gap:12px; justify-content:center; flex-wrap:wrap; margin-top:22px; }
.lp-contact-btn {
    text-decoration:none; font-weight:800; font-size:.82rem; padding:11px 22px; border-radius:12px;
    color:#fff; border:1px solid rgba(255,255,255,.12); background:rgba(255,255,255,.04);
    transition: all .2s ease;
}
.lp-contact-btn:hover { transform: translateY(-2px); border-color:transparent;
    background: linear-gradient(90deg,#ff3cac,#7c3aed,#22d3ee); color:#fff;
    box-shadow:0 10px 26px rgba(124,58,237,.45); }

/* ── Login card ─────────────────────────────────────────────────────────────── */
.lp-login-head { text-align:center; margin-bottom:6px; animation: fadeUp .7s ease both; }
.lp-login-ico {
    width:64px; height:64px; margin:0 auto 14px; border-radius:18px;
    background: conic-gradient(from 0deg,#ff3cac,#ffd200,#4ade80,#22d3ee,#4d9fff,#c084fc,#ff3cac);
    color:#0a0a14; display:flex; align-items:center; justify-content:center; font-size:1.9rem;
    box-shadow:0 8px 30px rgba(124,58,237,.5);
    animation: floaty 4s ease-in-out infinite, spin 16s linear infinite;
}
.lp-login-title { font-size:1.5rem; font-weight:900; color:#f2f5ff; }
.lp-login-sub   { font-size:.8rem; color:#8b97bf; margin-top:4px; }

/* ── Login form inputs + tabs (landing-only styling) ────────────────────────── */
.stTextInput input {
    background:#0a0e1f !important; color:#eaf1ff !important;
    border:1px solid rgba(124,58,237,.25) !important; border-radius:10px !important;
}
.stTextInput input:focus {
    border-color:#22d3ee !important; box-shadow:0 0 0 3px rgba(34,211,238,.15) !important;
}
.stTextInput label, .stCheckbox label span { color:#aeb9dd !important; }
div[data-testid="stTabs"] button[role="tab"] { font-weight:800 !important; color:#8b97bf !important; }
div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] { color:#22d3ee !important; }
div[data-testid="stTabs"] [data-baseweb="tab-highlight"] {
    background: linear-gradient(90deg,#ff3cac,#22d3ee,#c084fc) !important;
}

/* ── Keyframes ──────────────────────────────────────────────────────────────── */
@keyframes shimmer  { to { background-position: 220% center; } }
@keyframes floaty   { 0%,100%{ transform: translateY(0); } 50%{ transform: translateY(-7px); } }
@keyframes fadeUp   { from{ opacity:0; transform: translateY(18px);} to{ opacity:1; transform:none; } }
@keyframes fadeDown { from{ opacity:0; transform: translateY(-14px);} to{ opacity:1; transform:none; } }

.lp-foot { text-align:center; margin:34px 0 6px; font-size:.66rem; color:#3a4a78; }
.lp-foot a { color:#5a78c0; text-decoration:none; }
</style>

<div class="lp-bg">
  <div class="aurora"></div>
  <div class="stars"></div>
  <div class="planet"></div>
  <div class="grid"></div>
  <div class="orb a"></div>
  <div class="orb b"></div>
</div>
""", unsafe_allow_html=True)


def render_navbar():
    """Single-row professional navbar — brand · Home · About · Contact · Developer · Sign In."""
    nav = st.columns([3.0, 1.0, 1.0, 1.15, 1.45, 1.5], vertical_alignment="center")
    with nav[0]:
        st.markdown("""
<div class="lp-brand">
  <div class="lp-logo"><span>🖨️</span></div>
  <div>
    <div class="lp-brand-text">AI Document Intelligence</div>
    <div class="lp-brand-sub">Artwork · Navneet Processing System</div>
  </div>
</div>""", unsafe_allow_html=True)
    with nav[1]:
        if st.button("🏠 Home", use_container_width=True, key="nav_home"):
            st.session_state.landing_section = "home"; st.rerun()
    with nav[2]:
        if st.button("ℹ️ About", use_container_width=True, key="nav_about"):
            st.session_state.landing_section = "about"; st.rerun()
    with nav[3]:
        if st.button("✉️ Contact", use_container_width=True, key="nav_contact"):
            st.session_state.landing_section = "contact"; st.rerun()
    with nav[4]:
        if st.button("👨‍💻 Developer", use_container_width=True, key="nav_dev"):
            st.session_state.landing_section = "developer"; st.rerun()
    with nav[5]:
        if st.button("🔐 Sign In", type="primary", use_container_width=True, key="nav_signin"):
            st.session_state.app_stage = "login"; st.rerun()
    st.markdown('<div class="lp-navline"></div>', unsafe_allow_html=True)


def _render_dev_card():
    st.markdown(f"""
<div class="lp-dev">
  <div class="lp-dev-avatar">SJ</div>
  <div>
    <div class="lp-dev-name">{DEV_NAME}</div>
    <div class="lp-dev-role">{DEV_TITLE} · {DEV_COMPANY} · {DEV_YEAR}</div>
    <div class="lp-dev-links">
      <a href="mailto:{DEV_GMAIL}">✉️ Gmail</a>
      <a href="{DEV_LINKEDIN}" target="_blank">in LinkedIn</a>
    </div>
  </div>
</div>""", unsafe_allow_html=True)


def render_landing_page():
    inject_landing_css()
    render_navbar()

    section = st.session_state.get("landing_section", "home")

    if section == "home":
        st.markdown(f"""
<div class="lp-hero">
  <span class="lp-eyebrow">Navneet Education Limited · {DEV_YEAR}</span>
  <div class="lp-title">AI Document Intelligence &amp; Artwork Navneet Processing System</div>
  <p class="lp-sub">
    Production-grade document intelligence and prepress automation — dielines, cutmarks,
    dimensions and multi-format export (AI · PSD · EPS · SVG) with ISO 12647 accuracy,
    powered by a LangGraph + RAG neural pipeline.
  </p>
  <div class="lp-chips">
    <span class="lp-chip">LangGraph</span>
    <span class="lp-chip">LangChain RAG</span>
    <span class="lp-chip">OpenRouter LLM</span>
    <span class="lp-chip">OpenCV CV</span>
    <span class="lp-chip">7-Layer PSD</span>
    <span class="lp-chip">ISO 12647</span>
    <span class="lp-chip">Batch Mode</span>
  </div>
</div>""", unsafe_allow_html=True)

        cta = st.columns([2, 1.4, 2])
        with cta[1]:
            if st.button("🚀  Start Now", type="primary", use_container_width=True, key="lp_start"):
                st.session_state.app_stage = "login"; st.rerun()

    elif section == "about":
        st.markdown(f"""
<div class="lp-panel">
  <h3>ℹ️ About this System</h3>
  <p>
    The <b>AI Document Intelligence &amp; Artwork Navneet Processing System</b> automates the
    full prepress workflow for {DEV_COMPANY}. It ingests client PDFs and specifications,
    understands them with an LLM + RAG knowledge engine, and renders production-ready
    artwork with precise dielines, cutmarks and dimension annotations to ISO 12647.
  </p>
  <div class="lp-feature-row">
    <div class="lp-feature-ico">🧠</div>
    <div><div class="lp-feature-t">Neural Spec Parser</div>
    <div class="lp-feature-d">Hybrid NLP + LLM extracts every parameter from natural-language client briefs.</div></div>
  </div>
  <div class="lp-feature-row">
    <div class="lp-feature-ico">🎨</div>
    <div><div class="lp-feature-t">7-Layer CV Engine</div>
    <div class="lp-feature-d">Deterministic OpenCV geometry builds ARTWORK · BLEED · DIELINE · CUTMARKS · DIMENSIONS · MEASURE · TEXT.</div></div>
  </div>
  <div class="lp-feature-row">
    <div class="lp-feature-ico">📦</div>
    <div><div class="lp-feature-t">Multi-Format Export</div>
    <div class="lp-feature-d">Adobe Illustrator (.ai), Photoshop PSD, EPS and SVG — single file or full-book batch.</div></div>
  </div>
</div>""", unsafe_allow_html=True)

    elif section == "contact":
        st.markdown(f"""
<div class="lp-panel">
  <h3>✉️ Contact</h3>
  <p>
    Built and maintained by <b>{DEV_NAME}</b>, {DEV_TITLE} at {DEV_COMPANY}.
    Reach out for access, feature requests or production support.
  </p>
  <div class="lp-feature-row">
    <div class="lp-feature-ico">✉️</div>
    <div><div class="lp-feature-t">Gmail</div>
    <div class="lp-feature-d"><a href="mailto:{DEV_GMAIL}" style="color:#9fc4ff;">{DEV_GMAIL}</a></div></div>
  </div>
  <div class="lp-feature-row">
    <div class="lp-feature-ico">🔗</div>
    <div><div class="lp-feature-t">LinkedIn</div>
    <div class="lp-feature-d"><a href="{DEV_LINKEDIN}" target="_blank" style="color:#9fc4ff;">{DEV_LINKEDIN}</a></div></div>
  </div>
  <div class="lp-feature-row">
    <div class="lp-feature-ico">🏢</div>
    <div><div class="lp-feature-t">Organisation</div>
    <div class="lp-feature-d">{DEV_COMPANY} · {DEV_YEAR}</div></div>
  </div>
</div>""", unsafe_allow_html=True)

    elif section == "developer":
        st.markdown(f"""
<div class="lp-profile">
  <div class="lp-profile-avatar">SJ</div>
  <div class="lp-profile-name">{DEV_NAME}</div>
  <div class="lp-profile-role">{DEV_TITLE} · {DEV_COMPANY} · {DEV_YEAR}</div>
  <p class="lp-profile-bio">
    {DEV_NAME} is an {DEV_TITLE} at {DEV_COMPANY}, specialising in computer vision,
    LLM/RAG pipelines and production prepress automation. Designer and developer of the
    <b>AI Document Intelligence &amp; Artwork Navneet Processing System</b> — an end-to-end
    neural platform that turns client briefs and PDFs into ISO 12647-accurate, multi-format
    print-ready artwork.
  </p>
  <div class="lp-skill-row">
    <span class="lp-chip">Computer Vision</span>
    <span class="lp-chip">LLM / RAG</span>
    <span class="lp-chip">LangGraph</span>
    <span class="lp-chip">OpenCV</span>
    <span class="lp-chip">Python</span>
    <span class="lp-chip">Prepress AI</span>
    <span class="lp-chip">Streamlit</span>
  </div>
  <div class="lp-contact-row">
    <a class="lp-contact-btn" href="mailto:{DEV_GMAIL}">✉️ {DEV_GMAIL}</a>
    <a class="lp-contact-btn" href="{DEV_LINKEDIN}" target="_blank">in LinkedIn Profile</a>
  </div>
</div>""", unsafe_allow_html=True)

    st.markdown(f"""
<div class="lp-foot">
  © {DEV_YEAR} {DEV_COMPANY} · {APP_NAME} {APP_VERSION} ·
  Developed by {DEV_NAME}, {DEV_TITLE}
</div>""", unsafe_allow_html=True)


def _do_login(username: str, password: str):
    """Demo auth — accept any non-empty credentials, then open the main app."""
    if username.strip() and password.strip():
        st.session_state.authenticated = True
        st.session_state.user_name = username.strip()
        st.session_state.app_stage = "app"
        st.success("✓ Signed in — launching studio…")
        time.sleep(0.5)
        st.rerun()
    else:
        st.error("Please enter both a username and password.")


def render_login_page():
    inject_landing_css()

    # Back to landing
    top = st.columns([1.2, 6])
    with top[0]:
        if st.button("← Back", use_container_width=True, key="login_back"):
            st.session_state.app_stage = "landing"; st.rerun()

    _, mid, _ = st.columns([1, 1.25, 1])
    with mid:
        st.markdown(f"""
<div class="lp-login-head">
  <div class="lp-login-ico">🔐</div>
  <div class="lp-login-title">Welcome</div>
  <div class="lp-login-sub">Sign in or create an account for {APP_SHORT}</div>
</div>""", unsafe_allow_html=True)

        tab_in, tab_up = st.tabs(["🔓  Sign In", "✨  Create Account"])

        with tab_in:
            with st.form("login_form", clear_on_submit=False):
                username = st.text_input("Username / Email", placeholder="you@navneet.com")
                password = st.text_input("Password", type="password", placeholder="••••••••")
                st.checkbox("Remember me", value=True)
                submitted = st.form_submit_button("🔓  Sign In", type="primary",
                                                  use_container_width=True)
            if submitted:
                _do_login(username, password)
            st.markdown(
                '<div style="text-align:center;font-size:.72rem;color:#5a6899;margin-top:8px;">'
                'Demo access — any username &amp; password unlocks the studio.</div>',
                unsafe_allow_html=True,
            )

        with tab_up:
            with st.form("signup_form", clear_on_submit=False):
                full_name = st.text_input("Full Name", placeholder="Your name")
                new_email = st.text_input("Email", placeholder="you@navneet.com")
                new_pass  = st.text_input("Create Password", type="password", placeholder="••••••••")
                agree     = st.checkbox("I agree to the terms of use", value=True)
                created   = st.form_submit_button("✨  Create Account & Enter", type="primary",
                                                  use_container_width=True)
            if created:
                if not agree:
                    st.error("Please accept the terms to continue.")
                else:
                    _do_login(new_email or full_name, new_pass)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def render_app():
    inject_css()
    render_header()

    cfg = render_sidebar()
    sd: Dict[str, Any] = {}

    tabs = st.tabs([
        "📝 Job Setup",
        "🖨️ Process Artwork",
        "🔗 AI Pipeline",
        "🏗️ Architecture",
    ])

    with tabs[0]:
        sd = tab_job_setup(cfg)

    with tabs[1]:
        if sd:
            tab_process_artwork(sd, cfg)
        else:
            st.info("Complete Job Setup first.")

    with tabs[2]:
        if sd:
            tab_ai_pipeline(sd, cfg)
        else:
            st.info("Complete Job Setup first.")

    with tabs[3]:
        tab_architecture()

    # ── Dev footer ────────────────────────────────────────────────────────────
    st.markdown(
        f'<div style="text-align:center;margin-top:40px;padding:14px;'
        f'border-top:1px solid #1a2040;font-size:.62rem;color:#2d3a6a;">'
        f'{APP_NAME} {APP_VERSION} &nbsp;·&nbsp; '
        f'<span style="color:#22d3ee;">{DEV_NAME}</span> · {DEV_TITLE} · {DEV_COMPANY} · {DEV_YEAR} '
        f'&nbsp;·&nbsp; ISO 12647 Compliant Prepress Automation'
        f'</div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTER — landing → login → app
# ══════════════════════════════════════════════════════════════════════════════
def main():
    init_session()
    stage = st.session_state.get("app_stage", "landing")

    if stage == "app" and st.session_state.get("authenticated"):
        render_app()
    elif stage == "login":
        render_login_page()
    else:
        render_landing_page()


if __name__ == "__main__":
    main()
