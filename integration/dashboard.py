"""
Streamlit Dashboard — Multimodal Crime / Incident Report Analyzer

Interactive dashboard to visualize and query the final integrated dataset.
Run with: streamlit run dashboard.py
"""

import os
import sys
import pandas as pd
import streamlit as st

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Crime Incident Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# LOAD DATA
# ============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FINAL_CSV = os.path.join(SCRIPT_DIR, "final_incidents.csv")


@st.cache_data(ttl=10)
def load_data():
    """Load the integrated dataset. TTL=10s enables real-time refresh."""
    if os.path.exists(FINAL_CSV):
        return pd.read_csv(FINAL_CSV)

    st.error("No integrated dataset found. Run `python integrate.py` first.")
    st.stop()


df = load_data()

# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e3a5f;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1rem;
        color: #6b7280;
        margin-top: 0;
    }
    .severity-high { color: #ef4444; font-weight: bold; }
    .severity-medium { color: #f59e0b; font-weight: bold; }
    .severity-low { color: #10b981; font-weight: bold; }
    .stDataFrame { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# HEADER
# ============================================================================

st.markdown('<p class="main-header">🔍 Multimodal Crime / Incident Report Analyzer</p>',
            unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-powered unified incident analysis from Audio, PDF, Image, Video, and Text sources</p>',
            unsafe_allow_html=True)

st.divider()

# ============================================================================
# SIDEBAR FILTERS
# ============================================================================

# Real-time refresh controls
st.sidebar.header("⚡ Real-Time")

# Manual refresh button
if st.sidebar.button("🔄 Refresh Now", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

# Show real-time monitor status
monitor_lock = os.path.join(SCRIPT_DIR, ".processing.lock")
new_data_dir = os.path.join(os.path.dirname(SCRIPT_DIR), "new_data")
if os.path.exists(new_data_dir):
    pending = len([f for f in os.listdir(new_data_dir)
                   if not f.startswith('.') and f != '_processed'
                   and os.path.isfile(os.path.join(new_data_dir, f))])
    if os.path.exists(monitor_lock):
        st.sidebar.warning("⏳ Processing file...")
    elif pending > 0:
        st.sidebar.info(f"📂 {pending} file(s) pending")
    else:
        st.sidebar.success("✅ Monitor ready")
else:
    st.sidebar.caption("📁 new_data/ folder not created yet")

st.sidebar.markdown("---")

st.sidebar.header("🎛️ Filters")

# Severity filter
severity_options = sorted(df["Severity"].unique().tolist())
selected_severity = st.sidebar.multiselect(
    "Severity Level",
    severity_options,
    default=severity_options,
)

# Modality Count filter
modality_options = sorted(df["Modality_Count"].unique().tolist(), reverse=True)
selected_modality = st.sidebar.multiselect(
    "Modality Count",
    modality_options,
    default=modality_options,
)

# Incident ID range
inc_min = 1
inc_max = len(df)
selected_range = st.sidebar.slider(
    "Incident ID Range",
    min_value=inc_min,
    max_value=inc_max,
    value=(inc_min, inc_max),
)

# Apply filters
filtered = df[df["Severity"].isin(selected_severity)]
filtered = filtered[filtered["Modality_Count"].isin(selected_modality)]

# Apply ID range filter
filtered = filtered.iloc[selected_range[0] - 1:selected_range[1]]

# ============================================================================
# KEY METRICS
# ============================================================================

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Total Incidents", len(filtered))
with col2:
    high = len(filtered[filtered["Severity"] == "High"])
    st.metric("🔴 High Severity", high)
with col3:
    med = len(filtered[filtered["Severity"] == "Medium"])
    st.metric("🟡 Medium Severity", med)
with col4:
    low = len(filtered[filtered["Severity"] == "Low"])
    st.metric("🟢 Low Severity", low)
with col5:
    avg_modality = filtered["Modality_Count"].mean()
    st.metric("Avg Modalities", f"{avg_modality:.1f} / 5")

st.divider()

# ============================================================================
# VISUALIZATIONS
# ============================================================================

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.subheader("📊 Severity Distribution")
    severity_counts = filtered["Severity"].value_counts()
    st.bar_chart(severity_counts, color="#667eea")

with chart_col2:
    st.subheader("📈 Modality Coverage")
    modality_counts = filtered["Modality_Count"].value_counts().sort_index()
    st.bar_chart(modality_counts, color="#764ba2")

st.divider()

chart_col3, chart_col4 = st.columns(2)

with chart_col3:
    st.subheader("🎙️ Audio Events (Top 10)")
    if "Audio_Event" in filtered.columns:
        # Filter out missing value placeholders
        audio_real = filtered[~filtered["Audio_Event"].str.contains("No 911", na=False)]
        if not audio_real.empty:
            audio_counts = audio_real["Audio_Event"].value_counts().head(10)
            st.bar_chart(audio_counts, color="#10b981")
        else:
            st.info("No audio data in current filter")

with chart_col4:
    st.subheader("📝 Text Crime Types (Top 10)")
    if "Text_Crime_Type" in filtered.columns:
        text_real = filtered[~filtered["Text_Crime_Type"].str.contains("No media", na=False)]
        if not text_real.empty:
            text_counts = text_real["Text_Crime_Type"].value_counts().head(10)
            st.bar_chart(text_counts, color="#f59e0b")
        else:
            st.info("No text data in current filter")

st.divider()

# ============================================================================
# SUMMARY TABLE
# ============================================================================

st.subheader("📋 Incident Records")

# Highlight severity with colors
def color_severity(val):
    if val == "High":
        return "background-color: #fee2e2; color: #dc2626"
    elif val == "Medium":
        return "background-color: #fef3c7; color: #d97706"
    elif val == "Low":
        return "background-color: #d1fae5; color: #059669"
    return ""

styled = filtered.style.applymap(color_severity, subset=["Severity"])
st.dataframe(styled, use_container_width=True, height=400)

# ============================================================================
# PER-INCIDENT DETAIL VIEW
# ============================================================================

st.divider()
st.subheader("🔬 Incident Detail View")

incident_ids = filtered["Incident_ID"].tolist()
if incident_ids:
    selected_incident = st.selectbox("Select Incident", incident_ids)
    row = filtered[filtered["Incident_ID"] == selected_incident].iloc[0]

    detail_col1, detail_col2, detail_col3 = st.columns(3)

    with detail_col1:
        st.markdown(f"**🆔 {row['Incident_ID']}**")
        st.markdown(f"**Severity:** {row['Severity']}")
        st.markdown(f"**Sources:** {row['Sources_Available']}")
        st.markdown(f"**Modality Count:** {row['Modality_Count']} / 5")

    with detail_col2:
        st.markdown("**🎙️ Audio Event:**")
        st.info(row["Audio_Event"])
        st.markdown("**📄 PDF Doc Type:**")
        st.info(row["PDF_Doc_Type"])

    with detail_col3:
        st.markdown("**🖼️ Image Objects:**")
        st.info(row["Image_Objects"])
        st.markdown("**🎬 Video Event:**")
        st.info(row["Video_Event"])

    st.markdown("**📝 Text Crime Type:**")
    st.info(row["Text_Crime_Type"])

    # Show AI Summary if available
    if "AI_Summary" in row.index and pd.notna(row.get("AI_Summary")) and str(row.get("AI_Summary", "")).strip():
        st.markdown("**🤖 AI-Generated Summary** *(Flan-T5)*")
        st.success(row["AI_Summary"])

# ============================================================================
# PER-MODALITY BREAKDOWN
# ============================================================================

st.divider()
st.subheader("📂 Per-Modality Data Sources")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎙️ Audio", "📄 PDF", "🖼️ Image", "🎬 Video", "📝 Text"
])

BASE_DIR = os.path.dirname(SCRIPT_DIR)

with tab1:
    audio_path = os.path.join(BASE_DIR, "audio", "audio_output.csv")
    if os.path.exists(audio_path):
        audio_df = pd.read_csv(audio_path)
        st.write(f"**{len(audio_df)} records** from 911 emergency calls")
        st.dataframe(audio_df, use_container_width=True)
    else:
        st.info("Audio output not found")

with tab2:
    pdf_path = os.path.join(BASE_DIR, "pdf", "pdf_output.csv")
    if os.path.exists(pdf_path):
        pdf_df = pd.read_csv(pdf_path)
        st.write(f"**{len(pdf_df)} records** from police report PDFs")
        st.dataframe(pdf_df, use_container_width=True)
    else:
        st.info("PDF output not found")

with tab3:
    image_path = os.path.join(BASE_DIR, "images", "image_output.csv")
    if os.path.exists(image_path):
        image_df = pd.read_csv(image_path)
        st.write(f"**{len(image_df)} records** from scene photographs")
        st.dataframe(image_df, use_container_width=True)
    else:
        st.info("Image output not found")

with tab4:
    video_path = os.path.join(BASE_DIR, "video", "video_output.csv")
    if os.path.exists(video_path):
        video_df = pd.read_csv(video_path)
        st.write(f"**{len(video_df)} records** from CCTV surveillance footage")
        st.dataframe(video_df, use_container_width=True)
    else:
        st.info("Video output not found")

with tab5:
    text_path = os.path.join(BASE_DIR, "text", "text_output.csv")
    if os.path.exists(text_path):
        text_df = pd.read_csv(text_path)
        st.write(f"**{len(text_df)} records** from crime reports/news")
        st.dataframe(text_df, use_container_width=True)
    else:
        st.info("Text output not found")

# ============================================================================
# FOOTER
# ============================================================================

st.divider()

# Show last update time
if os.path.exists(FINAL_CSV):
    mod_time = os.path.getmtime(FINAL_CSV)
    from datetime import datetime
    last_updated = datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')
    st.caption(f"📅 Dataset last updated: {last_updated} | Auto-refresh every 10s")

st.markdown("""
<div style="text-align: center; color: #9ca3af; font-size: 0.85rem;">
    <p>Multimodal Crime / Incident Report Analyzer | AI for Engineers - Spring 2026</p>
    <p>Team: Lokeshwar | Neha | Nagraj | Rahul | Swet</p>
</div>
""", unsafe_allow_html=True)
