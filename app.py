import os
import re
import sys
import json
import time
import subprocess
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from dotenv import load_dotenv
from openai import OpenAI
import streamlit as st

# ── Resolve Paths ────────────────────────────────────────────────────────────
_APP_DIR = Path(__file__).resolve().parent
_WORKSPACE = _APP_DIR.parent
load_dotenv(dotenv_path=str(_WORKSPACE / ".env"))

_DATA = _APP_DIR / "Data"
_SRC = _APP_DIR / "src"
_MODEL_PATH = _DATA / "results" / "tfidf_logreg_model.pkl"
_RESULTS_DIR = _DATA / "results"
_PROCESSED_DIR = _DATA / "processed"

# Ensure dirs exist
os.makedirs(str(_RESULTS_DIR), exist_ok=True)

# ── Streamlit Page Configuration ──────────────────────────────────────────────
st.set_page_config(
    page_title="UrbanLLM Dispatcher",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS for Rich Glassmorphic Dark UI ──────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    /* Fonts & Typography */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-title {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        font-size: 3rem;
        background: linear-gradient(135deg, #60A5FA, #A78BFA, #F472B6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 5px;
        text-align: left;
    }
    
    .subtitle {
        color: #9CA3AF;
        font-size: 1.1rem;
        margin-bottom: 30px;
    }

    /* Glassmorphic Container Cards */
    .glass-card {
        background: rgba(30, 41, 59, 0.45);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease, border-color 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.4);
        border-color: rgba(255, 255, 255, 0.15);
    }
    
    .glass-card-header {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 600;
        font-size: 1.25rem;
        color: #F3F4F6;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Color Badges */
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .badge-critical {
        background-color: rgba(239, 68, 68, 0.15);
        color: #F87171;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    
    .badge-high {
        background-color: rgba(249, 115, 22, 0.15);
        color: #FB923C;
        border: 1px solid rgba(249, 115, 22, 0.3);
    }
    
    .badge-medium {
        background-color: rgba(234, 179, 8, 0.15);
        color: #FACC15;
        border: 1px solid rgba(234, 179, 8, 0.3);
    }
    
    .badge-low {
        background-color: rgba(34, 197, 94, 0.15);
        color: #4ADE80;
        border: 1px solid rgba(34, 197, 94, 0.3);
    }
    
    .badge-dept {
        background-color: rgba(147, 51, 234, 0.15);
        color: #C084FC;
        border: 1px solid rgba(147, 51, 234, 0.3);
    }

    /* Key Value Grid */
    .kv-row {
        display: flex;
        justify-content: space-between;
        padding: 8px 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    .kv-row:last-child {
        border-bottom: none;
    }
    
    .kv-key {
        color: #9CA3AF;
        font-weight: 500;
    }
    
    .kv-val {
        color: #F3F4F6;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar Configurations ────────────────────────────────────────────────────
st.sidebar.markdown("<h2 style='background: linear-gradient(135deg, #60A5FA, #A78BFA); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>🔧 System Settings</h2>", unsafe_allow_html=True)

# API Configurations
st.sidebar.subheader("LLM Configuration")
api_key = st.sidebar.text_input(
    "API Key",
    value=os.getenv("LLM_API_KEY", ""),
    type="password",
    help="LLM API Key from your .env or provider"
)
base_url = st.sidebar.text_input(
    "Base URL",
    value=os.getenv("LLM_BASE_URL", "https://api.morphllm.com/v1"),
    help="LLM Base Endpoint"
)
model_name = st.sidebar.text_input(
    "Model Name",
    value=os.getenv("LLM_MODEL", os.getenv("GLM_MODEL", "morph-glm52-744b")),
    help="Model identifier"
)

# Pipeline Configs
st.sidebar.markdown("---")
st.sidebar.subheader("Confidence Threshold")
confidence_threshold = st.sidebar.slider(
    "LLM Minimum Confidence",
    min_value=0.0,
    max_value=1.0,
    value=0.7,
    step=0.05,
    help="LLM predictions below this threshold will default to 'other' or trigger warnings."
)

# ── Setup LLM Client ─────────────────────────────────────────────────────────
@st.cache_resource
def get_openai_client(key, url):
    if not key:
        return None
    return OpenAI(api_key=key, base_url=url)

client = get_openai_client(api_key, base_url)

# ── Baseline ML Loader ───────────────────────────────────────────────────────
@st.cache_resource
def load_baseline_model():
    if _MODEL_PATH.exists():
        try:
            return joblib.load(str(_MODEL_PATH))
        except Exception as e:
            st.error(f"Error loading Baseline ML model: {e}")
            return None
    return None

baseline_model = load_baseline_model()

# ── Static Mappings ──────────────────────────────────────────────────────────
LABELS = [
    "road_infrastructure", "waste_management", "water_utilities",
    "traffic_management", "street_lighting", "noise",
    "public_safety", "environment", "other"
]

priority_rules = {
    "public_safety":      90,
    "traffic_management": 80,
    "water_utilities":    75,
    "road_infrastructure":65,
    "street_lighting":    55,
    "waste_management":   50,
    "environment":        45,
    "noise":              30,
    "other":              20,
}

critical_keywords = [
    "danger", "injury", "accident", "fire", "collapse", "flood",
    "blocked road", "traffic signal out", "water main break",
    "emergency", "hazard", "unsafe", "explosion", "gas leak",
]

sensitive_locations = ["school", "hospital", "playground", "daycare", "clinic"]

routing_table = {
    "road_infrastructure": "Department of Roads and Public Works",
    "waste_management":    "Sanitation Department",
    "water_utilities":     "Water Utility Department",
    "traffic_management":  "Traffic Operations Department",
    "street_lighting":     "Street Lighting Maintenance Division",
    "noise":               "Environmental Control / Noise Regulation Unit",
    "public_safety":       "Public Safety and Emergency Response Department",
    "environment":         "Environmental Protection Department",
    "other":               "General Municipal Services",
}

SYSTEM_PROMPT = """
You are an urban complaint classification assistant for a smart city platform.
Classify each citizen complaint into exactly one category.

Allowed categories:
road_infrastructure, waste_management, water_utilities, traffic_management,
street_lighting, noise, public_safety, environment, other.

Return only valid JSON:
{
  "category": "...",
  "confidence": 0.0,
  "reason": "short reason"
}
"""

# ── Logic Functions ──────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s.,!?-]", "", text)
    return text.strip()

def predict_baseline(text: str) -> str:
    if baseline_model is None:
        return "Model not loaded"
    cleaned = clean_text(text)
    try:
        pred = baseline_model.predict([cleaned])[0]
        return pred
    except Exception as e:
        return f"Error: {e}"

def predict_llm(text: str) -> dict:
    if not client:
        return {
            "category": "other",
            "confidence": 0.0,
            "reason": "API Key is not configured in settings."
        }
    
    user_prompt = f"Citizen complaint:\n{text}\n\nClassify the complaint into one allowed category."
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        parsed = json.loads(content)
        
        # Validation
        if not isinstance(parsed, dict):
            parsed = {}
        if "category" not in parsed:
            parsed["category"] = "other"
        if "confidence" not in parsed:
            parsed["confidence"] = 0.0
        if "reason" not in parsed:
            parsed["reason"] = "No reason provided by LLM."
            
        if parsed.get("category") not in LABELS:
            parsed["category"] = "other"
            
        return parsed
    except Exception as e:
        return {
            "category": "other",
            "confidence": 0.0,
            "reason": f"API Error: {str(e)}"
        }

def compute_priority(category: str, text: str) -> dict:
    text_lower = text.lower()
    base_score = priority_rules.get(category, 20)
    score = base_score
    escalations = []
    
    # Escalate for critical keywords
    found_keywords = [k for k in critical_keywords if k in text_lower]
    if found_keywords:
        score += 15
        escalations.append(f"Critical keywords (+15): {', '.join(found_keywords)}")
        
    # Escalate for sensitive locations
    found_locations = [loc for loc in sensitive_locations if loc in text_lower]
    if found_locations:
        score += 10
        escalations.append(f"Sensitive location proximity (+10): {', '.join(found_locations)}")
        
    score = min(score, 100)
    
    if score >= 85:
        level = "critical"
        badge_class = "badge-critical"
    elif score >= 65:
        level = "high"
        badge_class = "badge-high"
    elif score >= 40:
        level = "medium"
        badge_class = "badge-medium"
    else:
        level = "low"
        badge_class = "badge-low"
        
    return {
        "score": score,
        "level": level,
        "badge_class": badge_class,
        "base_score": base_score,
        "escalations": escalations
    }

def route_department(category: str) -> str:
    return routing_table.get(category, "General Municipal Services")

# ── Main Layout ──────────────────────────────────────────────────────────────
st.markdown("<h1 class='main-title'>🏙️ UrbanLLM Dispatcher</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Smart Citizen Complaint Classification, Priority Scoring, and Department Dispatch System</p>", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3 = st.tabs(["⚡ Single Analysis", "📂 Batch Processing", "📊 Analytics & Pipeline Control"])

# ── Tab 1: Single Analysis ───────────────────────────────────────────────────
with tab1:
    st.markdown("### Interactive Single Complaint Analysis")
    
    col_input, col_preset = st.columns([2, 1])
    
    with col_preset:
        st.markdown("**Sample Complaints (Click to copy/use)**")
        presets = [
            "Heavy water logging and flooding on 5th main road, cars are getting stuck near the hospital.",
            "Street lights are completely out on Nehru street, making it very unsafe for children returning from playground.",
            "There's a massive pothole in the middle of the road causing a major traffic jam and danger of accidents.",
            "Sanitation trucks did not pick up garbage here for 4 days. The pile is dirty and stinking near the daycare.",
            "Excessive noise from the construction site at midnight, people cannot sleep."
        ]
        selected_preset = st.selectbox("Select a sample preset:", [""] + presets)
        
    with col_input:
        user_input = st.text_area(
            "Enter citizen complaint text here:",
            value=selected_preset if selected_preset else "Write your complaint here...",
            height=120
        )
        
        analyze_btn = st.button("🚀 Analyze & Dispatch", type="primary", use_container_width=True)

    if analyze_btn and user_input and user_input != "Write your complaint here...":
        with st.spinner("Processing complaint..."):
            cleaned = clean_text(user_input)
            
            # Predict
            ml_pred = predict_baseline(cleaned)
            llm_result = predict_llm(cleaned)
            
            # Use LLM prediction if confidence is high, else fallback or use LLM directly
            final_category = llm_result["category"]
            confidence = llm_result["confidence"]
            
            # Compute Priority and Routing
            priority = compute_priority(final_category, user_input)
            department = route_department(final_category)
            
            # Outputs Layout
            st.markdown("### 📊 Classification & Dispatch Results")
            
            out_col1, out_col2, out_col3 = st.columns(3)
            
            # Card 1: Classifier Agreement
            with out_col1:
                st.markdown(f"""
                <div class="glass-card">
                    <div class="glass-card-header">🤖 Classification Models</div>
                    <div class="kv-row">
                        <span class="kv-key">Baseline ML Model:</span>
                        <span class="kv-val">{ml_pred.replace('_', ' ').title()}</span>
                    </div>
                    <div class="kv-row">
                        <span class="kv-key">LLM Classifier:</span>
                        <span class="kv-val">{llm_result['category'].replace('_', ' ').title()}</span>
                    </div>
                    <div class="kv-row">
                        <span class="kv-key">Confidence:</span>
                        <span class="kv-val">{confidence:.2f}</span>
                    </div>
                    <div class="kv-row">
                        <span class="kv-key">Decision Model:</span>
                        <span class="kv-val">Generative LLM</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
            # Card 2: Priority Assessment
            with out_col2:
                esc_html = "<br>".join([f"• {e}" for e in priority['escalations']]) if priority['escalations'] else "None"
                st.markdown(f"""
                <div class="glass-card">
                    <div class="glass-card-header">🚨 Priority Scoring</div>
                    <div class="kv-row">
                        <span class="kv-key">Priority Level:</span>
                        <span><span class="badge {priority['badge_class']}">{priority['level']}</span></span>
                    </div>
                    <div class="kv-row">
                        <span class="kv-key">Score (0-100):</span>
                        <span class="kv-val">{priority['score']} / 100</span>
                    </div>
                    <div class="kv-row">
                        <span class="kv-key">Base Category Score:</span>
                        <span class="kv-val">{priority['base_score']}</span>
                    </div>
                    <div class="kv-row" style="flex-direction: column; align-items: flex-start; padding-top: 10px;">
                        <span class="kv-key" style="margin-bottom: 4px;">Escalations applied:</span>
                        <span class="kv-val" style="font-size: 0.85rem; font-weight: normal; color: #D1D5DB;">{esc_html}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
            # Card 3: Department Dispatch
            with out_col3:
                st.markdown(f"""
                <div class="glass-card">
                    <div class="glass-card-header">📤 Department Dispatch</div>
                    <div class="kv-row">
                        <span class="kv-key">Category:</span>
                        <span class="kv-val">{final_category.replace('_', ' ').title()}</span>
                    </div>
                    <div class="kv-row" style="flex-direction: column; align-items: flex-start; padding-top: 10px;">
                        <span class="kv-key" style="margin-bottom: 4px;">Assigned Department:</span>
                        <span><span class="badge badge-dept" style="font-size: 0.9rem; padding: 6px 12px; white-space: normal; display: inline-block; word-break: break-word;">{department}</span></span>
                    </div>
                    <div class="kv-row">
                        <span class="kv-key">Status:</span>
                        <span class="kv-val" style="color: #60A5FA;">DISPATCHED / PENDING</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
            # LLM Reason Explanation
            st.markdown(f"""
            <div class="glass-card" style="margin-top: 10px;">
                <div class="glass-card-header">💡 LLM Decision Rationale</div>
                <p style="color: #E5E7EB; line-height: 1.5; margin: 0;">{llm_result['reason']}</p>
            </div>
            """, unsafe_allow_html=True)

# ── Tab 2: Batch Processing ──────────────────────────────────────────────────
with tab2:
    st.markdown("### Batch CSV Complaint Processing")
    st.markdown("Upload a CSV file containing complaints to batch process them through the pipeline.")
    
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df_batch = pd.read_csv(uploaded_file)
            st.write(f"📂 Loaded CSV with **{len(df_batch)}** rows.")
            
            # Column selector
            text_col = st.selectbox(
                "Select the column containing the complaint text:",
                options=df_batch.columns,
                index=0
            )
            
            model_choice = st.radio(
                "Select Classification Model:",
                options=["Baseline ML (Fast, Offline)", "Generative LLM (Detailed, API)"],
                help="Generative LLM takes longer due to API rate limits (approx 4 seconds per row)."
            )
            
            if st.button("⚙️ Process Batch Data", type="primary"):
                if text_col not in df_batch.columns:
                    st.error("Text column not found.")
                else:
                    progress_text = "Processing complaints. Please wait..."
                    progress_bar = st.progress(0.0, text=progress_text)
                    
                    results = []
                    start_time = time.time()
                    
                    total_rows = len(df_batch)
                    
                    for idx, row in df_batch.iterrows():
                        text = str(row[text_col])
                        cleaned = clean_text(text)
                        
                        # Predict
                        if "Baseline ML" in model_choice:
                            pred_cat = predict_baseline(cleaned)
                            confidence = 1.0
                            reason = "Predicted by TF-IDF + Logistic Regression Baseline"
                        else:
                            llm_res = predict_llm(cleaned)
                            pred_cat = llm_res["category"]
                            confidence = llm_res["confidence"]
                            reason = llm_res["reason"]
                            time.sleep(4.0)  # Free Tier API rate limiting (stay under 15 RPM)
                        
                        # Priority & Routing
                        p_res = compute_priority(pred_cat, text)
                        dept = route_department(pred_cat)
                        
                        results.append({
                            "cleaned_text": cleaned,
                            "predicted_category": pred_cat,
                            "confidence": confidence,
                            "priority_score": p_res["score"],
                            "priority_level": p_res["level"],
                            "assigned_department": dept,
                            "rationale": reason
                        })
                        
                        # Update progress
                        progress_pct = float(idx + 1) / total_rows
                        progress_bar.progress(progress_pct, text=f"Processed {idx+1}/{total_rows} complaints...")
                    
                    df_results = pd.concat([df_batch, pd.DataFrame(results)], axis=1)
                    st.success(f"Processed {total_rows} complaints in {time.time() - start_time:.1f}s!")
                    
                    # Preview results
                    st.markdown("### Processed Results Preview")
                    st.dataframe(df_results.head(100), use_container_width=True)
                    
                    # Download CSV
                    csv_data = df_results.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Full Processed CSV",
                        data=csv_data,
                        file_name="batch_processed_complaints.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
        except Exception as e:
            st.error(f"Error reading file: {e}")

# ── Tab 3: Analytics & Pipeline Control ──────────────────────────────────────
with tab3:
    st.markdown("### Analytics Dashboard & Pipeline Execution")
    
    # Check if we have routed complaints results
    routed_csv_path = _RESULTS_DIR / "routed_complaints.csv"
    
    if routed_csv_path.exists():
        df_analysis = pd.read_csv(routed_csv_path)
        
        # Summary Metrics
        st.markdown("#### Overall System Metrics")
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        
        total_complaints = len(df_analysis)
        critical_count = len(df_analysis[df_analysis["priority_level"] == "critical"])
        critical_pct = (critical_count / total_complaints * 100) if total_complaints > 0 else 0
        avg_priority_score = df_analysis["priority_score"].mean() if "priority_score" in df_analysis.columns else 0
        
        # Display nicely
        with m_col1:
            st.metric("Total Complaints Processed", f"{total_complaints}")
        with m_col2:
            st.metric("Average Priority Score", f"{avg_priority_score:.1f} / 100")
        with m_col3:
            st.metric("Critical Priority Complaints", f"{critical_count} ({critical_pct:.1f}%)")
        with m_col4:
            top_dept = df_analysis["assigned_department"].value_counts().idxmax() if "assigned_department" in df_analysis.columns else "N/A"
            st.metric("Highest Loaded Department", f"{top_dept[:25]}...")
            
        st.markdown("---")
        st.markdown("#### Visual Distribution Charts")
        
        fig_col1, fig_col2 = st.columns(2)
        
        with fig_col1:
            if "priority_level" in df_analysis.columns:
                fig, ax = plt.subplots(figsize=(6, 4))
                colors = {"low": "#4ADE80", "medium": "#FACC15", "high": "#FB923C", "critical": "#F87171"}
                counts = df_analysis["priority_level"].value_counts().reindex(["low", "medium", "high", "critical"], fill_value=0)
                
                sns.barplot(x=counts.index, y=counts.values, palette=[colors[k] for k in counts.index], ax=ax)
                ax.set_title("Complaints by Priority Level")
                ax.set_ylabel("Count")
                ax.set_xlabel("Priority Level")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
                
        with fig_col2:
            if "glm52_pred" in df_analysis.columns:
                fig, ax = plt.subplots(figsize=(6, 4))
                counts = df_analysis["glm52_pred"].value_counts()
                sns.barplot(y=counts.index, x=counts.values, palette="Blues_r", ax=ax)
                ax.set_title("Complaints by Category Classification")
                ax.set_xlabel("Count")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
                
        # Confusion Matrix Image
        cm_image_path = _RESULTS_DIR / "glm52_confusion_matrix.png"
        if cm_image_path.exists():
            st.markdown("#### LLM Classifier Evaluation - Confusion Matrix")
            st.image(str(cm_image_path), caption="GLM-5.2 Model Prediction vs Ground-Truth Category Matrix", use_container_width=True)
            
    else:
        st.info("No system results found. Run the pipeline below to process the default datasets and generate analytics.")

    # ── Pipeline Control ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Run Pipeline Scripts")
    st.markdown("Click the button below to execute the background Python processing scripts (`01_load_data.py` through `08_generate_tables.py`) in sequence. This will load the raw CSV data, preprocess it, classify a sample using the LLM, compute priority scores, route departments, and save the reports.")
    
    col_run, col_stop = st.columns([1, 4])
    
    # Store process run state in Session State
    if "pipeline_running" not in st.session_state:
        st.session_state["pipeline_running"] = False
        st.session_state["pipeline_output"] = ""
        
    run_btn = col_run.button("⚙️ Execute Pipeline", type="primary", disabled=st.session_state["pipeline_running"])
    
    if run_btn:
        st.session_state["pipeline_running"] = True
        st.session_state["pipeline_output"] = "Starting pipeline run...\n"
        
        # We will run the classification script which automatically runs downstream scripts
        classification_script = _SRC / "04_llm_classification.py"
        
        # We can run it in a subprocess and capture stdout/stderr
        st.info("Pipeline execution is running in the background. Check logs below.")
        
        log_area = st.empty()
        
        try:
            # We first run 01_load_data and 02_preprocess to make sure data is clean
            # (since 04_llm_classification starts by loading complaints_clean.csv)
            for script_name in ["01_load_data.py", "02_preprocess.py", "03_baseline_ml.py", "04_llm_classification.py"]:
                st.session_state["pipeline_output"] += f"\n>>> Running {script_name}...\n"
                log_area.code(st.session_state["pipeline_output"])
                
                # Executing script using the current virtual env python
                venv_python = str(_WORKSPACE / ".venv" / "bin" / "python")
                if not Path(venv_python).exists():
                     venv_python = sys.executable
                     
                script_path = _SRC / script_name
                
                # Set classification sample size to 15 in environment to keep it short
                env = os.environ.copy()
                env["CLASSIFICATION_SAMPLE_SIZE"] = "15"
                
                proc = subprocess.Popen(
                    [venv_python, str(script_path)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    cwd=str(_APP_DIR),
                    env=env
                )
                
                # Read stdout line by line
                while True:
                    line = proc.stdout.readline()
                    if not line:
                        break
                    st.session_state["pipeline_output"] += line
                    log_area.code(st.session_state["pipeline_output"])
                    
                proc.wait()
                if proc.returncode != 0:
                    st.session_state["pipeline_output"] += f"\n[ERROR] {script_name} failed with return code {proc.returncode}\n"
                    log_area.code(st.session_state["pipeline_output"])
                    break
            
            st.success("Pipeline executed successfully!")
            st.session_state["pipeline_running"] = False
            # Clear cache to reload analytics
            st.cache_resource.clear()
            st.rerun()
            
        except Exception as e:
            st.error(f"Error running pipeline: {e}")
            st.session_state["pipeline_running"] = False
