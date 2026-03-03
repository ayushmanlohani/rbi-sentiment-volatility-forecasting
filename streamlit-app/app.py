import streamlit as st
import pandas as pd
import numpy as np
import joblib
import yfinance as yf
import plotly.graph_objects as go
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import base64
import os

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="RBI Sentinel",
    page_icon="https://upload.wikimedia.org/wikipedia/en/thumb/0/09/Reserve_Bank_of_India_logo.svg/1200px-Reserve_Bank_of_India_logo.svg.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- LOAD CUSTOM CSS ---


def local_css(file_name):
    base_path = os.path.dirname(__file__)
    css_path = os.path.join(base_path, file_name)
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


local_css("style.css")

# --- MODEL LOADING (CACHED) ---


@st.cache_resource
def load_artifacts():
    # Paths adjusted for streamlit-app/ folder structure
    model = joblib.load('../models/rbi_vix_rf_model.pkl')
    scaler = joblib.load('../models/scaler.pkl')
    feature_names = joblib.load('../models/feature_names.pkl')
    shock_threshold = joblib.load('../models/shock_threshold.pkl')
    # Load FinBERT model + tokenizer (matching training pipeline)
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    finbert = AutoModelForSequenceClassification.from_pretrained(
        "ProsusAI/finbert")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    finbert.to(device)
    finbert.eval()
    return model, scaler, feature_names, shock_threshold, tokenizer, finbert, device


model, scaler, feature_names, shock_threshold, tokenizer, finbert, device = load_artifacts()


def get_sentiment_score(text):
    """
    Matches the training pipeline: sliding-window chunking over 510-token
    blocks, averaged softmax probabilities, score = prob(pos) - prob(neg).
    """
    if not isinstance(text, str) or len(text.strip()) == 0:
        return 0.0

    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    input_ids = inputs["input_ids"][0]

    chunk_size = 510
    chunks = input_ids.split(chunk_size)

    chunk_probs = []
    for chunk in chunks:
        chunk_with_special = torch.cat([
            torch.tensor([tokenizer.cls_token_id]),
            chunk,
            torch.tensor([tokenizer.sep_token_id])
        ]).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = finbert(chunk_with_special)
            probs = F.softmax(outputs.logits, dim=-1)
            chunk_probs.append(probs.cpu())

    # FinBERT labels: [0: positive, 1: negative, 2: neutral]
    avg_probs = torch.mean(torch.stack(chunk_probs), dim=0).squeeze()
    pos_score = avg_probs[0].item()
    neg_score = avg_probs[1].item()
    return pos_score - neg_score

# --- MARKET DATA FETCHING ---


@st.cache_data(ttl=3600)
def get_market_context():
    vix = yf.Ticker("^INDIAVIX").history(period="5d")
    nifty = yf.Ticker("^NSEI").history(period="10d")
    us_vix = yf.Ticker("^VIX").history(period="1d")

    curr_vix = vix['Close'].iloc[-1]
    prev_vix = vix['Close'].iloc[-2]

    # Log return to match training pipeline (np.log(P_t / P_{t-5}))
    nifty_5d_ret = np.log(nifty['Close'].iloc[-1] / nifty['Close'].iloc[-6])
    # Rolling std of log daily returns to match training pipeline
    nifty_log_daily_ret = np.log(nifty['Close'] / nifty['Close'].shift(1))
    nifty_std = nifty_log_daily_ret.tail(5).std()

    return curr_vix, prev_vix, nifty_5d_ret, nifty_std, us_vix['Close'].iloc[-1]


# --- SIDEBAR: LIVE MARKET CONTEXT ---
with st.sidebar:
    with open("logo.png", "rb") as img_file:
        logo_b64 = base64.b64encode(img_file.read()).decode()
    st.markdown(
        f'<div style="text-align:center; padding:12px 0 20px;">'
        f'<img src="data:image/png;base64,{logo_b64}" width="600" '
        f'style="image-rendering:-webkit-optimize-contrast; image-rendering:crisp-edges;" />'
        f'</div>',
        unsafe_allow_html=True
    )
    st.header("Live Market")

    curr_vix, prev_vix, nifty_ret, nifty_std, us_vix_val = get_market_context()

    st.metric("India VIX", f"{curr_vix:.2f}",
              f"{((curr_vix/prev_vix)-1)*100:.1f}%", delta_color="inverse")
    st.metric("Nifty 50 (5D Return)", f"{nifty_ret*100:.2f}%")
    st.metric("CBOE VIX", f"{us_vix_val:.2f}")

    st.divider()
    st.caption(f"THRESHOLD  ·  VIX RET > {shock_threshold:.4f}")
    st.caption("MODEL  ·  RANDOM FOREST v1.2")

# --- MAIN UI ---
st.title("RBI Sentinel")
st.subheader("Monetary policy sentiment → volatility shock prediction")

st.write("")

col1, col2 = st.columns([3, 1], gap="large")

with col1:
    user_text = st.text_area(
        "Paste RBI communication",
        placeholder="Paste the full text of an RBI Governor speech or MPC Minutes here...",
        height=320,
        label_visibility="collapsed")
    doc_type = st.selectbox("Document type", ["Speech", "MPC_Minutes"])

with col2:
    st.markdown("""
    **How it works**
    
    1. Paste a full RBI speech or MPC minutes.
    2. Choose the document type.
    3. The system runs **FinBERT** sentiment analysis and combines it with live market data to predict VIX shock probability.
    """)
    st.write("")
    predict_btn = st.button("Run Analysis", use_container_width=True)

if predict_btn and user_text:
    with st.spinner("Analyzing Sentiment & Market Regime..."):
        # 1. NLP Inference — matches training pipeline (sliding-window chunking)
        raw_score = get_sentiment_score(user_text)

        # 2. Feature Prep — matches the 5 features from training
        features = {
            'Sentiment_Intensity': abs(raw_score),
            'VIX_Lag1': curr_vix,
            'Nifty_5d_Ret': nifty_ret,
            'Nifty_5d_Std': nifty_std,
            'Is_MPC': 1 if doc_type == "MPC_Minutes" else 0
        }

        # Ensure correct order for scaler/model
        input_data = pd.DataFrame([features])[feature_names]
        input_scaled = scaler.transform(input_data)

        # 3. Predict
        prob = model.predict_proba(input_scaled)[0][1]

        # --- RESULTS UI ---
        st.write("")
        st.markdown("---")
        st.write("")

        risk_level = "High" if prob > 0.65 else "Moderate" if prob > 0.35 else "Low"
        risk_cls = risk_level.lower()
        risk_color = "#FF5252" if risk_level == "High" else "#FFAB40" if risk_level == "Moderate" else "#00E676"
        sentiment_label = "Positive" if raw_score > 0.05 else "Negative" if raw_score < - \
            0.05 else "Neutral"
        sent_color = "#00E676" if raw_score > 0.05 else "#FF5252" if raw_score < - \
            0.05 else "#8B8FA3"

        # Top row: 3 key numbers
        k1, k2, k3 = st.columns(3)
        with k1:
            st.markdown(f"""
            <div class="result-card card-{risk_cls}">
                <div class="stat-label">SHOCK PROBABILITY</div>
                <div class="prob-big {risk_cls}">{prob*100:.1f}%</div>
                <span class="risk-badge {risk_cls}">{risk_level} RISK</span>
            </div>
            """, unsafe_allow_html=True)
        with k2:
            st.markdown(f"""
            <div class="result-card card-{risk_cls}">
                <div class="stat-label">SENTIMENT SCORE</div>
                <div class="prob-big" style="color:{sent_color}; font-size:2.4rem;">{raw_score:+.3f}</div>
                <span style="font-size:0.75rem; color:#8B8FA3;">{sentiment_label} · Intensity {abs(raw_score):.2f}</span>
            </div>
            """, unsafe_allow_html=True)
        with k3:
            st.markdown(f"""
            <div class="result-card card-{risk_cls}">
                <div class="stat-label">MARKET REGIME</div>
                <div style="margin-top:12px;">
                    <div class="stat-row"><span class="stat-label">VIX</span><span class="stat-value">{curr_vix:.2f}</span></div>
                    <div class="stat-row"><span class="stat-label">NIFTY 5D</span><span class="stat-value">{nifty_ret*100:+.2f}%</span></div>
                    <div class="stat-row"><span class="stat-label">VOLATILITY</span><span class="stat-value">{nifty_std:.4f}</span></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.write("")

        # Gauge chart row
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={'suffix': '%', 'font': {'size': 48,
                                            'color': risk_color, 'family': 'JetBrains Mono'}},
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "VIX Shock Probability Gauge",
                   'font': {'size': 14, 'color': '#8B8FA3', 'family': 'Inter'}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': '#2A2D3A',
                         'tickfont': {'color': '#5C6076', 'family': 'JetBrains Mono', 'size': 11}},
                'bar': {'color': risk_color, 'thickness': 0.3},
                'bgcolor': '#1C1F2B',
                'borderwidth': 1,
                'bordercolor': '#2A2D3A',
                'steps': [
                    {'range': [0, 35], 'color': 'rgba(0,230,118,0.15)'},
                    {'range': [35, 65], 'color': 'rgba(255,171,64,0.15)'},
                    {'range': [65, 100], 'color': 'rgba(255,82,82,0.15)'}],
                'threshold': {
                    'line': {'color': '#00D4AA', 'width': 3},
                    'thickness': 0.8,
                    'value': prob * 100
                }
            }
        ))
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(t=60, b=20, l=60, r=60),
            height=280,
            font={'family': 'Inter'}
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            f"Combined with current market conditions, the model estimates a "
            f"**{prob*100:.1f}%** probability of a VIX shock following this communication.")

        # --- METHODOLOGY SECTION ---
        st.write("")
        st.write("")
        st.markdown("### How this is calculated")
        st.caption("A transparent look at every step behind the prediction.")
        st.write("")

        with st.expander("Data Pipeline", expanded=True):
            st.markdown(f"""
            <div class="pipeline-step">
                <div class="step-num">1</div>
                <div class="step-content">
                    <h4>Sentiment Extraction (FinBERT)</h4>
                    <p>Your text is split into <strong>510-token chunks</strong> (BERT's limit). Each chunk is passed through
                    <a href="https://huggingface.co/ProsusAI/finbert" target="_blank">ProsusAI/FinBERT</a>, a financial-domain
                    BERT model. Softmax probabilities for <em>positive</em>, <em>negative</em>, and <em>neutral</em> are
                    averaged across all chunks.<br>
                    <strong>Score = P(positive) &minus; P(negative)</strong><br>
                    Your text produced <strong>{len(tokenizer(user_text, add_special_tokens=False)['input_ids'])}</strong> tokens
                    across <strong>{len(tokenizer(user_text, add_special_tokens=False)['input_ids']) // 510 + 1}</strong> chunks.</p>
                </div>
            </div>
            <div class="pipeline-step">
                <div class="step-num">2</div>
                <div class="step-content">
                    <h4>Live Market Features</h4>
                    <p>The model also uses today's market state, fetched from Yahoo Finance in real-time:</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

            feat_df = pd.DataFrame({
                "Feature": ["Sentiment Intensity", "India VIX (Lag-1)", "Nifty 5-Day Log Return", "Nifty 5-Day Volatility", "Is MPC Minutes?"],
                "Value Used": [f"{abs(raw_score):.4f}", f"{curr_vix:.2f}", f"{nifty_ret:.4f}", f"{nifty_std:.4f}", f"{doc_type == 'MPC_Minutes'}"],
                "Description": [
                    "|Sentiment Score| — magnitude of policy tone",
                    "Previous day's India VIX close (forward-filled)",
                    "ln(Nifty_today / Nifty_5d_ago) — log return",
                    "Rolling 5-day std of daily log returns",
                    "1 if MPC minutes, 0 if speech"
                ]
            })
            st.dataframe(feat_df, use_container_width=True, hide_index=True)

            st.markdown(f"""
            <div class="pipeline-step">
                <div class="step-num">3</div>
                <div class="step-content">
                    <h4>Scaling &amp; Prediction</h4>
                    <p>The 5 features are standardised using the same <code>StandardScaler</code> fit during training
                    (mean/std from 2020–2024 RBI communication days). The scaled vector is passed to a
                    <strong>Random Forest Classifier</strong> (300 trees, max-depth 4, balanced class weights).
                    The output is the probability that <em>VIX_Returns &gt; 80th-percentile</em> of training data.</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with st.expander("Model Assumptions & Limitations"):
            st.markdown("""
            <div class="method-card">
                <h4>Key Assumptions</h4>
                <ul>
                    <li>RBI communications are the <em>primary</em> policy signal; other macro events on the same day are not isolated.</li>
                    <li>A <strong>VIX Shock</strong> is defined as a VIX return exceeding the <strong>80th percentile</strong> of training-period returns — a statistical, not economic, threshold.</li>
                    <li>Sentiment is computed identically to training: full-document sliding-window FinBERT, NOT a simple truncation.</li>
                    <li>Market features (VIX lag, Nifty momentum) are fetched live and assumed to reflect the <em>current</em> regime.</li>
                </ul>
            </div>
            <div class="method-card">
                <h4>Limitations</h4>
                <ul>
                    <li>The model was trained on ~130 RBI event-days (2020–2024). Small sample sizes may limit generalisation.</li>
                    <li>FinBERT was pre-trained on English financial text — nuances of RBI language may not be fully captured.</li>
                    <li>No causal claim is made; the model captures <em>association</em>, not causation.</li>
                    <li>Prediction accuracy depends on market conditions remaining within the distribution of training data.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

elif predict_btn and not user_text:
    st.warning("Please paste a speech to begin analysis.")
