import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
from datetime import datetime
from utils import predict, save_feedback, DATASET_PATH

def render_css():
    st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global styles */
    .main {
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom color variables */
    :root {
        --primary-color: #6366F1;
        --primary-dark: #4F46E5;
        --secondary-color: #10B981;
        --danger-color: #EF4444;
        --warning-color: #F59E0B;
        --success-color: #10B981;
        --text-primary: #111827;
        --text-secondary: #6B7280;
        --bg-light: #F9FAFB;
        --bg-white: #FFFFFF;
        --border-color: #E5E7EB;
        --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        --shadow-xl: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
        --swahili-color: #3B82F6;
        --english-color: #10B981;
        --mixed-color: #6B7280;
    }
    
    /* Hero section */
    .hero-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 4rem 2rem;
        text-align: center;
        border-radius: 1rem;
        margin-bottom: 3rem;
        box-shadow: var(--shadow-xl);
    }
    
    .hero-title {
        font-size: 3.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .hero-subtitle {
        font-size: 1.25rem;
        font-weight: 400;
        opacity: 0.9;
        max-width: 600px;
        margin: 0 auto;
        line-height: 1.6;
    }
    
    /* Modern card design */
    .modern-card {
        background: var(--bg-white);
        border-radius: 1rem;
        padding: 2rem;
        box-shadow: var(--shadow-md);
        border: 1px solid var(--border-color);
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .modern-card:hover {
        box-shadow: var(--shadow-lg);
        transform: translateY(-2px);
    }
    
    /* Enhanced button styling */
    .stButton > button {
        width: 100%;
        padding: 1rem 2rem;
        border-radius: 0.75rem;
        background: linear-gradient(135deg, var(--primary-color), var(--primary-dark));
        color: white;
        font-weight: 600;
        font-size: 1.1rem;
        border: none;
        transition: all 0.3s ease;
        box-shadow: var(--shadow-md);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-lg);
        background: linear-gradient(135deg, var(--primary-dark), #3730A3);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* Enhanced risk level badges */
    .risk-badge {
        padding: 0.75rem 1.5rem;
        border-radius: 2rem;
        font-weight: 600;
        font-size: 1.1rem;
        display: inline-block;
        margin: 1rem 0;
        box-shadow: var(--shadow-sm);
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .risk-high {
        background: linear-gradient(135deg, #FEE2E2, #FECACA);
        color: var(--danger-color);
        border: 2px solid #FCA5A5;
    }
    
    .risk-moderate {
        background: linear-gradient(135deg, #FEF3C7, #FDE68A);
        color: var(--warning-color);
        border: 2px solid #FBBF24;
    }
    
    .risk-low {
        background: linear-gradient(135deg, #D1FAE5, #A7F3D0);
        color: var(--success-color);
        border: 2px solid #34D399;
    }
    
    /* Code-mixing badge */
    .code-mixing-badge {
        background: linear-gradient(135deg, #DBEAFE, #BFDBFE);
        color: var(--primary-color);
        border: 2px solid var(--primary-color);
        padding: 0.75rem 1.5rem;
        border-radius: 2rem;
        font-weight: 600;
        font-size: 1.1rem;
        display: inline-block;
        margin: 1rem 0;
        box-shadow: var(--shadow-sm);
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Enhanced indicator cards */
    .indicator-card {
        background: var(--bg-white);
        padding: 1.5rem;
        border-radius: 1rem;
        margin: 1rem 0;
        border-left: 5px solid;
        transition: all 0.3s ease;
        box-shadow: var(--shadow-sm);
        position: relative;
        overflow: hidden;
    }
    
    .indicator-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, transparent, currentColor, transparent);
        opacity: 0.3;
    }
    
    .indicator-card:hover {
        transform: translateX(8px);
        box-shadow: var(--shadow-md);
    }
    
    .indicator-high {
        border-left-color: var(--danger-color);
        color: var(--danger-color);
    }
    
    .indicator-medium {
        border-left-color: var(--warning-color);
        color: var(--warning-color);
    }
    
    .indicator-low {
        border-left-color: var(--success-color);
        color: var(--success-color);
    }
    
    .indicator-header {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 0.5rem;
    }
    
    .indicator-icon {
        font-size: 1.5rem;
        filter: drop-shadow(0 2px 4px rgba(0,0,0,0.1));
    }
    
    .indicator-title {
        font-weight: 600;
        font-size: 1.1rem;
        color: var(--text-primary);
    }
    
    .indicator-detail {
        color: var(--text-secondary);
        line-height: 1.5;
        margin: 0;
        padding-left: 2.25rem;
    }
    
    /* Language tag styles */
    .swahili-tag {
        background-color: rgba(59, 130, 246, 0.1);
        color: var(--swahili-color);
        padding: 0.2rem 0.5rem;
        border-radius: 0.5rem;
        margin: 0.2rem;
        display: inline-block;
        font-size: 0.9rem;
        font-weight: 500;
    }
    
    .english-tag {
        background-color: rgba(16, 185, 129, 0.1);
        color: var(--english-color);
        padding: 0.2rem 0.5rem;
        border-radius: 0.5rem;
        margin: 0.2rem;
        display: inline-block;
        font-size: 0.9rem;
        font-weight: 500;
    }
    
    .mixed-tag {
        background-color: rgba(107, 114, 128, 0.1);
        color: var(--mixed-color);
        padding: 0.2rem 0.5rem;
        border-radius: 0.5rem;
        margin: 0.2rem;
        display: inline-block;
        font-size: 0.9rem;
        font-weight: 500;
    }
    
    /* Enhanced headers */
    h1, h2, h3, h4, h5, h6 {
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 1rem;
    }
    
    h1 { font-size: 2.5rem; }
    h2 { font-size: 2rem; }
    h3 { font-size: 1.5rem; }
    h4 { font-size: 1.25rem; }
    
    /* Enhanced tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: var(--bg-light);
        border-radius: 1rem;
        padding: 0.5rem;
        margin-bottom: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        padding: 0 1.5rem;
        border-radius: 0.75rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--primary-color);
        color: white;
        box-shadow: var(--shadow-md);
    }
    
    /* Enhanced text areas and inputs */
    .stTextArea textarea {
        border-radius: 0.75rem;
        border: 2px solid var(--border-color);
        padding: 1rem;
        font-size: 1rem;
        transition: all 0.3s ease;
        resize: vertical;
    }
    
    .stTextArea textarea:focus {
        border-color: var(--primary-color);
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
    }
    
    .stSelectbox > div > div {
        border-radius: 0.75rem;
        border: 2px solid var(--border-color);
        transition: all 0.3s ease;
    }
    
    .stSelectbox > div > div:focus-within {
        border-color: var(--primary-color);
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
    }
    
    /* Enhanced expanders */
    .streamlit-expanderHeader {
        background: var(--bg-light);
        border-radius: 0.75rem;
        padding: 1rem;
        border: 1px solid var(--border-color);
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .streamlit-expanderHeader:hover {
        background: var(--bg-white);
        box-shadow: var(--shadow-sm);
    }
    
    /* Protection tips styling */
    .protection-tip {
        background: linear-gradient(135deg, #F0F9FF, #E0F2FE);
        padding: 1.5rem;
        border-radius: 1rem;
        border-left: 5px solid var(--primary-color);
        margin: 1rem 0;
        box-shadow: var(--shadow-sm);
    }
    
    .protection-tip h4 {
        color: var(--primary-color);
        margin-bottom: 0.5rem;
    }
    
    /* Stats cards */
    .stats-card {
        background: var(--bg-white);
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        box-shadow: var(--shadow-md);
        border: 1px solid var(--border-color);
        transition: all 0.3s ease;
    }
    
    .stats-card:hover {
        transform: translateY(-4px);
        box-shadow: var(--shadow-lg);
    }
    
    .stats-number {
        font-size: 2.5rem;
        font-weight: 700;
        color: var(--primary-color);
        display: block;
    }
    
    .stats-label {
        color: var(--text-secondary);
        font-weight: 500;
        margin-top: 0.5rem;
    }
    
    /* Enhanced dataframe styling */
    .stDataFrame {
        border-radius: 1rem;
        overflow: hidden;
        box-shadow: var(--shadow-md);
    }
    
    /* Loading spinner enhancement */
    .stSpinner > div {
        border-top-color: var(--primary-color) !important;
    }
    
    /* Enhanced alerts */
    .stAlert {
        border-radius: 0.75rem;
        border: none;
        box-shadow: var(--shadow-sm);
    }
    
    /* Footer styling */
    .footer {
        background: var(--bg-light);
        padding: 3rem 2rem;
        text-align: center;
        border-radius: 1rem;
        margin-top: 4rem;
        border: 1px solid var(--border-color);
    }
    
    .footer-links {
        display: flex;
        justify-content: center;
        gap: 2rem;
        margin-top: 1rem;
        flex-wrap: wrap;
    }
    
    .footer-link {
        color: var(--primary-color);
        text-decoration: none;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .footer-link:hover {
        color: var(--primary-dark);
        text-decoration: underline;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .hero-title {
            font-size: 2.5rem;
        }
        
        .hero-subtitle {
            font-size: 1.1rem;
        }
        
        .modern-card {
            padding: 1.5rem;
        }
        
        .indicator-card {
            padding: 1rem;
        }
        
        .stats-number {
            font-size: 2rem;
        }
    }
</style>
""", unsafe_allow_html=True)

def render_hero_section():
    st.markdown("""
        <div class='hero-section'>
            <div class='hero-title'>🛡️ SMS Scam Shield</div>
            <div class='hero-subtitle'>
                Advanced AI-powered protection against SMS scams and phishing attempts. 
                Analyze messages instantly with our state-of-the-art detection system.
            </div>
        </div>
    """, unsafe_allow_html=True)

def render_footer():
    st.markdown("""
        <div class='footer'>
            <h3>🛡️ SMS Scam Shield</h3>
            <p style='color: var(--text-secondary); margin: 1rem 0;'>
                Protecting Kenyans from SMS scams with advanced AI technology. 
                Built with ❤️ for community safety and digital literacy.
            </p>
            <div class='footer-links'>
                <a href='https://mursimind.com' class='footer-link'>🌐 Visit Mursi Mind</a>
                <a href='#' class='footer-link'>📧 Contact Support</a>
                <a href='#' class='footer-link'>📚 Documentation</a>
                <a href='#' class='footer-link'>🔒 Privacy Policy</a>
            </div>
            <p style='color: var(--text-secondary); margin-top: 2rem; font-size: 0.9rem;'>
                © 2024 SMS Scam Shield. Empowering digital safety through AI innovation.
            </p>
        </div>
    """, unsafe_allow_html=True)

def get_prediction_explanation(prediction, confidence, indicators, language_analysis):
    explanation = []
    if prediction == "scam":
        explanation.append("🚫 **This message shows strong signs of being a scam.**")
    else:
        explanation.append("✅ **This message appears to be legitimate, but always stay vigilant.**")
    
    if confidence > 0.8:
        explanation.append(f"The AI system is very confident ({confidence:.0%}) about this assessment.")
    elif confidence > 0.6:
        explanation.append(f"The AI system is moderately confident ({confidence:.0%}) about this assessment.")
    else:
        explanation.append(f"The AI system is less certain ({confidence:.0%}) about this assessment, please use extra caution.")
    
    high_severity_count = sum(1 for ind in indicators if ind['severity'] == 'high')
    medium_severity_count = sum(1 for ind in indicators if ind['severity'] == 'medium')
    
    if high_severity_count > 0 or medium_severity_count > 0:
        explanation.append("\n**🔍 Key Concerns Detected:**")
        if high_severity_count > 0:
            explanation.append(f"- **{high_severity_count} High-Risk Indicators** 🔴")
        if medium_severity_count > 0:
            explanation.append(f"- **{medium_severity_count} Medium-Risk Indicators** 🟡")
        
        advice_given = set()
        for indicator in indicators:
            if indicator['severity'] in ['high', 'medium']:
                if indicator['type'] == 'Financial Bait' and 'financial' not in advice_given:
                    explanation.append("- 💡 **Financial Safety**: Be extremely cautious with money-related messages")
                    advice_given.add('financial')
                elif indicator['type'] == 'Suspicious Link' and 'links' not in advice_given:
                    explanation.append("- 💡 **Link Safety**: Never click unexpected links - they may be phishing attempts")
                    advice_given.add('links')
                elif indicator['type'] == 'Personal Info Request' and 'personal' not in advice_given:
                    explanation.append("- 💡 **Privacy Protection**: Legitimate services never ask for PINs or passwords via SMS")
                    advice_given.add('personal')
                elif indicator['type'] == 'Betting Lure' and 'betting' not in advice_given:
                    explanation.append("- 💡 **Gambling Awareness**: Be skeptical of unexpected betting opportunities")
                    advice_given.add('betting')
                elif indicator['type'] == 'Urgency Tactics' and 'urgency' not in advice_given:
                    explanation.append("- 💡 **Time Pressure**: Scammers create false urgency - take your time to verify")
                    advice_given.add('urgency')
    
    # Add linguistic analysis explanation
    explanation.append("\n**🗣️ Language Analysis:**")
    if language_analysis['code_mixing_intensity'] > 20:
        explanation.append("- **Code-mixing detected**: This message contains a mix of Swahili and English, which may indicate informal or targeted communication.")
    else:
        explanation.append("- **Language consistency**: The message is primarily in one language, which is typical for legitimate communications.")
    
    explanation.append(f"- **Linguistic Confidence**: {language_analysis['linguistic_confidence']:.1f}% (based on identified words)")
    
    return "\n".join(explanation)

import streamlit as st

import streamlit as st

def apply_rule_heuristics(text):
    import re
    lower_text = text.lower()
    
    # Expanded keyword categories
    financial_keywords = [
        "won", "money", "prize", "irs", "gift card", "lottery", "winner", "claim", "inheritance", 
        "cash", "free", "bonus", "refund", "payout", "bitcoin", "crypto", "investment"
    ]
    urgency_keywords = [
        "urgent", "immediate", "now", "limited time", "expire", "action required", "last chance", 
        "hurry", "today only", "don't miss", "final notice"
    ]
    phishing_keywords = [
        "click the link", "account suspended", "verify", "update", "login", "password", "security alert", 
        "confirm", "bank", "paypal", "amazon", "netflix", "apple id", "suspended", "locked"
    ]
    emotional_keywords = [
        "help", "family", "emergency", "charity", "donate", "sick", "accident", "trouble"
    ]
    
    # Detect matches
    financial_matches = [kw for kw in financial_keywords if kw in lower_text]
    urgency_matches = [kw for kw in urgency_keywords if kw in lower_text]
    phishing_matches = [kw for kw in phishing_keywords if kw in lower_text]
    emotional_matches = [kw for kw in emotional_keywords if kw in lower_text]
    
    # URL detection (enhanced)
    url_pattern = re.compile(r'(https?://|www\.|bit\.ly|tinyurl\.com|t\.co|\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b)', re.IGNORECASE)
    has_suspicious_url = bool(url_pattern.search(lower_text))
    suspicious_url_detail = "Suspicious URL detected (possible phishing link or shortener)" if has_suspicious_url else ""
    
    # Phone number detection
    phone_pattern = re.compile(r'(\b\d{3}[-.]\d{3}[-.]\d{4}\b|\b\d{10}\b|\b1-\d{3}-\d{3}-\d{4}\b|\+\d{1,3}\s?\d{3}\s?\d{3}\s?\d{4})', re.IGNORECASE)
    has_phone = bool(phone_pattern.search(text))
    phone_detail = "Phone number detected (scams often request calls)" if has_phone else ""
    
    # Collect fired rules as indicators
    rule_indicators = []
    if financial_matches:
        rule_indicators.append({
            'severity': 'high',
            'icon': '💰',
            'type': 'Financial Bait (Rule)',
            'detail': f"High-risk financial keywords: {', '.join(financial_matches)}"
        })
    if urgency_matches:
        rule_indicators.append({
            'severity': 'medium',
            'icon': '⏰',
            'type': 'Urgency Tactics (Rule)',
            'detail': f"Pressure keywords: {', '.join(urgency_matches)}"
        })
    if phishing_matches:
        rule_indicators.append({
            'severity': 'high',
            'icon': '🔗',
            'type': 'Phishing Attempt (Rule)',
            'detail': f"Account/security keywords: {', '.join(phishing_matches)}"
        })
    if emotional_matches:
        rule_indicators.append({
            'severity': 'medium',
            'icon': '😢',
            'type': 'Emotional Manipulation (Rule)',
            'detail': f"Emotional triggers: {', '.join(emotional_matches)}"
        })
    if has_suspicious_url:
        rule_indicators.append({
            'severity': 'high',
            'icon': '⚠️',
            'type': 'Suspicious Link (Rule)',
            'detail': suspicious_url_detail
        })
    if has_phone:
        rule_indicators.append({
            'severity': 'medium',
            'icon': '📞',
            'type': 'Call Request (Rule)',
            'detail': phone_detail
        })
    
    # Determine if rules fire strongly (for override)
    num_fired = len(rule_indicators)
    strong_fire = num_fired >= 3 or any(ind['severity'] == 'high' for ind in rule_indicators)
    should_override = strong_fire
    
    return rule_indicators, should_override, num_fired

def render_message_scanner_tab(lr_model, xgb_model, tfidf_vectorizer):
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.markdown("""
            <div class='modern-card'>
                <h3>📱 Message Analysis</h3>
                <p style='color: var(--text-secondary); margin-bottom: 1rem;'>
                    Paste the SMS message you want to analyze for potential scam indicators.
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        text_input = st.text_area(
            "Enter Message",
            height=120,
            help="Enter the complete SMS message you received",
            placeholder="Example: 'God bless you! you have won this amount of money'",
            key="text_input_tab1"
        )
        
        selected_model = st.selectbox(
            "Select Model",
            ["XGBoost", "Logistic Regression"],
            help="XGBoost provides detailed feature analysis, while Logistic Regression offers fast predictions",
            key="model_select_tab1"
        )
        
        if st.button("🔍 Analyze Message", use_container_width=True, key="analyze_button_tab1"):
            if text_input.strip():
                with st.spinner("🔄 **Analyzing message with AI...** This may take a few seconds."):
                    prediction, confidence, tokens, cleaned_text, features, indicators, top_features, language_analysis = predict(text_input, selected_model, lr_model, xgb_model, tfidf_vectorizer)
                    
                    # Rule-Based Heuristics as Fallback
                    rule_indicators, should_override, num_fired = apply_rule_heuristics(text_input)
                    indicators.extend(rule_indicators)
                    
                    if num_fired > 0:
                        if (prediction == "legit" and confidence < 0.80) or should_override:
                            prediction = "scam"
                            confidence = 0.90 if should_override else max(confidence, 0.85)  # Stronger boost for strong fires
                            st.info("📌 **Note:** Rule-based heuristics detected multiple high-risk indicators and overrode the initial prediction for added caution.")
                    
                    if prediction == "Invalid Input":
                        st.warning("⚠️ **Invalid Input** - Please enter a valid message to analyze.")
                    elif prediction == "Error in prediction":
                        st.error("❌ **Analysis Error** - Something went wrong. Please try again.")
                    else:
                        st.markdown("---")
                        st.markdown("## 📋 Analysis Results")
                        
                        # Risk Badge
                        risk_color = {"scam": "risk-high", "legit": "risk-low"}[prediction]
                        risk_emoji = {"scam": "🔴 LIKELY SCAM", "legit": "✅ APPEARS SAFE"}[prediction]
                        st.markdown(f"""
                            <div class='risk-badge {risk_color}'>
                                {risk_emoji} • {confidence:.0%} Confidence
                            </div>
                        """, unsafe_allow_html=True)
                        
                        # Warning or Confirmation Message
                        if prediction == "scam":
                            st.markdown("""
                                <div class='modern-card' style='border-left: 5px solid var(--danger-color);'>
                                    <p style='color: var(--danger-color); font-weight: 600;'>
                                        Warning: This message exhibits multiple characteristics of a common scam. You should delete it immediately.
                                    </p>
                                </div>
                            """, unsafe_allow_html=True)
                        
                        # Threat Breakdown
                        st.markdown("### 📊 Threat Breakdown")
                        for indicator in indicators:
                            severity_class = {'high': 'indicator-high', 'medium': 'indicator-medium', 'low': 'indicator-low'}[indicator['severity']]
                            severity_label = {'high': 'HIGH RISK', 'medium': 'MEDIUM RISK', 'low': 'LOW RISK'}[indicator['severity']]
                            st.markdown(f"""
                                <div class='indicator-card {severity_class}'>
                                    <div class='indicator-header'>
                                        <span class='indicator-icon'>{indicator['icon']}</span>
                                        <span class='indicator-title'>{indicator['type']} ({severity_label})</span>
                                    </div>
                                    <p class='indicator-detail'>{indicator['detail']}</p>
                                </div>
                            """, unsafe_allow_html=True)
                        
                        # Recommended Action
                        st.markdown("### 🛡️ Recommended Action")
                        if prediction == "scam":
                            # Define tailored recommendations based on indicator type
                            recommendations = []
                            has_high_risk = any(ind['severity'] == 'high' for ind in indicators)
                            primary_indicator = next((ind for ind in indicators if ind['severity'] == 'high'), indicators[0] if indicators else None)
                            
                            if primary_indicator:
                                if primary_indicator['type'] == 'Financial Bait':
                                    recommendations.extend([
                                        "<li><strong>Do not respond to prize or money offers</strong> - Legitimate lotteries don’t notify winners via unsolicited SMS.</li>",
                                        "<li><strong>Verify claims independently</strong> - Contact the organization using official contact details, not those provided in the message.</li>",
                                        "<li>Learn more about <a href='#' class='footer-link'>prize scams</a> in our Safety Education section.</li>"
                                    ])
                                elif primary_indicator['type'] == 'Suspicious Link':
                                    recommendations.extend([
                                        "<li><strong>Do not click any links</strong> - They may lead to phishing sites or malware.</li>",
                                        "<li><strong>Check URLs carefully</strong> - Visit official websites directly to verify claims.</li>",
                                        "<li>Learn more about <a href='#' class='footer-link'>phishing scams</a> in our Safety Education section.</li>"
                                    ])
                                elif primary_indicator['type'] == 'Urgency Tactics':
                                    recommendations.extend([
                                        "<li><strong>Take your time</strong> - Scammers use urgency to pressure you; legitimate offers don’t expire immediately.</li>",
                                        "<li><strong>Verify with trusted sources</strong> - Contact the sender through official channels.</li>",
                                        "<li>Learn more about <a href='#' class='footer-link'>urgency scams</a> in our Safety Education section.</li>"
                                    ])
                                else:
                                    recommendations.append("<li><strong>Be cautious</strong> - This message contains suspicious elements.</li>")
                            
                            # Add generic recommendations if not already included
                            if "<li><strong>Do not reply</strong> to the message.</li>" not in recommendations:
                                recommendations.append("<li><strong>Do not reply</strong> to the message.</li>")
                            if "<li><strong>Delete the message</strong> from your phone.</li>" not in recommendations:
                                recommendations.append("<li><strong>Delete the message</strong> from your phone.</li>")
                            
                            st.markdown(f"""
                                <div class='protection-tip'>
                                    <ul style='margin: 0; padding-left: 1.5rem;'>
                                        {''.join(recommendations)}
                                    </ul>
                                </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown("""
                                <div class='protection-tip'>
                                    <ul style='margin: 0; padding-left: 1.5rem;'>
                                        <li><strong>Stay vigilant</strong> - Even legitimate-looking messages may require verification.</li>
                                        <li><strong>Contact the sender directly</strong> - Use official contact details to confirm the message’s authenticity.</li>
                                        <li>Learn more about <a href='#' class='footer-link'>staying safe</a> in our Safety Education section.</li>
                                    </ul>
                                </div>
                            """, unsafe_allow_html=True)
                        
                        # AI Insights
                        st.markdown("### 🤖 AI Insight")
                        strongest_predictor = top_features[0][0] if top_features and isinstance(top_features[0], tuple) else top_features[0] if top_features else 'N/A'
                        st.markdown(f"""
                            <div class='modern-card'>
                                <p>Our {selected_model} and BERT models analyzed the message. The strongest predictor was the word/phrase <strong>"{strongest_predictor}"</strong>.</p>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        # Feedback Section
                        st.markdown("---")
                        st.markdown("### 📝 Was this analysis helpful?")
                        feedback_col1, feedback_col2 = st.columns([1, 1])
                        with feedback_col1:
                            if st.button("👍 Helpful", use_container_width=True, key="helpful_button_tab1"):
                                st.success("✅ **Thank you for your feedback!** This helps validate our AI accuracy.")
                                save_feedback(text_input, prediction, prediction, selected_model)
                        with feedback_col2:
                            if st.button("👎 Not Helpful", use_container_width=True, key="not_helpful_button_tab1"):
                                user_correction = st.radio(
                                    "**What was wrong with the analysis?**",
                                    ["This should be marked as SAFE", "This should be marked as SCAM"],
                                    key="feedback_correction_tab1"
                                )
                                if st.button("📤 Submit Feedback", use_container_width=True, key="submit_feedback_tab1"):
                                    corrected_label = "legit" if "SAFE" in user_correction else "scam"
                                    if save_feedback(text_input, prediction, corrected_label, selected_model):
                                        st.success("🙏 **Thank you!** Your feedback helps improve our detection accuracy.")
                                    else:
                                        st.error("❌ Failed to save feedback. Please try again.")
            else:
                st.warning("📝 **Please enter a message** to analyze before clicking the button.")

def render_safety_education_tab():
    st.markdown("## 🎓 SMS Safety Education Center")
    st.markdown("### 📊 Scam Statistics")
    stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
    
    with stats_col1:
        st.markdown("""
            <div class='stats-card'>
                <span class='stats-number'>85%</span>
                <div class='stats-label'>Detection Accuracy</div>
            </div>
        """, unsafe_allow_html=True)
    
    with stats_col2:
        st.markdown("""
            <div class='stats-card'>
                <span class='stats-number'>1M+</span>
                <div class='stats-label'>Messages Analyzed</div>
            </div>
        """, unsafe_allow_html=True)
    
    with stats_col3:
        st.markdown("""
            <div class='stats-card'>
                <span class='stats-number'>24/7</span>
                <div class='stats-label'>Protection Available</div>
            </div>
        """, unsafe_allow_html=True)
    
    with stats_col4:
        st.markdown("""
            <div class='stats-card'>
                <span class='stats-number'>0₵</span>
                <div class='stats-label'>Cost to Use</div>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 🎯 Common Scam Tactics to Watch For")
    tactics_col1, tactics_col2 = st.columns(2)
    
    with tactics_col1:
        st.markdown("""
            <div class='protection-tip'>
                <h4>⚡ Urgency & Pressure Tactics</h4>
                <p><strong>What to look for:</strong></p>
                <ul>
                    <li>Words like "urgent", "immediate", "expires today"</li>
                    <li>Threats of account closure or penalties</li>
                    <li>Limited time offers that seem too good to be true</li>
                </ul>
                <p><strong>Why scammers use this:</strong> They want you to act quickly without thinking or verifying.</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div class='protection-tip'>
                <h4>🔒 Personal Information Requests</h4>
                <p><strong>Red flags:</strong></p>
                <ul>
                    <li>Requests for PINs, passwords, or ID numbers</li>
                    <li>Asking to "verify" or "update" your details</li>
                    <li>Requests for M-PESA PIN or banking information</li>
                </ul>
                <p><strong>Remember:</strong> Legitimate companies never ask for sensitive information via SMS.</p>
            </div>
        """, unsafe_allow_html=True)
    
    with tactics_col2:
        st.markdown("""
            <div class='protection-tip'>
                <h4>💰 Financial Lures & Fake Prizes</h4>
                <p><strong>Common schemes:</strong></p>
                <ul>
                    <li>Fake lottery or competition winnings</li>
                    <li>Unexpected money transfers or refunds</li>
                    <li>Investment opportunities with guaranteed returns</li>
                </ul>
                <p><strong>Reality check:</strong> If you didn't enter a competition, you can't win it!</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div class='protection-tip'>
                <h4>🔗 Suspicious Links & Phishing</h4>
                <p><strong>Warning signs:</strong></p>
                <ul>
                    <li>Shortened URLs (bit.ly, tinyurl, etc.)</li>
                    <li>Misspelled website names</li>
                    <li>Unexpected links from unknown senders</li>
                </ul>
                <p><strong>Best practice:</strong> Never click links in suspicious messages. Visit official websites directly.</p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 🛡️ Your Protection Strategy")
    protection_col1, protection_col2 = st.columns(2)
    
    with protection_col1:
        st.markdown("""
            <div class='modern-card'>
                <h4>🕐 Take Your Time</h4>
                <ul>
                    <li><strong>Pause and think</strong> - Legitimate messages don't require immediate action</li>
                    <li><strong>Research the sender</strong> - Look up the company or phone number online</li>
                    <li><strong>Ask for help</strong> - Consult with family or friends if unsure</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div class='modern-card'>
                <h4>🔍 Verify Independently</h4>
                <ul>
                    <li><strong>Contact companies directly</strong> - Use official phone numbers or websites</li>
                    <li><strong>Don't use provided contact info</strong> - Scammers provide fake numbers</li>
                    <li><strong>Check official social media</strong> - Companies announce legitimate promotions publicly</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    with protection_col2:
        st.markdown("""
            <div class='modern-card'>
                <h4>🔐 Guard Your Information</h4>
                <ul>
                    <li><strong>Never share PINs or passwords</strong> - Not even with "bank representatives"</li>
                    <li><strong>Be cautious with personal details</strong> - Scammers use them for identity theft</li>
                    <li><strong>Use strong, unique passwords</strong> - Enable two-factor authentication when possible</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div class='modern-card'>
                <h4>📢 Report and Block</h4>
                <ul>
                    <li><strong>Report to your mobile provider</strong> - Help them block scam numbers</li>
                    <li><strong>Block suspicious numbers</strong> - Prevent future contact attempts</li>
                    <li><strong>Share with others</strong> - Warn friends and family about new scam tactics</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 📢 Report Scam Messages")
    st.markdown("""
        <div class='modern-card'>
            <h4>🚨 Help Protect Others - Report Scams</h4>
            <p>When you receive a scam message, reporting it helps protect the entire community:</p>
            <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem; margin-top: 1rem;'>
                <div>
                    <h5>📱 Mobile Providers</h5>
                    <ul>
                        <li>Safaricom: Forward to 333</li>
                        <li>Airtel: Report via customer care</li>
                        <li>Telkom: Contact customer support</li>
                    </ul>
                </div>
                <div>
                    <h5>🏛️ Government Agencies</h5>
                    <ul>
                        <li>Communications Authority of Kenya</li>
                        <li>Directorate of Criminal Investigations</li>
                        <li>Kenya Computer Incident Response Team</li>
                    </ul>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

def render_batch_analysis_tab(lr_model, xgb_model, tfidf_vectorizer):
    st.markdown("""
        <div class='modern-card'>
            <h3>🔄 Analyze Multiple Messages</h3>
            <p style='color: var(--text-secondary);'>
                Upload a CSV file containing multiple SMS messages to analyze them all at once. 
                Perfect for businesses or organizations that need to check multiple messages for threats.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "**Upload CSV File** (must contain 'message_content' column)",
        type=["csv"],
        help="Your CSV file should have a column named 'message_content' with the SMS messages to analyze",
        key="file_uploader_tab3"
    )
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.markdown("### 👀 File Preview")
            st.dataframe(df.head(), use_container_width=True)
            
            if 'message_content' in df.columns:
                st.success(f"✅ **File loaded successfully!** Found {len(df)} messages to analyze.")
                batch_model = st.selectbox(
                    "**Choose AI Model for Batch Analysis:**",
                    ["XGBoost", "Logistic Regression"],
                    help="XGBoost provides more detailed analysis but takes longer",
                    key="batch_model_select_tab3"
                )
                
                if st.button("🚀 **Start Batch Analysis**", use_container_width=True, key="batch_analyze_button_tab3"):
                    with st.spinner(f"📊 **Analyzing {len(df)} messages...** This may take a few minutes."):
                        results = []
                        progress_bar = st.progress(0)
                        
                        for idx, text in enumerate(df['message_content']):
                            pred, conf, _, _, _, inds, _, _ = predict(text, batch_model, lr_model, xgb_model, tfidf_vectorizer)
                            high_risk_indicators = [ind['type'] for ind in inds if ind['severity'] == 'high']
                            medium_risk_indicators = [ind['type'] for ind in inds if ind['severity'] == 'medium']
                            
                            results.append({
                                'Message Preview': str(text)[:60] + '...' if len(str(text)) > 60 else str(text),
                                'Risk Assessment': pred.upper(),
                                'Confidence': f"{conf:.1%}",
                                'High Risk Indicators': "; ".join(high_risk_indicators) if high_risk_indicators else "None",
                                'Medium Risk Indicators': "; ".join(medium_risk_indicators) if medium_risk_indicators else "None",
                                'Total Indicators': len(inds)
                            })
                            progress_bar.progress((idx + 1) / len(df))
                        
                        results_df = pd.DataFrame(results)
                        st.markdown("### 📈 Analysis Summary")
                        summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
                        
                        scam_count = len(results_df[results_df['Risk Assessment'] == 'SCAM'])
                        legit_count = len(results_df[results_df['Risk Assessment'] == 'LEGIT'])
                        high_confidence = len(results_df[results_df['Confidence'].str.rstrip('%').astype(float) >= 80])
                        
                        with summary_col1:
                            st.markdown(f"""
                                <div class='stats-card'>
                                    <span class='stats-number' style='color: var(--danger-color);'>{scam_count}</span>
                                    <div class='stats-label'>Potential Scams</div>
                                </div>
                            """, unsafe_allow_html=True)
                        
                        with summary_col2:
                            st.markdown(f"""
                                <div class='stats-card'>
                                    <span class='stats-number' style='color: var(--success-color);'>{legit_count}</span>
                                    <div class='stats-label'>Appear Safe</div>
                                </div>
                            """, unsafe_allow_html=True)
                        
                        with summary_col3:
                            st.markdown(f"""
                                <div class='stats-card'>
                                    <span class='stats-number'>{high_confidence}</span>
                                    <div class='stats-label'>High Confidence</div>
                                </div>
                            """, unsafe_allow_html=True)
                        
                        with summary_col4:
                            st.markdown(f"""
                                <div class='stats-card'>
                                    <span class='stats-number'>{len(df)}</span>
                                    <div class='stats-label'>Total Analyzed</div>
                                </div>
                            """, unsafe_allow_html=True)
                        
                        st.markdown("### 📋 Detailed Results")
                        st.dataframe(
                            results_df,
                            use_container_width=True,
                            hide_index=True
                        )
                        
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            "📥 **Download Complete Analysis Report**",
                            csv,
                            f"sms_scam_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            "text/csv",
                            use_container_width=True,
                            key="download_button_tab3"
                        )
                        st.success("✅ **Batch analysis completed successfully!**")
            else:
                st.error("❌ **Invalid file format** - Your CSV file must contain a column named 'message_content'.")
                st.info("💡 **Tip:** Make sure your CSV has a header row with 'message_content' as one of the column names.")
        except Exception as e:
            st.error(f"❌ **Error processing file:** {str(e)}")
            st.info("💡 **Tip:** Make sure your file is a valid CSV format with proper encoding (UTF-8).")

def render_dataset_explorer_tab():
    st.markdown("""
        <div class='modern-card'>
            <h3>🔎 Explore Our SMS Dataset</h3>
            <p style='color: var(--text-secondary);'>
                Browse through our dataset of labeled SMS messages. 
                Use the filters below to narrow down by label or language.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    try:
        df_explorer = pd.read_csv(DATASET_PATH)
        
        if len(df_explorer) == 0:
            st.warning("⚠️ Dataset is empty.")
        else:
            # Map is_kiswahili_sheng to language
            df_explorer['language'] = df_explorer['is_kiswahili_sheng'].map({True: 'Sheng/Swahili', False: 'English'})
            
            # Get unique values for filters
            labels = sorted(df_explorer['label'].unique()) if 'label' in df_explorer.columns else []
            languages = sorted(df_explorer['language'].unique()) if 'language' in df_explorer.columns else []
            
            # Filters
            st.markdown("### Filters")
            selected_labels = st.multiselect(
                "**Filter by Label:**",
                options=labels,
                default=None,
                help="Select one or more labels (scam or legit) to filter the dataset.",
                key="label_filter_tab4"
            )
            
            selected_languages = st.multiselect(
                "**Filter by Language:**",
                options=languages,
                default=None,
                help="Select one or more languages (English or Sheng/Swahili) to filter the dataset.",
                key="language_filter_tab4"
            )
            
            # Apply filters
            filtered_df = df_explorer.copy()
            if selected_labels:
                filtered_df = filtered_df[filtered_df['label'].isin(selected_labels)]
            if selected_languages:
                filtered_df = filtered_df[filtered_df['language'].isin(selected_languages)]
            
            # Display filtered dataset
            st.markdown(f"### 📋 Showing {len(filtered_df)} Messages")
            st.dataframe(
                filtered_df,
                use_container_width=True,
                hide_index=True
            )
            
            # Option to show random sample
            if st.button("🔀 Show Random Sample (10 Messages)", use_container_width=True, key="random_sample_button_tab4"):
                sample_df = filtered_df.sample(min(10, len(filtered_df)))
                st.dataframe(
                    sample_df,
                    use_container_width=True,
                    hide_index=True
                )
            
            # Dataset Statistics Dashboard
            st.markdown("### 📈 Dataset Statistics Dashboard")
            
            # Fraud Type (Label) Distribution
            st.markdown("#### Fraud Type Distribution")
            label_counts = df_explorer['label'].value_counts().reset_index()
            label_counts.columns = ['Label', 'Count']
            fig_label = px.bar(
                label_counts,
                x='Label',
                y='Count',
                color='Label',
                color_discrete_map={'scam': '#EF4444', 'legit': '#10B981'},
                title="Distribution of Scam vs Legit Messages",
                text='Count'
            )
            fig_label.update_layout(
                xaxis_title="Label",
                yaxis_title="Number of Messages",
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_label, use_container_width=True)
            
            # Language Mixing Patterns
            st.markdown("#### Language Mixing Patterns")
            language_counts = df_explorer['language'].value_counts().reset_index()
            language_counts.columns = ['Language', 'Count']
            fig_language = px.pie(
                language_counts,
                names='Language',
                values='Count',
                title="Distribution of Languages in the Dataset",
                color_discrete_sequence=px.colors.qualitative.Plotly
            )
            fig_language.update_traces(textinfo='percent+label')
            fig_language.update_layout(
                showlegend=True,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_language, use_container_width=True)
            
            # Annotation Confidence Distribution
            st.markdown("#### Annotation Confidence Distribution")
            if 'confidence' in df_explorer.columns:
                fig_confidence = ff.create_distplot(
                    [df_explorer['confidence'].dropna()],
                    group_labels=['Confidence'],
                    colors=['#6366F1'],
                    show_rug=False
                )
                fig_confidence.update_layout(
                    title="Distribution of Annotation Confidence Scores",
                    xaxis_title="Confidence Score",
                    yaxis_title="Density",
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig_confidence, use_container_width=True)
            else:
                st.warning("⚠️ Confidence column not found in the dataset.")
                
    except FileNotFoundError:
        st.error(f"❌ Dataset file not found at {DATASET_PATH}.")
    except Exception as e:
        st.error(f"❌ Error loading dataset: {str(e)}")