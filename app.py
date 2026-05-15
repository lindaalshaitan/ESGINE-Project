"""
Strategic Decision Intelligence
Cream theme · All-dark fonts · Logo support
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import warnings
warnings.filterwarnings('ignore')

from database import get_indicators, get_indicator_values, get_brent_data
from decision_advisor import analyze_decision, compute_indicator_correlations
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import IsolationForest

# ============================================================
# Page Config
# ============================================================
LOGO_PATH = "logo.png"
page_icon = LOGO_PATH if os.path.exists(LOGO_PATH) else "●"

st.set_page_config(
    page_title="Strategic Decision Intelligence",
    page_icon=page_icon,
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# Palette — Cream theme
# ============================================================
CREAM_BG = "#F5F1E8"
CREAM_CARD = "#FAF7EE"
CREAM_ELEVATED = "#EFEADC"
CREAM_BORDER = "#D4CFC0"

INK = "#0E2920"
INK_DEEP = "#061812"
INK_SOFT = "#1B4332"
TEXT_BODY = "#2D2A24"
TEXT_MUTED = "#5C5648"

GOLD = "#B8941F"
GOLD_DEEP = "#8B6F14"
GREEN = "#005A2D"
RED = "#8B1F1F"

# ============================================================
# Language System
# ============================================================
if 'lang' not in st.session_state:
    st.session_state.lang = 'en'

T = {
    'en': {
        'tagline': 'Strategic Decision Intelligence',
        'primary': 'PRIMARY', 'analysis': 'ANALYSIS', 'system': 'SYSTEM',
        'decision_advisor': 'Decision Advisor',
        'intelligence_workbench': 'Intelligence Workbench',
        'about': 'About the System',
        'executive_brief': 'Executive Brief',
        'da_eyebrow': 'Decision Intelligence · v1.0',
        'da_title': 'Decision Advisor',
        'da_desc': 'Analytical infrastructure supporting strategic interpretation. Enter a policy decision to receive a data-driven analytical brief.',
        'quick_examples': 'Quick examples',
        'decision_brief': 'Decision brief',
        'placeholder': 'Describe a policy, national project, or strategic decision...',
        'analyze': 'Analyze Decision',
        'domain': 'Domain Classification',
        'relevant_inds': 'Relevant indicators',
        'decision_signals': 'Decision Signals',
        'interpretation': 'Signal Interpretation',
        'strategic_interp': 'Strategic Interpretation',
        'systemic_link': 'Systemic Linkages',
        'systemic_desc': 'How indicators relevant to this decision move together with other indicators. These linkages reveal where policy effects may propagate.',
        'forecast': 'Exploratory Forecast',
        'forecast_desc': 'Five-year projection based on SARIMAX with Brent crude as external driver.',
        'strategic_sig': 'Strategic Signals',
        'strategic_recs': 'Strategic Recommendations',
        'lang_button': 'العربية',
        'high_conf': 'High Confidence', 'mod_conf': 'Moderate Confidence', 'exp_conf': 'Exploratory',
        'tab_explorer': 'Indicator Explorer',
        'tab_forecasting': 'Forecasting',
        'tab_scenarios': 'Scenarios',
        'tab_deviations': 'Pattern Deviations',
        'select_indicator': 'Select indicator',
        'pillar': 'PILLAR', 'unit': 'UNIT', 'sdg': 'SDG', 'weight': 'WEIGHT',
        'mean': 'MEAN', 'std_dev': 'STD DEV', 'min': 'MIN', 'max': 'MAX',
        'target_ind': 'Target indicator',
        'forecast_horizon': 'Forecast horizon (years)',
        'detect_btn': 'Detect Pattern Deviations',
        'about_title': 'About the System',
        'about_subtitle': 'What it is and how it works',
        'what_is': 'What is this system?',
        'what_is_text': 'A Strategic Decision Intelligence platform designed to support Kuwaiti decision-makers in evaluating policy decisions against real ESG data.',
        'how_it_works': 'How it works',
        'how_step1': '1. The decision-maker enters a strategic decision in natural language',
        'how_step2': '2. The system classifies the decision domain',
        'how_step3': '3. The system retrieves relevant indicators from 12 ESG metrics (2015-2024)',
        'how_step4': '4. Statistical models (SARIMAX, correlation) generate signals and forecasts',
        'how_step5': '5. A structured strategic brief is delivered',
        'philosophy': 'Philosophy',
        'philosophy_text': 'This is a Decision Intelligence Workflow, not an autonomous AI agent. Outputs support — not replace — human judgment.',
        'current_state': 'Current State (2024)',
        'strategic_cons': 'Strategic Considerations',
        'data_gov': 'Data Governance',
        'sources': 'Sources',
        'coverage': 'Coverage',
        'update_cadence': 'Update Cadence',
        'validation': 'Validation',
        'missing_data': 'Missing Data',
        'stat_limits': 'Statistical Limits',
        'disclaimer': 'Disclaimer',
        'pri_high': 'HIGH PRIORITY', 'pri_med': 'MEDIUM PRIORITY', 'pri_low': 'LOW PRIORITY',
        'scenario_desc': 'Select a driver variable and observe its projected impact based on historical correlations.',
        'driver_var': 'Driver variable (independent)',
        'change_pct': 'Projected change (%)',
        'impacted_inds': 'Impacted indicators',
        'run_scenario': 'Run Scenario',
        'scenario_results': 'Scenario Results',
        'historical_corr': 'Historical correlation',
        'projected_impact': 'Projected impact',
        'projected_value': 'Projected value',
        'current_val': 'Current value',
        'forecast_explanation': 'Forecast Explanation',
    },
    'ar': {
        'tagline': 'ذكاء القرار الاستراتيجي',
        'primary': 'الأساسي', 'analysis': 'التحليل', 'system': 'النظام',
        'decision_advisor': 'مستشار القرار',
        'intelligence_workbench': 'منصة التحليل',
        'about': 'عن النظام',
        'executive_brief': 'الموجز التنفيذي',
        'da_eyebrow': 'ذكاء القرار · الإصدار 1.0',
        'da_title': 'مستشار القرار',
        'da_desc': 'بنية تحليلية تدعم التفسير الاستراتيجي. أدخل قراراً سياسياً للحصول على موجز تحليلي قائم على البيانات.',
        'quick_examples': 'أمثلة سريعة',
        'decision_brief': 'صياغة القرار',
        'placeholder': 'صف سياسة أو مشروعاً وطنياً أو قراراً استراتيجياً...',
        'analyze': 'تحليل القرار',
        'domain': 'تصنيف المجال',
        'relevant_inds': 'المؤشرات ذات الصلة',
        'decision_signals': 'إشارات القرار',
        'interpretation': 'تفسير الإشارات',
        'strategic_interp': 'التفسير الاستراتيجي',
        'systemic_link': 'الترابطات النظمية',
        'systemic_desc': 'كيف تتحرك المؤشرات ذات الصلة بهذا القرار مع المؤشرات الأخرى. هذه الترابطات تكشف أين قد تنتقل آثار السياسات.',
        'forecast': 'التنبؤ الاستكشافي',
        'forecast_desc': 'إسقاط لخمس سنوات بناءً على SARIMAX مع سعر برنت كمحرّك خارجي.',
        'strategic_sig': 'الإشارات الاستراتيجية',
        'strategic_recs': 'التوصيات الاستراتيجية',
        'lang_button': 'English',
        'high_conf': 'ثقة عالية', 'mod_conf': 'ثقة متوسطة', 'exp_conf': 'استكشافي',
        'tab_explorer': 'مستكشف المؤشرات',
        'tab_forecasting': 'التنبؤ',
        'tab_scenarios': 'السيناريوهات',
        'tab_deviations': 'الانحرافات النمطية',
        'select_indicator': 'اختر مؤشراً',
        'pillar': 'الركيزة', 'unit': 'الوحدة', 'sdg': 'SDG', 'weight': 'الوزن',
        'mean': 'المتوسط', 'std_dev': 'الانحراف', 'min': 'الأدنى', 'max': 'الأعلى',
        'target_ind': 'المؤشر المستهدف',
        'forecast_horizon': 'أفق التنبؤ (سنوات)',
        'detect_btn': 'كشف الانحرافات النمطية',
        'about_title': 'عن النظام',
        'about_subtitle': 'ما هو وكيف يعمل',
        'what_is': 'ما هو هذا النظام؟',
        'what_is_text': 'منصة ذكاء قرار استراتيجي مصممة لدعم صنّاع القرار الكويتيين في تقييم القرارات السياسية مقابل بيانات حقيقية.',
        'how_it_works': 'كيف يعمل',
        'how_step1': '1. يُدخل صانع القرار قراراً استراتيجياً بلغة طبيعية',
        'how_step2': '2. يصنّف النظام مجال القرار',
        'how_step3': '3. يسترجع النظام المؤشرات ذات الصلة من 12 مؤشراً (2015-2024)',
        'how_step4': '4. تولّد النماذج الإحصائية إشارات وتنبؤات',
        'how_step5': '5. يُسلَّم موجز استراتيجي منظم',
        'philosophy': 'الفلسفة',
        'philosophy_text': 'هذا النظام هو سير عمل لذكاء القرار، وليس وكيلاً ذكياً مستقلاً. المخرجات تدعم — وليس تستبدل — الحكم البشري.',
        'current_state': 'الحالة الراهنة (2024)',
        'strategic_cons': 'الاعتبارات الاستراتيجية',
        'data_gov': 'حوكمة البيانات',
        'sources': 'المصادر',
        'coverage': 'التغطية',
        'update_cadence': 'وتيرة التحديث',
        'validation': 'التحقق',
        'missing_data': 'البيانات الناقصة',
        'stat_limits': 'الحدود الإحصائية',
        'disclaimer': 'إخلاء المسؤولية',
        'pri_high': 'أولوية عالية', 'pri_med': 'أولوية متوسطة', 'pri_low': 'أولوية منخفضة',
        'scenario_desc': 'اختر متغيراً محرّكاً وراقب تأثيره المتوقع بناءً على الترابطات التاريخية.',
        'driver_var': 'المتغير المحرّك',
        'change_pct': 'التغيير المتوقع (%)',
        'impacted_inds': 'المؤشرات المتأثرة',
        'run_scenario': 'تشغيل السيناريو',
        'scenario_results': 'نتائج السيناريو',
        'historical_corr': 'الترابط التاريخي',
        'projected_impact': 'التأثير المتوقع',
        'projected_value': 'القيمة المتوقعة',
        'current_val': 'القيمة الحالية',
        'forecast_explanation': 'تفسير التنبؤ',
    }
}

def t(key):
    return T[st.session_state.lang].get(key, key)

DIR = 'rtl' if st.session_state.lang == 'ar' else 'ltr'
TEXT_ALIGN = 'right' if st.session_state.lang == 'ar' else 'left'

# ============================================================
# CSS — Cream theme · All-dark fonts
# ============================================================
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Noto+Kufi+Arabic:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"], .main, .stApp, [data-testid="stAppViewContainer"] {{
        font-family: 'Inter', 'Noto Kufi Arabic', sans-serif !important;
        color: {INK_DEEP} !important;
        background: {CREAM_BG} !important;
    }}
    
    p, span, div, label, h1, h2, h3, h4, h5, h6, li, td, th, a {{
        color: {INK_DEEP} !important;
    }}
    
    .block-container {{
        background: {CREAM_BG} !important;
        padding-top: 2rem !important;
        max-width: 1300px !important;
    }}
    
    .page-header {{
        border-bottom: 2px solid {CREAM_BORDER};
        padding: 0.3rem 0 1.3rem 0;
        margin-bottom: 1.8rem;
    }}
    .page-header .eyebrow {{
        font-size: 0.7rem; letter-spacing: 3px; color: {TEXT_MUTED} !important;
        text-transform: uppercase; font-weight: 700; margin: 0;
    }}
    .page-header h1 {{
        font-size: 1.6rem; font-weight: 700; color: {INK_DEEP} !important;
        margin: 0.4rem 0 0.3rem 0; letter-spacing: -0.3px;
        word-wrap: break-word; overflow-wrap: anywhere;
    }}
    .page-header .subtitle {{
        font-size: 0.95rem; color: {TEXT_BODY} !important; font-weight: 500; margin: 0;
    }}
    
    .decision-hero {{
        background: linear-gradient(135deg, {CREAM_CARD} 0%, {CREAM_ELEVATED} 100%);
        border: 2px solid {CREAM_BORDER};
        color: {INK_DEEP} !important;
        padding: 1.8rem;
        border-radius: 6px;
        margin-bottom: 1.8rem;
        position: relative;
        box-shadow: 0 2px 8px rgba(14, 41, 32, 0.04);
    }}
    .decision-hero::before {{
        content: ''; position: absolute; top: 0; left: 0;
        width: 80px; height: 4px; background: {GOLD_DEEP};
    }}
    .decision-hero .eyebrow {{
        font-size: 0.7rem; letter-spacing: 3px;
        color: {GOLD_DEEP} !important; font-weight: 700;
        text-transform: uppercase; margin: 0;
    }}
    .decision-hero h1 {{
        font-size: 1.5rem !important;
        font-weight: 800;
        color: {INK_DEEP} !important;
        margin: 0.5rem 0 0.3rem 0;
        line-height: 1.3;
        word-wrap: break-word;
    }}
    .decision-hero p {{
        font-size: 0.92rem;
        color: {TEXT_BODY} !important;
        line-height: 1.7;
        margin: 0.7rem 0 0 0;
        max-width: 720px;
        font-weight: 500;
    }}
    
    @media (max-width: 768px) {{
        .decision-hero h1 {{ font-size: 1.25rem !important; }}
        .page-header h1 {{ font-size: 1.3rem; }}
    }}
    
    .signal-card {{
        background: {CREAM_CARD};
        padding: 1.3rem 1.4rem;
        border: 1.5px solid {CREAM_BORDER};
        border-radius: 6px;
        height: 100%;
    }}
    .signal-card:hover {{ border-color: {GOLD_DEEP}; }}
    .signal-card .label {{
        color: {TEXT_MUTED} !important;
        font-size: 0.68rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin: 0;
        font-weight: 700;
    }}
    .signal-card .value {{
        color: {INK_DEEP} !important;
        font-size: 1.5rem;
        font-weight: 800;
        margin: 0.6rem 0 0.3rem 0;
    }}
    .signal-card .trend {{
        font-size: 0.82rem;
        margin: 0;
        color: {TEXT_BODY} !important;
        font-weight: 600;
    }}
    
    .insight-panel {{
        background: {CREAM_CARD};
        border-left: 4px solid {GOLD_DEEP};
        padding: 1.3rem 1.5rem;
        margin: 1rem 0;
        border-radius: 0 6px 6px 0;
        border-top: 1px solid {CREAM_BORDER};
        border-right: 1px solid {CREAM_BORDER};
        border-bottom: 1px solid {CREAM_BORDER};
    }}
    .insight-panel h4 {{
        color: {INK_DEEP} !important;
        margin: 0 0 0.5rem 0;
        font-weight: 700;
        font-size: 1rem;
    }}
    .insight-panel p, .insight-panel li {{
        color: {TEXT_BODY} !important;
        line-height: 1.7;
        font-size: 0.92rem;
        margin: 0.3rem 0;
        font-weight: 500;
    }}
    .insight-panel strong {{
        color: {INK_DEEP} !important;
        font-weight: 700;
    }}
    
    .interpretation-panel {{
        background: linear-gradient(135deg, {CREAM_ELEVATED} 0%, #E6E0CF 100%);
        padding: 1.6rem 1.8rem;
        border-radius: 6px;
        margin: 1.5rem 0;
        position: relative;
        border: 1.5px solid {CREAM_BORDER};
    }}
    .interpretation-panel::before {{
        content: ''; position: absolute; top: 0; left: 0;
        width: 50px; height: 3px; background: {GOLD_DEEP};
    }}
    .interpretation-panel .eyebrow {{
        font-size: 0.7rem;
        letter-spacing: 3px;
        color: {GOLD_DEEP} !important;
        font-weight: 800;
        text-transform: uppercase;
        margin: 0 0 0.7rem 0;
    }}
    .interpretation-panel .narrative,
    .interpretation-panel p {{
        font-size: 0.95rem;
        line-height: 1.8;
        color: {INK_DEEP} !important;
        margin: 0;
        font-style: italic;
        font-weight: 600;
    }}
    
    .systemic-box {{
        background: {CREAM_CARD};
        border: 1.5px solid {CREAM_BORDER};
        border-left: 4px solid {INK_DEEP};
        padding: 1.4rem 1.6rem;
        margin: 1.2rem 0;
        border-radius: 0 6px 6px 0;
    }}
    .systemic-box h4 {{
        color: {INK_DEEP} !important;
        margin: 0 0 0.5rem 0;
        font-weight: 700;
        font-size: 1rem;
    }}
    .systemic-box .desc {{
        color: {TEXT_BODY} !important;
        font-size: 0.88rem;
        line-height: 1.6;
        margin-bottom: 1rem;
        font-weight: 500;
    }}
    .systemic-row {{
        display: flex;
        align-items: center;
        padding: 0.8rem 0;
        border-bottom: 1px solid {CREAM_BORDER};
        gap: 1rem;
        flex-wrap: wrap;
    }}
    .systemic-row:last-child {{ border-bottom: none; }}
    .systemic-pair {{
        flex: 2; min-width: 180px;
        font-weight: 600;
        color: {INK_DEEP} !important;
        font-size: 0.88rem;
    }}
    .systemic-corr {{
        flex: 1; min-width: 80px;
        text-align: center;
        font-weight: 800;
        font-size: 1rem;
    }}
    .systemic-tag {{
        flex: 1; min-width: 80px;
        text-align: center;
        font-size: 0.72rem;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        color: {TEXT_MUTED} !important;
        font-weight: 700;
    }}
    .systemic-interp {{
        flex: 3; min-width: 200px;
        color: {TEXT_BODY} !important;
        font-size: 0.84rem;
        line-height: 1.5;
        font-weight: 500;
    }}
    
    .confidence-badge {{
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.68rem;
        font-weight: 800;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-left: 0.8rem;
    }}
    .confidence-high {{ background: rgba(0,90,45,0.08); color: {GREEN} !important; border: 1.5px solid {GREEN}; }}
    .confidence-moderate {{ background: rgba(139,111,20,0.08); color: {GOLD_DEEP} !important; border: 1.5px solid {GOLD_DEEP}; }}
    .confidence-exploratory {{ background: rgba(92,86,72,0.08); color: {TEXT_BODY} !important; border: 1.5px solid {TEXT_BODY}; }}
    .confidence-dot {{ width: 6px; height: 6px; border-radius: 50%; display: inline-block; }}
    
    .governance-card {{
        background: {CREAM_CARD};
        border: 1.5px solid {CREAM_BORDER};
        border-radius: 6px;
        padding: 1.5rem 1.8rem;
        margin: 1rem 0;
    }}
    .governance-row {{
        display: flex;
        padding: 0.7rem 0;
        border-bottom: 1px solid {CREAM_BORDER};
        flex-wrap: wrap;
    }}
    .governance-row:last-child {{ border-bottom: none; }}
    .governance-row .label {{
        font-size: 0.72rem;
        letter-spacing: 1.5px;
        color: {TEXT_MUTED} !important;
        text-transform: uppercase;
        font-weight: 700;
        width: 180px;
        flex-shrink: 0;
    }}
    .governance-row .value {{
        font-size: 0.88rem;
        color: {INK_DEEP} !important;
        line-height: 1.6;
        font-weight: 500;
        flex: 1;
    }}
    
    .recommendation-panel {{
        background: {CREAM_CARD};
        border-left: 4px solid {INK_DEEP};
        border-top: 1px solid {CREAM_BORDER};
        border-right: 1px solid {CREAM_BORDER};
        border-bottom: 1px solid {CREAM_BORDER};
        padding: 1.3rem 1.5rem;
        margin: 1rem 0;
        border-radius: 0 6px 6px 0;
    }}
    .recommendation-panel h4 {{
        color: {INK_DEEP} !important;
        margin: 0.4rem 0 0.5rem 0;
        font-weight: 700;
        font-size: 1rem;
    }}
    .recommendation-panel p {{
        color: {TEXT_BODY} !important;
        line-height: 1.7;
        font-size: 0.9rem;
        font-weight: 500;
    }}
    .recommendation-panel strong {{ color: {INK_DEEP} !important; font-weight: 700; }}
    
    .signal-elevated {{ border-left-color: {RED} !important; }}
    .signal-stable {{ border-left-color: {GREEN} !important; }}
    
    /* SIDEBAR CREAM THEME */
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {CREAM_BG} 0%, {CREAM_ELEVATED} 100%) !important;
        border-right: 1px solid {CREAM_BORDER} !important;
        min-width: 290px !important;
    }}
    [data-testid="stSidebar"] * {{
        color: {INK_DEEP} !important;
    }}
    [data-testid="stSidebar"] .stButton button {{
        background: transparent !important;
        color: {INK_DEEP} !important;
        border: 1px solid {CREAM_BORDER} !important;
        font-weight: 600 !important;
    }}
    [data-testid="stSidebar"] .stButton button:hover {{
        border-color: {GOLD_DEEP} !important;
        background: rgba(212, 207, 192, 0.3) !important;
    }}
    [data-testid="stSidebar"] .stButton button[kind="primary"] {{
        background: linear-gradient(135deg, {CREAM_ELEVATED} 0%, #E6E0CF 100%) !important;
        color: {INK_DEEP} !important;
        border-color: {GOLD_DEEP} !important;
    }}
    
    .brand {{ padding: 1.2rem 0 1.5rem 0; text-align: center; }}
    .brand .tagline {{
        color: {TEXT_MUTED} !important;
        font-size: 0.68rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-top: 1rem;
        font-weight: 700;
    }}
    .brand .accent {{
        width: 40px;
        height: 2px;
        background: {GOLD_DEEP};
        margin: 0 auto;
    }}
    
    .nav-section {{
        color: {TEXT_MUTED} !important;
        font-size: 0.65rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin: 1.5rem 0 0.5rem 0;
        font-weight: 800;
    }}
    
    .section-title {{
        font-size: 1.05rem;
        font-weight: 700;
        color: {INK_DEEP} !important;
        margin: 2rem 0 1rem 0;
        letter-spacing: -0.3px;
        display: flex;
        align-items: center;
        flex-wrap: wrap;
    }}
    .section-desc {{
        color: {TEXT_BODY} !important;
        font-size: 0.9rem;
        line-height: 1.6;
        margin: 0 0 1rem 0;
        font-weight: 500;
    }}
    
    .method-note {{
        background: transparent;
        border-top: 1.5px solid {CREAM_BORDER};
        padding: 0.9rem 0 0 0;
        font-size: 0.78rem;
        color: {TEXT_BODY} !important;
        margin: 2rem 0 0 0;
        font-style: italic;
        line-height: 1.6;
        font-weight: 500;
    }}
    
    .stMetric {{
        background: {CREAM_CARD};
        padding: 0.8rem 1rem;
        border-radius: 6px;
        border: 1.5px solid {CREAM_BORDER};
    }}
    .stMetric label {{
        font-size: 0.7rem !important;
        letter-spacing: 1.5px !important;
        color: {TEXT_MUTED} !important;
        font-weight: 700 !important;
    }}
    .stMetric [data-testid="stMetricValue"] {{
        color: {INK_DEEP} !important;
        font-weight: 800 !important;
    }}
    
    .stTabs [data-baseweb="tab-list"] {{
        gap: 0;
        border-bottom: 2px solid {CREAM_BORDER};
        background: transparent !important;
    }}
    .stTabs [data-baseweb="tab"] {{
        padding: 0.8rem 1.4rem;
        font-weight: 600;
        font-size: 0.9rem;
        background: transparent !important;
        border: none;
        color: {TEXT_MUTED} !important;
    }}
    .stTabs [data-baseweb="tab"]:hover {{ color: {INK_DEEP} !important; }}
    .stTabs [aria-selected="true"] {{
        color: {INK_DEEP} !important;
        border-bottom: 3px solid {GOLD_DEEP} !important;
        font-weight: 700 !important;
    }}
    
    /* MAIN AREA BUTTONS */
    .stButton button {{ font-weight: 600 !important; }}
    .stButton button[kind="primary"] {{
        background: linear-gradient(135deg, {CREAM_ELEVATED} 0%, #E6E0CF 100%) !important;
        color: {INK_DEEP} !important;
        border: 1px solid {CREAM_BORDER} !important;
    }}
    .stButton button[kind="primary"]:hover {{
        background: #E6E0CF !important;
        border-color: {GOLD_DEEP} !important;
    }}
    
    .stTextArea textarea, .stTextInput input {{
        background: {CREAM_CARD} !important;
        color: {INK_DEEP} !important;
        border: 1.5px solid {CREAM_BORDER} !important;
        font-weight: 500 !important;
    }}
    .stTextArea textarea:focus, .stTextInput input:focus {{
        border-color: {GOLD_DEEP} !important;
    }}
    .stSelectbox > div > div {{
        background: {CREAM_CARD} !important;
        color: {INK_DEEP} !important;
        border: 1.5px solid {CREAM_BORDER} !important;
    }}
    .stMultiSelect > div > div {{
        background: {CREAM_CARD} !important;
        border: 1.5px solid {CREAM_BORDER} !important;
    }}
    
    .streamlit-expanderHeader {{
        background: {CREAM_CARD} !important;
        color: {INK_DEEP} !important;
        font-weight: 600 !important;
        border: 1.5px solid {CREAM_BORDER} !important;
    }}
    
    .stDataFrame {{
        border: 1.5px solid {CREAM_BORDER};
        border-radius: 6px;
    }}
    
    h1, h2, h3 {{
        letter-spacing: -0.3px;
        color: {INK_DEEP} !important;
        font-weight: 700;
    }}
    
    .js-plotly-plot text {{ fill: {INK_DEEP} !important; }}
    
    .footer-block {{
        text-align: center;
        color: {TEXT_BODY} !important;
        padding: 2.5rem 0 1rem 0;
        font-size: 0.82rem;
        border-top: 1.5px solid {CREAM_BORDER};
        margin-top: 3rem;
        font-weight: 500;
    }}
    .footer-block strong {{ color: {INK_DEEP} !important; font-weight: 700; }}
</style>
""", unsafe_allow_html=True)

# ============================================================
# Data
# ============================================================
@st.cache_data(ttl=300)
def load_data():
    return get_indicators(), get_indicator_values(), get_brent_data()

with st.spinner("Loading..."):
    indicators, values, brent = load_data()

if 'main_page' not in st.session_state:
    st.session_state.main_page = "Decision Advisor"

# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    if os.path.exists(LOGO_PATH):
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(LOGO_PATH, width=120)
    else:
        st.markdown(f"""
        <div style='text-align: center; margin-top: 1rem;'>
            <div style='width: 70px; height: 70px; border-radius: 50%; 
                        background: {CREAM_BORDER}; border: 2px solid {GOLD_DEEP}; margin: 0 auto;
                        display: flex; align-items: center; justify-content: center;
                        font-size: 2rem; font-weight: 800; color: {INK_DEEP};'>
                E
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="brand">
        <div class="accent"></div>
        <p class="tagline">{t('tagline')}</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button(f"🌐 {t('lang_button')}", use_container_width=True, key="lang_toggle"):
        st.session_state.lang = 'ar' if st.session_state.lang == 'en' else 'en'
        st.rerun()
    
    st.markdown(f'<p class="nav-section">{t("primary")}</p>', unsafe_allow_html=True)
    if st.button(t('decision_advisor'), use_container_width=True, 
                  type="primary" if st.session_state.main_page == "Decision Advisor" else "secondary",
                  key="nav_da"):
        st.session_state.main_page = "Decision Advisor"
        st.rerun()
    
    st.markdown(f'<p class="nav-section">{t("analysis")}</p>', unsafe_allow_html=True)
    if st.button(t('intelligence_workbench'), use_container_width=True,
                  type="primary" if st.session_state.main_page == "Intelligence Workbench" else "secondary",
                  key="nav_iw"):
        st.session_state.main_page = "Intelligence Workbench"
        st.rerun()
    
    st.markdown(f'<p class="nav-section">{t("system")}</p>', unsafe_allow_html=True)
    if st.button(t('about'), use_container_width=True,
                  type="primary" if st.session_state.main_page == "About" else "secondary",
                  key="nav_about"):
        st.session_state.main_page = "About"
        st.rerun()
    if st.button(t('executive_brief'), use_container_width=True,
                  type="primary" if st.session_state.main_page == "Executive Brief" else "secondary",
                  key="nav_eb"):
        st.session_state.main_page = "Executive Brief"
        st.rerun()
    
    st.markdown(f"""
    <div style='margin-top: 3rem; padding: 1rem 0.5rem;
                color: {TEXT_BODY}; font-size: 0.75rem; line-height: 1.6;
                border-top: 1px solid {CREAM_BORDER};'>
        <div style='font-weight: 800; color: {INK_DEEP};'>Linda Waleed ALSHAITAN</div>
        <div style='font-weight: 500;'>Youth Innovation Award 2026</div>
    </div>
    """, unsafe_allow_html=True)

page = st.session_state.main_page
lang = st.session_state.lang

def page_header(eyebrow, title, subtitle):
    st.markdown(f"""
    <div class="page-header">
        <p class="eyebrow">{eyebrow}</p>
        <h1>{title}</h1>
        <p class="subtitle">{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)


def generate_strategic_narrative(result, values, lang):
    cat = result['classification']['primary_category']
    gdp_data = values[values['code'] == 'ECO_GDP'].sort_values('year')
    gdp_volatility = gdp_data['value'].std()
    oil_rent_data = values[values['code'] == 'ECO_OIL_RENT'].sort_values('year')
    oil_rent_latest = oil_rent_data['value'].iloc[-1] if len(oil_rent_data) > 0 else 30
    renew_data = values[values['code'] == 'ENV_RENEW'].sort_values('year')
    renew_latest = renew_data['value'].iloc[-1] if len(renew_data) > 0 else 1
    
    if lang == 'ar':
        narratives = {
            'energy': f"تظل المرونة الاقتصادية الكلية مرتبطة بشدة بتقلبات أسعار النفط (σ={gdp_volatility:.1f})، بينما تستمر مؤشرات التحول للطاقة المتجددة دون المسار الاستراتيجي ({renew_latest:.2f}% مقابل هدف 15%).",
            'economic': f"التعرض الهيكلي للدورات السلعية يظل واضحاً — ريع النفط عند {oil_rent_latest:.1f}% من GDP يُبقي تركّز مخاطر الإيرادات. التقلبات التاريخية (σ={gdp_volatility:.1f}) تشير إلى أن الآليات المعاكسة للدورات تستحق الأولوية.",
            'social': f"تكشف مؤشرات سوق العمل عن جمود هيكلي يتفاعل مع ديناميكيات الاقتصاد الكلي. الاستثمار المستدام في رأس المال البشري يمثل تدخلاً عالي الأثر (تقلبات GDP σ={gdp_volatility:.1f}).",
            'environmental': f"تعكس المؤشرات البيئية ديناميكيات تحول في مرحلة مبكرة — قدرة الطاقة المتجددة ({renew_latest:.2f}%) دون مسار 2035، بينما تستمر الكثافة الكربونية الهيكلية."
        }
    else:
        narratives = {
            'energy': f"Macroeconomic resilience remains highly coupled to oil-price volatility (σ={gdp_volatility:.1f}), while renewable transition indicators continue below long-term trajectory ({renew_latest:.2f}% vs 15% target).",
            'economic': f"Structural exposure to commodity cycles remains pronounced — oil rent at {oil_rent_latest:.1f}% of GDP sustains revenue concentration risk. Historical volatility (σ={gdp_volatility:.1f}) suggests countercyclical mechanisms warrant strategic priority.",
            'social': f"Labor market indicators reveal structural rigidities that interact with broader macroeconomic dynamics. Sustained investment in human capital represents a high-leverage intervention (GDP σ={gdp_volatility:.1f}).",
            'environmental': f"Environmental indicators reflect early-stage transition dynamics — renewable capacity ({renew_latest:.2f}%) remains below the 2035 trajectory, while structural carbon intensity persists."
        }
    
    return narratives.get(cat, narratives['economic'])


def get_confidence_level(forecast, signals):
    if not forecast:
        return ("exploratory", t('exp_conf'))
    colors = [signals[k]['color'] for k in signals.keys()]
    red_count = colors.count('red')
    green_count = colors.count('green')
    if abs(red_count - green_count) >= 3:
        return ("high", t('high_conf'))
    elif abs(red_count - green_count) >= 1:
        return ("moderate", t('mod_conf'))
    else:
        return ("exploratory", t('exp_conf'))


def interpret_correlation(r, ind1, ind2, lang):
    strength = abs(r)
    direction = "positive" if r > 0 else "negative"
    
    if lang == 'ar':
        dir_ar = "إيجابي" if r > 0 else "عكسي"
        if strength > 0.7:
            return f"ترابط {dir_ar} قوي: عند تحرك {ind1}، يتحرك {ind2} في نفس الاتجاه" if r > 0 else f"ترابط {dir_ar} قوي: عند ارتفاع {ind1}، ينخفض {ind2}"
        elif strength > 0.4:
            return f"ترابط {dir_ar} متوسط — قد يتأثر {ind2} جزئياً بتحركات {ind1}"
        else:
            return f"ترابط {dir_ar} ضعيف"
    else:
        if strength > 0.7:
            return f"Strong {direction}: when {ind1} moves, {ind2} likely moves in the same direction" if r > 0 else f"Strong {direction}: when {ind1} rises, {ind2} tends to fall"
        elif strength > 0.4:
            return f"Moderate {direction} co-movement — {ind2} may partially respond to {ind1} shifts"
        else:
            return f"Weak {direction} correlation"


# ============================================================
# 1. DECISION ADVISOR
# ============================================================
if page == "Decision Advisor":
    st.markdown(f"""
    <div class="decision-hero" dir="{DIR}" style="text-align: {TEXT_ALIGN};">
        <p class="eyebrow">{t('da_eyebrow')}</p>
        <h1>{t('da_title')}</h1>
        <p>{t('da_desc')}</p>
    </div>
    """, unsafe_allow_html=True)
    
    examples_en = [
        "Expand desalination capacity by 30% in coastal zones",
        "Launch national green hydrogen export strategy",
        "Reform fuel subsidies over a 5-year horizon",
        "Mandate AI literacy across secondary education",
        "Increase women's workforce participation by 10%",
        "Establish national stabilization fund"
    ]
    examples_ar = [
        "توسيع طاقة التحلية بنسبة 30% في المناطق الساحلية",
        "إطلاق استراتيجية وطنية لتصدير الهيدروجين الأخضر",
        "إصلاح دعم الوقود على مدى 5 سنوات",
        "إلزام تعليم الذكاء الاصطناعي في الثانوية",
        "زيادة مشاركة النساء في العمل بنسبة 10%",
        "إنشاء صندوق استقرار وطني"
    ]
    examples = examples_ar if lang == 'ar' else examples_en
    
    example_choice = st.selectbox(t('quick_examples'), [""] + examples, index=0)
    
    decision_text = st.text_area(
        t('decision_brief'),
        value=example_choice if example_choice else "",
        height=100,
        placeholder=t('placeholder')
    )
    
    if st.button(t('analyze'), type="primary", use_container_width=True):
        if decision_text.strip():
            with st.spinner("Running Decision Intelligence Workflow..."):
                result = analyze_decision(decision_text, values, brent)
            st.session_state.last_analysis = result
    
    if 'last_analysis' in st.session_state:
        result = st.session_state.last_analysis
        
        cat = result['classification']['primary_category']
        cat_names_en = {'energy': 'Energy', 'economic': 'Economic', 'social': 'Social', 'environmental': 'Environmental'}
        cat_names_ar = {'energy': 'الطاقة', 'economic': 'الاقتصاد', 'social': 'الاجتماعي', 'environmental': 'البيئة'}
        cat_name = cat_names_ar[cat] if lang == 'ar' else cat_names_en[cat]
        
        st.markdown(f"""
        <div class="insight-panel" dir="{DIR}" style="text-align: {TEXT_ALIGN};">
            <p style="font-size: 0.7rem; letter-spacing: 2px; color: {TEXT_MUTED} !important; margin: 0; text-transform: uppercase; font-weight: 700;">{t('domain')}</p>
            <h4 style="font-size: 1.2rem; margin: 0.3rem 0;">{cat_name}</h4>
            <p style="margin: 0;">{t('relevant_inds')}: <strong>{", ".join(result['classification']['relevant_indicators'])}</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f'<div class="section-title">{t("decision_signals")}</div>', unsafe_allow_html=True)
        signals = result['signals']
        cols = st.columns(4)
        signal_keys = ['esg_signal', 'economic_outlook', 'social_conditions', 'environmental_trend']
        
        for col, key in zip(cols, signal_keys):
            s = signals[key]
            color_map = {'green': GREEN, 'gold': GOLD_DEEP, 'red': RED}
            color = color_map.get(s['color'], TEXT_BODY)
            label = s['label_ar'] if lang == 'ar' else s['label']
            with col:
                st.markdown(f"""
                <div class="signal-card">
                    <p class="label">{label}</p>
                    <p class="value">{s['score']}{s.get('unit', '')}</p>
                    <p class="trend" style="color: {color} !important;">{s['trend']}</p>
                </div>
                """, unsafe_allow_html=True)
        
        with st.expander(t('interpretation')):
            for key in signal_keys:
                s = signals[key]
                label = s['label_ar'] if lang == 'ar' else s['label']
                interp = s['interpretation_ar'] if lang == 'ar' else s['interpretation']
                st.write(f"**{label}**: {interp}")
        
        narrative = generate_strategic_narrative(result, values, lang)
        st.markdown(f"""
        <div class="interpretation-panel" dir="{DIR}" style="text-align: {TEXT_ALIGN};">
            <p class="eyebrow">{t('strategic_interp')}</p>
            <p class="narrative">"{narrative}"</p>
        </div>
        """, unsafe_allow_html=True)
        
        rel_corrs = result['relevant_correlations']
        if rel_corrs:
            st.markdown(f"""
            <div class="systemic-box" dir="{DIR}" style="text-align: {TEXT_ALIGN};">
                <h4>{t('systemic_link')}</h4>
                <p class="desc">{t('systemic_desc')}</p>
            </div>
            """, unsafe_allow_html=True)
            
            for c in rel_corrs[:5]:
                interp = interpret_correlation(c['correlation'], c['indicator_1'], c['indicator_2'], lang)
                strength_label = c['strength'].upper()
                if lang == 'ar':
                    strength_label = {'STRONG': 'قوي', 'MODERATE': 'متوسط', 'WEAK': 'ضعيف'}.get(strength_label, strength_label)
                color = GREEN if c['correlation'] > 0 else RED
                
                st.markdown(f"""
                <div style="background: {CREAM_CARD}; border: 1.5px solid {CREAM_BORDER}; border-left: 4px solid {color}; padding: 1rem 1.3rem; margin: 0.5rem 0; border-radius: 0 6px 6px 0;" dir="{DIR}">
                    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 0.5rem;">
                        <strong style="color: {INK_DEEP} !important; font-size: 0.95rem;">{c['indicator_1']} ↔ {c['indicator_2']}</strong>
                        <span style="color: {color} !important; font-weight: 800; font-size: 1rem;">r = {c['correlation']:+.2f} · {strength_label}</span>
                    </div>
                    <p style="margin: 0.5rem 0 0 0; color: {TEXT_BODY} !important; font-size: 0.85rem; line-height: 1.5;">{interp}</p>
                </div>
                """, unsafe_allow_html=True)
        
        forecast = result['forecast']
        if forecast:
            confidence_class, confidence_label = get_confidence_level(forecast, signals)
            
            st.markdown(f"""
            <div class="section-title">
                {t('forecast')}
                <span class="confidence-badge confidence-{confidence_class}">
                    <span class="confidence-dot" style="background: currentColor;"></span>
                    {confidence_label}
                </span>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f'<p class="section-desc">{t("forecast_desc")}</p>', unsafe_allow_html=True)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=forecast['historical_years'], y=forecast['historical_values'],
                mode='lines+markers', name='Historical' if lang == 'en' else 'تاريخي',
                line=dict(color=INK_DEEP, width=2.5), marker=dict(size=7, color=INK_DEEP)))
            fig.add_trace(go.Scatter(
                x=forecast['forecast_years'], y=forecast['forecast_values'],
                mode='lines+markers', name='Forecast' if lang == 'en' else 'تنبؤ',
                line=dict(color=GOLD_DEEP, width=2.5, dash='dot'), marker=dict(size=7, color=GOLD_DEEP)))
            fig.add_trace(go.Scatter(
                x=forecast['forecast_years'] + forecast['forecast_years'][::-1],
                y=forecast['ci_upper'] + forecast['ci_lower'][::-1],
                fill='toself', fillcolor='rgba(139, 111, 20, 0.15)',
                line=dict(color='rgba(255,255,255,0)'), name='95% CI'))
            fig.update_layout(
                height=360, hovermode='x unified',
                plot_bgcolor=CREAM_CARD, paper_bgcolor=CREAM_CARD,
                xaxis=dict(showgrid=False, showline=True, linecolor=CREAM_BORDER,
                          tickfont=dict(color=INK_DEEP, size=11)),
                yaxis=dict(showgrid=True, gridcolor=CREAM_BORDER, gridwidth=0.5,
                          tickfont=dict(color=INK_DEEP, size=11)),
                margin=dict(l=40, r=20, t=20, b=40),
                font=dict(family="Inter", size=11, color=INK_DEEP),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                           font=dict(color=INK_DEEP, size=11))
            )
            st.plotly_chart(fig, use_container_width=True)
            
            fc_vals = forecast['forecast_values']
            hist_avg = np.mean(forecast['historical_values'])
            fc_avg = np.mean(fc_vals)
            change = fc_avg - hist_avg
            
            if lang == 'ar':
                trend = "ارتفاع" if change > 0.5 else "انخفاض" if change < -0.5 else "استقرار"
                fc_exp = f"يتوقع النموذج {trend} نمو GDP إلى متوسط {fc_avg:.2f}% خلال السنوات الخمس القادمة، مقابل متوسط تاريخي {hist_avg:.2f}%. النطاق الواسع لفترة الثقة ({forecast['ci_lower'][0]:.1f} إلى {forecast['ci_upper'][0]:.1f}) يعكس عدم اليقين الناتج عن قاعدة بيانات محدودة واعتماد عالٍ على تقلبات النفط."
            else:
                trend = "rise" if change > 0.5 else "decline" if change < -0.5 else "remain stable"
                fc_exp = f"The model projects GDP growth will {trend} to an average of {fc_avg:.2f}% over the next 5 years, compared to a historical average of {hist_avg:.2f}%. The wide confidence interval ({forecast['ci_lower'][0]:.1f} to {forecast['ci_upper'][0]:.1f}) reflects uncertainty from limited training data."
            
            # Sanity check on forecast
            extreme_forecast = abs(fc_avg) > 10 or abs(forecast['ci_upper'][0]) > 30 or abs(forecast['ci_lower'][0]) > 30
            warning_text = ""
            if extreme_forecast:
                if lang == 'ar':
                    warning_text = "<p style='color: #8B1F1F !important; font-weight: 600; font-size: 0.85rem; margin-top: 0.7rem;'>⚠️ تحذير: التنبؤ يُظهر قيماً متطرفة بسبب محدودية البيانات (10 سنوات فقط) وكسر كوفيد-19 الهيكلي في 2020. يُنصح بمعاملة هذه الأرقام كاستكشافية فقط.</p>"
                else:
                    warning_text = "<p style='color: #8B1F1F !important; font-weight: 600; font-size: 0.85rem; margin-top: 0.7rem;'>⚠️ Caution: Forecast shows extreme values due to limited data (10 years only) and the 2020 COVID structural break. Treat these numbers as exploratory only.</p>"
            
            st.markdown(f"""
            <div class="insight-panel" dir="{DIR}" style="text-align: {TEXT_ALIGN};">
                <h4>{t('forecast_explanation')}</h4>
                <p>{fc_exp}</p>
                {warning_text}
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown(f'<div class="section-title">{t("strategic_sig")}</div>', unsafe_allow_html=True)
        for signal in result['strategic_signals']:
            severity_class = {'elevated': 'signal-elevated', 'advisory': '', 'stable': 'signal-stable'}.get(signal['severity'], '')
            sev_label_map_en = {'elevated': 'ELEVATED', 'advisory': 'ADVISORY', 'stable': 'STABLE'}
            sev_label_map_ar = {'elevated': 'مرتفع', 'advisory': 'استشاري', 'stable': 'مستقر'}
            severity_label = sev_label_map_ar[signal['severity']] if lang == 'ar' else sev_label_map_en[signal['severity']]
            
            title = signal['title_ar'] if lang == 'ar' else signal['title']
            detail = signal['detail_ar'] if lang == 'ar' else signal['detail']
            sector = signal['sector_ar'] if lang == 'ar' else signal['sector']
            horizon = signal['horizon_ar'] if lang == 'ar' else signal['horizon']
            
            st.markdown(f"""
            <div class="insight-panel {severity_class}" dir="{DIR}" style="text-align: {TEXT_ALIGN};">
                <p style="font-size: 0.7rem; letter-spacing: 2px; margin: 0; color: {TEXT_MUTED} !important; font-weight: 700;">
                    {severity_label} · {sector} · {horizon}
                </p>
                <h4>{title}</h4>
                <p>{detail}</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown(f'<div class="section-title">{t("strategic_recs")}</div>', unsafe_allow_html=True)
        for rec in result['recommendations']:
            priority_color = {'High': RED, 'Medium': GOLD_DEEP, 'Low': GREEN}.get(rec['priority'], TEXT_BODY)
            pri_label = {'High': t('pri_high'), 'Medium': t('pri_med'), 'Low': t('pri_low')}.get(rec['priority'], '')
            
            title = rec['title_ar'] if lang == 'ar' else rec['title']
            detail = rec['detail_ar'] if lang == 'ar' else rec['detail']
            timeline = rec['timeline_ar'] if lang == 'ar' else rec['timeline']
            
            st.markdown(f"""
            <div class="recommendation-panel" dir="{DIR}" style="text-align: {TEXT_ALIGN};">
                <p style="font-size: 0.7rem; letter-spacing: 2px; margin: 0; color: {TEXT_MUTED} !important; font-weight: 700;">
                    <span style="color: {priority_color} !important; font-weight: 800;">{pri_label}</span> · {timeline}
                </p>
                <h4>{rec['icon']} {title}</h4>
                <p>{detail}</p>
            </div>
            """, unsafe_allow_html=True)
        
        method_note = result['method_note_ar'] if lang == 'ar' else result['method_note']
        st.markdown(f'<div class="method-note">{method_note}</div>', unsafe_allow_html=True)


# ============================================================
# 2. INTELLIGENCE WORKBENCH
# ============================================================
elif page == "Intelligence Workbench":
    page_header(t('analysis'), t('intelligence_workbench'),
                "Unified analytical tools" if lang == 'en' else "أدوات تحليلية موحدة")
    
    tabs = st.tabs([t('tab_explorer'), t('tab_forecasting'), t('tab_scenarios'), t('tab_deviations')])
    
    with tabs[0]:
        if lang == 'ar':
            st.markdown('<p class="section-desc">تصفّح وحلّل المؤشرات الفردية بالتفصيل.</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="section-desc">Browse and analyze individual indicators in detail.</p>', unsafe_allow_html=True)
        
        name_field = 'name_ar' if lang == 'ar' else 'name_en'
        if name_field not in indicators.columns:
            name_field = 'name_ar'  # fallback
        indicator_options = {f"{row['code']} — {row[name_field]}": row['code'] 
                             for _, row in indicators.iterrows()}
        selected = st.selectbox(t('select_indicator'), list(indicator_options.keys()), key="ind_select")
        selected_code = indicator_options[selected]
        
        ind_data = values[values['code'] == selected_code].sort_values('year').copy()
        ind_info = indicators[indicators['code'] == selected_code].iloc[0]
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric(t('pillar'), ind_info['pillar'])
        col2.metric(t('unit'), ind_info['unit'])
        col3.metric(t('sdg'), ind_info['sdg_alignment'])
        col4.metric(t('weight'), f"{ind_info['weight']*100:.0f}%")
        
        fig = px.line(ind_data, x='year', y='value', markers=True)
        fig.update_traces(line_color=INK_DEEP, line_width=2.5, marker_size=8, marker_color=GOLD_DEEP)
        fig.add_hline(y=ind_data['value'].mean(), line_dash="dot", line_color=GOLD_DEEP, line_width=1,
                      annotation_text=f"Mean: {ind_data['value'].mean():.2f}",
                      annotation_font_size=10, annotation_font_color=INK_DEEP)
        fig.update_layout(
            height=360, hovermode='x unified',
            plot_bgcolor=CREAM_CARD, paper_bgcolor=CREAM_CARD,
            xaxis=dict(showgrid=False, showline=True, linecolor=CREAM_BORDER, tickfont=dict(color=INK_DEEP)),
            yaxis=dict(showgrid=True, gridcolor=CREAM_BORDER, gridwidth=0.5, tickfont=dict(color=INK_DEEP)),
            margin=dict(l=40, r=20, t=20, b=40),
            font=dict(family="Inter", size=11, color=INK_DEEP)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric(t('mean'), f"{ind_data['value'].mean():.2f}")
        col2.metric(t('std_dev'), f"{ind_data['value'].std():.2f}")
        col3.metric(t('min'), f"{ind_data['value'].min():.2f}")
        col4.metric(t('max'), f"{ind_data['value'].max():.2f}")
    
    with tabs[1]:
        if lang == 'ar':
            st.markdown('<p class="section-desc">تنبؤ SARIMAX مع برنت كمتغير خارجي. التنبؤات استكشافية واتجاهية.</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="section-desc">SARIMAX forecasting with Brent as exogenous regressor. Forecasts are exploratory.</p>', unsafe_allow_html=True)
        
        econ_options = ['ECO_GDP', 'ECO_INFLAT', 'ECO_OIL_RENT']
        forecast_target = st.selectbox(t('target_ind'), econ_options,
            format_func=lambda x: f"{x} — {indicators[indicators['code']==x].iloc[0][name_field if 'name_field' in dir() else 'name_ar']}",
            key="forecast_target")
        n_forecast = st.slider(t('forecast_horizon'), 1, 5, 3, key="forecast_horizon")
        
        if st.button("Run" if lang == 'en' else "تشغيل", type="primary", key="run_forecast"):
            with st.spinner("Training SARIMAX..."):
                try:
                    target_data = values[values['code'] == forecast_target].sort_values('year')
                    brent_sorted = brent.sort_values('year')
                    merged = pd.merge(
                        target_data[['year', 'value']].rename(columns={'value': 'target'}),
                        brent_sorted[['year', 'value']].rename(columns={'value': 'brent'}), on='year')
                    y_train = merged['target'].values
                    exog_train = merged['brent'].values.reshape(-1, 1)
                    model = SARIMAX(y_train, exog=exog_train, order=(1, 1, 1))
                    fitted = model.fit(disp=False)
                    future_brent = np.array([brent_sorted['value'].iloc[-3:].mean()] * n_forecast).reshape(-1, 1)
                    forecast = fitted.forecast(steps=n_forecast, exog=future_brent)
                    ci = fitted.get_forecast(steps=n_forecast, exog=future_brent).conf_int(alpha=0.05)
                    future_years = list(range(int(merged['year'].iloc[-1]) + 1, int(merged['year'].iloc[-1]) + 1 + n_forecast))
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=merged['year'], y=y_train, mode='lines+markers',
                                             name='Historical', line=dict(color=INK_DEEP, width=2.5)))
                    fig.add_trace(go.Scatter(x=future_years, y=forecast, mode='lines+markers',
                                             name='Forecast', line=dict(color=GOLD_DEEP, width=2.5, dash='dot')))
                    fig.add_trace(go.Scatter(x=future_years + future_years[::-1],
                                             y=list(ci[:, 1]) + list(ci[:, 0][::-1]),
                                             fill='toself', fillcolor='rgba(139,111,20,0.15)',
                                             line=dict(color='rgba(255,255,255,0)'), name='95% CI'))
                    fig.update_layout(
                        height=400, plot_bgcolor=CREAM_CARD, paper_bgcolor=CREAM_CARD,
                        xaxis=dict(showgrid=False, showline=True, linecolor=CREAM_BORDER, tickfont=dict(color=INK_DEEP)),
                        yaxis=dict(showgrid=True, gridcolor=CREAM_BORDER, gridwidth=0.5, tickfont=dict(color=INK_DEEP)),
                        margin=dict(l=40, r=20, t=20, b=40),
                        font=dict(family="Inter", size=11, color=INK_DEEP)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    forecast_df = pd.DataFrame({
                        'Year': future_years,
                        'Forecast': [f"{v:.2f}" for v in forecast],
                        'Lower 95%': [f"{v:.2f}" for v in ci[:, 0]],
                        'Upper 95%': [f"{v:.2f}" for v in ci[:, 1]]
                    })
                    st.dataframe(forecast_df, use_container_width=True, hide_index=True)
                    
                    hist_avg = np.mean(y_train)
                    fc_avg = np.mean(forecast)
                    if lang == 'ar':
                        trend = "ارتفاع" if fc_avg > hist_avg + 0.5 else "انخفاض" if fc_avg < hist_avg - 0.5 else "استقرار"
                        exp = f"يتوقع النموذج {trend} {forecast_target} إلى متوسط {fc_avg:.2f} (تاريخياً: {hist_avg:.2f})."
                    else:
                        trend = "rise" if fc_avg > hist_avg + 0.5 else "decline" if fc_avg < hist_avg - 0.5 else "stabilize"
                        exp = f"The model projects {forecast_target} will {trend} to an average of {fc_avg:.2f} (historical: {hist_avg:.2f})."
                    
                    st.markdown(f"""
                    <div class="insight-panel" dir="{DIR}" style="text-align: {TEXT_ALIGN};">
                        <h4>{t('forecast_explanation')}</h4>
                        <p>{exp}</p>
                        <p style="font-size: 0.8rem;">Model AIC: {fitted.aic:.2f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error: {e}")
    
    with tabs[2]:
        st.markdown(f'<p class="section-desc">{t("scenario_desc")}</p>', unsafe_allow_html=True)
        
        corr_data = compute_indicator_correlations(values)
        corr_matrix = corr_data['matrix']
        
        all_codes = list(indicators['code'].values)
        name_f = 'name_ar' if lang == 'ar' else ('name_en' if 'name_en' in indicators.columns else 'name_ar')
        ind_labels = {row['code']: f"{row['code']} — {row[name_f]}" for _, row in indicators.iterrows()}
        
        col1, col2 = st.columns(2)
        with col1:
            driver = st.selectbox(t('driver_var'), all_codes,
                format_func=lambda x: ind_labels.get(x, x), key="scenario_driver")
        with col2:
            change_pct = st.slider(t('change_pct'), -50, 50, 10, 5, key="scenario_change")
        
        available_impacts = [c for c in all_codes if c != driver]
        impacted = st.multiselect(t('impacted_inds'), available_impacts,
            default=available_impacts[:4],
            format_func=lambda x: ind_labels.get(x, x), key="scenario_impacted")
        
        if st.button(t('run_scenario'), type="primary", key="run_scenario"):
            if impacted:
                driver_current = values[values['code'] == driver].sort_values('year').iloc[-1]['value']
                driver_new = driver_current * (1 + change_pct/100)
                driver_change_abs = driver_new - driver_current
                
                results = []
                for ind in impacted:
                    if ind not in corr_matrix.columns or driver not in corr_matrix.columns:
                        continue
                    r = corr_matrix.loc[driver, ind]
                    if pd.isna(r):
                        continue
                    
                    ind_current = values[values['code'] == ind].sort_values('year').iloc[-1]['value']
                    ind_std = values[values['code'] == ind]['value'].std()
                    driver_std = values[values['code'] == driver]['value'].std()
                    
                    if driver_std > 0:
                        projected_change = r * (driver_change_abs / driver_std) * ind_std
                    else:
                        projected_change = 0
                    
                    projected_value = ind_current + projected_change
                    
                    results.append({
                        'code': ind, 'name': ind_labels.get(ind, ind),
                        'correlation': r, 'current': ind_current,
                        'change': projected_change, 'projected': projected_value
                    })
                
                st.markdown(f"""
                <div class="insight-panel">
                    <h4>{ind_labels.get(driver, driver)}</h4>
                    <p>{t('current_val')}: <strong>{driver_current:.2f}</strong> → {t('projected_value')}: <strong>{driver_new:.2f}</strong> ({change_pct:+}%)</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f'<div class="section-title">{t("scenario_results")}</div>', unsafe_allow_html=True)
                
                for r in results:
                    color = GREEN if abs(r['correlation']) > 0.7 else GOLD_DEEP if abs(r['correlation']) > 0.4 else TEXT_BODY
                    interp = interpret_correlation(r['correlation'], driver, r['code'], lang)
                    
                    st.markdown(f"""
                    <div class="systemic-box" dir="{DIR}" style="text-align: {TEXT_ALIGN};">
                        <h4>{r['name']}</h4>
                        <div class="systemic-row">
                            <div class="systemic-pair">{t('historical_corr')}: <span style="color: {color} !important; font-weight: 800;">r = {r['correlation']:+.2f}</span></div>
                            <div class="systemic-pair">{t('current_val')}: <strong>{r['current']:.2f}</strong></div>
                            <div class="systemic-pair">{t('projected_impact')}: <strong style="color: {color} !important;">{r['change']:+.2f}</strong></div>
                            <div class="systemic-pair">{t('projected_value')}: <strong>{r['projected']:.2f}</strong></div>
                        </div>
                        <p style="margin: 0.5rem 0 0 0; font-size: 0.85rem;">{interp}</p>
                    </div>
                    """, unsafe_allow_html=True)
    
    with tabs[3]:
        if lang == 'ar':
            st.markdown('<p class="section-desc">كشف القيم الشاذة باستخدام Isolation Forest.</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="section-desc">Statistical outlier detection using Isolation Forest.</p>', unsafe_allow_html=True)
        
        if st.button(t('detect_btn'), type="primary", key="detect_btn"):
            with st.spinner("Analyzing..."):
                anomalies_found = []
                for code in indicators['code']:
                    ind_data = values[values['code'] == code].sort_values('year').copy()
                    if len(ind_data) < 5:
                        continue
                    X = ind_data[['value']].values
                    iso = IsolationForest(contamination=0.15, random_state=42)
                    ind_data['anomaly'] = iso.fit_predict(X)
                    ind_data['anomaly_score'] = iso.score_samples(X)
                    for _, row in ind_data[ind_data['anomaly'] == -1].iterrows():
                        anomalies_found.append({
                            'Indicator': code, 'Name': row['name_ar'],
                            'Year': int(row['year']), 'Value': float(row['value']),
                            'Deviation Score': round(float(row['anomaly_score']), 3)
                        })
                
                if anomalies_found:
                    msg = f"تم اكتشاف {len(anomalies_found)} انحراف" if lang == 'ar' else f"Detected {len(anomalies_found)} pattern deviations"
                    st.markdown(f"<p style='color: {TEXT_BODY} !important; font-weight: 600;'>{msg}</p>", unsafe_allow_html=True)
                    df_anom = pd.DataFrame(anomalies_found).sort_values('Deviation Score')
                    st.dataframe(df_anom, use_container_width=True, hide_index=True)
                    
                    covid_count = len(df_anom[df_anom['Year'] == 2020])
                    if covid_count > 0:
                        if lang == 'ar':
                            msg = f"تجمعت {covid_count} انحرافات في 2020، متسقة مع كسر كوفيد-19 الهيكلي."
                        else:
                            msg = f"{covid_count} deviations concentrated in 2020, consistent with the documented COVID-19 structural break."
                        st.markdown(f"""
                        <div class="insight-panel">
                            <h4>{'كشف التجمعات' if lang == 'ar' else 'Cluster Detection'}</h4>
                            <p>{msg}</p>
                        </div>
                        """, unsafe_allow_html=True)


# ============================================================
# 3. ABOUT
# ============================================================
elif page == "About":
    page_header(t('system'), t('about_title'), t('about_subtitle'))
    
    st.markdown(f"""
    <div class="insight-panel" dir="{DIR}" style="text-align: {TEXT_ALIGN};">
        <h4>{t('what_is')}</h4>
        <p>{t('what_is_text')}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="systemic-box" dir="{DIR}" style="text-align: {TEXT_ALIGN};">
        <h4>{t('how_it_works')}</h4>
        <p>{t('how_step1')}</p>
        <p>{t('how_step2')}</p>
        <p>{t('how_step3')}</p>
        <p>{t('how_step4')}</p>
        <p>{t('how_step5')}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="interpretation-panel" dir="{DIR}" style="text-align: {TEXT_ALIGN};">
        <p class="eyebrow">{t('philosophy')}</p>
        <p class="narrative">{t('philosophy_text')}</p>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# 4. EXECUTIVE BRIEF
# ============================================================
elif page == "Executive Brief":
    page_header("", t('executive_brief'),
                "Kuwait 2024 strategic snapshot" if lang == 'en' else "لقطة استراتيجية للكويت 2024")
    
    gdp_data = values[values['code'] == 'ECO_GDP'].sort_values('year')
    inflat_data = values[values['code'] == 'ECO_INFLAT'].sort_values('year')
    renew_data = values[values['code'] == 'ENV_RENEW'].sort_values('year')
    latest_gdp = gdp_data.iloc[-1]['value']
    latest_inflat = inflat_data.iloc[-1]['value']
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("INDICATORS" if lang == 'en' else "المؤشرات", f"{len(indicators)}")
    col2.metric("DATA POINTS" if lang == 'en' else "نقاط البيانات", f"{len(values)}")
    col3.metric("GDP GROWTH" if lang == 'en' else "نمو GDP", f"{latest_gdp:.1f}%", "2024")
    col4.metric("INFLATION" if lang == 'en' else "التضخم", f"{latest_inflat:.1f}%", "2024")
    
    if lang == 'ar':
        st.markdown(f"""
        <div class="insight-panel" dir="rtl" style="text-align: right;">
            <h4>{t('current_state')}</h4>
            <ul>
                <li><strong>نمو GDP:</strong> {latest_gdp:.2f}% · {'مسار إيجابي' if latest_gdp > 0 else 'انكماش'}</li>
                <li><strong>التضخم:</strong> {latest_inflat:.2f}% · {'ضمن النطاق المعتاد' if latest_inflat < 4 else 'قراءة مرتفعة'}</li>
                <li><strong>حصة الطاقة المتجددة:</strong> {renew_data.iloc[-1]['value']:.2f}% · نمو قوي لكن دون مسار 2035</li>
            </ul>
        </div>
        <div class="recommendation-panel" dir="rtl" style="text-align: right;">
            <h4>{t('strategic_cons')}</h4>
            <ol>
                <li><strong>التنويع الاقتصادي:</strong> ريع النفط (27-44% من GDP) يُبقي التعرض لدورات السلع.</li>
                <li><strong>تسريع المتجددة:</strong> هدف 2035 (15%) يتطلب توسعاً مستداماً.</li>
                <li><strong>احتياطيات الاستقرار:</strong> دروس 2020 (-21.16% GDP) تُبرز قيمة المعاكسة للدورات.</li>
                <li><strong>رأس المال البشري:</strong> الاستثمار المستدام في التعليم والصحة.</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="insight-panel">
            <h4>{t('current_state')}</h4>
            <ul>
                <li><strong>GDP Growth:</strong> {latest_gdp:.2f}% · {'Positive trajectory' if latest_gdp > 0 else 'Contraction observed'}</li>
                <li><strong>Inflation:</strong> {latest_inflat:.2f}% · {'Within typical range' if latest_inflat < 4 else 'Elevated reading'}</li>
                <li><strong>Renewable Share:</strong> {renew_data.iloc[-1]['value']:.2f}% · Strong growth but below 2035 target trajectory</li>
            </ul>
        </div>
        <div class="recommendation-panel">
            <h4>{t('strategic_cons')}</h4>
            <ol>
                <li><strong>Economic diversification:</strong> Oil rent (27-44% of GDP) sustains exposure to commodity cycles.</li>
                <li><strong>Renewable acceleration:</strong> 2035 target of 15% requires sustained expansion.</li>
                <li><strong>Stabilization reserves:</strong> 2020 lessons (-21.16% GDP) underscore countercyclical value.</li>
                <li><strong>Human capital:</strong> Sustained education and health investment supports productivity.</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown(f'<div class="section-title">{t("data_gov")}</div>', unsafe_allow_html=True)
    
    if lang == 'ar':
        gov_html = f"""
        <div class="governance-card" dir="rtl" style="text-align: right;">
            <div class="governance-row"><div class="label">{t('sources')}</div><div class="value">البنك الدولي · IMF · EIA · Our World in Data</div></div>
            <div class="governance-row"><div class="label">{t('coverage')}</div><div class="value">12 مؤشر · 2015–2024 · الكويت · سنوي</div></div>
            <div class="governance-row"><div class="label">{t('update_cadence')}</div><div class="value">مواءمة يدوية · يُحدّث عند إصدار المصدر</div></div>
            <div class="governance-row"><div class="label">{t('validation')}</div><div class="value">تحقق متعدد المصادر · بيانات وصفية SDMX · كوفيد-19 ككسر هيكلي</div></div>
            <div class="governance-row"><div class="label">{t('missing_data')}</div><div class="value">استكمال بالاستيفاء · موثقة في ورقة المنهجية</div></div>
            <div class="governance-row"><div class="label">{t('stat_limits')}</div><div class="value">نافذة 10 سنوات تحدّ من التنبؤ طويل الأمد</div></div>
            <div class="governance-row"><div class="label">{t('disclaimer')}</div><div class="value">بنية تحليلية إرشادية · ليست بديلاً عن الإحصاءات الرسمية</div></div>
        </div>
        """
    else:
        gov_html = f"""
        <div class="governance-card">
            <div class="governance-row"><div class="label">{t('sources')}</div><div class="value">World Bank · IMF · U.S. EIA · Our World in Data</div></div>
            <div class="governance-row"><div class="label">{t('coverage')}</div><div class="value">12 indicators · 2015–2024 · Kuwait · Annual frequency</div></div>
            <div class="governance-row"><div class="label">{t('update_cadence')}</div><div class="value">Manual harmonization · refreshed at source release</div></div>
            <div class="governance-row"><div class="label">{t('validation')}</div><div class="value">Cross-source verification · SDMX-aligned metadata · COVID-19 flagged as structural break</div></div>
            <div class="governance-row"><div class="label">{t('missing_data')}</div><div class="value">Interpolated where gaps exist · documented in methodology</div></div>
            <div class="governance-row"><div class="label">{t('stat_limits')}</div><div class="value">10-year window limits long-horizon forecast precision</div></div>
            <div class="governance-row"><div class="label">{t('disclaimer')}</div><div class="value">Indicative analytical infrastructure · not a substitute for official statistics</div></div>
        </div>
        """
    st.markdown(gov_html, unsafe_allow_html=True)


# ============================================================
# Footer
# ============================================================
st.markdown(f"""
<div class="footer-block">
    <p style="margin: 0; letter-spacing: 1px;"><strong>{t('tagline')}</strong> · V1.0</p>
    <p style="margin: 0.5rem 0; font-size: 0.72rem;">
        Linda Waleed Alshaitan · Data: World Bank · IMF · EIA
    </p>
</div>
""", unsafe_allow_html=True)
