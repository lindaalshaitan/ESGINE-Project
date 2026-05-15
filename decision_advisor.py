"""
ESGINE Decision Intelligence Workflow
======================================
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# 1. Decision Domain Classification (Enhanced)
# ============================================================

DECISION_KEYWORDS = {
    'energy': {
        'ar': ['طاقة', 'كهرباء', 'شمسي', 'متجدد', 'هيدروجين', 'وقود', 'تحلية', 'نفط', 'غاز', 'بنزين'],
        'en': ['energy', 'electricity', 'solar', 'renewable', 'hydrogen', 'fuel', 'desalination', 'oil', 'gas', 'petroleum'],
        'indicators': ['ENV_RENEW', 'ECO_OIL_RENT', 'ENV_CO2', 'ENV_TEMP']
    },
    'economic': {
        'ar': ['اقتصاد', 'استثمار', 'ضرائب', 'دعم', 'ميزانية', 'صندوق', 'تضخم', 'تنويع', 'تجارة', 'سوق', 'مالية', 'صادرات', 'واردات'],
        'en': ['economic', 'economy', 'investment', 'tax', 'subsidy', 'budget', 'fund', 'inflation', 'diversification', 'trade', 'market', 'finance', 'export', 'import', 'fiscal'],
        'indicators': ['ECO_GDP', 'ECO_INFLAT', 'ECO_OIL_RENT', 'ECO_POP']
    },
    'social': {
        'ar': ['تعليم', 'صحة', 'بطالة', 'توظيف', 'سكان', 'وافدين', 'مواطنين', 'تكويتة', 'نساء', 'مرأة', 'شباب', 'عمال', 'عمل', 'وظائف', 'تدريب', 'مدارس', 'جامعات', 'مستشفيات', 'إسكان', 'تقاعد', 'رواتب', 'أجور'],
        'en': ['education', 'health', 'unemployment', 'employment', 'population', 'expat', 'citizen', 'women', 'female', 'gender', 'youth', 'workers', 'labor', 'jobs', 'training', 'schools', 'universities', 'hospitals', 'housing', 'pension', 'salary', 'wages', 'social', 'workforce'],
        'indicators': ['SOC_UNEMP_KW', 'SOC_UNEMP_EX', 'SOC_HEALTH', 'SOC_EDU']
    },
    'environmental': {
        'ar': ['بيئة', 'انبعاثات', 'كربون', 'مناخ', 'حرارة', 'كوارث', 'استدامة', 'تلوث', 'مياه', 'هواء', 'نفايات', 'إعادة تدوير'],
        'en': ['environment', 'environmental', 'emissions', 'carbon', 'climate', 'temperature', 'disaster', 'sustainability', 'pollution', 'water', 'air', 'waste', 'recycling'],
        'indicators': ['ENV_CO2', 'ENV_TEMP', 'ENV_DISASTER', 'ENV_RENEW']
    }
}


def classify_decision(decision_text):
    """تصنيف القرار حسب المجال السائد."""
    decision_lower = decision_text.lower()
    scores = {}
    
    for category, data in DECISION_KEYWORDS.items():
        score = 0
        for keyword in data['ar'] + data['en']:
            if keyword.lower() in decision_lower:
                score += 1
        scores[category] = score
    
    sorted_cats = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    if sorted_cats[0][1] == 0:
        primary = 'economic'
        relevant_indicators = ['ECO_GDP', 'ECO_INFLAT', 'ENV_RENEW', 'SOC_UNEMP_KW']
    else:
        primary = sorted_cats[0][0]
        relevant_indicators = DECISION_KEYWORDS[primary]['indicators']
    
    return {
        'primary_category': primary,
        'scores': scores,
        'relevant_indicators': relevant_indicators
    }


# ============================================================
# 2. Decision Signals
# ============================================================

def calculate_decision_signals(decision_text, values, brent):
    """حساب 4 إشارات تحليلية بناءً على البيانات التاريخية."""
    
    env_data = values[values['pillar'] == 'Environmental'].copy()
    renew_latest = env_data[env_data['code'] == 'ENV_RENEW']['value'].iloc[-1] if len(env_data[env_data['code'] == 'ENV_RENEW']) > 0 else 0
    co2_latest = env_data[env_data['code'] == 'ENV_CO2']['value'].iloc[-1] if len(env_data[env_data['code'] == 'ENV_CO2']) > 0 else 25
    esg_score = min(100, (renew_latest * 20) + ((30 - co2_latest) * 2.5) + 60)
    esg_score = max(0, esg_score)
    
    gdp_data = values[values['code'] == 'ECO_GDP'].sort_values('year')
    gdp_volatility = gdp_data['value'].std()
    if gdp_volatility > 10:
        eco_label = 'Volatile'
        eco_score = 75
    elif gdp_volatility > 5:
        eco_label = 'Moderate'
        eco_score = 50
    else:
        eco_label = 'Stable'
        eco_score = 25
    
    unemp_latest = values[values['code'] == 'SOC_UNEMP_KW'].sort_values('year')['value'].iloc[-1]
    if unemp_latest < 3:
        social_label = 'Favorable'
        social_score = 85
    elif unemp_latest < 5:
        social_label = 'Moderate'
        social_score = 60
    else:
        social_label = 'Elevated'
        social_score = 35
    
    temp_data = values[values['code'] == 'ENV_TEMP'].sort_values('year')
    temp_avg_recent = temp_data['value'].tail(3).mean()
    env_trend_score = min(100, max(0, 30 + (temp_avg_recent * 25)))
    
    return {
        'esg_signal': {
            'score': round(esg_score, 1),
            'label': 'ESG Signal',
            'label_ar': 'إشارة ESG',
            'unit': '/100',
            'trend': f'+{renew_latest:.2f}' if renew_latest > 1 else f'-{co2_latest/10:.1f}',
            'color': 'green' if esg_score > 70 else 'gold' if esg_score > 50 else 'red',
            'interpretation': 'Strong sustainability footprint' if esg_score > 70 else 'Moderate sustainability footprint' if esg_score > 50 else 'Sustainability signals warrant attention',
            'interpretation_ar': 'بصمة استدامة قوية' if esg_score > 70 else 'بصمة استدامة متوسطة' if esg_score > 50 else 'إشارات الاستدامة تستدعي الانتباه'
        },
        'economic_outlook': {
            'score': eco_label,
            'label': 'Economic Outlook',
            'label_ar': 'الأفق الاقتصادي',
            'numeric': eco_score,
            'trend': f'σ={gdp_volatility:.1f}',
            'color': 'red' if eco_label == 'Volatile' else 'gold' if eco_label == 'Moderate' else 'green',
            'interpretation': f'GDP volatility (σ={gdp_volatility:.1f}) suggests {eco_label.lower()} macroeconomic conditions',
            'interpretation_ar': f'تقلبات الناتج المحلي (σ={gdp_volatility:.1f}) تشير إلى ظروف اقتصادية {("متقلبة" if eco_label == "Volatile" else "متوسطة" if eco_label == "Moderate" else "مستقرة")}'
        },
        'social_conditions': {
            'score': social_label,
            'label': 'Social Conditions',
            'label_ar': 'الأوضاع الاجتماعية',
            'numeric': social_score,
            'trend': f'{unemp_latest:.1f}% unemployment',
            'color': 'green' if social_label == 'Favorable' else 'gold' if social_label == 'Moderate' else 'red',
            'interpretation': f'Labor market indicators ({unemp_latest:.1f}% unemployment) reflect {social_label.lower()} conditions',
            'interpretation_ar': f'مؤشرات سوق العمل (بطالة {unemp_latest:.1f}%) تعكس ظروفاً {("مواتية" if social_label == "Favorable" else "متوسطة" if social_label == "Moderate" else "مرتفعة")}'
        },
        'environmental_trend': {
            'score': round(env_trend_score, 1),
            'label': 'Environmental Trend',
            'label_ar': 'الاتجاه البيئي',
            'unit': '/100',
            'trend': f'+{temp_avg_recent:.2f}°C',
            'color': 'red' if env_trend_score > 70 else 'gold' if env_trend_score > 50 else 'green',
            'interpretation': f'Recent temperature change (+{temp_avg_recent:.2f}°C) signals warming pressure',
            'interpretation_ar': f'تغير الحرارة الأخير (+{temp_avg_recent:.2f}°C) يشير إلى ضغط احتراري'
        }
    }


# ============================================================
# 3. Exploratory Forecasting
# ============================================================

def generate_forecast(values, brent, target_code='ECO_GDP', n_years=5):
    """تنبؤ استكشافي باستخدام SARIMAX."""
    try:
        target_data = values[values['code'] == target_code].sort_values('year')
        brent_sorted = brent.sort_values('year')
        
        merged = pd.merge(
            target_data[['year', 'value']].rename(columns={'value': 'target'}),
            brent_sorted[['year', 'value']].rename(columns={'value': 'brent'}),
            on='year'
        )
        
        if len(merged) < 8:
            return None
        
        y_train = merged['target'].values
        exog_train = merged['brent'].values.reshape(-1, 1)
        
        model = SARIMAX(y_train, exog=exog_train, order=(1, 1, 1))
        fitted = model.fit(disp=False)
        
        future_brent = np.array([brent_sorted['value'].iloc[-3:].mean()] * n_years).reshape(-1, 1)
        forecast = fitted.forecast(steps=n_years, exog=future_brent)
        forecast_obj = fitted.get_forecast(steps=n_years, exog=future_brent)
        ci = forecast_obj.conf_int(alpha=0.05)
        
        future_years = list(range(int(merged['year'].iloc[-1]) + 1, 
                                   int(merged['year'].iloc[-1]) + 1 + n_years))
        
        return {
            'historical_years': merged['year'].tolist(),
            'historical_values': y_train.tolist(),
            'forecast_years': future_years,
            'forecast_values': forecast.tolist(),
            'ci_lower': ci[:, 0].tolist(),
            'ci_upper': ci[:, 1].tolist(),
            'aic': float(fitted.aic),
            'method': 'SARIMAX(1,1,1) with Brent as exogenous regressor',
            'note': 'Exploratory directional forecast — limited training data',
            'confidence_note': '95% confidence interval shown for directional interpretation only'
        }
    except Exception as e:
        return None


# ============================================================
# 4. Indicator Correlations
# ============================================================

def compute_indicator_correlations(values):
    """حساب مصفوفة الترابطات بين المؤشرات."""
    pivot = values.pivot_table(
        index='year', columns='code', values='value', aggfunc='first'
    )
    
    corr_matrix = pivot.corr(min_periods=5)
    
    correlations = []
    seen_pairs = set()
    
    for code1 in corr_matrix.columns:
        for code2 in corr_matrix.columns:
            if code1 == code2:
                continue
            pair = tuple(sorted([code1, code2]))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            
            corr_val = corr_matrix.loc[code1, code2]
            if pd.notna(corr_val):
                correlations.append({
                    'indicator_1': code1,
                    'indicator_2': code2,
                    'correlation': round(corr_val, 3),
                    'strength': 'strong' if abs(corr_val) > 0.7 else 'moderate' if abs(corr_val) > 0.4 else 'weak',
                    'direction': 'positive' if corr_val > 0 else 'negative'
                })
    
    correlations_sorted = sorted(correlations, key=lambda x: abs(x['correlation']), reverse=True)
    
    return {
        'matrix': corr_matrix,
        'top_correlations': correlations_sorted[:5]
    }


def get_relevant_correlations(values, relevant_indicators, top_n=5):
    """ترابطات للمؤشرات ذات الصلة بالقرار فقط."""
    pivot = values.pivot_table(
        index='year', columns='code', values='value', aggfunc='first'
    )
    
    corr_matrix = pivot.corr(min_periods=5)
    
    correlations = []
    seen_pairs = set()
    
    for code1 in relevant_indicators:
        if code1 not in corr_matrix.columns:
            continue
        for code2 in corr_matrix.columns:
            if code1 == code2:
                continue
            pair = tuple(sorted([code1, code2]))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            
            corr_val = corr_matrix.loc[code1, code2]
            if pd.notna(corr_val):
                correlations.append({
                    'indicator_1': code1,
                    'indicator_2': code2,
                    'correlation': round(corr_val, 3),
                    'strength': 'strong' if abs(corr_val) > 0.7 else 'moderate' if abs(corr_val) > 0.4 else 'weak',
                    'direction': 'positive' if corr_val > 0 else 'negative'
                })
    
    correlations_sorted = sorted(correlations, key=lambda x: abs(x['correlation']), reverse=True)
    return correlations_sorted[:top_n]


# ============================================================
# 5. Strategic Signals
# ============================================================

def detect_strategic_signals(values, decision_category):
    """رصد الإشارات الاستراتيجية."""
    signals = []
    
    unemp_data = values[values['code'] == 'SOC_UNEMP_KW'].sort_values('year')
    if len(unemp_data) > 0:
        latest_unemp = unemp_data['value'].iloc[-1]
        if latest_unemp > 3:
            signals.append({
                'severity': 'elevated' if latest_unemp > 4 else 'advisory',
                'sector': 'Labor Market',
                'sector_ar': 'سوق العمل',
                'title': 'Citizen unemployment above benchmark',
                'title_ar': 'بطالة المواطنين فوق المعدل المرجعي',
                'detail': f'Latest reading: {latest_unemp:.2f}% in {int(unemp_data["year"].iloc[-1])}. May require further policy evaluation.',
                'detail_ar': f'القراءة الأخيرة: {latest_unemp:.2f}% في {int(unemp_data["year"].iloc[-1])}. قد يستدعي مراجعة سياسات.',
                'horizon': '12-month window',
                'horizon_ar': 'نافذة 12 شهراً'
            })
    
    gdp_data = values[values['code'] == 'ECO_GDP'].sort_values('year')
    if len(gdp_data) > 0:
        latest_gdp = gdp_data['value'].iloc[-1]
        if latest_gdp < 0:
            signals.append({
                'severity': 'elevated',
                'sector': 'Macroeconomy',
                'sector_ar': 'الاقتصاد الكلي',
                'title': 'GDP contraction observed',
                'title_ar': 'انكماش في الناتج المحلي',
                'detail': f'Reading: {latest_gdp:.2f}% in {int(gdp_data["year"].iloc[-1])}. Warrants closer policy monitoring.',
                'detail_ar': f'القراءة: {latest_gdp:.2f}% في {int(gdp_data["year"].iloc[-1])}. يستدعي مراقبة سياسية أدق.',
                'horizon': 'Near-term attention',
                'horizon_ar': 'متابعة قريبة المدى'
            })
    
    renew_data = values[values['code'] == 'ENV_RENEW'].sort_values('year')
    if len(renew_data) >= 2:
        renew_latest = renew_data['value'].iloc[-1]
        renew_growth = renew_latest / renew_data['value'].iloc[0] if renew_data['value'].iloc[0] > 0 else 0
        if renew_growth > 100:
            signals.append({
                'severity': 'stable',
                'sector': 'Energy Transition',
                'sector_ar': 'تحول الطاقة',
                'title': 'Renewable capacity expansion sustained',
                'title_ar': 'توسع متواصل في الطاقة المتجددة',
                'detail': f'Capacity expanded {renew_growth:.0f}x since 2015. On constructive trajectory.',
                'detail_ar': f'توسعت السعة بـ {renew_growth:.0f}x منذ 2015. مسار بنّاء.',
                'horizon': 'Multi-year trend',
                'horizon_ar': 'اتجاه متعدد السنوات'
            })
    
    if len(signals) < 3:
        signals.append({
            'severity': 'advisory',
            'sector': 'Sustainability',
            'sector_ar': 'الاستدامة',
            'title': 'Renewable share below 2035 target trajectory',
            'title_ar': 'حصة الطاقة المتجددة دون مسار 2035',
            'detail': 'Current share remains below the 15% target. Acceleration would benefit alignment.',
            'detail_ar': 'الحصة الحالية دون هدف 15%. التسارع سيُحسّن المحاذاة.',
            'horizon': 'Medium-term',
            'horizon_ar': 'متوسط الأمد'
        })
    
    return signals[:3]


# ============================================================
# 6. Strategic Recommendations
# ============================================================

def generate_recommendations(decision_text, classification, signals, values):
    """توليد 3 توصيات استراتيجية."""
    category = classification['primary_category']
    
    recs_by_cat = {
        'energy': [
            {'priority': 'High', 'priority_ar': 'عالية',
             'title': 'Renewable capacity acceleration',
             'title_ar': 'تسريع سعة الطاقة المتجددة',
             'detail': 'Doubling solar investment may improve alignment with 15% target by 2035.',
             'detail_ar': 'مضاعفة الاستثمار الشمسي قد يُحسّن المحاذاة مع هدف 15% بحلول 2035.',
             'timeline': '6-12 months', 'timeline_ar': '6-12 شهراً', 'icon': '☀️'},
            {'priority': 'Medium', 'priority_ar': 'متوسطة',
             'title': 'Hydrocarbon revenue diversification',
             'title_ar': 'تنويع إيرادات الهيدروكربون',
             'detail': 'Oil rent (27-44% of GDP) sustains exposure to global price cycles.',
             'detail_ar': 'ريع النفط (27-44% من GDP) يُبقي تعرضاً لدورات الأسعار العالمية.',
             'timeline': '2-3 years', 'timeline_ar': '2-3 سنوات', 'icon': '🛢️'},
            {'priority': 'High', 'priority_ar': 'عالية',
             'title': 'Energy efficiency standards',
             'title_ar': 'معايير كفاءة الطاقة',
             'detail': 'Applying efficiency standards in public buildings could yield 12-20% savings.',
             'detail_ar': 'تطبيق معايير الكفاءة في المباني العامة قد يحقق 12-20% توفيراً.',
             'timeline': 'Near-term', 'timeline_ar': 'قريب الأمد', 'icon': '⚡'}
        ],
        'economic': [
            {'priority': 'High', 'priority_ar': 'عالية',
             'title': 'Strengthen stabilization mechanisms',
             'title_ar': 'تعزيز آليات الاستقرار',
             'detail': '2020 contraction (-21%) underscores value of robust countercyclical reserves.',
             'detail_ar': 'انكماش 2020 (-21%) يُبرز قيمة الاحتياطيات المعاكسة للدورات.',
             'timeline': 'Near-term', 'timeline_ar': 'قريب الأمد', 'icon': '💰'},
            {'priority': 'High', 'priority_ar': 'عالية',
             'title': 'Revenue base diversification',
             'title_ar': 'تنويع قاعدة الإيرادات',
             'detail': 'Reducing oil dependency would moderate macroeconomic volatility.',
             'detail_ar': 'تقليل الاعتماد على النفط قد يُخفّف تقلبات الاقتصاد الكلي.',
             'timeline': '2-3 years', 'timeline_ar': '2-3 سنوات', 'icon': '📊'},
            {'priority': 'Medium', 'priority_ar': 'متوسطة',
             'title': 'Private sector activation',
             'title_ar': 'تنشيط القطاع الخاص',
             'detail': 'Entrepreneurship incentives could reduce public-sector employment concentration.',
             'detail_ar': 'حوافز ريادة الأعمال قد تُقلل تركّز التوظيف في القطاع العام.',
             'timeline': '1-2 years', 'timeline_ar': '1-2 سنة', 'icon': '🏢'}
        ],
        'social': [
            {'priority': 'High', 'priority_ar': 'عالية',
             'title': 'Human capital investment',
             'title_ar': 'الاستثمار في رأس المال البشري',
             'detail': 'Sustained education and health spending supports long-term productivity.',
             'detail_ar': 'الإنفاق المستدام على التعليم والصحة يدعم الإنتاجية طويلة الأمد.',
             'timeline': '1-3 years', 'timeline_ar': '1-3 سنوات', 'icon': '🎓'},
            {'priority': 'High', 'priority_ar': 'عالية',
             'title': 'Private-sector skills alignment',
             'title_ar': 'محاذاة المهارات للقطاع الخاص',
             'detail': 'Targeted training programs may improve citizen employment in private sector.',
             'detail_ar': 'برامج تدريب موجهة قد تُحسّن توظيف المواطنين في القطاع الخاص.',
             'timeline': 'Near-term', 'timeline_ar': 'قريب الأمد', 'icon': '👔'},
            {'priority': 'Medium', 'priority_ar': 'متوسطة',
             'title': 'Labor market modernization',
             'title_ar': 'تحديث سوق العمل',
             'detail': 'Reviewing labor regulations could improve employment dynamics.',
             'detail_ar': 'مراجعة لوائح العمل قد تُحسّن ديناميكيات التوظيف.',
             'timeline': '6-12 months', 'timeline_ar': '6-12 شهراً', 'icon': '⚖️'}
        ],
        'environmental': [
            {'priority': 'High', 'priority_ar': 'عالية',
             'title': 'Emissions reduction pathway',
             'title_ar': 'مسار تقليل الانبعاثات',
             'detail': 'Per-capita emissions remain among the highest globally — structured reduction warranted.',
             'detail_ar': 'انبعاثات الفرد من الأعلى عالمياً — يستدعي تقليلاً منهجياً.',
             'timeline': '1-2 years', 'timeline_ar': '1-2 سنة', 'icon': '🌍'},
            {'priority': 'Medium', 'priority_ar': 'متوسطة',
             'title': 'Climate adaptation infrastructure',
             'title_ar': 'البنية التحتية للتكيف المناخي',
             'detail': 'Rising temperatures suggest infrastructure adaptation value.',
             'detail_ar': 'ارتفاع الحرارة يقترح قيمة تكيّف البنية التحتية.',
             'timeline': '2-5 years', 'timeline_ar': '2-5 سنوات', 'icon': '🌡️'},
            {'priority': 'High', 'priority_ar': 'عالية',
             'title': 'Marine environment protection',
             'title_ar': 'حماية البيئة البحرية',
             'detail': 'Enhanced monitoring of industrial discharge supports coastal ecosystem health.',
             'detail_ar': 'مراقبة محسّنة للصرف الصناعي تدعم صحة النظام الساحلي.',
             'timeline': 'Near-term', 'timeline_ar': 'قريب الأمد', 'icon': '🌊'}
        ]
    }
    
    return recs_by_cat.get(category, recs_by_cat['economic'])


# ============================================================
# 7. Main Analysis
# ============================================================

def analyze_decision(decision_text, values, brent):
    """Decision Intelligence Workflow."""
    classification = classify_decision(decision_text)
    signals = calculate_decision_signals(decision_text, values, brent)
    forecast = generate_forecast(values, brent, target_code='ECO_GDP', n_years=5)
    correlations = compute_indicator_correlations(values)
    relevant_correlations = get_relevant_correlations(values, classification['relevant_indicators'])
    strategic_signals = detect_strategic_signals(values, classification['primary_category'])
    recommendations = generate_recommendations(decision_text, classification, signals, values)
    
    return {
        'decision_text': decision_text,
        'classification': classification,
        'signals': signals,
        'forecast': forecast,
        'correlations': correlations,
        'relevant_correlations': relevant_correlations,
        'strategic_signals': strategic_signals,
        'recommendations': recommendations,
        'method_note': 'ESGINE applies a Decision Intelligence Workflow combining rule-based classification, heuristic signals, statistical forecasting (SARIMAX), and correlation analysis. Outputs are indicative — not investment or policy directives.',
        'method_note_ar': 'يطبّق ESGINE سير عمل لذكاء القرار يجمع بين التصنيف القائم على القواعد والإشارات الاستدلالية والتنبؤ الإحصائي (SARIMAX) وتحليل الترابطات. المخرجات إرشادية — وليست توجيهات استثمارية أو سياسية.',
        'category_emoji': {
            'energy': '⚡', 'economic': '💰',
            'social': '👥', 'environmental': '🌱'
        }.get(classification['primary_category'], '📊')
    }