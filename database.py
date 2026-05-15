"""
ESGINE - وحدة الاتصال بقاعدة البيانات
============================================
هذا الملف مسؤول عن:
1. الاتصال بـ Supabase
2. قراءة بيانات المؤشرات الـ12
3. قراءة بيانات سعر برنت
"""

import os
from supabase import create_client, Client
from dotenv import load_dotenv
import pandas as pd

# تحميل المفاتيح من ملف .env
load_dotenv()

# إنشاء اتصال بـ Supabase
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# التحقق من وجود المفاتيح
if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("⚠️ مفاتيح Supabase غير موجودة في ملف .env")

# الاتصال بقاعدة البيانات
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


def get_indicators():
    """
    جلب قائمة المؤشرات الـ12
    """
    response = supabase.table("indicators").select("*").execute()
    return pd.DataFrame(response.data)


def get_indicator_values(indicator_code=None):
    """
    جلب القيم الزمنية للمؤشرات
    إذا تم تحديد رمز مؤشر، يجلب قيمه فقط
    """
    # جلب جدول المؤشرات أولاً للحصول على الـ ids
    indicators = get_indicators()
    
    # جلب القيم
    response = supabase.table("indicator_values").select("*").execute()
    values = pd.DataFrame(response.data)
    
    # دمج البيانات للحصول على الأسماء
    merged = values.merge(
        indicators[['id', 'code', 'name_ar', 'name_en', 'pillar', 'unit']],
        left_on='indicator_id',
        right_on='id',
        suffixes=('', '_indicator')
    )
    
    # تصفية حسب الرمز إذا تم تحديده
    if indicator_code:
        merged = merged[merged['code'] == indicator_code]
    
    # ترتيب حسب السنة
    merged = merged.sort_values(['code', 'year'])
    
    return merged


def get_brent_data():
    """
    جلب بيانات سعر برنت (المتغير الخارجي)
    """
    response = supabase.table("exogenous_variables").select("*").eq(
        "variable_name", "BRENT_CRUDE"
    ).execute()
    df = pd.DataFrame(response.data)
    return df.sort_values('year')


# اختبار الاتصال عند تشغيل الملف مباشرة
if __name__ == "__main__":
    print("🔍 اختبار الاتصال بـ Supabase...")
    print("=" * 60)
    
    try:
        # اختبار 1: المؤشرات
        indicators = get_indicators()
        print(f"✅ المؤشرات: {len(indicators)} مؤشر")
        print(f"   الأمثلة: {indicators['code'].head(3).tolist()}")
        
        # اختبار 2: القيم
        values = get_indicator_values()
        print(f"✅ القيم الزمنية: {len(values)} نقطة بيانات")
        print(f"   السنوات: {values['year'].min()} - {values['year'].max()}")
        
        # اختبار 3: برنت
        brent = get_brent_data()
        print(f"✅ بيانات برنت: {len(brent)} سنة")
        print(f"   متوسط السعر: ${brent['value'].mean():.2f}/برميل")
        
        print("=" * 60)
        print("🎉 الاتصال يعمل بشكل ممتاز!")
        
    except Exception as e:
        print(f"❌ خطأ: {e}")