"""
🌿 FreshScan — Fruit & Vegetable Freshness Detector
Streamlit UI with Image Upload + Live Camera Detection
Run: streamlit run app.py
"""

import streamlit as st
import numpy as np
import cv2
import json
import os
import time
from PIL import Image
import tensorflow as tf

# ═══════════════════════════════════════════════════════════════
# PAGE CONFIG — must be first Streamlit call
# ═══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="FreshScan — AI Freshness Detector",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════════════
# CUSTOM CSS — Dark organic theme
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;900&family=JetBrains+Mono:wght@400;600&display=swap');

/* ── Root & Background ─────────────────────────────────────── */
:root {
    --fresh:   #4ADE80;
    --rotten:  #F87171;
    --gold:    #FBBF24;
    --bg:      #0D1117;
    --card:    #161B22;
    --border:  #30363D;
    --text:    #E6EDF3;
    --muted:   #8B949E;
    --accent:  #238636;
}

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
    background-color: var(--bg);
    color: var(--text);
}

.stApp {
    background: radial-gradient(ellipse at 20% 0%, #0f2a1a 0%, #0D1117 50%, #1a0d0d 100%);
    min-height: 100vh;
}

/* ── Hide Streamlit branding ───────────────────────────────── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Sidebar ───────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #0D1117;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] .stMarkdown { color: var(--muted); }

/* ── Hero header ───────────────────────────────────────────── */
.hero {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    background: linear-gradient(135deg, #0f2a1a 0%, #1a1f2e 50%, #2a1010 100%);
    border-radius: 20px;
    border: 1px solid var(--border);
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(circle at 50% 50%, rgba(74,222,128,0.04) 0%, transparent 60%);
    pointer-events: none;
}
.hero-title {
    font-size: 3rem;
    font-weight: 900;
    letter-spacing: -1px;
    background: linear-gradient(135deg, #4ADE80, #86EFAC, #FBBF24);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    line-height: 1.1;
}
.hero-sub {
    color: var(--muted);
    font-size: 1.05rem;
    margin-top: 0.5rem;
    font-weight: 300;
    letter-spacing: 0.5px;
}
.hero-badge {
    display: inline-block;
    background: rgba(74,222,128,0.1);
    border: 1px solid rgba(74,222,128,0.3);
    color: var(--fresh);
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-top: 1rem;
    font-family: 'JetBrains Mono', monospace;
}

/* ── Tab styling ───────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--card);
    border-radius: 14px;
    padding: 6px;
    border: 1px solid var(--border);
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px;
    font-family: 'Outfit', sans-serif;
    font-weight: 600;
    font-size: 1rem;
    color: var(--muted);
    padding: 10px 28px;
    transition: all 0.2s;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #1a3a26, #1e2a3a) !important;
    color: var(--fresh) !important;
    border: 1px solid rgba(74,222,128,0.3) !important;
}
.stTabs [data-baseweb="tab-panel"] {
    padding-top: 1.5rem;
}

/* ── Cards ─────────────────────────────────────────────────── */
.card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}
.card-title {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--muted);
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: 0.75rem;
}

/* ── Result badges ─────────────────────────────────────────── */
.result-fresh {
    background: linear-gradient(135deg, rgba(74,222,128,0.15), rgba(74,222,128,0.05));
    border: 2px solid var(--fresh);
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    margin: 1rem 0;
}
.result-rotten {
    background: linear-gradient(135deg, rgba(248,113,113,0.15), rgba(248,113,113,0.05));
    border: 2px solid var(--rotten);
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    margin: 1rem 0;
}
.result-label {
    font-size: 2rem;
    font-weight: 900;
    letter-spacing: -0.5px;
}
.result-conf {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.95rem;
    margin-top: 0.4rem;
    opacity: 0.8;
}
.result-produce {
    font-size: 1.1rem;
    font-weight: 600;
    margin-top: 0.3rem;
    opacity: 0.9;
}

/* ── Nutrition pills ───────────────────────────────────────── */
.nutr-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 0.5rem;
}
.nutr-pill {
    background: rgba(255,255,255,0.05);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 6px 12px;
    font-size: 0.82rem;
    font-family: 'JetBrains Mono', monospace;
}
.nutr-key { color: var(--muted); }
.nutr-val { color: var(--gold); font-weight: 600; margin-left: 6px; }

/* ── Confidence bar ────────────────────────────────────────── */
.conf-bar-wrap {
    background: rgba(255,255,255,0.06);
    border-radius: 8px;
    height: 10px;
    overflow: hidden;
    margin-top: 6px;
}
.conf-bar-fill-fresh  { background: linear-gradient(90deg, #4ADE80, #86EFAC); height: 100%; border-radius: 8px; transition: width 0.5s ease; }
.conf-bar-fill-rotten { background: linear-gradient(90deg, #F87171, #FCA5A5); height: 100%; border-radius: 8px; transition: width 0.5s ease; }

/* ── Info rows ─────────────────────────────────────────────── */
.info-row {
    display: flex;
    justify-content: space-between;
    padding: 8px 0;
    border-bottom: 1px solid var(--border);
    font-size: 0.9rem;
}
.info-row:last-child { border-bottom: none; }
.info-key { color: var(--muted); }
.info-val { color: var(--text); font-weight: 500; text-align: right; max-width: 60%; }

/* ── Benefit tags ──────────────────────────────────────────── */
.benefit-tag {
    display: inline-block;
    background: rgba(74,222,128,0.08);
    border: 1px solid rgba(74,222,128,0.2);
    color: var(--fresh);
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.78rem;
    margin: 3px 3px 3px 0;
    font-weight: 500;
}

/* ── Upload zone ───────────────────────────────────────────── */
[data-testid="stFileUploader"] {
    background: rgba(74,222,128,0.03);
    border: 2px dashed rgba(74,222,128,0.25);
    border-radius: 16px;
    padding: 1rem;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(74,222,128,0.5);
}

/* ── Buttons ───────────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #1a3a26, #2d4a36) !important;
    color: var(--fresh) !important;
    border: 1px solid rgba(74,222,128,0.4) !important;
    border-radius: 10px !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    padding: 0.55rem 1.4rem !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #1f4a30, #375a40) !important;
    border-color: var(--fresh) !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(74,222,128,0.2) !important;
}
.stop-btn > button {
    background: linear-gradient(135deg, #3a1a1a, #4a2d2d) !important;
    color: var(--rotten) !important;
    border-color: rgba(248,113,113,0.4) !important;
}
.stop-btn > button:hover {
    border-color: var(--rotten) !important;
    box-shadow: 0 4px 20px rgba(248,113,113,0.2) !important;
}

/* ── Camera live frame ─────────────────────────────────────── */
.cam-container {
    border: 2px solid rgba(74,222,128,0.3);
    border-radius: 16px;
    overflow: hidden;
    position: relative;
}
.cam-label {
    position: absolute;
    top: 10px; left: 10px;
    background: rgba(0,0,0,0.7);
    color: var(--fresh);
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 1px;
    font-family: 'JetBrains Mono', monospace;
    text-transform: uppercase;
}

/* ── Selectbox / Radio ─────────────────────────────────────── */
.stSelectbox > div > div,
.stRadio > div {
    background: var(--card) !important;
    border-color: var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
}

/* ── Metrics ───────────────────────────────────────────────── */
[data-testid="metric-container"] {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem;
}

/* ── Divider ───────────────────────────────────────────────── */
hr { border-color: var(--border); opacity: 0.5; }

/* ── Scrollbar ─────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# PRODUCE INFO DATABASE
# ═══════════════════════════════════════════════════════════════
PRODUCE_INFO = {
    'apple':      {'emoji':'🍎','type':'Fruit','nutrition':{'Calories':'95 kcal','Carbs':'25g','Fiber':'4.4g','Vitamin C':'14%DV','Potassium':'195mg'},'health_benefits':['Reduces heart disease risk','Aids digestion','Supports weight management','Rich in antioxidants'],'shelf_life':{'Room temp':'1-2 weeks','Refrigerator':'4-6 weeks'},'storage_tips':'Store in fridge crisper. Keep away from other produce — apples emit ethylene gas.','fresh_signs':'Firm, smooth, bright skin with no soft spots.','rot_signs':'Soft/mushy, brown spots, wrinkled skin, mold, fermented smell.'},
    'banana':     {'emoji':'🍌','type':'Fruit','nutrition':{'Calories':'89 kcal','Carbs':'23g','Fiber':'2.6g','Vitamin B6':'25%DV','Potassium':'422mg'},'health_benefits':['Boosts energy','Supports heart health','Improves digestion','Reduces muscle cramps'],'shelf_life':{'Room temp':'2-7 days','Refrigerator':'7-10 days'},'storage_tips':'Room temperature. Wrap stem in plastic to slow ripening.','fresh_signs':'Bright yellow, slightly firm, no brown patches.','rot_signs':'Heavily blackened, extremely mushy, sour/fermented smell.'},
    'orange':     {'emoji':'🍊','type':'Fruit','nutrition':{'Calories':'62 kcal','Carbs':'15g','Fiber':'3.1g','Vitamin C':'116%DV','Folate':'10%DV'},'health_benefits':['Immune boost','Skin health','Reduces inflammation','Lowers cholesterol'],'shelf_life':{'Room temp':'1-2 weeks','Refrigerator':'3-4 weeks'},'storage_tips':'Store loosely in fridge. Avoid airtight bags — moisture causes mold.','fresh_signs':'Firm, heavy for size, bright color, no soft spots.','rot_signs':'Mold, mushy spots, sour smell, very wrinkled skin.'},
    'mango':      {'emoji':'🥭','type':'Fruit','nutrition':{'Calories':'99 kcal','Carbs':'25g','Fiber':'2.6g','Vitamin C':'67%DV','Vitamin A':'10%DV'},'health_benefits':['Supports immunity','Improves digestion','Eye health','Anti-inflammatory'],'shelf_life':{'Room temp':'2-3 days ripe','Refrigerator':'5-7 days ripe'},'storage_tips':'Ripen at room temperature, then refrigerate.','fresh_signs':'Slightly yielding to pressure, fruity aroma, vibrant color.','rot_signs':'Very soft/mushy, black patches, fermented smell, mold.'},
    'strawberry': {'emoji':'🍓','type':'Fruit','nutrition':{'Calories':'49 kcal','Carbs':'12g','Fiber':'3g','Vitamin C':'149%DV','Folate':'9%DV'},'health_benefits':['Heart health','Blood sugar regulation','Anti-inflammatory','Rich in antioxidants'],'shelf_life':{'Room temp':'1-2 days','Refrigerator':'3-7 days'},'storage_tips':'Do not wash until ready to eat. Store in single layers with paper towel.','fresh_signs':'Bright red, firm, intact hull, fresh fruity smell.','rot_signs':'Mold, mushy texture, dull color, sour smell.'},
    'grapes':     {'emoji':'🍇','type':'Fruit','nutrition':{'Calories':'69 kcal','Carbs':'18g','Fiber':'0.9g','Vitamin K':'22%DV','Vitamin C':'6%DV'},'health_benefits':['Antioxidant-rich','Heart health','Anti-inflammatory','Brain health'],'shelf_life':{'Room temp':'1-2 days','Refrigerator':'1-2 weeks'},'storage_tips':'Store unwashed in perforated bag in fridge. Wash just before eating.','fresh_signs':'Plump, firm, attached to stem, bloom intact.','rot_signs':'Shriveled, leaking juice, brown spots, mold.'},
    'tomato':     {'emoji':'🍅','type':'Vegetable','nutrition':{'Calories':'18 kcal','Carbs':'3.9g','Fiber':'1.2g','Vitamin C':'28%DV','Lycopene':'High'},'health_benefits':['Reduces cancer risk','Heart health','Skin health','Bone health'],'shelf_life':{'Room temp':'3-5 days ripe','Refrigerator':'1-2 weeks ripe'},'storage_tips':'Ripen at room temperature stem-side down. Refrigerate only once fully ripe.','fresh_signs':'Vibrant red, slightly yielding, fresh earthy smell.','rot_signs':'Very mushy, leaking juice, mold, extremely wrinkled.'},
    'potato':     {'emoji':'🥔','type':'Vegetable','nutrition':{'Calories':'77 kcal','Carbs':'17g','Fiber':'2.2g','Vitamin C':'30%DV','Potassium':'620mg'},'health_benefits':['Energy source','Blood pressure control','Digestive health','Immunity boost'],'shelf_life':{'Cool dark place':'3-5 weeks','Refrigerator':'Not recommended'},'storage_tips':'Store in cool, dark, ventilated space. Keep away from onions.','fresh_signs':'Firm, smooth skin, no sprouts, no green tinge.','rot_signs':'Soft/wrinkled, foul smell, green patches, large sprouts, mold.'},
    'carrot':     {'emoji':'🥕','type':'Vegetable','nutrition':{'Calories':'41 kcal','Carbs':'10g','Fiber':'2.8g','Vitamin A':'334%DV','Vitamin K':'13%DV'},'health_benefits':['Eye health','Cancer prevention','Blood sugar control','Immune support'],'shelf_life':{'Refrigerator':'3-4 weeks','Freezer':'10-12 months blanched'},'storage_tips':'Remove tops, wrap in damp paper towel, store in fridge crisper.','fresh_signs':'Bright orange, firm, crunchy, smooth skin.','rot_signs':'Limp/rubbery, white fuzzy mold, slimy texture, foul smell.'},
    'bellpepper': {'emoji':'🫑','type':'Vegetable','nutrition':{'Calories':'31 kcal','Carbs':'6g','Fiber':'2.1g','Vitamin C':'169%DV','Vitamin B6':'20%DV'},'health_benefits':['Immune support','Eye health','Iron absorption','Anti-inflammatory'],'shelf_life':{'Room temp':'1-2 days','Refrigerator':'1-2 weeks'},'storage_tips':'Store whole in fridge crisper. Once cut, refrigerate airtight — use within 3 days.','fresh_signs':'Firm, glossy, vibrant color, intact stem.','rot_signs':'Soft/wrinkled, dark sunken spots, mold, slimy inside.'},
    'cucumber':   {'emoji':'🥒','type':'Vegetable','nutrition':{'Calories':'16 kcal','Carbs':'3.6g','Fiber':'0.5g','Vitamin K':'16%DV','Water':'96%'},'health_benefits':['Hydration','Weight management','Bone health','Anti-inflammatory'],'shelf_life':{'Room temp':'1-2 days','Refrigerator':'7-10 days'},'storage_tips':'Wrap in paper towel and store in fridge. Sensitive to cold.','fresh_signs':'Dark green, firm, smooth skin, no soft ends.','rot_signs':'Soft/mushy ends, yellow patches, sliminess, mold.'},
    'spinach':    {'emoji':'🥬','type':'Vegetable','nutrition':{'Calories':'23 kcal','Carbs':'3.6g','Fiber':'2.2g','Iron':'15%DV','Vitamin K':'460%DV'},'health_benefits':['Bone health','Blood pressure control','Cancer prevention','Eye health'],'shelf_life':{'Refrigerator':'5-7 days'},'storage_tips':'Store in slightly damp paper towel in airtight container in fridge.','fresh_signs':'Bright green, crisp leaves, no yellowing.','rot_signs':'Yellowed, slimy, musty smell, dark wilted leaves.'},
    'pomegranate':{'emoji':'🍎','type':'Fruit','nutrition':{'Calories':'83 kcal','Carbs':'19g','Fiber':'4g','Vitamin C':'17%DV','Vitamin K':'16%DV'},'health_benefits':['Powerful antioxidants','Reduces blood pressure','Anti-inflammatory','Memory improvement'],'shelf_life':{'Room temp':'1-2 weeks','Refrigerator':'1-2 months'},'storage_tips':'Whole pomegranates last months in cool/dry conditions. Refrigerate once cut.','fresh_signs':'Heavy for size, deep red/purple color, firm leathery skin.','rot_signs':'Lightweight, soft/sunken spots, mold at crown, fermented smell.'},
}

def get_produce_name(class_name):
    name = class_name.lower()
    for prefix in ['fresh', 'rotten', 'stale', 'spoiled', 'bad', 'good']:
        name = name.replace(prefix, '')
    name = name.strip('_- ')
    aliases = {
        'apples':'apple','bananas':'banana','oranges':'orange',
        'mangoes':'mango','mangos':'mango','strawberries':'strawberry',
        'tomatoes':'tomato','potatoes':'potato','carrots':'carrot',
        'peppers':'bellpepper','bell_pepper':'bellpepper','bellpeppers':'bellpepper',
        'cucumbers':'cucumber',
    }
    return aliases.get(name, name)

def is_fresh(class_name):
    return 'fresh' in class_name.lower() or 'good' in class_name.lower()


# ═══════════════════════════════════════════════════════════════
# MODEL LOADING
# ═══════════════════════════════════════════════════════════════
@st.cache_resource
def load_model_and_classes():
    """Load model and class names once and cache them."""
    model, class_names = None, {}
    model_path = st.session_state.get('model_path', 'freshness_model.h5')
    class_path = st.session_state.get('class_path', 'class_names.json')

    if os.path.exists(model_path):
        try:
            model = tf.keras.models.load_model(model_path)
        except Exception as e:
            st.error(f"Model load error: {e}")

    if os.path.exists(class_path):
        try:
            with open(class_path) as f:
                raw = json.load(f)
            class_names = {int(k): v for k, v in raw.items()}
        except Exception as e:
            st.error(f"Class names load error: {e}")

    return model, class_names


def predict_image(img_array, model, class_names, img_size=(224, 224)):
    """Run model inference on a numpy RGB image array."""
    img = Image.fromarray(img_array).resize(img_size)
    arr = np.expand_dims(np.array(img) / 255.0, axis=0)
    preds   = model.predict(arr, verbose=0)[0]
    top_idx = np.argmax(preds)
    top5    = sorted(enumerate(preds), key=lambda x: x[1], reverse=True)[:5]
    return {
        'class':      class_names[top_idx],
        'produce':    get_produce_name(class_names[top_idx]),
        'is_fresh':   is_fresh(class_names[top_idx]),
        'confidence': float(preds[top_idx]) * 100,
        'top5':       [(class_names[i], float(p)*100) for i, p in top5],
    }


# ═══════════════════════════════════════════════════════════════
# RESULT DISPLAY COMPONENT
# ═══════════════════════════════════════════════════════════════
def show_result(result):
    fresh   = result['is_fresh']
    produce = result['produce']
    conf    = result['confidence']
    info    = PRODUCE_INFO.get(produce)
    emoji   = info['emoji'] if info else '🌿'
    color   = '#4ADE80' if fresh else '#F87171'
    status  = '✅ FRESH' if fresh else '❌ ROTTEN / SPOILED'
    cls_    = 'result-fresh' if fresh else 'result-rotten'

    # Main result badge
    st.markdown(f"""
    <div class="{cls_}">
        <div style="font-size:2.5rem">{emoji}</div>
        <div class="result-label" style="color:{color}">{status}</div>
        <div class="result-produce">{produce.title()}</div>
        <div class="result-conf">Confidence: {conf:.1f}%</div>
        <div class="conf-bar-wrap" style="margin:10px auto;max-width:260px">
            <div class="conf-bar-fill-{'fresh' if fresh else 'rotten'}" style="width:{conf:.1f}%"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if info:
        col1, col2 = st.columns(2)

        # ── Nutrition ──────────────────────────────────────────
        with col1:
            st.markdown('<div class="card"><div class="card-title">📊 Nutrition per 100g</div>', unsafe_allow_html=True)
            pills = ''.join([
                f'<span class="nutr-pill"><span class="nutr-key">{k}</span><span class="nutr-val">{v}</span></span>'
                for k, v in info['nutrition'].items()
            ])
            st.markdown(f'<div class="nutr-grid">{pills}</div></div>', unsafe_allow_html=True)

        # ── Health Benefits ────────────────────────────────────
        with col2:
            st.markdown('<div class="card"><div class="card-title">💪 Health Benefits</div>', unsafe_allow_html=True)
            tags = ''.join([f'<span class="benefit-tag">{b}</span>' for b in info['health_benefits']])
            st.markdown(f'{tags}</div>', unsafe_allow_html=True)

        col3, col4 = st.columns(2)

        # ── Shelf Life & Storage ───────────────────────────────
        with col3:
            st.markdown('<div class="card"><div class="card-title">⏳ Shelf Life</div>', unsafe_allow_html=True)
            rows = ''.join([
                f'<div class="info-row"><span class="info-key">{k}</span><span class="info-val">{v}</span></div>'
                for k, v in info['shelf_life'].items()
            ])
            st.markdown(f'{rows}<div class="info-row"><span class="info-key">Storage</span><span class="info-val" style="font-size:0.82rem">{info["storage_tips"]}</span></div></div>', unsafe_allow_html=True)

        # ── Freshness / Spoilage Signs ─────────────────────────
        with col4:
            sign_key   = 'fresh_signs' if fresh else 'rot_signs'
            sign_title = '✅ Signs of Freshness' if fresh else '⚠️ Signs of Spoilage'
            sign_color = '#4ADE80' if fresh else '#F87171'
            st.markdown(f"""
            <div class="card">
                <div class="card-title">{sign_title}</div>
                <p style="color:{sign_color};font-size:0.9rem;margin:0;line-height:1.6">{info[sign_key]}</p>
            </div>
            """, unsafe_allow_html=True)

        # ── Top 5 predictions ──────────────────────────────────
        st.markdown('<div class="card"><div class="card-title">🎯 Top Predictions</div>', unsafe_allow_html=True)
        for cls_name, prob in result['top5']:
            bar_color = '#4ADE80' if is_fresh(cls_name) else '#F87171'
            st.markdown(f"""
            <div style="margin-bottom:8px">
                <div style="display:flex;justify-content:space-between;font-size:0.85rem;margin-bottom:3px">
                    <span style="color:#E6EDF3">{cls_name}</span>
                    <span style="font-family:'JetBrains Mono',monospace;color:{bar_color}">{prob:.1f}%</span>
                </div>
                <div class="conf-bar-wrap">
                    <div style="background:{bar_color};width:{prob:.1f}%;height:100%;border-radius:8px;opacity:0.8"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:1rem 0">
        <div style="font-size:2.5rem">🌿</div>
        <div style="font-size:1.2rem;font-weight:700;color:#4ADE80">FreshScan</div>
        <div style="font-size:0.75rem;color:#8B949E;letter-spacing:1px">AI FRESHNESS DETECTOR</div>
    </div>
    <hr style="border-color:#30363D;margin:0.5rem 0 1rem">
    """, unsafe_allow_html=True)

    st.markdown("**⚙️ Model Settings**")

    model_path = st.text_input("Model path (.h5)", value="freshness_model.h5",
                                help="Path to your trained .h5 model file")
    class_path = st.text_input("Class names (.json)", value="class_names.json",
                                help="Path to your class_names.json file")

    st.session_state['model_path'] = model_path
    st.session_state['class_path'] = class_path

    img_size_opt = st.selectbox("Image size", [224, 128, 256, 299], index=0)
    IMG_SIZE = (img_size_opt, img_size_opt)

    cam_index = st.number_input("Camera index", min_value=0, max_value=5, value=0,
                                 help="0 = built-in webcam, 1 = external camera")

    detect_every = st.slider("Detect every N frames", 5, 60, 15,
                              help="Higher = faster but less frequent predictions")

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.75rem;color:#8B949E;line-height:1.8">
        <b style="color:#E6EDF3">📁 Required files:</b><br>
        • freshness_model.h5<br>
        • class_names.json<br><br>
        <b style="color:#E6EDF3">🎥 Camera controls:</b><br>
        • Click Stop to end feed<br>
        • Screenshot saves frame
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    model_exists = os.path.exists(model_path)
    class_exists = os.path.exists(class_path)
    st.markdown(f"""
    <div style="font-size:0.8rem">
        {'✅' if model_exists else '❌'} Model file<br>
        {'✅' if class_exists else '❌'} Class names
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# MAIN CONTENT
# ═══════════════════════════════════════════════════════════════

# Hero
st.markdown("""
<div class="hero">
    <div class="hero-title">🌿 FreshScan</div>
    <div class="hero-sub">AI-powered freshness detection for fruits & vegetables</div>
    <div class="hero-badge">MobileNetV2 · Transfer Learning</div>
</div>
""", unsafe_allow_html=True)

# Load model
model, class_names = load_model_and_classes()

if not model or not class_names:
    st.markdown("""
    <div style="background:rgba(251,191,36,0.08);border:1px solid rgba(251,191,36,0.3);
                border-radius:12px;padding:1.2rem;margin-bottom:1.5rem">
        <b style="color:#FBBF24">⚠️ Model not loaded</b><br>
        <span style="color:#8B949E;font-size:0.9rem">
        Set correct paths in the sidebar and make sure <code>freshness_model.h5</code>
        and <code>class_names.json</code> exist in your project folder.
        </span>
    </div>
    """, unsafe_allow_html=True)

# ── Tabs ────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🖼️  Image Detection", "🎥  Live Camera"])


# ════════════════════════════════════════════════════════════════
# TAB 1 — IMAGE DETECTION
# ════════════════════════════════════════════════════════════════
with tab1:
    col_upload, col_result = st.columns([1, 1.2], gap="large")

    with col_upload:
        st.markdown("""
        <div class="card-title" style="margin-bottom:0.5rem">
            📤 Upload an Image
        </div>
        """, unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "Drop a fruit or vegetable image here",
            type=['jpg', 'jpeg', 'png', 'webp', 'bmp'],
            label_visibility="collapsed"
        )

        if uploaded:
            img = Image.open(uploaded).convert('RGB')
            st.image(img, caption="Uploaded image", use_container_width=True)

            col_a, col_b = st.columns(2)
            with col_a:
                analyze_btn = st.button("🔍 Analyze Image", use_container_width=True)
            with col_b:
                clear_btn = st.button("🗑️ Clear", use_container_width=True)

            if clear_btn:
                st.session_state.pop('img_result', None)
                st.rerun()

            if analyze_btn:
                if model and class_names:
                    with st.spinner("Analyzing..."):
                        arr = np.array(img)
                        result = predict_image(arr, model, class_names, IMG_SIZE)
                        st.session_state['img_result'] = result
                else:
                    st.error("❌ Model not loaded. Check sidebar settings.")

        else:
            # Empty state
            st.markdown("""
            <div style="text-align:center;padding:3rem 1rem;
                        border:2px dashed rgba(74,222,128,0.2);
                        border-radius:16px;color:#8B949E">
                <div style="font-size:3rem;margin-bottom:1rem">🍎</div>
                <div style="font-size:1rem;font-weight:600;color:#E6EDF3">
                    Drop your image here
                </div>
                <div style="font-size:0.85rem;margin-top:0.3rem">
                    JPG, PNG, WEBP supported
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Quick tips
            st.markdown("---")
            st.markdown("""
            <div class="card-title">💡 Tips for best results</div>
            <div style="font-size:0.85rem;color:#8B949E;line-height:2">
                • Use well-lit, clear images<br>
                • Center the fruit/vegetable<br>
                • Avoid heavy shadows or blur<br>
                • Single item per image works best
            </div>
            """, unsafe_allow_html=True)

    with col_result:
        st.markdown("""
        <div class="card-title" style="margin-bottom:0.5rem">
            📋 Detection Result
        </div>
        """, unsafe_allow_html=True)

        if 'img_result' in st.session_state:
            show_result(st.session_state['img_result'])
        else:
            st.markdown("""
            <div style="text-align:center;padding:4rem 1rem;color:#8B949E">
                <div style="font-size:3rem;opacity:0.3">📊</div>
                <div style="font-size:0.95rem;margin-top:0.5rem">
                    Results will appear here after analysis
                </div>
            </div>
            """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# TAB 2 — LIVE CAMERA
# ════════════════════════════════════════════════════════════════
with tab2:
    # Initialize session state
    if 'cam_running' not in st.session_state:
        st.session_state['cam_running'] = False
    if 'cam_result' not in st.session_state:
        st.session_state['cam_result'] = None
    if 'screenshot_count' not in st.session_state:
        st.session_state['screenshot_count'] = 0

    col_cam, col_cam_result = st.columns([1.2, 1], gap="large")

    with col_cam:
        st.markdown("""
        <div class="card-title" style="margin-bottom:0.5rem">
            🎥 Live Camera Feed
        </div>
        """, unsafe_allow_html=True)

        # Control buttons
        ctrl1, ctrl2, ctrl3 = st.columns(3)
        with ctrl1:
            start_btn = st.button("▶ Start Camera", use_container_width=True,
                                   disabled=st.session_state['cam_running'])
        with ctrl2:
            with st.container():
                st.markdown('<div class="stop-btn">', unsafe_allow_html=True)
                stop_btn = st.button("⏹ Stop Camera", use_container_width=True,
                                      disabled=not st.session_state['cam_running'])
                st.markdown('</div>', unsafe_allow_html=True)
        with ctrl3:
            shot_btn = st.button("📸 Screenshot", use_container_width=True,
                                  disabled=not st.session_state['cam_running'])

        # Camera frame placeholder
        frame_placeholder = st.empty()
        status_placeholder = st.empty()

        # Info box
        st.markdown("""
        <div class="card" style="margin-top:1rem">
            <div class="card-title">ℹ️ How to use</div>
            <div style="font-size:0.85rem;color:#8B949E;line-height:2">
                1. Click <b style="color:#4ADE80">▶ Start Camera</b> to begin<br>
                2. Hold a fruit/vegetable in front of camera<br>
                3. Results update every few frames automatically<br>
                4. Click <b style="color:#F87171">⏹ Stop Camera</b> to end session<br>
                5. Use <b style="color:#E6EDF3">📸 Screenshot</b> to save a frame
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_cam_result:
        st.markdown("""
        <div class="card-title" style="margin-bottom:0.5rem">
            📋 Live Detection Result
        </div>
        """, unsafe_allow_html=True)

        result_placeholder = st.empty()

        if st.session_state['cam_result']:
            show_result(st.session_state['cam_result'])
        else:
            result_placeholder.markdown("""
            <div style="text-align:center;padding:4rem 1rem;color:#8B949E">
                <div style="font-size:3rem;opacity:0.3">🎯</div>
                <div style="font-size:0.95rem;margin-top:0.5rem">
                    Start camera to see live results
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Camera loop ────────────────────────────────────────────
    if start_btn:
        st.session_state['cam_running'] = True
        st.rerun()

    if stop_btn:
        st.session_state['cam_running'] = False
        frame_placeholder.markdown("""
        <div style="text-align:center;padding:4rem;color:#8B949E;
                    border:2px dashed rgba(74,222,128,0.15);border-radius:16px">
            <div style="font-size:2rem">⏹</div>
            <div style="margin-top:0.5rem">Camera stopped</div>
        </div>
        """, unsafe_allow_html=True)
        status_placeholder.empty()
        st.rerun()

    if st.session_state['cam_running']:
        if not model or not class_names:
            st.error("❌ Model not loaded. Check sidebar settings.")
            st.session_state['cam_running'] = False
        else:
            cap = cv2.VideoCapture(int(cam_index))

            if not cap.isOpened():
                st.error(f"❌ Cannot open camera (index {cam_index}). Try a different index in the sidebar.")
                st.session_state['cam_running'] = False
            else:
                status_placeholder.markdown("""
                <div style="display:flex;align-items:center;gap:8px;
                            background:rgba(74,222,128,0.08);border:1px solid rgba(74,222,128,0.2);
                            border-radius:8px;padding:8px 14px;font-size:0.85rem;color:#4ADE80">
                    <span style="width:8px;height:8px;background:#4ADE80;border-radius:50%;
                                 display:inline-block;animation:pulse 1s infinite"></span>
                    Camera active — detecting...
                </div>
                """, unsafe_allow_html=True)

                frame_count  = 0
                last_result  = None
                COLOR_FRESH  = (50, 205,  50)
                COLOR_ROTTEN = (50,  50, 220)

                try:
                    while st.session_state['cam_running']:
                        ret, frame = cap.read()
                        if not ret:
                            status_placeholder.error("❌ Failed to grab frame.")
                            break

                        frame_count += 1

                        # Run model every N frames
                        if frame_count % int(detect_every) == 0:
                            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            result = predict_image(rgb, model, class_names, IMG_SIZE)
                            last_result = result
                            st.session_state['cam_result'] = result

                            # Update result panel
                            with result_placeholder.container():
                                show_result(result)

                        # Draw overlay
                        h, w = frame.shape[:2]
                        color = COLOR_FRESH if (last_result and last_result['is_fresh']) else COLOR_ROTTEN
                        if last_result is None:
                            color = (150, 150, 150)

                        # Detection box
                        cx, cy   = w//2, h//2
                        bs       = min(w, h) // 2
                        x1, y1   = cx - bs//2, cy - bs//2
                        x2, y2   = cx + bs//2, cy + bs//2
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                        # Corner accents
                        cl = 22
                        for px, py, dx, dy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
                            cv2.line(frame, (px, py), (px+cl*dx, py), color, 4)
                            cv2.line(frame, (px, py), (px, py+cl*dy), color, 4)

                        # Label bar
                        if last_result:
                            label = f"{last_result['produce'].upper()} — {'FRESH' if last_result['is_fresh'] else 'ROTTEN'}  {last_result['confidence']:.0f}%"
                            cv2.rectangle(frame, (0, 0), (w, 50), (20, 20, 20), -1)
                            cv2.putText(frame, label, (12, 34),
                                        cv2.FONT_HERSHEY_DUPLEX, 0.75, color, 2)

                        # Screenshot
                        if shot_btn:
                            fname = f"screenshot_{st.session_state['screenshot_count']:03d}.jpg"
                            cv2.imwrite(fname, frame)
                            st.session_state['screenshot_count'] += 1
                            st.toast(f"📸 Screenshot saved: {fname}", icon="✅")

                        # Show frame in notebook
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame_placeholder.image(rgb_frame, channels='RGB',
                                                use_container_width=True)

                        time.sleep(0.03)   # ~30 fps cap

                except Exception as e:
                    st.error(f"Camera error: {e}")
                finally:
                    cap.release()
