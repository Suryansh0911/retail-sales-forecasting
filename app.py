import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import json
import requests
import io
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Retail Sales Forecasting",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .main-title { font-size: 2.2rem; font-weight: 700; color: #ffffff; letter-spacing: -0.5px; }
    .subtitle   { font-size: 0.95rem; color: #8b8fa8; margin-top: -8px; }
    .section-title {
        font-size: 0.85rem; font-weight: 600; color: #a0aec0;
        text-transform: uppercase; letter-spacing: 1px; margin: 20px 0 10px 0;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem; font-weight: 700; font-family: 'DM Mono', monospace;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.8px; color: #8b8fa8;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px; background: #1a1d2e; padding: 6px; border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 7px; padding: 6px 16px; font-size: 0.85rem; font-weight: 500;
    }
    .stTabs [aria-selected="true"] { background: #4A90D9 !important; color: white !important; }
    .store-badge {
        display: inline-block; background: #4A90D9; color: white;
        padding: 3px 12px; border-radius: 20px; font-size: 0.85rem;
        font-weight: 600; font-family: 'DM Mono', monospace;
    }
    .divider { border: none; border-top: 1px solid #2d3148; margin: 16px 0; }
    .upload-box {
        border: 2px dashed #4A90D9; border-radius: 12px;
        padding: 40px; text-align: center; background: #1a1d2e;
    }
    .mapping-card {
        background: #1a1d2e; border-radius: 8px;
        padding: 12px 16px; margin: 6px 0;
        border-left: 3px solid #4A90D9;
    }
    .warning-badge {
        display: inline-block; background: #4a3a1a; color: #fbbf24;
        padding: 3px 12px; border-radius: 20px; font-size: 0.85rem; font-weight: 600;
    }
    section[data-testid="stSidebar"] { background: #1a1d2e; }
</style>
""", unsafe_allow_html=True)

TEMPLATE        = "plotly_dark"
COLOR_ACTUAL    = "#4A90D9"
COLOR_PREDICTED = "#E05C5C"

# ── Sample Dataset ────────────────────────────────────────────
SAMPLE_CSV = """date,store_id,product_id,category,units_sold,price,event,is_holiday
2023-01-01,STORE_A,ITEM_001,FOODS,12,1.99,New Year,1
2023-01-01,STORE_A,ITEM_002,HOUSEHOLD,3,9.99,New Year,1
2023-01-01,STORE_B,ITEM_001,FOODS,8,1.99,New Year,1
2023-01-02,STORE_A,ITEM_001,FOODS,15,1.99,,0
2023-01-02,STORE_A,ITEM_002,HOUSEHOLD,5,9.99,,0
2023-01-02,STORE_B,ITEM_001,FOODS,11,1.99,,0
2023-01-03,STORE_A,ITEM_001,FOODS,9,1.89,,0
2023-01-03,STORE_A,ITEM_002,HOUSEHOLD,4,9.49,,0
2023-01-03,STORE_B,ITEM_001,FOODS,14,1.89,,0
2023-01-04,STORE_A,ITEM_001,FOODS,18,1.99,,0
2023-01-04,STORE_A,ITEM_002,HOUSEHOLD,6,9.99,,0
2023-01-04,STORE_B,ITEM_001,FOODS,13,1.99,,0
2023-01-05,STORE_A,ITEM_001,FOODS,20,1.79,Weekend Sale,0
2023-01-05,STORE_A,ITEM_002,HOUSEHOLD,8,8.99,Weekend Sale,0
2023-01-05,STORE_B,ITEM_001,FOODS,16,1.79,Weekend Sale,0
"""

# ── Load M5 Data ──────────────────────────────────────────────
@st.cache_data
def load_m5_data():
    try:
        preds   = pd.read_csv('store_predictions/all_store_preds.csv', parse_dates=['date'])
        metrics = pd.read_csv('store_predictions/per_store_metrics.csv', index_col='store')
        with open('store_predictions/feat_imp.json') as f:
            feat_imp = json.load(f)
        return preds, metrics, feat_imp, True
    except Exception:
        return None, None, None, False


# ── Claude API — Column Mapping ───────────────────────────────
def map_columns_with_claude(columns: list, sample_rows: str) -> dict:
    prompt = f"""You are a data analyst. A user uploaded a retail sales CSV dataset.

Column names: {columns}

First 3 rows sample:
{sample_rows}

Map these columns to the required fields below. Return ONLY a valid JSON object, no explanation, no markdown fences.

Required fields:
- "date": column containing dates/timestamps
- "store_id": column identifying store/location/outlet
- "category": column for product category/department/type (null if missing)
- "sales": column for units sold/quantity/sales volume
- "price": column for price/cost (null if missing)
- "event": column for events/holidays/promotions (null if missing)
- "item_id": column for product/item/SKU identifier (null if missing)

If date, sales or store_id are missing set: {{"error": "missing mandatory columns: <list>"}}

Return exactly this format with real column names or null:
{{"date": "col","store_id": "col","category": "col_or_null","sales": "col","price": "col_or_null","event": "col_or_null","item_id": "col_or_null"}}"""

    try:
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"Content-Type": "application/json"},
            json={
                "model"     : "claude-sonnet-4-20250514",
                "max_tokens": 500,
                "messages"  : [{"role": "user", "content": prompt}]
            }
        )
        text = response.json()['content'][0]['text'].strip()
        text = text.replace('```json', '').replace('```', '').strip()
        return json.loads(text)
    except Exception as e:
        return {"error": str(e)}


# ── Feature Engineering ───────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(['store_id', 'date']).reset_index(drop=True)

    df['wday']        = df['date'].dt.dayofweek.astype(np.int8)
    df['month']       = df['date'].dt.month.astype(np.int8)
    df['year']        = df['date'].dt.year.astype(np.int16)
    df['day_of_year'] = df['date'].dt.dayofyear.astype(np.int16)
    df['is_weekend']  = (df['wday'] >= 5).astype(np.int8)
    df['quarter']     = df['date'].dt.quarter.astype(np.int8)

    if 'price' in df.columns:
        df['price']          = pd.to_numeric(df['price'], errors='coerce').fillna(1.0)
        item_mean            = df.groupby('store_id')['price'].transform('mean')
        df['price_rel_mean'] = (df['price'] / (item_mean + 1e-9)).astype(np.float32)
        df['price_change']   = df.groupby('store_id')['price'].diff().fillna(0).astype(np.float32)
    else:
        df['price']          = 1.0
        df['price_rel_mean'] = 1.0
        df['price_change']   = 0.0

    df['is_event'] = (df['event'].notna() & (df['event'].astype(str) != '')).astype(np.int8) \
                     if 'event' in df.columns else 0

    grp          = df.groupby('store_id')['sales']
    df['lag_7']  = grp.shift(7).fillna(0).astype(np.float32)
    df['lag_14'] = grp.shift(14).fillna(0).astype(np.float32)
    df['lag_28'] = grp.shift(28).fillna(0).astype(np.float32)

    def rolling_feats(g):
        s = g['sales'].shift(1).fillna(0)
        g['rolling_mean_7']  = s.rolling(7,  min_periods=1).mean().astype(np.float32)
        g['rolling_mean_28'] = s.rolling(28, min_periods=1).mean().astype(np.float32)
        g['rolling_std_7']   = s.rolling(7,  min_periods=1).std().fillna(0).astype(np.float32)
        return g

    df = df.groupby('store_id', group_keys=False).apply(rolling_feats)

    for col in ['store_id', 'category']:
        if col in df.columns:
            df[col + '_enc'] = df[col].astype('category').cat.codes.astype(np.int16)

    return df


# ── Train & Predict ───────────────────────────────────────────
def train_and_predict(df: pd.DataFrame):
    feature_cols = [c for c in [
        'store_id_enc', 'category_enc', 'wday', 'month', 'year',
        'day_of_year', 'is_weekend', 'quarter', 'price', 'price_rel_mean',
        'price_change', 'is_event', 'lag_7', 'lag_14', 'lag_28',
        'rolling_mean_7', 'rolling_mean_28', 'rolling_std_7'
    ] if c in df.columns]

    df['sales_log'] = np.log1p(df['sales']).astype(np.float32)
    split_date      = df['date'].quantile(0.8)
    train           = df[df['date'] <= split_date]
    test            = df[df['date'] >  split_date]

    if len(test) < 10:
        return None, None, "Not enough data for train/test split. Upload at least 60 days of data."

    model = lgb.LGBMRegressor(
        n_estimators=300, learning_rate=0.05, num_leaves=64,
        max_depth=6, min_child_samples=5, subsample=0.8,
        colsample_bytree=0.8, n_jobs=-1, random_state=42, verbose=-1
    )
    model.fit(train[feature_cols], train['sales_log'])
    y_pred = model.predict(test[feature_cols])

    metrics = {
        'rmse': round(np.sqrt(mean_squared_error(test['sales_log'], y_pred)), 4),
        'mae' : round(mean_absolute_error(test['sales_log'], y_pred), 4),
        'r2'  : round(r2_score(test['sales_log'], y_pred), 4)
    }

    pred_df              = test[['date', 'store_id', 'category', 'sales']].copy()
    pred_df['predicted'] = np.expm1(y_pred).clip(0).round(2)
    pred_df['actual']    = pred_df['sales']
    pred_df['residual']  = pred_df['actual'] - pred_df['predicted']

    feat_imp = pd.DataFrame({
        'feature'   : feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=True)

    return pred_df, feat_imp, metrics, model


# ── Shared Dashboard Renderer ─────────────────────────────────
def render_dashboard(preds_df, metrics_df, feat_imp, selected_store,
                     selected_cat, date_range, global_r2=0.5544):

    store_df = preds_df[preds_df['store_id'] == selected_store].copy()
    if selected_cat != 'All':
        store_df = store_df[store_df['category'] == selected_cat]
    store_df = store_df[
        (store_df['date'] >= pd.to_datetime(date_range[0])) &
        (store_df['date'] <= pd.to_datetime(date_range[1]))
    ]
    store_df['residual'] = store_df['actual'] - store_df['predicted']

    m = metrics_df if isinstance(metrics_df, dict) \
        else (metrics_df.loc[selected_store].to_dict()
              if selected_store in metrics_df.index else {'r2':0,'rmse':0,'mae':0})

    total_actual    = store_df['actual'].sum()
    total_predicted = store_df['predicted'].sum()
    accuracy_pct    = 100 - abs(total_actual - total_predicted) / (total_actual + 1e-9) * 100

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("R² Score",          f"{m['r2']:.4f}",
              delta=f"{m['r2']-global_r2:+.4f} vs baseline")
    c2.metric("RMSE",              f"{m['rmse']:.4f}")
    c3.metric("MAE",               f"{m['mae']:.4f}")
    c4.metric("Total Actual",      f"{int(total_actual):,}")
    c5.metric("Forecast Accuracy", f"{accuracy_pct:.1f}%")

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    t1, t2, t3, t4 = st.tabs([
        "📈 Forecast", "🛒 Category Breakdown",
        "🔍 Feature Importance", "📉 Residual Analysis"
    ])

    # TAB 1 — Forecast
    with t1:
        st.markdown("<div class='section-title'>Daily Actual vs Predicted</div>",
                    unsafe_allow_html=True)
        daily = (store_df.groupby('date')
                 .agg(actual=('actual','sum'), predicted=('predicted','sum'))
                 .reset_index())
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=daily['date'], y=daily['actual'],
                                  name='Actual', line=dict(color=COLOR_ACTUAL, width=2)))
        fig.add_trace(go.Scatter(x=daily['date'], y=daily['predicted'],
                                  name='Predicted',
                                  line=dict(color=COLOR_PREDICTED, width=2, dash='dash')))
        fig.update_layout(height=400, template=TEMPLATE,
                           xaxis_title='Date', yaxis_title='Units Sold', margin=dict(t=20))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("<div class='section-title'>Weekly Sales Trend</div>",
                    unsafe_allow_html=True)
        weekly = (store_df.copy()
                  .assign(week=pd.to_datetime(store_df['date'])
                          .dt.to_period('W').astype(str))
                  .groupby('week')
                  .agg(actual=('actual','sum'), predicted=('predicted','sum'))
                  .reset_index())
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=weekly['week'], y=weekly['actual'],
                               name='Actual', marker_color=COLOR_ACTUAL, opacity=0.85))
        fig2.add_trace(go.Bar(x=weekly['week'], y=weekly['predicted'],
                               name='Predicted', marker_color=COLOR_PREDICTED, opacity=0.85))
        fig2.update_layout(barmode='group', height=350, template=TEMPLATE,
                            xaxis_title='Week', yaxis_title='Units Sold', margin=dict(t=20))
        st.plotly_chart(fig2, use_container_width=True)

    # TAB 2 — Category
    with t2:
        cat_data = (preds_df[preds_df['store_id'] == selected_store]
                    .groupby('category')
                    .agg(actual=('actual','sum'), predicted=('predicted','sum'))
                    .reset_index())
        c1, c2 = st.columns(2)
        with c1:
            fig_c = go.Figure()
            fig_c.add_trace(go.Bar(x=cat_data['category'], y=cat_data['actual'],
                                    name='Actual', marker_color=COLOR_ACTUAL))
            fig_c.add_trace(go.Bar(x=cat_data['category'], y=cat_data['predicted'],
                                    name='Predicted', marker_color=COLOR_PREDICTED))
            fig_c.update_layout(barmode='group', height=350, template=TEMPLATE,
                                 title=f'Category Sales — {selected_store}', margin=dict(t=40))
            st.plotly_chart(fig_c, use_container_width=True)
        with c2:
            fig_p = px.pie(cat_data, values='actual', names='category',
                            title='Actual Sales Share',
                            color_discrete_sequence=px.colors.qualitative.Set2,
                            template=TEMPLATE)
            fig_p.update_layout(height=350, margin=dict(t=40))
            st.plotly_chart(fig_p, use_container_width=True)

        cat_daily = (preds_df[preds_df['store_id'] == selected_store]
                     .groupby(['date','category'])['actual'].sum().reset_index())
        fig_ct = px.line(cat_daily, x='date', y='actual', color='category',
                          height=350, title='Daily Sales by Category', template=TEMPLATE,
                          color_discrete_sequence=px.colors.qualitative.Set2)
        fig_ct.update_layout(margin=dict(t=40))
        st.plotly_chart(fig_ct, use_container_width=True)

        # Gap metrics
        cat_data['gap_pct'] = ((cat_data['actual'] - cat_data['predicted'])
                               / (cat_data['actual'] + 1e-9) * 100).round(1)
        cols = st.columns(len(cat_data))
        for i, (_, row) in enumerate(cat_data.iterrows()):
            cols[i].metric(row['category'], f"{int(row['actual']):,}",
                           delta=f"{row['gap_pct']:+.1f}% gap")

    # TAB 3 — Feature Importance
    with t3:
        if feat_imp is not None:
            fi = feat_imp.sort_values('importance', ascending=True) \
                 if isinstance(feat_imp, pd.DataFrame) \
                 else pd.DataFrame(feat_imp[selected_store]).sort_values('importance', ascending=True)
            c1, c2 = st.columns([2, 1])
            with c1:
                fig_fi = go.Figure(go.Bar(
                    x=fi['importance'], y=fi['feature'], orientation='h',
                    marker=dict(color=fi['importance'], colorscale='Blues', showscale=False)
                ))
                fig_fi.update_layout(height=520, template=TEMPLATE,
                                      xaxis_title='Importance Score',
                                      margin=dict(t=20, l=140))
                st.plotly_chart(fig_fi, use_container_width=True)
            with c2:
                st.markdown("**🏆 Top 5 Features**")
                for rank, (_, row) in enumerate(
                        fi.sort_values('importance', ascending=False).head(5).iterrows(), 1):
                    st.markdown(f"`#{rank}` **{row['feature']}**  \nScore: {int(row['importance']):,}")
                    st.markdown("")
                st.markdown("---")
                st.markdown("**⚠️ Weakest Features**")
                for _, row in fi.head(5).iterrows():
                    st.markdown(f"· {row['feature']} — {int(row['importance']):,}")
        else:
            st.info("Feature importance not available.")

    # TAB 4 — Residuals
    with t4:
        c1, c2 = st.columns(2)
        with c1:
            fig_r = px.histogram(store_df, x='residual', nbins=60,
                                  title='Residual Distribution',
                                  color_discrete_sequence=[COLOR_ACTUAL], template=TEMPLATE)
            fig_r.add_vline(x=0, line_dash='dash', line_color='red',
                             annotation_text='Zero Error', annotation_font_color='red')
            fig_r.update_layout(height=350, margin=dict(t=40))
            st.plotly_chart(fig_r, use_container_width=True)
        with c2:
            fig_s = px.scatter(store_df, x='actual', y='predicted', opacity=0.6,
                                color='category', title='Actual vs Predicted Scatter',
                                template=TEMPLATE,
                                color_discrete_sequence=px.colors.qualitative.Set2)
            mv = int(store_df['actual'].max())
            fig_s.add_shape(type='line', x0=0, y0=0, x1=mv, y1=mv,
                             line=dict(color='red', dash='dash'))
            fig_s.update_layout(height=350, margin=dict(t=40))
            st.plotly_chart(fig_s, use_container_width=True)

        daily_res = store_df.groupby('date')['residual'].mean().reset_index()
        fig_rt    = px.line(daily_res, x='date', y='residual',
                             title='Mean Daily Residual Over Time',
                             color_discrete_sequence=[COLOR_PREDICTED], template=TEMPLATE)
        fig_rt.add_hline(y=0, line_dash='dash', line_color='gray')
        fig_rt.update_layout(height=320, margin=dict(t=40))
        st.plotly_chart(fig_rt, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            cat_res = store_df.groupby('category')['residual'].mean().reset_index()
            fig_cr  = px.bar(cat_res, x='category', y='residual', color='residual',
                              color_continuous_scale='RdBu', title='Mean Residual by Category',
                              template=TEMPLATE)
            fig_cr.add_hline(y=0, line_dash='dash', line_color='gray')
            fig_cr.update_layout(height=300, margin=dict(t=40))
            st.plotly_chart(fig_cr, use_container_width=True)
        with c2:
            st.markdown("**📊 Residual Stats**")
            st.dataframe(store_df['residual'].describe().round(2).to_frame('Value'),
                         use_container_width=True)
            st.metric("Over-predicted",  f"{(store_df['residual']<0).mean()*100:.1f}%")
            st.metric("Under-predicted", f"{(store_df['residual']>0).mean()*100:.1f}%")


# ══════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════
preds_df, metrics_df, feat_imp, m5_loaded = load_m5_data()

with st.sidebar:
    st.markdown("## 🛒 Navigation")
    page = st.radio("", ["📊 M5 Dashboard", "📁 Upload Your Dataset"],
                    label_visibility="collapsed")
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("**Dataset:** M5 Walmart")
    st.markdown("10 Stores · 3 States · 2011–2016")


# ══════════════════════════════════════════════════════════════
# PAGE 1 — M5 DASHBOARD
# ══════════════════════════════════════════════════════════════
if page == "📊 M5 Dashboard":

    if not m5_loaded:
        st.error("M5 data files not found. Ensure `store_predictions/` folder exists with the 3 required files.")
        st.stop()

    with st.sidebar:
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        stores         = sorted(preds_df['store_id'].unique())
        categories     = ['All'] + sorted(preds_df['cat_id'].unique())
        selected_store = st.selectbox("🏪 Select Store", stores)
        selected_cat   = st.selectbox("📦 Category", categories)
        date_range     = st.date_input(
            "📅 Date Range",
            value=[preds_df['date'].min(), preds_df['date'].max()],
            min_value=preds_df['date'].min(), max_value=preds_df['date'].max()
        )

    c1, c2 = st.columns([3, 1])
    with c1:
        st.markdown("<div class='main-title'>🛒 Retail Sales Forecasting</div>",
                    unsafe_allow_html=True)
        st.markdown("<div class='subtitle'>M5 Walmart Dataset · LightGBM Per-Store Models · Test Period: Jan–Apr 2016</div>",
                    unsafe_allow_html=True)
    with c2:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"<div style='text-align:right'><span class='store-badge'>{selected_store}</span></div>",
                    unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>All Stores Overview</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        perf   = metrics_df.reset_index().sort_values('r2', ascending=False)
        fig_r2 = px.bar(perf, x='store', y='r2', color='r2',
                         color_continuous_scale='Blues', title='R² Score by Store',
                         template=TEMPLATE)
        fig_r2.add_hline(y=metrics_df['r2'].mean(), line_dash='dash', line_color='yellow',
                          annotation_text=f"Avg: {metrics_df['r2'].mean():.3f}",
                          annotation_font_color='yellow')
        fig_r2.update_layout(height=320, margin=dict(t=40))
        st.plotly_chart(fig_r2, use_container_width=True)
    with col2:
        preds_renamed = preds_df.rename(columns={'cat_id': 'category'})
        store_totals  = (preds_renamed.groupby('store_id')
                         .agg(actual=('actual','sum'), predicted=('predicted','sum'))
                         .reset_index().sort_values('actual', ascending=False))
        fig_st = go.Figure()
        fig_st.add_trace(go.Bar(x=store_totals['store_id'], y=store_totals['actual'],
                                 name='Actual', marker_color=COLOR_ACTUAL))
        fig_st.add_trace(go.Bar(x=store_totals['store_id'], y=store_totals['predicted'],
                                 name='Predicted', marker_color=COLOR_PREDICTED))
        fig_st.update_layout(barmode='group', height=320, template=TEMPLATE,
                              title='Total Sales by Store', margin=dict(t=40))
        st.plotly_chart(fig_st, use_container_width=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown(f"<div class='main-title' style='font-size:1.4rem'>Store Detail — "
                f"<span class='store-badge'>{selected_store}</span></div>",
                unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    render_dashboard(
        preds_df       = preds_df.rename(columns={'cat_id': 'category'}),
        metrics_df     = metrics_df,
        feat_imp       = feat_imp,
        selected_store = selected_store,
        selected_cat   = selected_cat,
        date_range     = date_range,
        global_r2      = 0.5544
    )

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.dataframe(metrics_df.round(4)
                 .style.highlight_max(subset=['r2'], color='#1a4a2e')
                 .highlight_min(subset=['rmse','mae'], color='#1a4a2e'),
                 use_container_width=True)


# ══════════════════════════════════════════════════════════════
# PAGE 2 — UPLOAD YOUR DATASET
# ══════════════════════════════════════════════════════════════
elif page == "📁 Upload Your Dataset":

    st.markdown("<div class='main-title'>📁 Upload Your Retail Dataset</div>",
                unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Upload any retail CSV — Claude AI auto-detects columns, "
                "engineers features, trains LightGBM and generates your dashboard.</div>",
                unsafe_allow_html=True)
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("### What You Need")
        st.markdown("""
Your CSV can have **any column names** — Claude will map them automatically.

**Required (minimum):**
- A **date** column
- A **store / location** column  
- A **sales / units sold** column

**Recommended for better forecasts:**
- Category / department
- Price per unit
- Event / holiday flags
- Product / SKU ID
        """)
    with col2:
        st.markdown("### Sample Dataset")
        st.download_button(
            label="⬇️ Download Sample CSV", data=SAMPLE_CSV,
            file_name="sample_retail_data.csv", mime="text/csv"
        )
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='warning-badge'>⚡ Any column names accepted</div>",
                    unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload your retail sales CSV", type=['csv'],
        help="Any CSV with at minimum: date, store and sales columns"
    )

    if uploaded_file is not None:

        # Step 1 — Load
        st.markdown("### Step 1 — Data Preview")
        try:
            raw_df = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded **{len(raw_df):,} rows** × **{len(raw_df.columns)} columns**")
            st.dataframe(raw_df.head(5), use_container_width=True)
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            st.stop()

        # Step 2 — Claude mapping
        st.markdown("### Step 2 — AI Column Detection")
        with st.spinner("🤖 Claude is analyzing your columns..."):
            mapping = map_columns_with_claude(
                raw_df.columns.tolist(),
                raw_df.head(3).to_string(index=False)
            )

        if "error" in mapping:
            st.error(f"Column mapping failed: {mapping['error']}")
            st.stop()

        st.markdown("**Claude's mapping:**")
        icons = {"date":"📅","store_id":"🏪","category":"📦",
                 "sales":"📈","price":"💰","event":"🎉","item_id":"🏷️"}
        cols  = st.columns(4)
        for i, (field, col_name) in enumerate(mapping.items()):
            with cols[i % 4]:
                color  = "#4A90D9" if (col_name and col_name != "null") else "#7a5a1a"
                label  = col_name if (col_name and col_name != "null") else "not found"
                st.markdown(
                    f"<div class='mapping-card' style='border-left-color:{color}'>"
                    f"{icons.get(field,'•')} <b>{field}</b><br><code>{label}</code></div>",
                    unsafe_allow_html=True
                )

        with st.expander("✏️ Override column mapping"):
            all_cols = ['(not available)'] + raw_df.columns.tolist()
            oc1, oc2, oc3 = st.columns(3)
            def safe_idx(col): return all_cols.index(col) if col in all_cols else 0
            with oc1:
                mapping['date']     = st.selectbox("Date",     all_cols, index=safe_idx(mapping.get('date','')))
                mapping['store_id'] = st.selectbox("Store",    all_cols, index=safe_idx(mapping.get('store_id','')))
            with oc2:
                mapping['sales']    = st.selectbox("Sales",    all_cols, index=safe_idx(mapping.get('sales','')))
                mapping['category'] = st.selectbox("Category", all_cols, index=safe_idx(mapping.get('category','')))
            with oc3:
                mapping['price']    = st.selectbox("Price",    all_cols, index=safe_idx(mapping.get('price','')))
                mapping['event']    = st.selectbox("Event",    all_cols, index=safe_idx(mapping.get('event','')))
            mapping = {k: (v if v != '(not available)' else None) for k, v in mapping.items()}

        # Step 3 — Feature Engineering
        st.markdown("### Step 3 — Feature Engineering")
        try:
            mapped = pd.DataFrame()
            mapped['date']     = pd.to_datetime(raw_df[mapping['date']], errors='coerce')
            mapped['store_id'] = raw_df[mapping['store_id']].astype(str)
            mapped['sales']    = pd.to_numeric(raw_df[mapping['sales']], errors='coerce').fillna(0)
            mapped['category'] = raw_df[mapping['category']].astype(str) \
                                 if (mapping.get('category') and mapping['category'] in raw_df.columns) \
                                 else 'GENERAL'
            if mapping.get('price') and mapping['price'] in raw_df.columns:
                mapped['price'] = pd.to_numeric(raw_df[mapping['price']], errors='coerce')
            if mapping.get('event') and mapping['event'] in raw_df.columns:
                mapped['event'] = raw_df[mapping['event']]
            mapped = mapped.dropna(subset=['date', 'sales'])

            with st.spinner("⚙️ Engineering features..."):
                mapped = engineer_features(mapped)

            st.success(f"✅ {len(mapped):,} rows · {len(mapped.columns)} features ready")

        except Exception as e:
            st.error(f"Feature engineering failed: {e}")
            st.stop()

        # Step 4 — Train
        st.markdown("### Step 4 — Training LightGBM")
        with st.spinner("🚀 Training model on your data..."):
            result = train_and_predict(mapped)

        if len(result) == 3 and isinstance(result[2], str):
            st.error(result[2])
            st.stop()

        pred_df, feat_imp_custom, metrics_custom, _ = result

        c1, c2, c3 = st.columns(3)
        c1.metric("R² Score", f"{metrics_custom['r2']:.4f}")
        c2.metric("RMSE",     f"{metrics_custom['rmse']:.4f}")
        c3.metric("MAE",      f"{metrics_custom['mae']:.4f}")
        st.success("✅ Model trained! Dashboard ready.")
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)

        # Step 5 — Dashboard
        st.markdown("### Step 5 — Your Dashboard")

        stores_c = sorted(pred_df['store_id'].unique())
        cats_c   = ['All'] + sorted(pred_df['category'].unique())

        with st.sidebar:
            st.markdown("<hr class='divider'>", unsafe_allow_html=True)
            st.markdown("**Your Dataset**")
            sel_store = st.selectbox("🏪 Store",    stores_c, key='cs')
            sel_cat   = st.selectbox("📦 Category", cats_c,   key='cc')
            dr        = st.date_input(
                "📅 Date Range",
                value=[pred_df['date'].min(), pred_df['date'].max()],
                min_value=pred_df['date'].min(), max_value=pred_df['date'].max(),
                key='cd'
            )

        c1, c2 = st.columns([3, 1])
        with c1:
            st.markdown("<div class='main-title'>📊 Custom Dataset Dashboard</div>",
                        unsafe_allow_html=True)
            st.markdown(f"<div class='subtitle'>{uploaded_file.name} · "
                        f"LightGBM · Auto-mapped by Claude AI</div>",
                        unsafe_allow_html=True)
        with c2:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"<div style='text-align:right'>"
                        f"<span class='store-badge'>{sel_store}</span></div>",
                        unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        render_dashboard(
            preds_df=pred_df, metrics_df=metrics_custom, feat_imp=feat_imp_custom,
            selected_store=sel_store, selected_cat=sel_cat, date_range=dr,
            global_r2=metrics_custom['r2']
        )

        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        st.markdown("### 📥 Download Predictions")
        st.download_button(
            label="⬇️ Download Predictions CSV",
            data=pred_df.to_csv(index=False),
            file_name=f"predictions_{uploaded_file.name}",
            mime="text/csv"
        )

    else:
        st.markdown("""
        <div class='upload-box'>
            <h3 style='color:#4A90D9'>⬆️ Upload a CSV to get started</h3>
            <p style='color:#8b8fa8'>Any retail CSV with date, store and sales columns</p>
            <p style='color:#8b8fa8'>Claude AI maps your columns automatically</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 👀 Sample Dataset Format")
        st.dataframe(pd.read_csv(io.StringIO(SAMPLE_CSV)), use_container_width=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("<center style='color:#4a4f6a; font-size:0.8rem;'>"
                "🛒 Retail Sales Forecasting · AI Column Detection · "
                "LightGBM · Streamlit + Plotly</center>",
                unsafe_allow_html=True)
