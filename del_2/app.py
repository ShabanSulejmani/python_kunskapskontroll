import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import time
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go


# Konfigurera sidan
st.set_page_config(
    page_title="Guldfynd Diamantanalys",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set Seaborn style
sns.set_style("whitegrid")

# Ladda rensad data med simulerad realtidsdata
@st.cache_data
def load_data():
    df = pd.read_csv("diamonds_clean.csv")
    # Simulera realtidspriser genom att lägga till små variationer
    np.random.seed(42)
    df['price_variation'] = np.random.normal(0, 0.02, len(df))  # ±2% variation
    return df

# Simulera realtidspriser
def update_realtime_prices(df):
    current_time = datetime.now()
    # Simulera marknadsrörelser baserat på tid
    time_factor = np.sin(current_time.second / 10) * 0.01  # Små cykliska rörelser
    noise = np.random.normal(0, 0.005, len(df))  # Liten slumpmässig brus

    df['current_price'] = df['price'] * (1 + df['price_variation'] + time_factor + noise)
    df['price_change'] = ((df['current_price'] - df['price']) / df['price']) * 100
    return df

df = load_data()

# Titel och introduktion
st.title("💎 Interaktiv Diamantanalys för Guldfynd")
st.markdown("""
**Realtidsanalys av diamantpriser och marknadsinsikter**  
Använd filtren för att analysera olika diamantsegment och få insikter för lager, prissättning och försäljning.
""")

# Realtidsuppdatering
if 'last_update' not in st.session_state:
    st.session_state.last_update = time.time()

# Auto-refresh toggle
auto_refresh = st.sidebar.checkbox("🔄 Auto-refresh priser (5s)", value=False)

if auto_refresh:
    if time.time() - st.session_state.last_update > 5:
        st.session_state.last_update = time.time()
        st.rerun()

# Uppdatera priser
df = update_realtime_prices(df)

# Sidebar för filter
st.sidebar.header("🔍 Filter")
cut_options = df['cut'].unique()
selected_cut = st.sidebar.multiselect("Välj slipkvalitet", cut_options, default=cut_options)

color_options = df['color'].unique()
selected_color = st.sidebar.multiselect("Välj färg", color_options, default=color_options)

clarity_options = df['clarity'].unique()
selected_clarity = st.sidebar.multiselect("Välj klarhet", clarity_options, default=clarity_options)

carat_min, carat_max = st.sidebar.slider("Carat-intervall",
                                         float(df['carat'].min()),
                                         float(df['carat'].max()),
                                         (float(df['carat'].min()), float(df['carat'].max())))

# Filtrera datan
filtered_df = df[
    (df['cut'].isin(selected_cut)) &
    (df['color'].isin(selected_color)) &
    (df['clarity'].isin(selected_clarity)) &
    (df['carat'] >= carat_min) &
    (df['carat'] <= carat_max)
    ]

# Visualiseringsalternativ
st.sidebar.header("📊 Visualiseringsalternativ")
color_by = st.sidebar.selectbox("Färglägg scatterplot efter:", ['cut', 'color', 'clarity'])
pie_variable = st.sidebar.selectbox("Visa fördelning för:", ['cut', 'color', 'clarity'])

# Huvudlayout med kolumner
col1, col2, col3 = st.columns([2, 2, 1])

# Realtidspris-dashboard
with col3:
    st.subheader("📈 Realtidspriser")
    st.caption(f"Uppdaterad: {datetime.now().strftime('%H:%M:%S')}")

    # Visa top 5 dyraste diamanter med prisändring
    top_diamonds = filtered_df.nlargest(5, 'current_price')[['carat', 'cut', 'color', 'clarity', 'current_price', 'price_change']]

    for idx, row in top_diamonds.iterrows():
        price_color = "🟢" if row['price_change'] >= 0 else "🔴"
        st.metric(
            label=f"{row['carat']}ct {row['cut']} {row['color']}-{row['clarity']}",
            value=f"${row['current_price']:,.0f}",
            delta=f"{row['price_change']:+.2f}%"
        )

# Datavisning i kolumner
with col1:
    st.subheader("💍 Diamantdata")

    # Visa filtrerad data med realtidspriser
    display_df = filtered_df[['carat', 'cut', 'color', 'clarity', 'current_price', 'price_change']].copy()
    display_df['current_price'] = display_df['current_price'].round(0).astype(int)
    display_df['price_change'] = display_df['price_change'].round(2)
    display_df.columns = ['Karat', 'Slip', 'Färg', 'Klarhet', 'Aktuellt Pris ($)', 'Förändring (%)']

    st.dataframe(
        display_df,
        use_container_width=True,
        height=400,
        hide_index=True,
        column_config={
            "Aktuellt Pris ($)": st.column_config.NumberColumn(
                format="$%d"
            ),
            "Förändring (%)": st.column_config.NumberColumn(
                format="%.2f%%"
            )
        }
    )

with col2:
    st.subheader("📊 Marknadsöversikt")

    # Skapa en sammanfattningstabell
    summary_stats = pd.DataFrame({
        'Metrik': ['Totalt antal', 'Genomsnittspris', 'Mediankarat', 'Högsta pris', 'Lägsta pris'],
        'Värde': [
            f"{len(filtered_df):,}",
            f"${filtered_df['current_price'].mean():,.0f}",
            f"{filtered_df['carat'].median():.2f} ct",
            f"${filtered_df['current_price'].max():,.0f}",
            f"${filtered_df['current_price'].min():,.0f}"
        ]
    })

    st.dataframe(summary_stats, use_container_width=True, hide_index=True)

    # Prisfördelning histogram
    fig_hist = px.histogram(
        filtered_df,
        x='current_price',
        nbins=30,
        title="Prisfördelning",
        labels={'current_price': 'Pris ($)', 'count': 'Antal'}
    )
    fig_hist.update_layout(height=300)
    st.plotly_chart(fig_hist, use_container_width=True)

# Sektion 1: Korrelation mellan karat och pris
st.subheader("💎 Korrelation mellan karat och pris")
correlation_carat = filtered_df[['carat', 'current_price']].corr().iloc[0, 1]

col1, col2 = st.columns([3, 1])
with col2:
    st.metric("Korrelationskoefficient", f"{correlation_carat:.3f}")

with col1:
    fig_scatter = px.scatter(
        filtered_df,
        x='carat',
        y='current_price',
        color=color_by,
        size='carat',
        hover_data=['cut', 'color', 'clarity'],
        title="Samband mellan karat och pris",
        labels={'current_price': 'Pris ($)', 'carat': 'Karat'}
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

# Sektion 2: Genomsnittspris per karat-intervall
st.subheader("📊 Genomsnittspris per karat-intervall")
bins = [0, 0.5, 1.0, 1.5, 2.0, float('inf')]
labels = ['0-0.5', '0.5-1.0', '1.0-1.5', '1.5-2.0', '2.0+']
filtered_df['carat_range'] = pd.cut(filtered_df['carat'], bins=bins, labels=labels)
avg_price_by_carat_range = filtered_df.groupby('carat_range', observed=True)['current_price'].mean().reset_index()

fig_bar = px.bar(
    avg_price_by_carat_range,
    x='carat_range',
    y='current_price',
    title="Genomsnittspris per karat-intervall",
    labels={'current_price': 'Genomsnittspris ($)', 'carat_range': 'Karat-intervall'}
)
st.plotly_chart(fig_bar, use_container_width=True)

# Sektion 3: Separata visualiseringar för central- och spridningsmått
st.subheader("🎯 Statistisk analys per karat-intervall")

# Beräkna statistik
stats_by_carat_range = filtered_df.groupby('carat_range', observed=True)['current_price'].agg(['mean', 'median', 'std', 'var']).reset_index()

# Skapa två kolumner för graferna
col1, col2 = st.columns(2)

# Graf 1: Median och Medelvärde
with col1:
    st.subheader("📊 Median och Medelvärde")
    fig_median = go.Figure()
    fig_median.add_trace(go.Bar(x=stats_by_carat_range['carat_range'], y=stats_by_carat_range['median'], name='Median', marker_color='lightblue'))
    fig_median.add_trace(go.Scatter(x=stats_by_carat_range['carat_range'], y=stats_by_carat_range['mean'], mode='lines+markers', name='Medelvärde', line=dict(color='red', width=3)))
    fig_median.update_layout(title="Median och Medelvärde per karat-intervall", xaxis_title="Karat-intervall", yaxis_title="Pris ($)", height=400)
    st.plotly_chart(fig_median, use_container_width=True)

# Graf 2: Varians och Standardavvikelse
with col2:
    st.subheader("📈 Varians och Standardavvikelse")
    fig_var = go.Figure()
    fig_var.add_trace(go.Bar(x=stats_by_carat_range['carat_range'], y=stats_by_carat_range['var'], name='Varians', marker_color='orange', opacity=0.7))
    fig_var.add_trace(go.Scatter(x=stats_by_carat_range['carat_range'], y=stats_by_carat_range['mean'], error_y=dict(type='data', array=stats_by_carat_range['std'], visible=True, color='darkred', thickness=3, width=10), mode='markers', name='Medelvärde ± Std', marker=dict(color='darkred', size=8)))
    fig_var.update_layout(title="Varians och Standardavvikelse per karat-intervall", xaxis_title="Karat-intervall", yaxis_title="Varians ($²) / Pris ($)", height=400)
    st.plotly_chart(fig_var, use_container_width=True)

# Statistiktabell
st.subheader("📋 Detaljerad statistik")
stats_display = stats_by_carat_range.copy()
stats_display.columns = ['Karat-intervall', 'Medelvärde', 'Median', 'Standardavvikelse', 'Varians']
for col in ['Medelvärde', 'Median', 'Standardavvikelse', 'Varians']:
    stats_display[col] = stats_display[col].round(0).astype(int)
st.dataframe(stats_display, use_container_width=True, hide_index=True)

# Tolkning
st.markdown("""
**Tolkning av statistiken:**
- **Median och Medelvärde**: Om medelvärdet är högre än medianen kan det tyda på att några få dyra diamanter drar upp genomsnittet.
- **Varians och Standardavvikelse**: En hög varians visar stor prisvariation inom ett karat-intervall, vilket kan indikera risk eller prisfluktuationer.
""")

# Sektion 4: Genomsnittspris per klarhet
st.subheader("💍 Genomsnittspris per klarhet")
avg_price_by_clarity = filtered_df.groupby('clarity', observed=True)['current_price'].mean().reset_index()

fig_clarity = px.bar(
    avg_price_by_clarity,
    x='clarity',
    y='current_price',
    title="Genomsnittspris per klarhet",
    labels={'current_price': 'Genomsnittspris ($)', 'clarity': 'Klarhet'}
)
st.plotly_chart(fig_clarity, use_container_width=True)

# Sektion 5: Korrelations-heatmap
st.subheader("🔥 Korrelationsmatris för numeriska variabler")
corr_matrix = filtered_df[['carat', 'current_price', 'depth', 'table', 'x', 'y', 'z']].corr()

fig_heatmap = px.imshow(
    corr_matrix,
    title="Korrelationsmatris",
    color_continuous_scale="RdBu_r",
    zmin=-1,
    zmax=1,
    text_auto=True
)
st.plotly_chart(fig_heatmap, use_container_width=True)

# Sektion 6: Cirkeldiagram
st.subheader(f"🥧 Fördelning av diamanter per {pie_variable}")
pie_data = filtered_df[pie_variable].value_counts()

fig_pie = px.pie(
    values=pie_data.values,
    names=pie_data.index,
    title=f"Fördelning av diamanter per {pie_variable}"
)
st.plotly_chart(fig_pie, use_container_width=True)

# Expandable insikter och rekommendationer
with st.expander("🎯 Viktiga insikter"):
    st.markdown(f"""
    - **Karat vs. pris**: Stark korrelation ({correlation_carat:.3f}) - större diamanter är betydligt dyrare
    - **Klarhet vs. pris**: Klarhet påverkar priset, men mindre än karat
    - **Prisvariabilitet**: Större karat-intervall visar högre prisvariation och risk
    - **Marknadsfördelning**: {pie_variable.capitalize()} visar populära kategorier för lagerplanering
    - **Realtidstrend**: Prisrörelser följer marknadsvolatilitet
    """)

with st.expander("💡 Rekommendationer för Guldfynd"):
    st.markdown("""
    ### Lagerstrategi
    - **Fokusområde**: 0.5–1.5 karat med Ideal/Premium-slipning
    - **Kvalitet**: Prioritera VS1-IF klarhet för bästa värde
    
    ### Marknadsföring
    - **Premiumlinje**: >1.5 karat med D-F färg och VS1-IF klarhet
    - **Volymförsäljning**: 0.5-1.0 karat för bredare kundkrets
    
    ### Prissättning
    - **Basera på**: Primärt karat, sekundärt klarhet och slipkvalitet
    - **Riskhantering**: Var försiktig med stora karat p.g.a. hög prisvariation
    
    ### Kundsegmentering
    - **Prisvärt segment**: 0.5–1.0 karat för vardagssmycken
    - **Exklusivt segment**: >1.5 karat för speciella tillfällen
    """)

# Footer med realtidsinfo
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Totalt antal diamanter", f"{len(filtered_df):,}")
with col2:
    st.metric("Genomsnittlig prisändring", f"{filtered_df['price_change'].mean():+.3f}%")
with col3:
    if auto_refresh:
        st.success("🔄 Auto-refresh aktiverad")
    else:
        st.info("⏸️ Auto-refresh pausad")