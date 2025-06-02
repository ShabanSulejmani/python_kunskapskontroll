import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Set Seaborn style
sns.set_style("whitegrid")

# Ladda rensad data
@st.cache_data
def load_data():
    return pd.read_csv("diamonds_clean.csv")

df = load_data()

# Titel och introduktion
st.title("Interaktiv Diamantanalys för Guldfynd")
st.markdown("""
Denna analys använder rensad data för att undersöka hur karat, klarhet och andra faktorer påverkar diamantpriser. 
Använd filtren och dropdown-menynerna för att dynamiskt ändra visualiseringarna och få insikter för att optimera lager, prissättning och försäljning.
""")

# Sidebar för filter
st.sidebar.header("Filter")
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

# Dropdown för färgkodning i scatterplot
st.sidebar.header("Visualiseringsalternativ")
color_by = st.sidebar.selectbox("Färglägg scatterplot efter:", ['cut', 'color', 'clarity'])

# Dropdown för cirkeldiagram
pie_variable = st.sidebar.selectbox("Visa fördelning för:", ['cut', 'color', 'clarity'])

# Sektion 1: Korrelation mellan karat och pris med färg per karat
st.subheader("Korrelation mellan karat och pris (färg per karat)")
correlation_carat = filtered_df[['carat', 'price']].corr().iloc[0, 1]
st.write(f"Korrelationskoefficient (karat vs. pris): {correlation_carat:.3f}")
st.markdown("Varje karatvärde är färglagt för att visa hur priset varierar med karat.")

fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(x='carat', y='price', hue='carat', size='carat', palette='viridis', data=filtered_df, ax=ax)
ax.set_title("Samband mellan karat och pris (färg per karat)", fontsize=14, pad=10)
ax.set_xlabel("Karat", fontsize=12)
ax.set_ylabel("Pris (USD)", fontsize=12)
ax.legend(title="Karat", loc='upper left', bbox_to_anchor=(1, 1))
st.pyplot(fig)

# Sektion 2: Genomsnittspris per karat-intervall
st.subheader("Genomsnittspris per karat-intervall")
bins = [0, 0.5, 1.0, 1.5, 2.0]
labels = ['0-0.5', '0.5-1.0', '1.0-1.5', '1.5-2.0']
filtered_df['carat_range'] = pd.cut(filtered_df['carat'], bins=bins, labels=labels)
avg_price_by_carat_range = filtered_df.groupby('carat_range', observed=True)['price'].mean().reset_index()
avg_price_by_carat_range.columns = ['carat_range', 'mean_price']

fig, ax = plt.subplots(figsize=(8, 6))
sns.barplot(x='carat_range', y='mean_price', hue='carat_range', data=avg_price_by_carat_range, palette='Blues_d', legend=False, ax=ax)
ax.set_title("Genomsnittspris per karat-intervall", fontsize=14, pad=10)
ax.set_xlabel("Karat-intervall", fontsize=12)
ax.set_ylabel("Genomsnittspris (USD)", fontsize=12)
ax.tick_params(axis='x', rotation=45, labelsize=12)
plt.tight_layout()
st.pyplot(fig)

# Sektion 3: Varians och standardavvikelse per karat-intervall
st.subheader("Varians och standardavvikelse för pris per karat-intervall")
stats_by_carat_range = filtered_df.groupby('carat_range', observed=True)['price'].agg(['mean', 'std', 'var', 'median']).reset_index()
stats_by_carat_range.columns = ['carat_range', 'mean_price', 'std_price', 'var_price', 'median_price']

fig, ax = plt.subplots(figsize=(8, 6))
sns.barplot(x='carat_range', y='var_price', hue='carat_range', data=stats_by_carat_range, palette='Oranges_d', legend=False, ax=ax)
for i, row in stats_by_carat_range.iterrows():
    ax.axhline(y=row['mean_price'], xmin=i/len(stats_by_carat_range)+0.05, xmax=(i+1)/len(stats_by_carat_range)-0.05,
               color='blue', linestyle='--', label='Medelvärde' if i == 0 else "")
    ax.axhline(y=row['median_price'], xmin=i/len(stats_by_carat_range)+0.05, xmax=(i+1)/len(stats_by_carat_range)-0.05,
               color='green', linestyle='-', label='Median' if i == 0 else "")
    ax.errorbar(x=i, y=row['mean_price'], yerr=row['std_price'], fmt='none', c='red', capsize=5, elinewidth=2,
                label='Standardavvikelse' if i == 0 else "")
ax.set_title("Varians och standardavvikelse för pris per karat-intervall", fontsize=14, pad=10)
ax.set_xlabel("Karat-intervall", fontsize=12)
ax.set_ylabel("Varians (USD²) / Pris (USD)", fontsize=12)
ax.tick_params(axis='x', rotation=45, labelsize=12)
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()
st.pyplot(fig)

# Sektion 4: Genomsnittspris per klarhet
st.subheader("Genomsnittspris per klarhet")
avg_price_by_clarity = filtered_df.groupby('clarity', observed=True)['price'].mean().reset_index()
st.markdown("Klarhet påverkar priset, men effekten är mindre än karat.")

fig, ax = plt.subplots(figsize=(8, 6))
sns.barplot(x='clarity', y='price', hue='clarity', data=avg_price_by_clarity, palette='Purples_d', legend=False, ax=ax)
ax.set_title("Genomsnittspris per klarhet", fontsize=14, pad=10)
ax.set_xlabel("Klarhet", fontsize=12)
ax.set_ylabel("Genomsnittspris (USD)", fontsize=12)
ax.tick_params(axis='x', rotation=45, labelsize=12)
st.pyplot(fig)

# Sektion 5: Korrelations-heatmap
st.subheader("Korrelationsmatris för numeriska variabler")
st.markdown("Röda färger indikerar stark positiv korrelation.")

fig, ax = plt.subplots(figsize=(10, 8))
corr_matrix = filtered_df[['carat', 'price', 'depth', 'table', 'x', 'y', 'z']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0,
            fmt='.2f', linewidths=0.5, annot_kws={'size': 12}, ax=ax)
ax.set_title("Korrelationsmatris", fontsize=14, pad=25)
ax.tick_params(axis='x', rotation=45, labelsize=12)
ax.tick_params(axis='y', labelsize=12)
st.pyplot(fig)

# Sektion 6: Cirkeldiagram för fördelning
st.subheader(f"Fördelning av diamanter per {pie_variable}")
pie_data = filtered_df[pie_variable].value_counts()
fig, ax = plt.subplots(figsize=(8, 8))
sns.color_palette("Set2")
ax.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("Set2", len(pie_data)))
ax.set_title(f"Fördelning av diamanter per {pie_variable}", fontsize=14, pad=10)
plt.tight_layout()
st.pyplot(fig)

# Insikter
st.subheader("Viktiga insikter")
st.markdown(f"""
- **Karat vs. pris**: Stark korrelation ({correlation_carat:.3f}), större diamanter är dyrare.
- **Klarhet vs. pris**: Klarhet påverkar priset, men mindre än karat. Högre klarhet (t.ex. IF) ger högre priser.
- **Varians och standardavvikelse**: Större karat-intervall har högre varians och prisvariation, vilket indikerar risker i prissättning.
- **Fördelning**: {pie_variable.capitalize()} visar populära kategorier för lagerplanering.
""")

# Rekommendationer
st.subheader("Rekommendationer för Guldfynd")
st.markdown("""
1. **Lagerstrategi**: Prioritera 0.5–1.5 karat med Ideal/Premium-slipning och hög klarhet (VS1-IF).
2. **Marknadsföring**: Skapa en premiumlinje för >1.5 karat med hög kvalitet (D-F, VS1-IF).
3. **Prissättning**: Basera priser på karat, med premium för klarhet och slipkvalitet. Var försiktig med stora karat p.g.a. hög prisvariation.
4. **Kundsegmentering**: Erbjud prisvärda smycken (0.5–1.0 karat) och exklusiva produkter (>1.5 karat).
""")