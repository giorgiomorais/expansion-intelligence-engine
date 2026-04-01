import os
os.environ["OMP_NUM_THREADS"] = "1"

import streamlit as st
import pandas as pd
import numpy as np
import warnings
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import poisson
from io import BytesIO
from datetime import datetime

warnings.filterwarnings('ignore')

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Analytics Engine - Expansão Comercial", layout="wide")

# Estilização Sandbox
st.markdown("""
    <style>
    .main { background-color: #F8F9FA; }
    h1, h2, h3 { color: #1E3A8A; }
    .stMetric { background-color: #FFFFFF; padding: 15px; border-radius: 10px; border: 1px solid #E5E7EB; }
    </style>
    """, unsafe_allow_html=True)

st.title("🚀 Engine Preditiva para Expansão de Filiais")
st.caption("Framework de Clusterização, Similaridade de Frota e Otimização de Inventário")

# --- GERADOR DE DADOS SINTÉTICOS (MODO SANDBOX) ---
@st.cache_data
def carregar_dados_demo():
    # Simulação de Cadastro de Cidades
    cidades = pd.DataFrame({
        'CODIGO': [str(i) for i in range(100, 200)],
        'NOME': [f"CIDADE_{i}" for i in range(100, 200)],
        'UF': np.random.choice(['SP', 'MG', 'RJ', 'PR'], 100)
    })

    # Simulação de Potencial de Mercado
    potencial = pd.DataFrame({
        'CODIGO': cidades['CODIGO'],
        'POTENCIAL': np.random.uniform(50000, 500000, 100),
        'FROTA': np.random.uniform(1000, 10000, 100)
    })

    # Simulação de Histórico de Vendas (Últimos 12 meses)
    vendas = []
    for cod in cidades['CODIGO'][:30]: # Apenas 30 cidades com histórico
        for mes in range(1, 13):
            vendas.append({
                'CODIGO': cod,
                'CODPRODUTO': np.random.randint(1000, 1050),
                'FATURADO_BRUTO': np.random.uniform(500, 5000),
                'QUANTIDADE_BRUTA': np.random.randint(1, 20),
                'MES_REF': f"2025{str(mes).zfill(2)}"
            })
    df_vendas = pd.DataFrame(vendas)

    # Simulação de Catálogo
    produtos = pd.DataFrame({
        'CODPRODUTO': range(1000, 1050),
        'PRODUTO': [f"SKU_{i}" for i in range(1000, 1050)],
        'CATEGORIA': np.random.choice(['MOTOR', 'TRANSMISSAO', 'ELETRICA', 'CHASSI'], 50),
        'MARCA': np.random.choice(['MARCA_A', 'MARCA_B', 'MARCA_C'], 50)
    })

    # Simulação de Frota Detalhada (Para Similaridade de Cosseno)
    frota_list = []
    modelos = ['MODELO_X', 'MODELO_Y', 'MODELO_Z']
    for cod in cidades['CODIGO']:
        for mod in modelos:
            frota_list.append({
                'CODIGO': cod,
                'MODELANO': f"{mod}-2024",
                'QTD': np.random.randint(10, 500),
                'CIL': np.random.choice([125, 150, 160, 250, 600]),
                'MARCA': np.random.choice(['HONDA', 'YAMAHA', 'OUTROS'])
            })
    df_frota = pd.DataFrame(frota_list)
    tabela_frota = df_frota.pivot_table(index='CODIGO', columns='MODELANO', values='QTD', fill_value=0)

    return df_vendas, potencial, cidades, produtos, tabela_frota, df_frota

# --- FUNÇÕES AUXILIARES ---
def formatar_kpi(valor):
    return f"R$ {valor:,.2f}"

def calcular_matriz_rampagem(valor_final, venda_atual, modo, meses):
    ajustes = {"Conservadora": (1.25, 0.5), "Base": (2.3, -0.1), "Agressiva": (3.5, -0.5)}
    k, x0 = ajustes[modo]
    t = np.linspace(-2.0, 2.0, meses + 1)
    curva = 1 / (1 + np.exp(-k * (t - x0)))
    curva_norm = (curva - curva[0]) / (curva[-1] - curva[0])
    return venda_atual + (valor_final - venda_atual) * curva_norm

def preparar_base(cidades, vendas, potencial):
    base = cidades.merge(potencial, on='CODIGO', how='left')
    venda_media = vendas.groupby('CODIGO')['FATURADO_BRUTO'].mean().reset_index()
    base = base.merge(venda_media, on='CODIGO', how='left').fillna(0)

    scaler = StandardScaler()
    base['CLUSTER'] = KMeans(n_clusters=5, random_state=42).fit_predict(scaler.fit_transform(base[['POTENCIAL', 'FATURADO_BRUTO']]))
    return base

# --- LÓGICA DE SIMILARIDADE E PROJEÇÃO ---
def processar_expansao(base, uf_sel, tabela_frota, agressividade, meses_cobertura, lead_time, dt_abertura):
    df_alvo = base[base['UF'] == uf_sel].copy()
    candidatas = base[(base['UF'] != uf_sel) & (base['FATURADO_BRUTO'] > 0)]

    # Similaridade de Cosseno (Perfil de Frota)
    vetor_outros = tabela_frota[tabela_frota.index.isin(candidatas['CODIGO'])]
    ids_alvo = [i for i in df_alvo['CODIGO'] if i in tabela_frota.index]

    dict_gemea = {}
    if not vetor_outros.empty and ids_alvo:
        sim = cosine_similarity(tabela_frota.loc[ids_alvo], vetor_outros)
        for i, id_a in enumerate(ids_alvo):
            dict_gemea[id_a] = {'id': vetor_outros.index[np.argmax(sim[i])], 'score': np.max(sim[i])}

    df_alvo['ID_Gemea'] = df_alvo['CODIGO'].map(lambda x: dict_gemea.get(x, {}).get('id'))
    df_alvo = df_alvo.merge(base[['CODIGO', 'FATURADO_BRUTO']], left_on='ID_Gemea', right_on='CODIGO', how='left', suffixes=('', '_G'))

    df_alvo['V_Prevista'] = np.where(df_alvo['ID_Gemea'].notnull(), df_alvo['FATURADO_BRUTO_G'] * 1.1, base['FATURADO_BRUTO'].mean())

    def aplicar_rampa(row):
        return calcular_matriz_rampagem(row['V_Prevista'], 0, agressividade, 12)

    df_alvo['Venda_Acumulada_Rampa'] = df_alvo.apply(aplicar_rampa, axis=1)
    df_alvo['Investimento_Estoque'] = df_alvo['Venda_Acumulada_Rampa'].apply(lambda x: sum(x[:meses_cobertura]))

    # Backtesting
    df_alvo['Meses_Avaliados'] = 3
    df_alvo['Realizado_Acum'] = df_alvo['Investimento_Estoque'] * np.random.uniform(0.8, 1.1)
    df_alvo['Vendas_Reais_Lista'] = df_alvo['Venda_Acumulada_Rampa'].apply(lambda x: [v * np.random.uniform(0.8, 1.2) for v in x])

    return df_alvo

# --- CÁLCULO DE MIX (LÓGICA HUB) ---
def calcular_mix_sandbox(df_res, v_raw, p_cad):
    setup_total = df_res['Investimento_Estoque'].sum()
    mix = p_cad.copy()
    mix['Share'] = np.random.dirichlet(np.ones(len(mix)), size=1)[0]
    mix['Venda_Moeda'] = setup_total * mix['Share']
    mix['Preco'] = np.random.uniform(10, 500, len(mix))
    mix['mu'] = mix['Venda_Moeda'] / mix['Preco']

    mix = mix.sort_values('Venda_Moeda', ascending=False)
    mix['Pct'] = mix['Venda_Moeda'].cumsum() / mix['Venda_Moeda'].sum()
    mix['SLA'] = np.select([mix['Pct'] <= 0.8, mix['Pct'] <= 0.95], [0.95, 0.85], 0.75)
    mix['Estoque_Qtd'] = np.ceil(poisson.ppf(mix['SLA'], mix['mu']))
    mix['Valor_Setup'] = mix['Estoque_Qtd'] * mix['Preco']

    return mix[mix['Estoque_Qtd'] > 0]

# --- UI SIDEBAR ---
with st.sidebar:
    st.header("Configurações")
    uf_sel = st.selectbox("Estado Alvo", ['SP', 'MG', 'RJ', 'PR'])
    agres = st.select_slider("Agressividade", ["Conservadora", "Base", "Agressiva"], "Base")
    m_est = st.slider("Meses Cobertura", 1, 12, 4)
    lt = st.number_input("Lead Time (Dias)", 1, 30, 15)
    btn = st.button("Simular Expansão")

# --- EXECUÇÃO ---
v_raw, pot, cids, p_cad, tab_f, df_f_raw = carregar_dados_demo()

if btn:
    base_p = preparar_base(cids, v_raw, pot)
    df_res = processar_expansao(base_p, uf_sel, tab_f, agres, m_est, lt, "2025-11-17")
    df_mix = calcular_mix_sandbox(df_res, v_raw, p_cad)

    # --- DASHBOARD ---
    st.header("📈 Resultados da Simulação")
    k1, k2, k3 = st.columns(3)
    k1.metric("Projeção Venda (Mês 12)", formatar_kpi(df_res['V_Prevista'].sum()))
    k2.metric("Setup Inicial Inv.", formatar_kpi(df_mix['Valor_Setup'].sum()))

    real, prev = df_res['Realizado_Acum'].sum(), df_res['Investimento_Estoque'].sum()
    k3.metric("Aderência Backtesting", f"{max(1 - abs(real-prev)/real, 0):.1%}")

    t1, t2, t3 = st.tabs(["🚀 Rampa Comercial", "🏍️ Análise de Frota", "📊 Mix & Pareto"])

    with t1:
        m_cols = [f"Mês_{str(i).zfill(2)}" for i in range(13)]
        for i in range(13): df_res[m_cols[i]] = df_res['Venda_Acumulada_Rampa'].apply(lambda x: x[i])
        st.dataframe(df_res[['NOME', 'POTENCIAL', 'V_Prevista'] + m_cols], use_container_width=True)

        v_real = np.zeros(13)
        for l in df_res['Vendas_Reais_Lista']:
            for idx, v in enumerate(l): v_real[idx] += v

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=m_cols, y=df_res[m_cols].sum(), name="Previsto", line=dict(color='#1E3A8A', width=3)))
        fig.add_trace(go.Scatter(x=m_cols, y=v_real, name="Realizado (Fake)", line=dict(color='#DC2626', width=3)))
        st.plotly_chart(fig, use_container_width=True)

    with t2:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("🏍️ Pareto de Frota")
            f_p = df_f_raw.groupby('CIL')['QTD'].sum().reset_index().sort_values('QTD', ascending=False)
            f_p['CIL'] = f_p['CIL'].astype(str) + 'cc'
            f_p['Acc'] = (f_p['QTD'].cumsum() / f_p['QTD'].sum()) * 100
            fig_p = go.Figure()
            fig_p.add_trace(go.Bar(x=f_p['CIL'], y=f_p['QTD'], marker_color='#1E3A8A'))
            fig_p.add_trace(go.Scatter(x=f_p['CIL'], y=f_p['Acc'], yaxis="y2", line=dict(color='#DC2626')))
            fig_p.update_layout(yaxis2=dict(overlaying='y', side='right', range=[0, 105]), xaxis=dict(type='category'))
            st.plotly_chart(fig_p, use_container_width=True)
        with c2:
            st.plotly_chart(px.pie(df_f_raw.groupby('MARCA')['QTD'].sum().reset_index(), values='QTD', names='MARCA', hole=0.4))

    with t3:
        st.dataframe(df_mix[['PRODUTO', 'CATEGORIA', 'Estoque_Qtd', 'Valor_Setup', 'SLA']], use_container_width=True)
        abc = df_mix.groupby('CATEGORIA')['Valor_Setup'].sum().reset_index().sort_values('Valor_Setup', ascending=False)
        st.plotly_chart(px.bar(abc, x='CATEGORIA', y='Valor_Setup', title="Investimento por Categoria", color_discrete_sequence=['#1E3A8A']))
else:
    st.info("Clique em 'Simular Expansão' para gerar os dados do Sandbox.")
