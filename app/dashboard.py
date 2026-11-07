import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

# Configuración de página
st.set_page_config(
    page_title="Markov CLV & Retention Simulator",
    layout="wide"
)

# Estilo CSS personalizado
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1E3A8A; margin-bottom: 0.2rem; }
    .sub-header { font-size: 1.1rem; color: #4B5563; margin-bottom: 1.5rem; }
    .card { background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 0.5rem; padding: 1.25rem; }
</style>
""", unsafe_allow_html=True)


# Funciones de notebook 2 y 3


# Definición de constantes
ETAPAS = ["E1: 1-12", "E2: 13-24", "E3: 25-48", "E4: 49-72"]
ESTADOS = ["E1: 1-12", "E2: 13-24", "E3: 25-48", "E4: 49-72", "E5: Cancelado"]
DURACION_ETAPAS = [12, 12, 24, 24]

# cargar datos con caching para mejorar el rendimiento
@st.cache_data
def cargar_datos():
    ruta = "data/processed/telco_estados.csv"
    if not os.path.exists(ruta):
        ruta = "telco_estados.csv"
    return pd.read_csv(ruta)

# Funciones de cálculo de tasas, matrices y CLV
def estimar_tasas_cancelacion(df_data, reduccion_churn=0.0):
    """Estima la tasa mensual de cancelación (q_s,c) por etapa y tipo de contrato."""
    resultados = {}
    contratos = ["Mensual", "1 año", "2 años"]
    limites = [(1, 12), (13, 24), (25, 48), (49, 72)]
    
    for c in contratos:
        df_c = df_data[df_data["contrato"] == c]
        q_dict = {}
        for i, etapa in enumerate(ETAPAS):
            m_min, m_max = limites[i]
            eventos = len(df_c[(df_c["churn"] == 1) & (df_c["tenure"] >= m_min) & (df_c["tenure"] <= m_max)])
            
            meses_riesgo = 0
            for _, row in df_c.iterrows():
                t = row["tenure"]
                if t >= m_min:
                    meses_riesgo += min(t, m_max) - m_min + 1
                    
            q = eventos / meses_riesgo if meses_riesgo > 0 else 0.001
            q_adj = q * (1.0 - reduccion_churn)
            q_dict[etapa] = q_adj
        resultados[c] = q_dict
    return resultados

# Construcción de la matriz de transición
def construir_matriz_transicion(q_dict):
    """Construye la matriz de transición P (5x5) de la Cadena de Markov."""
    P = np.zeros((5, 5))
    for i in range(4):
        q = q_dict[ETAPAS[i]]
        L = DURACION_ETAPAS[i]
        p_stay = 1.0 - (1.0 / L) - q
        p_advance = 1.0 / L
        
        P[i, i] = max(0.0, p_stay)
        P[i, 4] = q
        if i < 3:
            P[i, i+1] = max(0.0, p_advance)
        else:
            P[i, 3] += max(0.0, p_advance)
            
    P[4, 4] = 1.0  # Estado Absorbente (Cancelado)
    return P

# Cálculo de la matriz fundamental y tiempo esperado hasta absorción
def calcular_matriz_fundamental(P):
    """Calcula N = (I - Q)^(-1) y el tiempo esperado hasta absorción."""
    Q = P[:4, :4]
    I = np.eye(4)
    try:
        N = np.linalg.inv(I - Q)
    except np.linalg.LinAlgError:
        N = np.eye(4)
    tiempo_esperado = N.sum(axis=1)
    return N, tiempo_esperado

# Cálculo del CLV probabilístico ajustado por descuento y costo de retención
def calcular_clv_probabilistico(P, pago_mensual, tasa_descuento_anual=0.10, costo_retencion_mensual=0.0):
    """Calcula el CLV esperado ajustado por descuento financiero (WACC) y costo de retención."""
    d_mensual = (1 + tasa_descuento_anual)**(1/12) - 1
    Q = P[:4, :4]
    I = np.eye(4)
    N_d = np.linalg.inv(I - (Q / (1 + d_mensual)))
    margen_neto = max(0.0, pago_mensual - costo_retencion_mensual)
    clv_por_etapa = N_d.sum(axis=1) * margen_neto
    return clv_por_etapa

# Cargar datos
df_raw = cargar_datos()

# interfaz de usuario con Streamlit
st.markdown('<div class="main-header">Modelo Markoviano de Churn & CLV Probabilístico</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Cuantificación del tiempo esperado hasta cancelación y simulación del ROI de estrategias de retención</div>', unsafe_allow_html=True)

st.sidebar.header("Parámetros de Retención")

# Parámetros de la campaña de retención
reduccion_churn_pct = st.sidebar.slider(
    "Reducción de Cancelación (%)",
    min_value=0, max_value=50, value=15, step=1,
    help="Porcentaje en que disminuye la tasa de churn mensual con la campaña"
) / 100.0

# Parámetro de descuento al cliente
descuento_precio_pct = st.sidebar.slider(
    "Descuento al Cliente (%)",
    min_value=0, max_value=30, value=5, step=1,
    help="Descuento ofrecido en la tarifa mensual"
) / 100.0

# Costo operativo de retención
costo_retencion_cliente = st.sidebar.number_input(
    "Costo Operativo de Retención ($/cliente/mes)",
    min_value=0.0, max_value=50.0, value=2.0, step=0.5,
    help="Costo directo mensual por aplicar la campaña de retención"
)

# Tasa de descuento anual (WACC)
tasa_descuento_anual = st.sidebar.slider(
    "Tasa de Descuento Anual (WACC %)",
    min_value=1, max_value=25, value=10, step=1
) / 100.0

# Estimación de matrices de transición
tasas_base = estimar_tasas_cancelacion(df_raw, reduccion_churn=0.0)
tasas_retencion = estimar_tasas_cancelacion(df_raw, reduccion_churn=reduccion_churn_pct)

# pestañas de la aplicación
tab1, tab2, tab3 = st.tabs(["Impacto Económico General", "Segmentación & Inversión", "Simulador de Cliente"])

# impacto económico general de la cartera
with tab1:
    st.subheader("Simulación Económica de la Cartera de Clientes")
    
    df_calc = df_raw.copy()
    clv_base_list, clv_ret_list, tiempo_base_list, tiempo_ret_list = [], [], [], []
    
    for idx, row in df_calc.iterrows():
        c = row["contrato"]
        etapa_idx = ETAPAS.index(row["etapa"]) if row["etapa"] in ETAPAS else 0
        pago_base = row["MonthlyCharges"]
        pago_ret = pago_base * (1.0 - descuento_precio_pct)
        
        # Base
        P_base = construir_matriz_transicion(tasas_base[c])
        _, t_esp_base = calcular_matriz_fundamental(P_base)
        clv_b = calcular_clv_probabilistico(P_base, pago_base, tasa_descuento_anual, 0.0)[etapa_idx]
        
        # Con Programa
        P_ret = construir_matriz_transicion(tasas_retencion[c])
        _, t_esp_ret = calcular_matriz_fundamental(P_ret)
        clv_r = calcular_clv_probabilistico(P_ret, pago_ret, tasa_descuento_anual, costo_retencion_cliente)[etapa_idx]
         
        # Agregar resultados a las listas
        clv_base_list.append(clv_b)
        clv_ret_list.append(clv_r)
        tiempo_base_list.append(t_esp_base[etapa_idx])
        tiempo_ret_list.append(t_esp_ret[etapa_idx])

    # Agregar resultados al DataFrame
    df_calc["CLV_Base"] = clv_base_list
    df_calc["CLV_Retencion"] = clv_ret_list
    df_calc["Vida_Base"] = tiempo_base_list
    df_calc["Vida_Retencion"] = tiempo_ret_list
    df_calc["Diff_CLV"] = df_calc["CLV_Retencion"] - df_calc["CLV_Base"]
    # Filtrar clientes activos (churn = 0)
    df_activos = df_calc[df_calc["churn"] == 0]

    # Cálculo de métricas clave
    clv_total_base = df_activos["CLV_Base"].sum()
    clv_total_retencion = df_activos["CLV_Retencion"].sum()
    ganancia_neta_total = clv_total_retencion - clv_total_base

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Valor Total Cartera (Base)", f"${clv_total_base:,.0f}")
    with col2:
        st.metric("Valor Total (Con Programa)", f"${clv_total_retencion:,.0f}")
    with col3:
        st.metric("Impacto Neto Ganancia/Pérdida", f"${ganancia_neta_total:,.0f}", 
                  delta=f"{(ganancia_neta_total/clv_total_base)*100:.2f}%")
    with col4:
        vida_prom_base = df_activos["Vida_Base"].mean()
        vida_prom_ret = df_activos["Vida_Retencion"].mean()
        st.metric("Vida Media Esperada", f"{vida_prom_ret:.1f} meses", 
                  delta=f"{vida_prom_ret - vida_prom_base:+.1f} meses")
        
    st.markdown("---")

    # Visualización del impacto económico total
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        fig_clv = px.bar(
            x=["Escenario Base", "Con Programa Retención"],
            y=[clv_total_base, clv_total_retencion],
            color=["Base", "Retención"],
            color_discrete_sequence=["#64748B", "#2563EB"],
            title="Valor del Portafolio de Clientes Activos ($)",
            labels={"x": "Escenario", "y": "Valor Presente Neto CLV ($)"}
        )
        st.plotly_chart(fig_clv, use_container_width=True)

    # Visualización del impacto por tipo de contrato
    with col_g2:
        df_contrato_summary = df_activos.groupby("contrato")[["CLV_Base", "CLV_Retencion"]].sum().reset_index()
        fig_cont = px.bar(
            df_contrato_summary, x="contrato", y=["CLV_Base", "CLV_Retencion"],
            barmode="group",
            title="Valor Presente Neto por Tipo de Contrato",
            labels={"value": "CLV Total ($)", "contrato": "Tipo de Contrato"}
        )
        st.plotly_chart(fig_cont, use_container_width=True)

# tab 2: segmentación y análisis de inversión
with tab2:
    st.subheader("¿En qué segmentos conviene invertir en retención?")

    # Resumen por segmento
    resumen_seg = df_activos.groupby(["contrato", "etapa"]).agg(
        Clientes=("customerID", "count"),
        Pago_Promedio=("MonthlyCharges", "mean"),
        CLV_Base_Prom=("CLV_Base", "mean"),
        CLV_Ret_Prom=("CLV_Retencion", "mean"),
        Impacto_Unitario=("Diff_CLV", "mean"),
        Impacto_Total=("Diff_CLV", "sum")
    ).reset_index()

    # Visualización de la matriz de decisión de inversión
    fig_bubble = px.scatter(
        resumen_seg,
        x="CLV_Base_Prom",
        y="Impacto_Unitario",
        size="Clientes",
        color="contrato",
        hover_name="etapa",
        text="etapa",
        title="Matriz de Decisión de Inversión (Ganancia Neta por Cliente vs. CLV Base)",
        labels={"CLV_Base_Prom": "CLV Promedio Actual ($)", "Impacto_Unitario": "Ganancia Neta Promedio por Cliente ($)"}
    )

    # Añadir línea de referencia en y=0 para indicar el umbral de rentabilidad
    fig_bubble.add_hline(y=0, line_dash="dash", line_color="red")
    st.plotly_chart(fig_bubble, use_container_width=True)

    # Resumen tabular de la evaluación de rentabilidad por segmento
    st.markdown("### Tabla de Evaluación de Rentabilidad por Segmento")
    st.dataframe(resumen_seg.style.format({
        "Pago_Promedio": "${:.2f}",
        "CLV_Base_Prom": "${:.2f}",
        "CLV_Ret_Prom": "${:.2f}",
        "Impacto_Unitario": "${:.2f}",
        "Impacto_Total": "${:,.0f}"
    }), use_container_width=True)

# tab 3: simulador individual de cliente
with tab3:
    st.subheader("Simulador Individual de Cliente")

    # Inputs del simulador
    c1, c2, c3 = st.columns(3)
    with c1:
        sim_contrato = st.selectbox("Tipo de Contrato", ["Mensual", "1 año", "2 años"])
    with c2:
        sim_tenure = st.number_input("Antigüedad (Meses)", min_value=1, max_value=72, value=6)
    with c3:
        sim_pago = st.number_input("Pago Mensual ($)", min_value=10.0, max_value=200.0, value=65.0, step=5.0)

    # Determinación de la etapa del cliente según su antigüedad    
    if sim_tenure <= 12:
        sim_etapa = "E1: 1-12"
    elif sim_tenure <= 24:
        sim_etapa = "E2: 13-24"
    elif sim_tenure <= 48:
        sim_etapa = "E3: 25-48"
    else:
        sim_etapa = "E4: 49-72"
        
    idx_etapa = ETAPAS.index(sim_etapa)
    
    # Cálculos
    P_ind_base = construir_matriz_transicion(tasas_base[sim_contrato])
    _, t_esp_ind_base = calcular_matriz_fundamental(P_ind_base)
    clv_ind_base = calcular_clv_probabilistico(P_ind_base, sim_pago, tasa_descuento_anual, 0.0)[idx_etapa]

    # Cálculo del CLV con programa de retención
    pago_ret_ind = sim_pago * (1.0 - descuento_precio_pct)
    P_ind_ret = construir_matriz_transicion(tasas_retencion[sim_contrato])
    _, t_esp_ind_ret = calcular_matriz_fundamental(P_ind_ret)
    clv_ind_ret = calcular_clv_probabilistico(P_ind_ret, pago_ret_ind, tasa_descuento_anual, costo_retencion_cliente)[idx_etapa]
    # Diferencia de CLV
    diff_clv_ind = clv_ind_ret - clv_ind_base
    
    st.markdown("---")
    res_c1, res_c2, res_c3, res_c4 = st.columns(4)
    with res_c1:
        st.metric("Etapa Asignada", sim_etapa)
    with res_c2:
        st.metric("Tiempo Esperado de Permanencia", f"{t_esp_ind_ret[idx_etapa]:.1f} meses", 
                  delta=f"{t_esp_ind_ret[idx_etapa] - t_esp_ind_base[idx_etapa]:+.1f} meses")
    with res_c3:
        st.metric("CLV Esperado (Base)", f"${clv_ind_base:,.2f}")
    with res_c4:
        st.metric("CLV Esperado (Con Programa)", f"${clv_ind_ret:,.2f}", 
                  delta=f"${diff_clv_ind:+,.2f}")

    # Visualización de la matriz de transición para el cliente simulado
    st.markdown("### Matriz de Transición de Markov (5x5)")
    df_matriz = pd.DataFrame(P_ind_base, index=ESTADOS, columns=ESTADOS)
    st.dataframe(df_matriz.style.format("{:.4f}").background_gradient(cmap="Blues"), use_container_width=True)