import streamlit as st
import numpy as np
import pandas as pd
import scipy.stats as stats
import plotly.graph_objects as go
import plotly.express as px

# --- 1. CONFIGURACIÓN DE PÁGINA (CORREGIDO) ---
st.set_page_config(
    page_title="Reliarisk FlowCast",
    page_icon="📉",  # CORREGIDO: Emoji entre comillas
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. ESTILOS CSS (CORREGIDO) ---
st.markdown("""
    <style>
    .main { background-color: #f4f6f9; }
    .stButton>button {
        width: 100%;
        background-color: #004B87; 
        color: white;
        border-radius: 5px;
        font-weight: bold;
    }
    .metric-container {
        background-color: white;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #004B87;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True) # CORREGIDO: Cierre correcto de comillas triples

# --- 3. FUNCIONES MATEMÁTICAS (BACKEND) ---

def generate_samples(dist_type, params, n_iter):
    """Generador de variables estocásticas vectorizado."""
    if dist_type == 'Normal':
        return np.random.normal(params['mean'], params['std'], n_iter)
    elif dist_type == 'Lognormal':
        sigma = np.sqrt(np.log(1 + (params['std']/params['mean'])**2))
        mu = np.log(params['mean']) - 0.5 * sigma**2
        return np.random.lognormal(mu, sigma, n_iter)
    elif dist_type == 'Triangular':
        return np.random.triangular(params['min'], params['mode'], params['max'], n_iter)
    elif dist_type == 'BetaPERT':
        mn, md, mx = params['min'], params['mode'], params['max']
        alpha = (4 * md + mx - 5 * mn) / (mx - mn)
        beta = (5 * mx - mn - 4 * md) / (mx - mn)
        return mn + (mx - mn) * np.random.beta(alpha, beta, n_iter)
    elif dist_type == 'Uniforme':
        return np.random.uniform(params['min'], params['max'], n_iter)
    elif dist_type == 'Determinístico':
        return np.full(n_iter, params['value'])
    return np.zeros(n_iter)

# --- ECUACIONES DE AFLUENCIA (MÓDULO I) ---
def ipr_oil_darcy(k, h, Pr, Pwf, mu, Bo, re, rw, S):
    numerator = k * h * (Pr - Pwf)
    denominator = 141.2 * Bo * mu * (np.log(re/rw) + S)
    q = numerator / denominator
    return np.maximum(q, 0)

def ipr_oil_vogel(qmax, Pr, Pwf):
    ratio = Pwf / Pr
    q = qmax * (1 - 0.2 * ratio - 0.8 * (ratio**2))
    return np.maximum(q, 0)

def ipr_gas_backpressure(C, Pr, Pwf, n):
    term = (Pr**2 - Pwf**2)
    term = np.maximum(term, 0)
    q = C * (term**n)
    return q

# --- ECUACIONES DE DECLINACIÓN (MÓDULO II) ---
def arps_forecast(t_array, qi_vec, di_vec, b_vec):
    di_m = di_vec / 12.0
    qi = qi_vec[:, np.newaxis]
    b = b_vec[:, np.newaxis]
    di = di_m[:, np.newaxis]
    t = t_array[np.newaxis, :]
    
    is_hyp = b > 0.001
    
    term_hyp = (1 + b * di * t)
    q_hyp = qi / (term_hyp ** (1.0 / np.maximum(b, 1e-9)))
    q_exp = qi * np.exp(-di * t)
    
    return np.where(is_hyp, q_hyp, q_exp)

# --- HELPER PARA INPUTS ---
def render_dist_input(label, key, default_mode, default_min, default_max):
    dist = st.selectbox(f"Distribución {label}", 
                       ['BetaPERT', 'Lognormal', 'Normal', 'Triangular', 'Determinístico'],
                       key=f"d_{key}")
    p = {}
    cols = st.columns(3)
    if dist == 'BetaPERT':
        p['min'] = cols[0].number_input("Mín", value=float(default_min), key=f"mn_{key}")
        p['mode'] = cols[1].number_input("Moda", value=float(default_mode), key=f"md_{key}")
        p['max'] = cols[2].number_input("Máx", value=float(default_max), key=f"mx_{key}")
    elif dist == 'Normal':
        p['mean'] = cols[0].number_input("Media", value=float(default_mode), key=f"nm_{key}")
        p['std'] = cols[1].number_input("Std Dev", value=float((default_max-default_min)/6), key=f"ns_{key}")
    elif dist == 'Triangular':
        p['min'] = cols[0].number_input("Mín", value=float(default_min), key=f"tm_{key}")
        p['mode'] = cols[1].number_input("Moda", value=float(default_mode), key=f"tmd_{key}")
        p['max'] = cols[2].number_input("Máx", value=float(default_max), key=f"tmx_{key}")
    elif dist == 'Determinístico':
        p['value'] = cols[0].number_input("Valor", value=float(default_mode), key=f"dt_{key}")
    elif dist == 'Lognormal':
        p['mean'] = cols[0].number_input("Media", value=float(default_mode), key=f"lm_{key}")
        p['std'] = cols[1].number_input("Std Dev", value=float((default_max-default_min)/4), key=f"ls_{key}")
        
    return {'dist': dist, 'params': p}

# --- 4. APLICACIÓN PRINCIPAL ---

def main():
    # Sidebar
    with st.sidebar:
        try:
            st.image("mi-logo.png", use_container_width=True)
        except:
            # CORREGIDO: Emoji dentro de comillas
            st.warning("⚠️ Logo no cargado") 
        
        st.title("Configuración Global")
        fluid_type = st.radio("Tipo de Fluido", ["Aceite", "Gas"])
        n_iters = st.selectbox("Iteraciones Montecarlo", [1000, 5000, 10000], index=2)
    
    st.title("Reliarisk FlowCast")
    st.markdown("Plataforma Probabilística de Afluencia y Pronóstico de Producción")

    # CORREGIDO: Emojis dentro de comillas en las Tabs
    tab1, tab2 = st.tabs(["🔹 Módulo I: Prod. Inicial (Afluencia)", "🔹 Módulo II: Pronóstico (Declinación)"])

    # --- MÓDULO I: AFLUENCIA ---
    with tab1:
        st.header("Módulo I: Cálculo de Producción Inicial ($q_i$)")
        st.markdown("Este módulo calcula la capacidad de aporte del pozo (Afluencia) basándose en propiedades físicas.")
        
        col_m1_1, col_m1_2 = st.columns([1, 2])
        
        with col_m1_1:
            st.subheader("Modelo de Flujo")
            if fluid_type == "Aceite":
                model_ipr = st.selectbox("Seleccione Ecuación IPR", ["Darcy (Flujo Radial)", "Vogel (Saturado)"])
            else:
                model_ipr = st.selectbox("Seleccione Ecuación IPR", ["Back Pressure (C & n)"])
            
            st.subheader("Variables Estocásticas")
            
            inputs_m1 = {}
            if fluid_type == "Aceite" and model_ipr == "Darcy (Flujo Radial)":
                inputs_m1['k'] = render_dist_input("Permeabilidad k (mD)", "k", 50, 10, 100)
                inputs_m1['h'] = render_dist_input("Espesor h (ft)", "h", 100, 50, 150)
                inputs_m1['Pr'] = render_dist_input("Presión Yac. Pr (psi)", "pr", 3000, 2500, 3500)
                inputs_m1['Pwf'] = render_dist_input("Presión Fondo Pwf (psi)", "pwf", 2000, 1500, 2500)
                inputs_m1['mu'] = render_dist_input("Viscosidad (cp)", "mu", 1.5, 1.0, 2.0)
                inputs_m1['Bo'] = render_dist_input("Factor Vol. Bo", "bo", 1.2, 1.1, 1.3)
                inputs_m1['S'] = render_dist_input("Daño (Skin)", "s", 0, -2, 5)
                re = st.number_input("Radio de Drene re (ft)", value=1000.0)
                rw = st.number_input("Radio del Pozo rw (ft)", value=0.328)

            elif fluid_type == "Aceite" and model_ipr == "Vogel (Saturado)":
                inputs_m1['qmax'] = render_dist_input("Qmax (AOF) bbl/d", "qmax", 5000, 3000, 8000)
                inputs_m1['Pr'] = render_dist_input("Presión Yac. Pr (psi)", "pr_v", 3000, 2500, 3500)
                inputs_m1['Pwf'] = render_dist_input("Presión Fondo Pwf (psi)", "pwf_v", 2000, 1500, 2500)
            
            elif fluid_type == "Gas":
                inputs_m1['C'] = render_dist_input("Coeficiente C", "c_gas", 0.1, 0.01, 0.5)
                inputs_m1['n'] = render_dist_input("Exponente de Turbulencia n", "n_gas", 0.8, 0.5, 1.0)
                inputs_m1['Pr'] = render_dist_input("Presión Yac. Pr (psi)", "pr_g", 3000, 2500, 3500)
                inputs_m1['Pwf'] = render_dist_input("Presión Fondo Pwf (psi)", "pwf_g", 1500, 1000, 2000)

            btn_calc_m1 = st.button("Calcular Producción Inicial ($q_i$)", key="btn_m1")

        with col_m1_2:
            if btn_calc_m1:
                # 1. Generar muestras
                samples = {k: generate_samples(v['dist'], v['params'], n_iters) for k, v in inputs_m1.items()}
                
                # 2. Calcular Qi
                if fluid_type == "Aceite" and model_ipr == "Darcy (Flujo Radial)":
                    qi_result = ipr_oil_darcy(samples['k'], samples['h'], samples['Pr'], samples['Pwf'], 
                                             samples['mu'], samples['Bo'], re, rw, samples['S'])
                    unit = "bbl/d"
                elif fluid_type == "Aceite" and model_ipr == "Vogel (Saturado)":
                    qi_result = ipr_oil_vogel(samples['qmax'], samples['Pr'], samples['Pwf'])
                    unit = "bbl/d"
                elif fluid_type == "Gas":
                    qi_result = ipr_gas_backpressure(samples['C'], samples['Pr'], samples['Pwf'], samples['n'])
                    unit = "MMPCD"

                # 3. Guardar en Session State
                st.session_state['qi_distribution'] = qi_result
                st.session_state['qi_unit'] = unit
                st.session_state['run_m1'] = True
                
                # 4. Visualización
                stats_qi = {
                    'P90': np.percentile(qi_result, 10),
                    'P50': np.percentile(qi_result, 50),
                    'P10': np.percentile(qi_result, 90)
                }
                
                st.success("¡Cálculo Exitoso! Los datos han sido transferidos al Módulo II.")
                
                c1, c2, c3 = st.columns(3)
                c1.metric("P90 (Conservador)", f"{stats_qi['P90']:.1f} {unit}")
                c2.metric("P50 (Base)", f"{stats_qi['P50']:.1f} {unit}")
                c3.metric("P10 (Optimista)", f"{stats_qi['P10']:.1f} {unit}")
                
                fig_hist = px.histogram(qi_result, nbins=50, title=f"Distribución de Producción Inicial ({unit})",
                                       color_discrete_sequence=['#004B87'])
                fig_hist.add_vline(x=stats_qi['P50'], line_dash="dash", line_color="orange", annotation_text="P50")
                st.plotly_chart(fig_hist, use_container_width=True)

    # --- MÓDULO II: PRONÓSTICO ---
    with tab2:
        st.header("Módulo II: Pronóstico de Producción (Arps)")
        
        has_m1_data = st.session_state.get('run_m1', False)
        
        col_m2_1, col_m2_2 = st.columns([1, 2])
        
        with col_m2_1:
            st.subheader("Configuración de Entrada ($q_i$)")
            
            use_m1 = False
            if has_m1_data:
                # CORREGIDO: Emoji entre comillas
                st.success(f"✅ Datos del Módulo I disponibles ({st.session_state['qi_unit']})")
                use_m1 = st.checkbox("Usar Probabilidad Calculada en Módulo I", value=True)
                
                if use_m1:
                    qi_dist = st.session_state['qi_distribution']
                    st.info(f"Se usarán 10,000 escenarios con Media: {np.mean(qi_dist):.1f}")
            
            if not use_m1:
                st.warning("Usando carga manual para $q_i$ (No vinculado a IPR)")
                input_qi_manual = render_dist_input("Gasto Inicial Qi", "qi_man", 1000, 500, 1500)

            st.subheader("Parámetros de Declinación")
            input_di = render_dist_input("Declinación Inicial Di (Anual %)", "di", 0.20, 0.10, 0.40)
            input_b = render_dist_input("Exponente b", "b", 0.4, 0.0, 0.9)
            
            st.subheader("Tiempos")
            years = st.slider("Años a pronosticar", 1, 30, 20)
            qa_limit = st.number_input("Gasto de Abandono", value=10.0)
            
            btn_calc_m2 = st.button("Ejecutar Pronóstico Estocástico", key="btn_m2")

        with col_m2_2:
            if btn_calc_m2:
                # 1. Preparar Qi Vector
                if use_m1:
                    qi_vec = st.session_state['qi_distribution']
                    if len(qi_vec) != n_iters:
                        qi_vec = np.random.choice(qi_vec, n_iters)
                else:
                    qi_vec = generate_samples(input_qi_manual['dist'], input_qi_manual['params'], n_iters)
                
                # 2. Preparar Di y b
                di_vec = generate_samples(input_di['dist'], input_di['params'], n_iters)
                b_vec = generate_samples(input_b['dist'], input_b['params'], n_iters)
                
                # 3. Calcular Perfiles (Arps)
                months = np.arange(0, years * 12 + 1)
                q_profiles = arps_forecast(months, qi_vec, di_vec, b_vec)
                
                # Límite económico
                q_profiles = np.where(q_profiles < qa_limit, 0, q_profiles)
                
                # 4. Calcular EUR (Acumulada)
                days_per_month = 30.4167
                np_profiles = np.cumsum(q_profiles, axis=1) * days_per_month
                eur_vec = np_profiles[:, -1] / 1e6 # MM units
                
                # 5. Visualización
                p10_q = np.percentile(q_profiles, 90, axis=0)
                p50_q = np.percentile(q_profiles, 50, axis=0)
                p90_q = np.percentile(q_profiles, 10, axis=0)
                
                st.subheader("Perfil de Producción Probabilista")
                
                fig_q = go.Figure()
                fig_q.add_trace(go.Scatter(
                    x=np.concatenate([months, months[::-1]]),
                    y=np.concatenate([p90_q, p10_q[::-1]]),
                    fill='toself', fillcolor='rgba(0, 75, 135, 0.2)',
                    line=dict(color='rgba(255,255,255,0)'), name='Rango P90-P10'
                ))
                fig_q.add_trace(go.Scatter(x=months, y=p50_q, name='P50', line=dict(color='#004B87', width=3)))
                fig_q.add_trace(go.Scatter(x=months, y=p10_q, name='P10', line=dict(color='green', dash='dot')))
                fig_q.add_trace(go.Scatter(x=months, y=p90_q, name='P90', line=dict(color='red', dash='dot')))
                
                fig_q.update_layout(title="Gasto vs Tiempo", xaxis_title="Meses", yaxis_title="Gasto", template="plotly_white", yaxis_type="log")
                st.plotly_chart(fig_q, use_container_width=True)
                
                # Métricas EUR
                eur_p90 = np.percentile(eur_vec, 10)
                eur_p50 = np.percentile(eur_vec, 50)
                eur_p10 = np.percentile(eur_vec, 90)
                
                st.markdown("### Reservas Recuperables Estimadas (EUR)")
                m1, m2, m3 = st.columns(3)
                m1.markdown(f"<div class='metric-container'><h3>P90</h3><h2>{eur_p90:.2f}</h2><p>MM</p></div>", unsafe_allow_html=True)
                m2.markdown(f"<div class='metric-container'><h3>P50</h3><h2>{eur_p50:.2f}</h2><p>MM</p></div>", unsafe_allow_html=True)
                m3.markdown(f"<div class='metric-container'><h3>P10</h3><h2>{eur_p10:.2f}</h2><p>MM</p></div>", unsafe_allow_html=True)

                # Tornado Chart
                st.markdown("---")
                st.subheader("Análisis de Sensibilidad (Drivers del EUR)")
                
                df_sens = pd.DataFrame({
                    'Qi (Inicial)': qi_vec,
                    'Di (Declinación)': di_vec,
                    'b (Exponente)': b_vec,
                    'EUR': eur_vec
                })
                corr = df_sens.corr(method='spearman')['EUR'].drop('EUR').sort_values()
                
                fig_torn = px.bar(corr, orientation='h', title="Impacto en Reservas (Correlación de Rango)",
                                 color=corr, color_continuous_scale='RdBu_r')
                st.plotly_chart(fig_torn, use_container_width=True)

if __name__ == "__main__":
    main()