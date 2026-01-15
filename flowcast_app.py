import streamlit as st
import numpy as np
import pandas as pd
import scipy.stats as stats
import plotly.graph_objects as go
import plotly.express as px

# --- CONFIGURACIÓN DE PÁGINA (UX/UI) ---
st.set_page_config(
    page_title="Reliarisk FlowCast",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ESTILOS CSS PERSONALIZADOS (MODERN LOOK) ---
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button {
        width: 100%;
        background-color: #0066CC;
        color: white;
        border-radius: 5px;
        font-weight: bold;
    }
    .metric-container {
        background-color: white;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        border-left: 5px solid #0066CC;
    }
    .stExpander { background-color: white; border-radius: 5px; }
    h1, h2, h3 { color: #2C3E50; }
    </style>
    """, unsafe_allow_html=True)

# --- FUNCIONES NÚCLEO (BACKEND MATEMÁTICO) ---

def generate_samples(dist_type, params, n_iter):
    """Genera vector de muestras aleatorias según distribución."""
    if dist_type == 'Normal':
        return np.random.normal(params['mean'], params['std'], n_iter)
    elif dist_type == 'Lognormal':
        # Conversión a parámetros mu y sigma subyacentes
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

def arps_forecast(t_array, qi, di, b):
    """
    Calcula el perfil de producción q(t) usando Arps.
    t_array: array de tiempos (meses)
    qi: gasto inicial (bbl/d o mmpcd)
    di: declinación nominal anual (%) convertida a mensual
    b: exponente hiperbólico
    """
    # Conversión de Di anual a tasa efectiva mensual instantánea aproximada
    # Di_mensual = Di_anual / 12
    di_m = di / 12.0
    
    # Manejo de vectores para b (puede ser escalar o array)
    # Evitar división por cero si b=0 (Exponencial)
    
    # Pre-allocating result array
    q_t = np.zeros((len(qi), len(t_array)))
    
    # Caso 1: Hiperbólico (b > 0)
    # q(t) = qi / (1 + b * Di * t)^(1/b)
    
    # Caso 2: Exponencial (b = 0)
    # q(t) = qi * exp(-Di * t)

    # Para eficiencia vectorial, asumimos b > 0.001 como hiperbólico
    # Si b es muy cercano a 0, tratamos como exponencial
    
    # Broadcasting: qi shape (N,1), t_array shape (1, T) -> Result (N, T)
    qi_vec = qi[:, np.newaxis]
    b_vec = b[:, np.newaxis]
    di_vec = di_m[:, np.newaxis]
    t_mat = t_array[np.newaxis, :]
    
    # Máscara para exponencial vs hiperbólico
    is_hyp = b_vec > 0.001
    
    # Cálculo Hiperbólico
    term_hyp = (1 + b_vec * di_vec * t_mat)
    # Evitar warnings de float con np.where
    term_hyp = np.maximum(term_hyp, 1e-9) 
    q_hyp = qi_vec / (term_hyp ** (1.0 / np.maximum(b_vec, 1e-9)))
    
    # Cálculo Exponencial
    q_exp = qi_vec * np.exp(-di_vec * t_mat)
    
    # Combinar
    q_final = np.where(is_hyp, q_hyp, q_exp)
    
    return q_final

# --- INTERFAZ DE USUARIO (SIDEBAR) ---

with st.sidebar:
    try:
        st.image("mi-logo.png", use_container_width=True)
    except:
        st.warning("⚠️ Logo no encontrado (mi-logo.png)")
        
    st.title("Configuración del Pronóstico")
    
    # Configuración Global
    with st.expander("⚙️ Parámetros Generales", expanded=True):
        fluid_type = st.selectbox("Fluido", ["Aceite (bbl/d)", "Gas (MMPCD)"])
        time_years = st.number_input("Horizonte de Tiempo (Años)", 1, 50, 20)
        n_iters = st.selectbox("Iteraciones (Montecarlo)", [1000, 5000, 10000], index=1)
        q_abandono = st.number_input("Gasto de Abandono (qa)", 0.0, 1000.0, 10.0)

    st.markdown("### Variables Estocásticas (Arps)")
    
    # Función generadora de inputs de distribución
    def input_distribution(label, key_prefix, default_mode, default_min, default_max):
        dist = st.selectbox(f"Distribución {label}", 
                           ['BetaPERT', 'Lognormal', 'Normal', 'Triangular', 'Uniforme', 'Determinístico'],
                           key=f"d_{key_prefix}")
        
        params = {}
        col_input = st.container()
        if dist == 'BetaPERT':
            c1, c2, c3 = col_input.columns(3)
            params['min'] = c1.number_input(f"Mín {label}", value=float(default_min), key=f"min_{key_prefix}")
            params['mode'] = c2.number_input(f"Moda {label}", value=float(default_mode), key=f"mod_{key_prefix}")
            params['max'] = c3.number_input(f"Máx {label}", value=float(default_max), key=f"max_{key_prefix}")
        elif dist == 'Triangular':
            c1, c2, c3 = col_input.columns(3)
            params['min'] = c1.number_input(f"Mín {label}", value=float(default_min), key=f"tmin_{key_prefix}")
            params['mode'] = c2.number_input(f"Moda {label}", value=float(default_mode), key=f"tmod_{key_prefix}")
            params['max'] = c3.number_input(f"Máx {label}", value=float(default_max), key=f"tmax_{key_prefix}")
        elif dist == 'Normal':
            c1, c2 = col_input.columns(2)
            params['mean'] = c1.number_input(f"Media {label}", value=float(default_mode), key=f"nmu_{key_prefix}")
            params['std'] = c2.number_input(f"StdDev {label}", value=float((default_max-default_min)/6), key=f"nstd_{key_prefix}")
        elif dist == 'Lognormal':
            c1, c2 = col_input.columns(2)
            params['mean'] = c1.number_input(f"Media {label}", value=float(default_mode), key=f"lmu_{key_prefix}")
            params['std'] = c2.number_input(f"StdDev {label}", value=float((default_max-default_min)/6), key=f"lstd_{key_prefix}")
        elif dist == 'Uniforme':
            c1, c2 = col_input.columns(2)
            params['min'] = c1.number_input(f"Mín {label}", value=float(default_min), key=f"umin_{key_prefix}")
            params['max'] = c2.number_input(f"Máx {label}", value=float(default_max), key=f"umax_{key_prefix}")
        elif dist == 'Determinístico':
            params['value'] = st.number_input(f"Valor {label}", value=float(default_mode), key=f"det_{key_prefix}")
            
        return {'dist': dist, 'params': params}

    # Inputs para Qi, Di, b
    config_qi = input_distribution("Gasto Inicial (qi)", "qi", 1000, 800, 1500)
    config_di = input_distribution("Declinación Inicial (Di anual %)", "di", 0.20, 0.10, 0.40)
    config_b = input_distribution("Exponente b", "b", 0.4, 0.0, 1.0)
    
    run_btn = st.button("🚀 EJECUTAR PRONÓSTICO", use_container_width=True)

# --- APP PRINCIPAL ---

st.title("Reliarisk FlowCast")
st.markdown("**Pronóstico Probabilístico de Producción y Reservas (DCA + Montecarlo)**")

if run_btn:
    with st.spinner('Realizando simulación estocástica...'):
        # 1. Muestreo de Variables
        qi_samples = generate_samples(config_qi['dist'], config_qi['params'], n_iters)
        di_samples = generate_samples(config_di['dist'], config_di['params'], n_iters)
        b_samples = generate_samples(config_b['dist'], config_b['params'], n_iters)
        
        # Validación física (Di y b >= 0)
        qi_samples = np.maximum(qi_samples, 0)
        di_samples = np.maximum(di_samples, 0)
        b_samples = np.maximum(b_samples, 0) # b puede ser 0
        
        # 2. Vector de Tiempo
        months = np.arange(0, time_years * 12 + 1)
        
        # 3. Cálculo de Perfiles (Matriz: Iteraciones x Tiempo)
        # di entra como fracción (ej. 20% -> 0.20)
        q_profiles = arps_forecast(months, qi_samples, di_samples, b_samples)
        
        # Aplicar límite económico (Gasto de Abandono)
        q_profiles = np.where(q_profiles < q_abandono, 0, q_profiles)
        
        # 4. Cálculo de Acumulada (Np/Gp)
        # Integración numérica simple (Trapezoidal o suma mensual)
        # Asumiendo q es tasa mensual promedio, Np = sum(q * 30.416)
        days_per_month = 30.4167
        np_profiles = np.cumsum(q_profiles, axis=1) * days_per_month
        eur_samples = np_profiles[:, -1] / 1e6 # En millones (MMbbl o BCF)
        
        # 5. Estadísticas por paso de tiempo (P10, P50, P90)
        # Axis 0 son las iteraciones
        p10_q = np.percentile(q_profiles, 90, axis=0) # P10 High Case
        p50_q = np.percentile(q_profiles, 50, axis=0)
        p90_q = np.percentile(q_profiles, 10, axis=0) # P90 Low Case
        
        p10_np = np.percentile(np_profiles, 90, axis=0) / 1e6
        p50_np = np.percentile(np_profiles, 50, axis=0) / 1e6
        p90_np = np.percentile(np_profiles, 10, axis=0) / 1e6
        
        # Resultados Escalares (EUR)
        eur_stats = {
            'P90': np.percentile(eur_samples, 10),
            'P50': np.percentile(eur_samples, 50),
            'P10': np.percentile(eur_samples, 90),
            'Mean': np.mean(eur_samples)
        }

    # --- DASHBOARD DE RESULTADOS ---
    
    # 1. Tarjetas de Métricas (EUR)
    st.markdown("### 📊 Reservas Recuperables Estimadas (EUR)")
    c1, c2, c3, c4 = st.columns(4)
    unit = "MMbbls" if "Aceite" in fluid_type else "BCF"
    
    c1.markdown(f"<div class='metric-container'><h3>P90 (Probado)</h3><h2>{eur_stats['P90']:.2f}</h2><p>{unit}</p></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='metric-container'><h3>P50 (Base)</h3><h2>{eur_stats['P50']:.2f}</h2><p>{unit}</p></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='metric-container'><h3>P10 (Posible)</h3><h2>{eur_stats['P10']:.2f}</h2><p>{unit}</p></div>", unsafe_allow_html=True)
    c4.markdown(f"<div class='metric-container'><h3>Media</h3><h2>{eur_stats['Mean']:.2f}</h2><p>{unit}</p></div>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # 2. Gráficos Interactivos (Plotly)
    tab1, tab2, tab3 = st.tabs(["📈 Perfil de Producción", "🛢️ Acumulada", "🌪️ Análisis de Sensibilidad"])
    
    with tab1:
        fig_q = go.Figure()
        
        # Áreas sombreadas (Incertidumbre)
        # Truco: Rellenar entre P10 y P90
        fig_q.add_trace(go.Scatter(
            x=np.concatenate([months, months[::-1]]),
            y=np.concatenate([p90_q, p10_q[::-1]]),
            fill='toself',
            fillcolor='rgba(0, 102, 204, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            name='Rango P90-P10'
        ))
        
        # Líneas Principales
        fig_q.add_trace(go.Scatter(x=months, y=p10_q, name='P10 (Optimista)', line=dict(color='green', dash='dot')))
        fig_q.add_trace(go.Scatter(x=months, y=p50_q, name='P50 (Base)', line=dict(color='#0066CC', width=3)))
        fig_q.add_trace(go.Scatter(x=months, y=p90_q, name='P90 (Conservador)', line=dict(color='red', dash='dot')))
        
        # Límite económico
        fig_q.add_hline(y=q_abandono, line_dash="dash", line_color="gray", annotation_text="Límite Económico")

        fig_q.update_layout(
            title="Pronóstico de Producción (Gasto vs Tiempo)",
            xaxis_title="Meses",
            yaxis_title=fluid_type,
            template="plotly_white",
            hovermode="x unified",
            yaxis_type="log"  # Logarítmico por defecto para análisis de declinación
        )
        
        # Toggle para escala lineal/log
        use_log = st.checkbox("Escala Logarítmica en Eje Y", value=True)
        if not use_log:
            fig_q.update_yaxes(type="linear")
            
        st.plotly_chart(fig_q, use_container_width=True)
        
    with tab2:
        fig_np = go.Figure()
        fig_np.add_trace(go.Scatter(x=months, y=p10_np, name='P10 Acum', line=dict(color='green', dash='dot')))
        fig_np.add_trace(go.Scatter(x=months, y=p50_np, name='P50 Acum', line=dict(color='#0066CC', width=3)))
        fig_np.add_trace(go.Scatter(x=months, y=p90_np, name='P90 Acum', line=dict(color='red', dash='dot')))
        
        fig_np.update_layout(
            title="Producción Acumulada (EUR Progresivo)",
            xaxis_title="Meses",
            yaxis_title=f"Acumulada ({unit})",
            template="plotly_white",
            hovermode="x unified"
        )
        st.plotly_chart(fig_np, use_container_width=True)
        
    with tab3:
        # Diagrama de Tornado (Correlación de Rango - Spearman)
        # Mide qué variable de entrada (qi, di, b) influye más en el EUR
        
        st.subheader("Drivers de Incertidumbre (Diagrama de Tornado)")
        st.info("Este gráfico muestra la correlación entre las variables de entrada y la Reserva Final (EUR). Barras más largas indican mayor impacto.")
        
        # Crear DataFrame temporal para análisis
        df_corr = pd.DataFrame({
            'Qi': qi_samples,
            'Di': di_samples,
            'b': b_samples,
            'EUR': eur_samples
        })
        
        correlations = df_corr.corr(method='spearman')['EUR'].drop('EUR')
        corr_df = correlations.sort_values(ascending=True).to_frame(name='Correlación')
        
        fig_torn = px.bar(
            corr_df, 
            x='Correlación', 
            y=corr_df.index, 
            orientation='h',
            color='Correlación',
            color_continuous_scale='RdBu_r', # Rojo negativo (Di), Azul positivo (Qi, b)
            range_x=[-1, 1]
        )
        st.plotly_chart(fig_torn, use_container_width=True)

else:
    st.info("👈 Configura los parámetros en el menú lateral y presiona 'EJECUTAR PRONÓSTICO'.")