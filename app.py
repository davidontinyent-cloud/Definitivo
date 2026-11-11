import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import f_oneway
from scipy.integrate import trapezoid
import os

# ---------------------------------------------------------
# 1. CONFIGURACIÓN E INTERFAZ
# ---------------------------------------------------------
st.set_page_config(page_title="DURAPAV - Análisis Factorial", layout="wide")

st.sidebar.markdown("### Laboratorio de Caminos - UPV")
st.sidebar.markdown("**Proyecto DURAPAV**")
st.sidebar.markdown("---")

st.title("Análisis Dinámico de Parámetros: Ensayo Ideal-CT")

# --- DESCRIPCIÓN INICIAL ---
st.markdown("El dashboard permite explorar gráficamente los resultados obtenidos mediante el ensayo Ideal-CT. Su objetivo es facilitar la comparación entre mezclas y condiciones de ensayo, evaluando cómo cambian la tenacidad, la ductilidad, la fragilidad y la resistencia en función del envejecimiento, la incorporación de material fresado (RAP) y la velocidad de ensayo.")

col_desc1, col_desc2 = st.columns(2)
with col_desc1:
    st.info("""
    **Variables empleadas como filtros**
    * **Envejecimiento:** Control; E1; E2. (E3 y E4 excluidos en este análisis).
    * **RAP:** Sin_RAP = mezcla sin asfalto recuperado; Con_RAP = mezcla con asfalto recuperado.
    * **Velocidad:** fijada a 50 mm/min (los resultados a velocidad de 2 mm/min han sido excluidos).
    * **Mezcla:** AC16, AC22, AUTL 11s, AUTL11c, BBTM11B, BBTMc, BBTMs, SMA16c, SMA16m.
    *(Nota: Mezclas tipo AC sin RAP excluidas, salvo las AC16)*
    """)
with col_desc2:
    st.info("""
    **Parámetros del ensayo Ideal-CT**
    * **Gf:** tenacidad o energía de fractura (J/m²).
    * **Gf prepico y postpico:** contribución energética antes y después de la carga máxima.
    * **l75:** ductilidad, asociada a la capacidad de deformación.
    * **m75 / m_xx:** fragilidad, asociada a la pendiente postpico y propagación de fisura.
    * **Carga Pico:** resistencia asociada al inicio de la formación de la fisura.
    * **CT_Index:** índice global de tolerancia a la fisuración.
    * **Centroide Pre-pico:** Momento estático real integrado de la curva inicial.
    """)

@st.cache_data
def cargar_datos():
    if not os.path.exists('parametros_mezclas.xlsx'):
        st.error("No se encontró el archivo 'parametros_mezclas.xlsx'. Asegúrate de que el archivo Excel esté en el mismo directorio.")
        st.stop()
        
    try:
        df = pd.read_excel('parametros_mezclas.xlsx', engine='openpyxl')
        df.columns = df.columns.astype(str).str.strip()
        df = df.replace('-', np.nan)
        
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    df[col] = [val.strip() if isinstance(val, str) else val for val in df[col]]
                except Exception:
                    pass
        
        cols_numericas = ['Carga_pico', 'CT_Index', 'Gf', 'Gf_prepico', 'Gf_postpico', 'm75', 'l75', 'Rigidez_20', 'Rigidez_15']
        for col in cols_numericas:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        if 'Espesor' in df.columns and 'Diametro' in df.columns:
            df['Tension_Rotura'] = (df['Carga_pico'] * 1000) / (df['Espesor'] * df['Diametro'])
            
        if 'Mezcla' in df.columns:
            if 'RAP' in df.columns and 'Envejecimiento' in df.columns:
                df['Todas'] = df['Mezcla'] + " | RAP: " + df['RAP'].astype(str) + " | Env: " + df['Envejecimiento'].astype(str)
            else:
                df['Todas'] = df['Mezcla']
        else:
            df['Todas'] = df.iloc[:, 0]
            
        # --- FILTROS GLOBALES SOLICITADOS ---
        if 'Mezcla' in df.columns and 'RAP' in df.columns:
            # Excluimos mezclas AC sin RAP, pero desbloqueamos las que contengan 'AC16'
            mask_ac_sin_rap = (df['Mezcla'].str.startswith('AC', na=False)) & (~df['Mezcla'].str.contains('AC16', na=False)) & (df['RAP'].astype(str).str.lower().isin(['sin_rap', '0']))
            df = df[~mask_ac_sin_rap]
            
        if 'Envejecimiento' in df.columns:
            df = df[df['Envejecimiento'].isin(['Control', 'E1', 'E2'])]
            
        # --- BLOQUEO DE VELOCIDAD 2 ---
        if 'Velocidad' in df.columns:
            df = df[df['Velocidad'] != 2]
            
        return df
        
    except Exception as e:
        st.error(f"Hubo un error cargando el archivo de datos: {e}")
        st.stop()
    return None

@st.cache_data
def agregar_centroide_prepico(df_main):
    if not os.path.exists('datos_brutos.xlsx'):
        return df_main
    try:
        df_b = pd.read_excel('datos_brutos.xlsx', sheet_name=0, header=None)
        resultados = []
        num_columnas = df_b.shape[1]
        
        for i in range(0, num_columnas, 2):
            mezcla_orig = str(df_b.iloc[0, i]).strip()
            if mezcla_orig.lower() == 'nan': continue
            
            start_row = 4
            for row_idx in range(4, 10):
                try:
                    float(df_b.iloc[row_idx, i])
                    start_row = row_idx
                    break
                except:
                    pass
            
            x_data = pd.to_numeric(df_b.iloc[start_row:, i], errors='coerce').dropna().values
            y_data = pd.to_numeric(df_b.iloc[start_row:, i+1], errors='coerce').dropna().values
            
            min_len = min(len(x_data), len(y_data))
            if min_len == 0: continue
            
            x = x_data[:min_len]
            y = y_data[:min_len]
            
            idx_peak = np.argmax(y)
            if idx_peak == 0: continue 
            
            x_peak = x[idx_peak] 
            y_peak = y[idx_peak] 
            
            x_pre = x[:idx_peak+1]
            y_pre = y[:idx_peak+1]
            
            Qx_real = 0.5 * trapezoid(y_pre**2, x_pre)
            area_real = trapezoid(y_pre, x_pre)
            
            centroide_y = Qx_real / area_real if area_real > 0 else np.nan
            
            resultados.append({
                'Mezcla_Orig': mezcla_orig,
                'y_peak': y_peak,
                'Centroide_Prepico': centroide_y
            })
            
        df_calc = pd.DataFrame(resultados)
        if df_calc.empty: return df_main
        
        df_calc['Mezcla_Merge'] = df_calc['Mezcla_Orig'].astype(str).str.strip().str.upper().str.replace(' ', '').str.replace('BBTM11B', 'BBTM11')
        
        df_main_merge = df_main.copy()
        df_main_merge['_orig_index'] = df_main_merge.index
        df_main_merge['Mezcla_Merge'] = df_main_merge['Mezcla'].astype(str).str.strip().str.upper().str.replace(' ', '').str.replace('BBTM11B', 'BBTM11')
        
        df_merged = pd.merge(df_main_merge, df_calc, on=['Mezcla_Merge'], how='left')
        df_merged['diff_pico'] = abs(df_merged['y_peak'] - df_merged['Carga_pico'])
        df_merged = df_merged.sort_values('diff_pico')
        
        df_merged = df_merged.drop_duplicates(subset=['_orig_index'])
        df_merged.loc[df_merged['diff_pico'] >= 1.0, 'Centroide_Prepico'] = np.nan
        df_merged = df_merged.sort_values('_orig_index').drop(columns=['Mezcla_Merge', 'Mezcla_Orig', 'y_peak', 'diff_pico', '_orig_index']).reset_index(drop=True)
        
        return df_merged
    except Exception as e:
        st.warning(f"Error procesando centroides: {e}")
        return df_main

@st.cache_data
def obtener_curvas_ejemplo():
    if not os.path.exists('datos_brutos.xlsx'): return None
    try:
        df_b = pd.read_excel('datos_brutos.xlsx', sheet_name=0, header=None)
        curvas = {'AC16': None, 'AC22': None, 'SMA': None, 'BBTM': None}
        
        for i in range(0, df_b.shape[1], 2):
            mezcla_orig = str(df_b.iloc[0, i]).upper().strip()
            if 'AC16' in mezcla_orig: tipo = 'AC16'
            elif 'AC22' in mezcla_orig: tipo = 'AC22'
            elif 'SMA' in mezcla_orig: tipo = 'SMA'
            elif 'BBTM' in mezcla_orig: tipo = 'BBTM'
            else: continue
            
            env = str(df_b.iloc[2, i]).strip()
            rap = str(df_b.iloc[1, i]).upper().strip()
            
            if curvas[tipo] is None or (env.lower() == 'control' and 'SIN' in rap):
                start_row = 4
                for r in range(4, 10):
                    try:
                        float(df_b.iloc[r, i])
                        start_row = r
                        break
                    except: pass
                
                x = pd.to_numeric(df_b.iloc[start_row:, i], errors='coerce').dropna().values
                y = pd.to_numeric(df_b.iloc[start_row:, i+1], errors='coerce').dropna().values
                
                min_len = min(len(x), len(y))
                if min_len > 20: 
                    curvas[tipo] = (x[:min_len], y[:min_len], f"{mezcla_orig} ({env})")
        return curvas
    except:
        return None

df = cargar_datos()
df = agregar_centroide_prepico(df)

if 'l75' in df.columns and 'm75' in df.columns:
    df['abs_m75'] = df['m75'].abs()
    df['Ratio_Flexibilidad'] = (df['l75'] / df['abs_m75'].replace(0, np.nan)) * 1e6

if 'Gf_prepico' in df.columns and 'Tension_Rotura' in df.columns and 'abs_m75' in df.columns:
    df['Tension_Rotura_SI'] = df['Tension_Rotura'] * 1e6
    df['Num_Fisuracion'] = df['Gf_prepico'] * (df['Tension_Rotura_SI'] / 3)
    df['Indice_Fisuracion'] = df['Num_Fisuracion'] / df['abs_m75'].replace(0, np.nan)

if 'Gf_prepico' in df.columns and 'Centroide_Prepico' in df.columns and 'abs_m75' in df.columns and 'Espesor' in df.columns and 'Diametro' in df.columns:
    df['Tension_Centroide'] = (df['Centroide_Prepico'] * 1000) / (df['Espesor'] * df['Diametro'])
    df['Tension_Centroide_SI'] = df['Tension_Centroide'] * 1e6
    df['Num_Fisuracion_Centroide'] = df['Gf_prepico'] * df['Tension_Centroide_SI']
    df['Indice_Fisuracion_Centroide'] = df['Num_Fisuracion_Centroide'] / df['abs_m75'].replace(0, np.nan)

if 'Gf_prepico' in df.columns:
    if 'Centroide_Prepico' in df.columns:
        df['Gf_x_Centroide'] = df['Gf_prepico'] * df['Centroide_Prepico']
    if 'Tension_Rotura' in df.columns:
        df['Gf_x_TR'] = df['Gf_prepico'] * df['Tension_Rotura']

if 'Centroide_Prepico' in df.columns and 'Carga_pico' in df.columns:
    df['Cociente_Forma'] = df['Centroide_Prepico'] / df['Carga_pico']

st.markdown("---")

# ---------------------------------------------------------
# 2. PESTAÑAS DE NAVEGACIÓN
# ---------------------------------------------------------
tab_parametrico, tab_fisuracion = st.tabs([
    "Análisis Paramétrico", 
    "Índice de fisuración"
])

# --- PESTAÑA 1: ANÁLISIS PARAMÉTRICO ---
with tab_parametrico:
    st.header("1. Análisis Paramétrico Integrado")
    
    st.subheader("1.0. Morfología de las Curvas Fuerza-Desplazamiento")
    st.markdown("Visualización de una curva representativa real en estado de **Control (Sin RAP)** para cada tipo de mezcla extraída de los datos brutos.")
    
    curvas_dict = obtener_curvas_ejemplo()
    if curvas_dict:
        fig_curvas = go.Figure()
        colores_mix = {'AC16': '#e41a1c', 'AC22': '#377eb8', 'SMA': '#984ea3', 'BBTM': '#4daf4a'}
        
        for tipo, data in curvas_dict.items():
            if data is not None:
                x_val, y_val, label = data
                fig_curvas.add_trace(go.Scatter(x=x_val, y=y_val, mode='lines', name=label, line=dict(color=colores_mix.get(tipo, 'black'), width=2.5)))
                
        fig_curvas.update_layout(height=450, xaxis_title="Desplazamiento (mm)", yaxis_title="Fuerza (kN)", margin=dict(t=30))
        st.plotly_chart(fig_curvas, use_container_width=True)
    else:
        st.info("No se pudieron cargar las curvas representativas desde 'datos_brutos.xlsx'.")

    st.markdown("---")
    
    opciones_var = ['Carga_pico', 'CT_Index', 'Gf', 'Gf_prepico', 'Gf_postpico', 'm75', 'l75', 'Ratio_Flexibilidad']
    if 'Tension_Rotura' in df.columns: opciones_var.append('Tension_Rotura')
    if 'Centroide_Prepico' in df.columns: opciones_var.append('Centroide_Prepico')
    
    st.subheader("1.1. Comparativa Directa por Parámetro (Medias)")
    col1A, col1B, col1C = st.columns(3)
    with col1A:
        param_bar = st.selectbox("Parámetro (Eje Y):", opciones_var, key="bar_y")
    with col1B:
        eje_x_bar = st.selectbox("Agrupar por (Eje X):", ['Todas', 'Mezcla', 'Envejecimiento', 'RAP'], key="bar_x")
    with col1C:
        color_bar = st.selectbox("Separar colores por:", ['Envejecimiento', 'RAP', 'Mezcla', 'Todas'], key="bar_color")

    agrupacion = [eje_x_bar] if eje_x_bar == color_bar else [eje_x_bar, color_bar]
    df_barras = df.groupby(agrupacion)[param_bar].mean().reset_index()
    df_barras[color_bar] = df_barras[color_bar].astype(str)
    
    modo_barra_t1 = 'relative' if eje_x_bar == color_bar or eje_x_bar == 'Todas' else 'group'
    fmt_texto_t2 = '.2f'
    
    fig_bar = px.bar(
        df_barras, x=eje_x_bar, y=param_bar, color=color_bar, 
        barmode=modo_barra_t1, text_auto=fmt_texto_t2, color_discrete_sequence=px.colors.qualitative.Alphabet if color_bar == 'Todas' else px.colors.qualitative.Set2
    )
    
    for m in df['Mezcla'].dropna().unique():
        for r in df['RAP'].dropna().unique():
            df_ctrl = df[(df['Mezcla'] == m) & (df['RAP'] == r) & (df['Envejecimiento'] == 'Control')]
            df_e2 = df[(df['Mezcla'] == m) & (df['RAP'] == r) & (df['Envejecimiento'] == 'E2')]
            
            if not df_ctrl.empty and not df_e2.empty:
                x_ctrl = df_ctrl['Todas'].iloc[0] if eje_x_bar == 'Todas' else df_ctrl[eje_x_bar].iloc[0]
                x_e2 = df_e2['Todas'].iloc[0] if eje_x_bar == 'Todas' else df_e2[eje_x_bar].iloc[0]
                
                if x_ctrl != x_e2:
                    y_ctrl = df_ctrl[param_bar].mean()
                    y_e2 = df_e2[param_bar].mean()
                    
                    fig_bar.add_trace(go.Scatter(
                        x=[x_ctrl, x_e2],
                        y=[y_ctrl, y_e2],
                        mode='lines+markers',
                        line=dict(width=2.5, dash='dot', color='rgba(50,50,50,0.8)'),
                        marker=dict(size=7, color='black', symbol='circle'),
                        showlegend=False,
                        hoverinfo='skip'
                    ))

    fig_bar.update_layout(height=450, xaxis_tickangle=-45)
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")
    
    st.subheader("1.2. Análisis de Dispersión y Variabilidad")
    st.markdown("Las cajas muestran la dispersión real de los datos, la mediana central y los posibles valores atípicos (*outliers*).")
    
    col1D, col1E, col1F, col1G = st.columns(4)
    with col1D:
        mezclas_disp_t3 = df['Mezcla'].dropna().unique().tolist()
        filtro_mezcla_t3 = st.selectbox("Filtrar Mezcla:", ["Todas"] + mezclas_disp_t3, key="box_mezcla")
    with col1E:
        param_box = st.selectbox("Parámetro (Eje Y):", opciones_var, key="box_y")
    with col1F:
        eje_x_box = st.selectbox("Agrupar por (Eje X):", ['Todas', 'Mezcla', 'Envejecimiento', 'RAP'], key="box_x")
    with col1G:
        color_box = st.selectbox("Separar colores por:", ['RAP', 'Envejecimiento', 'Mezcla', 'Todas'], key="box_color")

    if filtro_mezcla_t3 != "Todas":
        df_cajas = df[df['Mezcla'] == filtro_mezcla_t3].copy()
    else:
        df_cajas = df.copy()

    df_cajas[color_box] = df_cajas[color_box].astype(str)

    fig_box = px.box(
        df_cajas, x=eje_x_box, y=param_box, color=color_box, 
        points="all", 
        color_discrete_sequence=px.colors.qualitative.Alphabet if color_box == 'Todas' else px.colors.qualitative.Pastel
    )

    fig_box.update_layout(height=500, xaxis_tickangle=-45)
    st.plotly_chart(fig_box, use_container_width=True)

    st.markdown("---")

    st.subheader("1.3. Balance Energético de Fractura")
    if 'Gf_prepico' in df.columns and 'Gf_postpico' in df.columns:
        col3A, col3B, col3C = st.columns(3)
        with col3A:
            filtro_mezcla_t5 = st.selectbox("Elegir Mezcla:", ["Todas"] + mezclas_disp_t3, key="t5_mezcla_ener")
        with col3B:
            eje_x_energia = st.selectbox("Agrupar gráficas por (Eje X):", ["Todas", "Envejecimiento", "RAP"], key="t5_var_ener")
        with col3C:
            tipo_vista = st.radio("Modo de representación:", ["Valores Absolutos (J/m²)", "Porcentaje Relativo (100%)"], horizontal=False, key="energia_modo")
            
        df_energia = df[df['Mezcla'] == filtro_mezcla_t5].copy() if filtro_mezcla_t5 != "Todas" else df.copy()
            
        if not df_energia.empty:
            st.markdown("**Velocidad: 50 mm/min**")
            
            if 'Velocidad' in df_energia.columns:
                df_vel = df_energia[df_energia['Velocidad'] == 50]
            else:
                df_vel = df_energia
                
            if not df_vel.empty:
                df_g_mean = df_vel.groupby(eje_x_energia)[['Gf_prepico', 'Gf_postpico']].mean().reset_index()
                if tipo_vista == "Porcentaje Relativo (100%)":
                    total_gf = df_g_mean['Gf_prepico'] + df_g_mean['Gf_postpico']
                    df_g_mean['Gf_prepico'] = (df_g_mean['Gf_prepico'] / total_gf) * 100
                    df_g_mean['Gf_postpico'] = (df_g_mean['Gf_postpico'] / total_gf) * 100
                    eje_y_titulo = "Porcentaje de Energía (%)"
                else:
                    eje_y_titulo = "Energía (J/m²)"
                    
                df_melted = pd.melt(df_g_mean, id_vars=[eje_x_energia], value_vars=['Gf_prepico', 'Gf_postpico'], var_name='Fase_Energia', value_name='Valor')
                fig_stack = px.bar(df_melted, x=eje_x_energia, y='Valor', color='Fase_Energia', barmode='stack', text_auto='.1f', color_discrete_map={'Gf_prepico': '#3498db', 'Gf_postpico': '#e67e22'})
                fig_stack.update_layout(height=400, xaxis_tickangle=-45, yaxis_title=eje_y_titulo)
                st.plotly_chart(fig_stack, use_container_width=True)
            else:
                st.info("No hay datos para 50 mm/min con los filtros actuales.")
        else:
            st.warning("No hay datos disponibles tras aplicar los filtros de exclusión.")

    st.markdown("---")

    st.subheader("1.4. Análisis de Poder Discriminatorio (ANOVA - 1 Vía)")
    st.markdown("""
    Calcula el Valor F aislando un solo factor a **50 mm/min**. 
    **A mayor F, mayor poder separador tiene la variable.**
    """)
    
    col_tgt1, col_tgt2 = st.columns(2)
    with col_tgt1:
        target_col = st.radio("Selecciona la variable objetivo a clasificar:", ["Mezcla", "Envejecimiento"], horizontal=True, key="anova_target")
    
    df_ml = df[df['Velocidad'] == 50].copy() if 'Velocidad' in df.columns else df.copy()
    
    with col_tgt2:
        if target_col == "Envejecimiento":
            st.info("💡 Sugerencia: Al evaluar el envejecimiento, es mejor aislar una mezcla concreta.")
            filtro_mezcla_ml = st.selectbox("Filtrar datos por Mezcla:", ["Todas juntas"] + df_ml['Mezcla'].dropna().unique().tolist(), key="anova_mix")
            if filtro_mezcla_ml != "Todas juntas":
                df_ml = df_ml[df_ml['Mezcla'] == filtro_mezcla_ml]
        else:
            st.info("💡 Sugerencia: Al clasificar mezclas, puedes aislar un estado de envejecimiento concreto.")
            filtro_env_ml = st.selectbox("Filtrar datos por Envejecimiento:", ["Todos juntos"] + df_ml['Envejecimiento'].dropna().unique().tolist(), key="anova_env")
            if filtro_env_ml != "Todos juntos":
                df_ml = df_ml[df_ml['Envejecimiento'] == filtro_env_ml]
    
    features_obj = ['Carga_pico', 'Gf_prepico', 'Gf_postpico', 'abs_m75', 'l75', 'Tension_Rotura', 'Centroide_Prepico']
    valid_features = [f for f in features_obj if f in df_ml.columns]
    df_ml = df_ml.dropna(subset=valid_features + [target_col])
    
    if len(df_ml[target_col].unique()) > 1 and len(df_ml) >= 5:
        f_data = []
        for col in valid_features:
            groups = [group[col].values for name, group in df_ml.groupby(target_col) if len(group) > 0]
            if len(groups) > 1:
                f_stat, p_val = f_oneway(*groups)
                f_data.append({'Parámetro': col, 'F-Value': f_stat, 'p-value': p_val})
        
        if f_data:
            df_f = pd.DataFrame(f_data).sort_values('F-Value', ascending=False)
            df_f['p-value_fmt'] = df_f['p-value'].apply(lambda x: f"{x:.2e}" if x < 0.001 else f"{x:.3f}")
            
            fig_anova = px.bar(
                df_f, x='F-Value', y='Parámetro', orientation='h', text_auto='.1f', 
                color='F-Value', color_continuous_scale='Blues',
                hover_data={'p-value_fmt': True, 'F-Value': ':.2f', 'p-value': False},
                labels={'p-value_fmt': 'p-value'}
            )
            fig_anova.update_layout(yaxis={'categoryorder':'total ascending'}, height=400, coloraxis_showscale=False)
            st.plotly_chart(fig_anova, use_container_width=True)
        else:
            st.warning("No se pudo calcular el ANOVA para estos grupos.")
    else:
        st.warning("No hay suficientes datos válidos para entrenar el modelo.")

    st.markdown("---")

    st.subheader("1.5. Resumen Global: ANOVA Factorial (Mezcla y Envejecimiento)")
    st.markdown("Tabla resumen con los Valores F y p-values de los parámetros seleccionados a **50 mm/min**. Evalúa el efecto aislado de la Mezcla, el Envejecimiento y la Interacción simultánea entre ambos.")
    
    df_fact_base = df[df['Velocidad'] == 50].copy() if 'Velocidad' in df.columns else df.copy()
    resultados_factorial = []
    
    for param in features_obj:
        if param in df_fact_base.columns:
            df_temp = df_fact_base[[param, 'Mezcla', 'Envejecimiento']].dropna()
            if len(df_temp['Mezcla'].unique()) > 1 and len(df_temp['Envejecimiento'].unique()) > 1:
                df_temp = df_temp.rename(columns={param: 'Y', 'Mezcla': 'F1', 'Envejecimiento': 'F2'})
                df_temp['F1'] = df_temp['F1'].astype(str)
                df_temp['F2'] = df_temp['F2'].astype(str)
                try:
                    modelo_fact = smf.ols('Y ~ C(F1) + C(F2) + C(F1):C(F2)', data=df_temp).fit()
                    anova_res = sm.stats.anova_lm(modelo_fact, typ=2)
                    
                    resultados_factorial.append({
                        'Parámetro': param,
                        'F (Mezcla)': anova_res.loc['C(F1)', 'F'],
                        'p-value (Mezcla)': anova_res.loc['C(F1)', 'PR(>F)'],
                        'F (Env.)': anova_res.loc['C(F2)', 'F'],
                        'p-value (Env.)': anova_res.loc['C(F2)', 'PR(>F)'],
                        'F (Inter.)': anova_res.loc['C(F1):C(F2)', 'F'],
                        'p-value (Inter.)': anova_res.loc['C(F1):C(F2)', 'PR(>F)']
                    })
                except Exception:
                    pass
                    
    if resultados_factorial:
        df_res_fact = pd.DataFrame(resultados_factorial)
        
        def highlight_p_val(val):
            if pd.isna(val): return ''
            color = 'rgba(46, 204, 113, 0.2)' if val < 0.05 else 'rgba(231, 76, 60, 0.1)'
            return f'background-color: {color}'
            
        st.dataframe(
            df_res_fact.style
            .map(highlight_p_val, subset=['p-value (Mezcla)', 'p-value (Env.)', 'p-value (Inter.)'])
            .format({
                'F (Mezcla)': '{:.2f}', 'p-value (Mezcla)': '{:.4e}',
                'F (Env.)': '{:.2f}', 'p-value (Env.)': '{:.4e}',
                'F (Inter.)': '{:.2f}', 'p-value (Inter.)': '{:.4e}'
            }),
            use_container_width=True, height=400
        )
    else:
        st.warning("No se pudo calcular el ANOVA factorial para los datos actuales.")
        
    st.markdown("---")
    
    st.subheader("1.6. Análisis del Cociente de Forma Pre-pico (Centroide / Carga Pico)")
    st.info("Si ese cociente es prácticamente constante, significa que la forma normalizada de la curva pre-pico es relativamente estable, y el centroide exacto aporta poca información adicional respecto a una aproximación basada en la carga máxima.\n\nSi hay diferencias entre mezclas y/o envejecimientos, entonces el centroide nos aporta información extra.")

    df_forma = df[df['Velocidad'] == 50].dropna(subset=['Cociente_Forma', 'Mezcla', 'Envejecimiento']) if 'Velocidad' in df.columns else df.dropna(subset=['Cociente_Forma', 'Mezcla', 'Envejecimiento'])
    if not df_forma.empty:
        fig_forma = px.box(
            df_forma, x='Mezcla', y='Cociente_Forma', color='Envejecimiento',
            points='all', hover_data=['RAP', 'Probeta'],
            labels={'Cociente_Forma': 'Cociente (Centroide / Carga Pico)'},
            color_discrete_sequence=px.colors.qualitative.Set1
        )
        
        fig_forma.update_layout(height=450)
        st.plotly_chart(fig_forma, use_container_width=True)
        
    st.markdown("---")
    
    st.subheader("1.7. Análisis de Componentes Principales (PCA)")
    st.markdown("Visualización de las cargas (*loadings*) de los componentes principales basados en las variables del ANOVA a **50 mm/min**.")
    
    anova_vars = ['Carga_pico', 'Gf_prepico', 'Gf_postpico', 'abs_m75', 'l75', 'Tension_Rotura', 'Centroide_Prepico']
    anova_vars_presentes = [c for c in anova_vars if c in df.columns]
    
    if len(anova_vars_presentes) > 1:
        df_pca = df[df['Velocidad'] == 50][anova_vars_presentes].dropna() if 'Velocidad' in df.columns else df[anova_vars_presentes].dropna()
        if len(df_pca) > 5:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(df_pca)
            pca = PCA(n_components=2)
            pca.fit(X_scaled)
            loadings = pca.components_.T
            
            fig_pca, ax_pca = plt.subplots(figsize=(10, 8))
            ax_pca.set_xlim(-1, 1)
            ax_pca.set_ylim(-1, 1)
            ax_pca.axhline(0, color='grey', linestyle='--', alpha=0.5)
            ax_pca.axvline(0, color='grey', linestyle='--', alpha=0.5)
            ax_pca.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% Varianza)")
            ax_pca.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% Varianza)")
            
            texts_positions = []
            for i, feature in enumerate(anova_vars_presentes):
                x, y = loadings[i, 0], loadings[i, 1]
                ax_pca.arrow(0, 0, x, y, head_width=0.03, head_length=0.03, fc='crimson', ec='crimson', alpha=0.8)
                
                tx, ty = x * 1.15, y * 1.15
                
                colision = True
                intentos = 0
                while colision and intentos < 10:
                    colision = False
                    for pos_x, pos_y in texts_positions:
                        if abs(tx - pos_x) < 0.25 and abs(ty - pos_y) < 0.07:
                            ty += 0.04 if ty >= pos_y else -0.04
                            colision = True
                    intentos += 1
                        
                texts_positions.append((tx, ty))
                
                ax_pca.text(tx, ty, feature, color='black', ha='center', va='center', fontsize=10, 
                            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.2'))
                
            ax_pca.grid(True, linestyle=':', alpha=0.6)
            
            col_pcaA, col_pcaB, col_pcaC = st.columns([1, 2, 1])
            with col_pcaB:
                st.pyplot(fig_pca)
        else:
            st.warning("No hay suficientes datos válidos para realizar el PCA.")
    else:
        st.warning("Faltan variables necesarias para realizar el PCA.")

# --- PESTAÑA 2: ÍNDICE DE FISURACIÓN (Originalmente Pestaña 3) ---
with tab_fisuracion:
    st.header("2. Espacio de Diseño: Tolerancia a la Fisuración y Durabilidad")
    st.markdown("Análisis basado en la aproximación integral empleando la Tensión Equivalente del Centroide Pre-pico.")
    
    def render_indice_fisuracion(col_idx, lbl_idx, id_suffix, sec_num_1, sec_num_2, threshold_x=None):
        df_tab_lab = df[df['Velocidad'] == 50].copy() if 'Velocidad' in df.columns else df.copy()
        if col_idx not in df_tab_lab.columns:
            st.warning(f"No se pudo calcular el {lbl_idx}. Faltan datos necesarios para su formulación.")
            return
            
        st.subheader(f"{sec_num_1}. Comparativa del Índice por Mezcla")
        
        col_labA, col_labB = st.columns(2)
        with col_labA:
            agrup_x_lab = st.selectbox("Eje X:", ['Todas', 'Mezcla', 'Envejecimiento', 'RAP'], key=f"lab_x_fis_{id_suffix}", index=0)
        with col_labB:
            color_lab = st.selectbox("Color:", ['Todas', 'Envejecimiento', 'RAP', 'Mezcla'], key=f"lab_color_fis_{id_suffix}", index=0)
            
        agrupacion_lab = [agrup_x_lab] if agrup_x_lab == color_lab else [agrup_x_lab, color_lab]
        df_barras_lab = df_tab_lab.dropna(subset=[col_idx]).groupby(agrupacion_lab)[col_idx].mean().reset_index()
        df_barras_lab[color_lab] = df_barras_lab[color_lab].astype(str)
        
        modo_barra_lab = 'relative' if agrup_x_lab == color_lab or agrup_x_lab == 'Todas' else 'group'
        
        fig_bar_lab = px.bar(
            df_barras_lab, x=agrup_x_lab, y=col_idx, color=color_lab, 
            barmode=modo_barra_lab, text_auto='.2f', color_discrete_sequence=px.colors.qualitative.Set2
        )
        
        for m in df_tab_lab['Mezcla'].dropna().unique():
            for r in df_tab_lab['RAP'].dropna().unique():
                df_ctrl = df_tab_lab[(df_tab_lab['Mezcla'] == m) & (df_tab_lab['RAP'] == r) & (df_tab_lab['Envejecimiento'] == 'Control')]
                df_e2 = df_tab_lab[(df_tab_lab['Mezcla'] == m) & (df_tab_lab['RAP'] == r) & (df_tab_lab['Envejecimiento'] == 'E2')]
                
                if not df_ctrl.empty and not df_e2.empty:
                    if agrup_x_lab == 'Todas':
                        x_ctrl = df_ctrl['Todas'].iloc[0]
                        x_e2 = df_e2['Todas'].iloc[0]
                    else:
                        x_ctrl = df_ctrl[agrup_x_lab].iloc[0]
                        x_e2 = df_e2[agrup_x_lab].iloc[0]
                    
                    if x_ctrl != x_e2:
                        y_ctrl = df_ctrl[col_idx].mean()
                        y_e2 = df_e2[col_idx].mean()
                        
                        fig_bar_lab.add_trace(go.Scatter(
                            x=[x_ctrl, x_e2],
                            y=[y_ctrl, y_e2],
                            mode='lines+markers',
                            line=dict(width=2.5, dash='dot', color='rgba(50,50,50,0.8)'),
                            marker=dict(size=7, color='black', symbol='circle'),
                            showlegend=False,
                            hoverinfo='skip'
                        ))

        fig_bar_lab.update_layout(height=450, xaxis_tickangle=-45, yaxis_title=lbl_idx)
        st.plotly_chart(fig_bar_lab, use_container_width=True)
        
        st.markdown("---")
        
        df_dur = df_tab_lab.groupby(['Mezcla', 'RAP', 'Envejecimiento'])[col_idx].mean().reset_index()
        try:
            df_pivot = df_dur.pivot(index=['Mezcla', 'RAP'], columns='Envejecimiento', values=col_idx).reset_index()
            if 'Control' in df_pivot.columns and 'E2' in df_pivot.columns:
                
                df_pivot['Indice_Durabilidad_E2 (%)'] = (df_pivot['E2'] / df_pivot['Control']) * 100
                
                st.markdown(f"**{sec_num_2}. Análisis de Durabilidad: Retención de Fisuración**")
                col_durA, col_durB = st.columns([1.5, 1])
                
                with col_durA:
                    tab_limpio, tab_div = st.tabs(["Gráfico Limpio", "Gráfico con Divisiones"])
                    
                    fig_limpio = px.scatter(
                        df_pivot, x='Control', y='Indice_Durabilidad_E2 (%)', 
                        color='Mezcla', symbol='RAP', hover_data=['Mezcla', 'RAP', 'E2'], 
                        labels={'Control': f'{lbl_idx} (Control)', 'Indice_Durabilidad_E2 (%)': 'Índice de Durabilidad E2 (%)'}, 
                        color_discrete_sequence=px.colors.qualitative.Set1
                    )
                    fig_limpio.update_traces(marker=dict(size=14, line=dict(width=1, color='DarkSlateGrey')))
                    
                    if threshold_x is None:
                        threshold_x = df_pivot['Control'].median()
                        if pd.isna(threshold_x) or threshold_x == 0:
                            threshold_x = df_pivot['Control'].max() / 2
                            
                    max_x_dur = df_pivot['Control'].max() * 1.15
                    if max_x_dur < threshold_x * 1.5:
                        max_x_dur = threshold_x * 1.5
                        
                    max_y_dur = max(df_pivot['Indice_Durabilidad_E2 (%)'].max() * 1.15, 120) if not df_pivot.empty else 120
                    
                    fig_limpio.update_layout(
                        height=550, 
                        title=f"Espacio de Diseño de Durabilidad - {lbl_idx}",
                        xaxis=dict(range=[0, max_x_dur], rangemode='tozero'),
                        yaxis=dict(range=[0, max_y_dur], rangemode='tozero')
                    )
                    
                    fig_div = go.Figure(fig_limpio)
                    
                    fig_div.add_shape(type="line", x0=threshold_x, y0=0, x1=threshold_x, y1=max_y_dur, line=dict(color="black", dash="dash", width=2), layer="below")
                    fig_div.add_shape(type="line", x0=0, y0=80, x1=max_x_dur, y1=80, line=dict(color="red", dash="dash", width=2), layer="below")
                    fig_div.add_shape(type="line", x0=threshold_x, y0=30, x1=max_x_dur, y1=30, line=dict(color="red", dash="dash", width=2), layer="below")
                    
                    arrow_style = dict(showarrow=False, font=dict(size=40, color="rgba(200, 50, 50, 0.15)"))
                    arrow_style_green = dict(showarrow=False, font=dict(size=40, color="rgba(50, 200, 50, 0.15)"))
                    
                    mid_x_left = threshold_x / 2
                    mid_x_right = threshold_x + (max_x_dur - threshold_x) / 2
                    
                    fig_div.add_annotation(x=mid_x_left, y=80 + (max_y_dur - 80)/2, text="⬆⬆", **arrow_style_green)
                    fig_div.add_annotation(x=mid_x_left, y=40, text="⬇⬇", **arrow_style)
                    fig_div.add_annotation(x=mid_x_right, y=80 + (max_y_dur - 80)/2, text="⬆⬆", **arrow_style_green)
                    fig_div.add_annotation(x=mid_x_right, y=30 + (80 - 30)/2, text="⬆", **arrow_style_green)
                    fig_div.add_annotation(x=mid_x_right, y=15, text="⬇⬇", **arrow_style)
                    
                    with tab_limpio:
                        st.plotly_chart(fig_limpio, use_container_width=True)
                    with tab_div:
                        st.plotly_chart(fig_div, use_container_width=True)
                        
                with col_durB:
                    st.markdown("**Tabla de Diagnóstico**")
                    cols_show = [c for c in ['Mezcla', 'RAP', 'Control', 'E1', 'E2', 'Indice_Durabilidad_E2 (%)'] if c in df_pivot.columns or c in ['Mezcla', 'RAP', 'Indice_Durabilidad_E2 (%)']]
                    st.dataframe(df_pivot[cols_show].style.format({'Control': '{:.2f}', 'E1': '{:.2f}', 'E2': '{:.2f}', 'Indice_Durabilidad_E2 (%)': '{:.1f}%'}, na_rep="-"))
        except Exception as e:
            st.warning(f"No se pudo generar el espacio de durabilidad: {e}")

    # --- Renderizado único del bloque de integración ---
    st.markdown("### Análisis mediante Integración Exacta (Centroide Pre-pico)")
    render_indice_fisuracion('Indice_Fisuracion_Centroide', 'Índice de Fisuración (Tensión Equivalente Centroide)', 'CENT', '2.1', '2.2', threshold_x=None)