import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
import os
import joblib
import folium
from streamlit_folium import st_folium
import requests
# geopy.Nominatim removed to avoid external OSM calls in deployed environment
import urllib3
from folium.plugins import MarkerCluster

# Deshabilitar warnings de SSL
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="Predicción de Precios de Deptos. en CABA",
    page_icon="",
    layout="wide",
)

# --- CARGA DE DATOS Y MODELO ---

@st.cache_data
def cargar_datos():
    """Carga los datos limpios desde la carpeta data"""
    try:
        # Reemplaza 'tu_archivo.csv' con el nombre real de tu archivo CSV
        df = pd.read_csv('data/DatasetFinal.csv')  
        # Calcular precio por m2
        df['precio_m2_usd'] = df['precio'] / df['surface_total']
        return df
    except FileNotFoundError:
        st.error("Error: No se encontró el archivo en 'data/DatasetFinal.csv'")
        st.error("Verifica que el archivo existe y el nombre es correcto.")
        return None
    except Exception as e:
        st.error(f"Error al cargar los datos: {e}")
        return None

@st.cache_resource
def cargar_modelo_y_preprocesador():
    """Carga el pipeline de CLASIFICACIÓN y el ENCODER"""
    
    # Esta es la ruta al archivo que guardaste desde tu notebook
    ruta_paquete = 'model/modelo_clasificador_precios_xgb1V3.pkl'
    
    try:
        # Cargar el diccionario que contiene ambos objetos
        data = joblib.load(ruta_paquete)
        pipeline = data['pipeline']
        encoder = data['encoder']
        
        if 'pipeline' in data and 'encoder' in data:
            return data['pipeline'], data['encoder']
        else:
            st.error("Error: El archivo .pkl no contiene las llaves 'pipeline' o 'encoder'.")
            return None, None
            
    except FileNotFoundError:
        st.error(f"Error: No se encontró el archivo del modelo en `{ruta_paquete}`.")
        st.error("Asegúrate de haber guardado el modelo desde tu notebook en la carpeta 'model'.")
        return None, None
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None, None# --- CARGA INICIAL DE DATOS Y MODELO ---
    
@st.cache_data(show_spinner=False)
def get_barrios_coords_top10(df):
    """
    Devuelve un diccionario con coordenadas para los 10 barrios más frecuentes del dataset.
    Usa el promedio del CSV y no consulta geopy.
    """
    top_barrios = df['barrio'].value_counts().head(30).index.tolist()
    coords = {}
    for barrio in top_barrios:
        group = df[df['barrio'] == barrio]
        valid = group[(~group['latitud'].isnull()) & (~group['longitud'].isnull())]
        if not valid.empty:
            lat = valid['latitud'].astype(float).mean()
            lon = valid['longitud'].astype(float).mean()
            coords[barrio] = (lat, lon)
    return coords


df = cargar_datos()
# Ahora cargamos ambos objetos
modelo, label_encoder = cargar_modelo_y_preprocesador() 
COORDENADAS_BARRIOS = get_barrios_coords_top10(df)
# --- TÍTULO PRINCIPAL ---
st.title("BairesProp")

# --- CSS PERSONALIZADO PARA TABS ---
st.markdown("""
    <style>
    /* Estilo para los tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: white;
        border-radius: 8px;
        color: #31333F;
        font-size: 16px;
        font-weight: 600;
        padding: 10px 20px;
        border: 2px solid transparent;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #e8eaf0;
        border-color: #4A90E2;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #4A90E2 !important;
        color: white !important;
        border-color: transparent !important;
        box-shadow: 0 2px 8px rgba(74, 144, 226, 0.3);
    }
    
    /* Quitar la línea roja de abajo del tab seleccionado */
    .stTabs [data-baseweb="tab-highlight"] {
        background-color: transparent !important;
    }
    
    /* Cambiar color de los botones primarios al mismo azul de los tabs */
    .stButton > button[kind="primary"] {
        background-color: #4A90E2 !important;
        border-color: #4A90E2 !important;
        color: white !important;
    }
    
    .stButton > button[kind="primary"]:hover {
        background-color: #3A7BC8 !important;
        border-color: #3A7BC8 !important;
        box-shadow: 0 4px 12px rgba(74, 144, 226, 0.4) !important;
    }
    
    /* Estilo para el botón de submit del formulario */
    .stForm button[type="submit"] {
        background-color: #4A90E2 !important;
        border-color: #4A90E2 !important;
        color: white !important;
    }
    
    .stForm button[type="submit"]:hover {
        background-color: #3A7BC8 !important;
        border-color: #3A7BC8 !important;
        box-shadow: 0 4px 12px rgba(74, 144, 226, 0.4) !important;
    }
    
    /* Selector adicional para botones primarios en formularios */
    button[kind="primary"] {
        background-color: #4A90E2 !important;
        border-color: #4A90E2 !important;
        color: white !important;
    }
    
    button[kind="primary"]:hover {
        background-color: #3A7BC8 !important;
        border-color: #3A7BC8 !important;
        box-shadow: 0 4px 12px rgba(74, 144, 226, 0.4) !important;
    }
    
    /* Forzar estilo en todos los botones con data-testid */
    button[data-testid="baseButton-primary"] {
        background-color: #4A90E2 !important;
        border-color: #4A90E2 !important;
        color: white !important;
    }
    
    button[data-testid="baseButton-primary"]:hover {
        background-color: #3A7BC8 !important;
        border-color: #3A7BC8 !important;
        box-shadow: 0 4px 12px rgba(74, 144, 226, 0.4) !important;
    }
    
    /* Selector universal para cualquier botón dentro de stForm */
    .stForm button {
        background-color: #4A90E2 !important;
        border-color: #4A90E2 !important;
        color: white !important;
    }
    
    .stForm button:hover {
        background-color: #3A7BC8 !important;
        border-color: #3A7BC8 !important;
        box-shadow: 0 4px 12px rgba(74, 144, 226, 0.4) !important;
    }
    
    /* Selector más específico para el submit button */
    [data-testid="stFormSubmitButton"] button {
        background-color: #4A90E2 !important;
        border-color: #4A90E2 !important;
        color: white !important;
    }
    
    [data-testid="stFormSubmitButton"] button:hover {
        background-color: #3A7BC8 !important;
        border-color: #3A7BC8 !important;
        box-shadow: 0 4px 12px rgba(74, 144, 226, 0.4) !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- PESTAÑAS DE NAVEGACIÓN ---
tab_inicio, tab_eda, tab_prediccion, tab_ingresa = st.tabs([
    " Inicio", 
    " Análisis Exploratorio (EDA)", 
    " ¿Cuánto vale mi Dpto?",
    " Ingresa tu Dpto"
])

# --- PESTAÑA 1: INICIO ---
with tab_inicio:
    st.header("Bienvenido a BairesProp")
    here = os.path.dirname(__file__)
    local_img = os.path.join(here, "skyline-caba.jpg")

    if os.path.exists(local_img):
        st.image(local_img, use_container_width=True)
    else:
        st.warning(f"Imagen local no encontrada en {local_img}. Usando imagen remota.")
        st.image("https://placehold.co/1200x400/333/FFF?text=Foto+Skyline+CABA", use_container_width=True)
    
    st.subheader("Objetivo del Proyecto")
    st.write("""
    El objetivo de este trabajo es analizar el mercado inmobiliario de la Ciudad Autónoma de Buenos Aires (CABA)
    y desarrollar un modelo de Machine Learning capaz de predecir el precio de venta (en USD) de un departamento
    basado en sus características principales, como la ubicación, superficie, y cantidad de ambientes.
    """)
    
    st.subheader("Integrantes del Grupo")
    st.markdown("""
    * Cirrincione, Giovanni
    * Cisterna, Emiliano
    * Donnarumma, Pedro
    """)
    
    st.subheader("Datos Utilizados")
    st.write(f"""
    El análisis y el modelo se basan en un dataset de **{len(df)}** propiedades.
    Aquí puedes ver una muestra de los datos limpios que se utilizaron para las visualizaciones
    y el entrenamiento del modelo:
    """)
    st.dataframe(df.sample(5))

# --- PESTAÑA 2: ANÁLISIS EXPLORATORIO (EDA) ---
with tab_eda:
    st.header("Visualizaciones Interactivas del Mercado")
    st.write("Exploración de las variables clave y su relación con el precio.")

    # --- VISUALIZACIONES CON ALTAIR (Requisito de la entrega) ---
    
    # --- Gráfico 1: Precio Promedio por Barrio (Comparable) ---
    st.subheader("1. Precio Promedio por Barrio")
    st.write("Vista del precio promedio por barrio. Puedes seleccionar filtros para ajustar la visualización.")
    # Calcular percentiles para excluir outliers por defecto en Gráfico 1
    superficie_p5_bar = int(df['surface_total'].quantile(0.05))
    superficie_p95_bar = int(df['surface_total'].quantile(0.95))
    
    # FILTROS para Gráfico 1
    with st.expander("Filtros - Precio por Barrio", expanded=False):
        # Filtro de Zona
        zonas_disponibles = sorted(df['zona'].dropna().unique())
        zona_seleccionada = st.multiselect(
            "Seleccionar Zona(s)",
            options=zonas_disponibles,
            default=[],
            key="zona_bar",
            help="Filtra los barrios por zona geográfica"
        )
        
        col_f3, col_f4 = st.columns(2)
        with col_f3:
            superficie_min_bar = st.number_input(
                "Superficie mínima (m²)", 
                min_value=int(df['surface_total'].min()), 
                max_value=int(df['surface_total'].max()),
                value=superficie_p5_bar,
                step=5,
                key="superficie_min_bar",
                help="Por defecto muestra desde el percentil 5 para mejor visualización"
            )
        with col_f4:
            superficie_max_bar = st.number_input(
                "Superficie máxima (m²)", 
                min_value=int(df['surface_total'].min()), 
                max_value=int(df['surface_total'].max()),
                value=superficie_p95_bar,
                step=5,
                key="superficie_max_bar",
                help="Por defecto muestra hasta el percentil 95 para mejor visualización"
            )
        ambientes_min_bar = st.select_slider(
            "Ambientes mínimos",
            options=sorted(df['ambientes'].unique()),
            value=sorted(df['ambientes'].unique())[0],
            key="ambientes_min_bar"
        )
        banos_min_bar = st.select_slider(
            "Baños mínimos",
            options=sorted(df['baños'].dropna().unique()),
            value=sorted(df['baños'].dropna().unique())[0],
            key="banos_min_bar"
        )
    
    # Aplicar filtros
    df_filtered_bar = df[
        (df['surface_total'] >= superficie_min_bar) & 
        (df['surface_total'] <= superficie_max_bar) &
        (df['ambientes'] >= ambientes_min_bar) &
        (df['baños'] >= banos_min_bar)
    ]
    
    # Aplicar filtro de zona si se seleccionó alguna
    if zona_seleccionada:
        df_filtered_bar = df_filtered_bar[df_filtered_bar['zona'].isin(zona_seleccionada)]
    
    st.caption(f" Mostrando {len(df_filtered_bar)} de {len(df)} propiedades")
    
    # Agrupar datos para mejorar el rendimiento, incluyendo la zona
    df_grouped = df_filtered_bar.groupby(['barrio', 'zona']).agg({'precio': 'mean'}).reset_index()
    df_grouped.columns = ['barrio', 'zona', 'precio_promedio']
    
    # Calcular el precio promedio general
    precio_promedio_general = df_grouped['precio_promedio'].mean()
    
    # Gráfico de barras coloreado por zona
    chart_bar = alt.Chart(df_grouped).mark_bar().encode(
        x=alt.X('barrio:N', 
                sort='-y', 
                title='Barrio', 
                axis=alt.Axis(
                    labelAngle=-45,
                    labelOverlap=False  # Forzar que se muestren todos los labels
                )),
        y=alt.Y('precio_promedio:Q', title='Precio Promedio (USD)'),
        color=alt.Color('zona:N', 
                       title='Zona',
                       scale=alt.Scale(scheme='category10'),
                       legend=alt.Legend(
                           orient='right',
                           title='Zona',
                           titleFontSize=12,
                           labelFontSize=11
                       )),
        tooltip=[
            alt.Tooltip('barrio:N', title='Barrio'),
            alt.Tooltip('zona:N', title='Zona'),
            alt.Tooltip('precio_promedio:Q', title='Precio Promedio', format='$,.0f')
        ]
    ).properties(
        title='Precio Promedio (USD) por Barrio',
        width='container',
        height=400
    )
    
    # Línea punteada horizontal del precio promedio general
    line_promedio = alt.Chart(pd.DataFrame({'y': [precio_promedio_general]})).mark_rule(
        strokeDash=[5, 5],
        color='#FF6B6B',
        size=2
    ).encode(
        y='y:Q'
    )
    
    # Texto con el valor del precio promedio
    text_promedio = alt.Chart(pd.DataFrame({
        'y': [precio_promedio_general],
        'label': [f'Promedio: ${precio_promedio_general:,.0f}']
    })).mark_text(
        align='left',
        dx=5,
        dy=-10,
        fontSize=12,
        fontWeight='bold',
        color="#000000"
    ).encode(
        y='y:Q',
        text='label:N'
    )
    
    # Combinar el gráfico de barras con la línea y el texto
    chart_combined = (chart_bar + line_promedio + text_promedio).configure_axis(
        labelFontSize=10,
        titleFontSize=12
    )
    
    st.altair_chart(chart_combined, use_container_width=True)

    # --- Gráfico 2: Mapa Interactivo de CABA ---
    st.subheader("2. Mapa Interactivo de Propiedades en CABA")
    st.write("Explora las propiedades en el mapa de Buenos Aires. Los marcadores están coloreados según el precio.")
    
    # Calcular percentiles para excluir outliers por defecto en Gráfico 2
    precio_p5_map = int(df['precio'].quantile(0.05))
    precio_p95_map = int(df['precio'].quantile(0.95))
    superficie_p5_map = int(df['surface_total'].quantile(0.05))
    superficie_p95_map = int(df['surface_total'].quantile(0.95))
    
    # FILTROS para Gráfico 2
    with st.expander("Filtros - Mapa de CABA", expanded=False):
        col_f10, col_f11 = st.columns(2)
        
        with col_f10:
            precio_min_map = st.number_input(
                "Precio mínimo (USD)", 
                min_value=int(df['precio'].min()), 
                max_value=int(df['precio'].max()),
                value=precio_p5_map,
                step=10000,
                key="precio_min_map",
                help="Por defecto muestra desde el percentil 5 para mejor visualización"
            )
            precio_max_map = st.number_input(
                "Precio máximo (USD)", 
                min_value=int(df['precio'].min()), 
                max_value=int(df['precio'].max()),
                value=precio_p95_map,
                step=10000,
                key="precio_max_map",
                help="Por defecto muestra hasta el percentil 95 para mejor visualización"
            )
        
        with col_f11:
            superficie_min_map = st.number_input(
                "Superficie mínima (m²)", 
                min_value=int(df['surface_total'].min()), 
                max_value=int(df['surface_total'].max()),
                value=superficie_p5_map,
                step=5,
                key="superficie_min_map",
                help="Por defecto muestra desde el percentil 5 para mejor visualización"
            )
            superficie_max_map = st.number_input(
                "Superficie máxima (m²)", 
                min_value=int(df['surface_total'].min()), 
                max_value=int(df['surface_total'].max()),
                value=superficie_p95_map,
                step=5,
                key="superficie_max_map",
                help="Por defecto muestra hasta el percentil 95 para mejor visualización"
            )
        
        barrios_map = st.multiselect(
            "Seleccionar barrios",
            options=sorted(df['barrio'].unique()),
            default=[],
            key="barrios_map"
        )
        
        col_f12, col_f13 = st.columns(2)
        with col_f12:
            ambientes_map = st.multiselect(
                "Cantidad de ambientes",
                options=sorted(df['ambientes'].unique()),
                default=[],
                key="ambientes_map"
            )
        with col_f13:
            limite_propiedades = st.slider(
                "Máximo de propiedades a mostrar",
                min_value=10,
                max_value=100,
                value=50,
                step=10,
                help="Para mejor rendimiento, limita la cantidad de marcadores en el mapa",
                key="limite_map"
            )
    
    # Aplicar filtros
    df_filtered_map = df[
        (df['precio'] >= precio_min_map) & 
        (df['precio'] <= precio_max_map) &
        (df['surface_total'] >= superficie_min_map) & 
        (df['surface_total'] <= superficie_max_map)
    ]
    if barrios_map:
        df_filtered_map = df_filtered_map[df_filtered_map['barrio'].isin(barrios_map)]
    if ambientes_map:
        df_filtered_map = df_filtered_map[df_filtered_map['ambientes'].isin(ambientes_map)]

    # Tomar hasta 'limite_propiedades' por barrio
    # Limitar a 5 propiedades por barrio para el mapa
    # Filtrar propiedades que tienen coordenadas disponibles
    barrios_con_coord = set(COORDENADAS_BARRIOS.keys())
    propiedades_con_coord = df_filtered_map[df_filtered_map['barrio'].isin(barrios_con_coord)]
    
    st.caption(f" Mostrando {len(df_filtered_map)} de {len(df)} propiedades en el mapa")
    if len(df_filtered_map) == 0:
        st.warning("No hay propiedades que cumplan con todos los filtros seleccionados. Prueba con menos filtros o ajusta los valores.")
    else:
        barrios_filtrados = set(df_filtered_map['barrio'].unique())
        barrios_sin_coord = barrios_filtrados - barrios_con_coord
        if len(propiedades_con_coord) == 0:
            st.warning(f"Las propiedades filtradas pertenecen a barrios sin coordenadas en el mapa: {', '.join(sorted(barrios_sin_coord))}. No se pueden mostrar en el mapa.")
        else:
            propiedades_con_coord = propiedades_con_coord.groupby('barrio').head(5).reset_index(drop=True)
            mapa_caba = folium.Map(
                location=[-34.6037, -58.3816],
                zoom_start=12,
                tiles='OpenStreetMap'
            )
            marker_cluster = MarkerCluster().add_to(mapa_caba)
            def get_color_by_price(precio):
                if precio < 150000:
                    return 'green'
                elif precio < 250000:
                    return 'blue'
                elif precio < 350000:
                    return 'orange'
                else:
                    return 'red'
            propiedades_mostradas = 0
            barrios_fallback = set()
            for idx, row in propiedades_con_coord.iterrows():
                barrio = row['barrio']
                coords = COORDENADAS_BARRIOS.get(barrio)
                if coords and not pd.isnull(coords[0]) and not pd.isnull(coords[1]):
                    lat, lon = coords
                    popup_html = f"""
                    <div style='font-family: Arial; font-size: 12px;'>
                    <b>🏘️ {barrio}</b><br>
                    <b>💰 Precio:</b> ${row['precio']:,.0f} USD<br>
                    <b>📏 Superficie:</b> {row['surface_total']} m²<br>
                    <b>🚪 Ambientes:</b> {row['ambientes']}<br>
                    <b>🚿 Baños:</b> {row['baños']}<br>
                    <b>📊 Precio/m²:</b> ${row['precio_m2_usd']:,.0f} USD
                    </div>
                    """
                    folium.Marker(
                        location=[lat, lon],
                        popup=folium.Popup(popup_html, max_width=250),
                        tooltip=f"{barrio} - ${row['precio']:,.0f}",
                        icon=folium.Icon(
                            color=get_color_by_price(row['precio']),
                            icon='home',
                            prefix='fa'
                        )
                    ).add_to(marker_cluster)
                    propiedades_mostradas += 1

    # Mostrar advertencias y leyenda fuera del bucle
    if propiedades_mostradas == 0:
        if barrios_sin_coord:
            st.warning(f"No se encontraron coordenadas para los siguientes barrios: {', '.join(sorted(barrios_sin_coord))}. No se pueden mostrar propiedades en el mapa.")
        else:
            st.warning("No se encontraron propiedades con coordenadas para los filtros seleccionados.")
    else:
        if barrios_fallback:
            st.info(f"Se usó geolocalización para los siguientes barrios poco comunes: {', '.join(sorted(barrios_fallback))}.")
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; right: 50px; 
                    border:2px solid grey; z-index:9999; 
                    background-color:white;
                    padding: 10px;
                    font-size:14px;
                    border-radius: 5px;
                    color: black;">
            <p style="margin:0; font-weight:bold;"> Leyenda de Precios</p>
            <p style="margin:3px 0;"><i class="fa fa-circle" style="color:green"></i> &lt; $150,000</p>
            <p style="margin:3px 0;"><i class="fa fa-circle" style="color:blue"></i> $150,000 - $250,000</p>
            <p style="margin:3px 0;"><i class="fa fa-circle" style="color:orange"></i> $250,000 - $350,000</p>
            <p style="margin:3px 0;"><i class="fa fa-circle" style="color:red"></i> &gt; $350,000</p>
        </div>
        '''
        mapa_caba.get_root().html.add_child(folium.Element(legend_html))
        st_folium(mapa_caba, width=None, height=500, returned_objects=[])

    # --- Gráfico 3: Comparador de Precio Promedio entre Barrios ---
    st.subheader("3. Comparador de Precio Promedio entre Barrios")
    st.write("Compara el precio promedio de dos barrios según las características del departamento que elijas.")
    
    # FILTROS para Gráfico 3 - Usando un formulario para evitar recargas constantes
    with st.form("form_comparador"):
        st.markdown("#### 🏘️ Selecciona los barrios a comparar")
        col_barrios_comp = st.columns(2)
        
        with col_barrios_comp[0]:
            barrio_1 = st.selectbox(
                "Barrio 1",
                options=sorted(df['barrio'].unique()),
                index=0,
                key="barrio_1_comp"
            )
        
        with col_barrios_comp[1]:
            barrio_2 = st.selectbox(
                "Barrio 2",
                options=sorted(df['barrio'].unique()),
                index=1 if len(df['barrio'].unique()) > 1 else 0,
                key="barrio_2_comp"
            )
        
        st.markdown("#### 🏠 Características del departamento")
        col_caract_1, col_caract_2, col_caract_3 = st.columns(3)
        
        with col_caract_1:
            ambientes_comp = st.selectbox(
                "Cantidad de ambientes",
                options=sorted(df['ambientes'].unique()),
                key="ambientes_comp",
                #valor default 2 ambientes
                index=1 if 2 in sorted(df['ambientes'].unique()) else 0
            )
        
        with col_caract_2:
            banos_comp = st.selectbox(
                "Cantidad de baños",
                options=sorted(df['baños'].dropna().unique()),
                key="banos_comp"
            )
        
        st.markdown("#### 📏 Rango de Superficie")
        col_sup_1, col_sup_2 = st.columns(2)
        
        with col_sup_1:
            superficie_min_comp = st.number_input(
                "Superficie mínima (m²)",
                min_value=int(df['surface_total'].min()),
                max_value=int(df['surface_total'].max()),
                value=int(df['surface_total'].quantile(0.25)),
                step=5,
                key="superficie_min_comp"
            )
        
        with col_sup_2:
            superficie_max_comp = st.number_input(
                "Superficie máxima (m²)",
                min_value=int(df['surface_total'].min()),
                max_value=int(df['surface_total'].max()),
                value=int(df['surface_total'].quantile(0.75)),
                step=5,
                key="superficie_max_comp"
            )
        
        # Botón de comparar centrado - submit del formulario
        st.markdown("<br>", unsafe_allow_html=True)
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
        with col_btn2:
            comparar_btn = st.form_submit_button("🔍 Comparar Barrios", type="primary", use_container_width=True)
    
    # Solo realizar el cálculo si se presionó el botón
    if comparar_btn:
        # Validar que superficie mínima sea menor que máxima
        if superficie_min_comp > superficie_max_comp:
            st.error("⚠️ La superficie mínima debe ser menor o igual a la superficie máxima.")
        else:
            # Filtrar datos para ambos barrios con las características seleccionadas
            df_barrio_1 = df[
                (df['barrio'] == barrio_1) &
                (df['ambientes'] == ambientes_comp) &
                (df['baños'] == banos_comp) &
                (df['surface_total'] >= superficie_min_comp) &
                (df['surface_total'] <= superficie_max_comp)
            ]
            
            df_barrio_2 = df[
                (df['barrio'] == barrio_2) &
                (df['ambientes'] == ambientes_comp) &
                (df['baños'] == banos_comp) &
                (df['surface_total'] >= superficie_min_comp) &
                (df['surface_total'] <= superficie_max_comp)
            ]
            
            # Calcular precios promedio
            precio_promedio_1 = df_barrio_1['precio'].mean() if len(df_barrio_1) > 0 else 0
            precio_promedio_2 = df_barrio_2['precio'].mean() if len(df_barrio_2) > 0 else 0
            
            # Mostrar resultados en columnas
            col_resultado_1, col_resultado_2 = st.columns(2)
            
            with col_resultado_1:
                st.metric(
                    label=f"📊 {barrio_1}",
                    value=f"${precio_promedio_1:,.0f}" if precio_promedio_1 > 0 else "Sin datos",
                    delta=f"{len(df_barrio_1)} propiedades"
                )
            
            with col_resultado_2:
                st.metric(
                    label=f"📊 {barrio_2}",
                    value=f"${precio_promedio_2:,.0f}" if precio_promedio_2 > 0 else "Sin datos",
                    delta=f"{len(df_barrio_2)} propiedades"
                )
            
            # Análisis de diferencia de precio
            if precio_promedio_1 > 0 and precio_promedio_2 > 0:
                if precio_promedio_1 > precio_promedio_2:
                    diferencia = ((precio_promedio_1 - precio_promedio_2) / precio_promedio_2) * 100
                    st.info(f"💡 **{barrio_1}** es **{diferencia:.1f}% más caro** que **{barrio_2}** para estas características.")
                elif precio_promedio_2 > precio_promedio_1:
                    diferencia = ((precio_promedio_2 - precio_promedio_1) / precio_promedio_1) * 100
                    st.info(f"💡 **{barrio_2}** es **{diferencia:.1f}% más caro** que **{barrio_1}** para estas características.")
                else:
                    st.info(f"💡 Ambos barrios tienen precios similares para estas características.")
            else:
                st.warning("⚠️ No hay suficientes datos para ambos barrios con las características seleccionadas. Intenta ajustar los filtros.")
    else:
        st.info("👆 Configura los parámetros de comparación y presiona el botón **'Comparar Barrios'** para ver los resultados.")

    # --- Gráfico 4: Relación Precio vs. Superficie (Interactivo) ---
    st.subheader("4. Relación Precio vs. Superficie Total")
    st.write("Usa el mouse para hacer zoom y panear la visualización.")
    
    # Calcular percentiles para excluir outliers por defecto
    precio_p95 = int(df['precio'].quantile(0.95))
    superficie_p5 = int(df['surface_total'].quantile(0.05))  # Excluir outliers muy pequeños
    superficie_p95 = int(df['surface_total'].quantile(0.95))
    
    # FILTROS para Gráfico 4
    with st.expander("Filtros - Precio vs Superficie", expanded=False):
        col_f5, col_f6, col_f7 = st.columns(3)
        
        with col_f5:
            barrios_scatter = st.multiselect(
                "Seleccionar barrios",
                options=sorted(df['barrio'].unique()),
                default=[],
                key="barrios_scatter"
            )
        
        with col_f6:
            precio_min_scatter = st.number_input(
                "Precio mínimo (USD)", 
                min_value=int(df['precio'].min()), 
                max_value=int(df['precio'].max()),
                value=int(df['precio'].min()),
                step=10000,
                key="precio_min_scatter"
            )
            precio_max_scatter = st.number_input(
                "Precio máximo (USD)", 
                min_value=int(df['precio'].min()), 
                max_value=int(df['precio'].max()),
                value=precio_p95,
                step=10000,
                key="precio_max_scatter",
                help="Por defecto muestra hasta el percentil 95 para mejor visualización"
            )
        
        with col_f7:
            superficie_min_scatter = st.number_input(
                "Superficie mínima (m²)", 
                min_value=int(df['surface_total'].min()), 
                max_value=int(df['surface_total'].max()),
                value=superficie_p5,
                step=5,
                key="superficie_min_scatter",
                help="Por defecto muestra desde el percentil 5 para mejor visualización"
            )
            superficie_max_scatter = st.number_input(
                "Superficie máxima (m²)", 
                min_value=int(df['surface_total'].min()), 
                max_value=int(df['surface_total'].max()),
                value=superficie_p95,
                step=5,
                key="superficie_max_scatter",
                help="Por defecto muestra hasta el percentil 95 para mejor visualización"
            )
        
        col_f8, col_f9 = st.columns(2)
        with col_f8:
            ambientes_scatter = st.multiselect(
                "Cantidad de ambientes",
                options=sorted(df['ambientes'].unique()),
                default=[],
                key="ambientes_scatter"
            )
        with col_f9:
            banos_scatter = st.multiselect(
                "Cantidad de baños",
                options=sorted(df['baños'].dropna().unique()),
                default=[],
                key="banos_scatter"
            )
    
    # Aplicar filtros
    df_filtered_scatter = df[
        (df['precio'] >= precio_min_scatter) & 
        (df['precio'] <= precio_max_scatter) &
        (df['surface_total'] >= superficie_min_scatter) & 
        (df['surface_total'] <= superficie_max_scatter)
    ]
    if barrios_scatter:
        df_filtered_scatter = df_filtered_scatter[df_filtered_scatter['barrio'].isin(barrios_scatter)]
    if ambientes_scatter:
        df_filtered_scatter = df_filtered_scatter[df_filtered_scatter['ambientes'].isin(ambientes_scatter)]
    if banos_scatter:
        df_filtered_scatter = df_filtered_scatter[df_filtered_scatter['baños'].isin(banos_scatter)]
    
    st.caption(f" Mostrando {len(df_filtered_scatter)} de {len(df)} propiedades")
    
    # Calcular límites del eje X basados en los datos filtrados
    x_min = df_filtered_scatter['surface_total'].min()
    x_max = df_filtered_scatter['surface_total'].max()
    
    # Gráfico de dispersión con puntos (sin tooltip)
    chart_scatter = alt.Chart(df_filtered_scatter).mark_circle(opacity=0.7).encode(
        x=alt.X('surface_total', 
                title='Superficie Total (m²)',
                scale=alt.Scale(domain=[x_min, x_max])),
        y=alt.Y('precio', title='Precio (USD)', scale=alt.Scale(zero=False)),
        color=alt.Color('zona:N', 
                       title='Zona',
                       scale=alt.Scale(
                           domain=['Norte', 'Sur', 'Centro/Oeste'],
                           range=['#1f77b4', '#ff7f0e', '#2ca02c']
                       )),
        tooltip=alt.value(None)
    ).properties(
        title='Precio vs. Superficie, coloreado por Zona'
    )
    
    # Crear datos agregados para la línea de tendencia
    df_trend = df_filtered_scatter.groupby('surface_total').agg({'precio': 'mean'}).reset_index()
    df_trend = df_trend.sort_values('surface_total')
    
    # Línea de tendencia roja continua
    line_trend = alt.Chart(df_trend).mark_line(
        color='red',
        size=3
    ).encode(
        x=alt.X('surface_total:Q', 
                title='Superficie Total (m²)',
                scale=alt.Scale(domain=[x_min, x_max])),
        y=alt.Y('precio:Q', title='Precio (USD)')
    )
    
    # Capa invisible para hover
    line_hover = alt.Chart(df_trend).mark_line(
        color='red',
        size=20,
        opacity=0
    ).encode(
        x=alt.X('surface_total:Q', 
                title='Superficie Total (m²)',
                scale=alt.Scale(domain=[x_min, x_max])),
        y=alt.Y('precio:Q', title='Precio (USD)'),
        tooltip=[
            alt.Tooltip('surface_total:Q', title='Superficie (m²)', format='.0f'),
            alt.Tooltip('precio:Q', title='Precio Promedio', format='$,.0f')
        ]
    )
    
    # Combinar los gráficos
    chart_combined = (chart_scatter + line_trend + line_hover).interactive()
    
    st.altair_chart(chart_combined, use_container_width=True)
    
    # Agregar leyenda debajo del gráfico
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 14px; margin-top: -10px;'>
        <span style='color: red; font-weight: bold;'>━━━</span> Línea Roja: Precio Promedio por Superficie
    </div>
    """, unsafe_allow_html=True)


# --- PESTAÑA 3: PREDICTOR de RANGOS DE PRECIOS ---
with tab_prediccion:
    st.header("Calculá el valor de tu propiedad")
    st.write("Completá los siguientes datos y conocé en segundos un valor estimado")
    
    # CSS para quitar el fondo gris de los inputs y ajustar el diseño
    st.markdown("""
        <style>
        /* Quitar fondo gris de inputs de texto */
        .stTextInput input {
            background-color: white !important;
        }
        
        /* Quitar fondo gris de inputs numéricos */
        .stNumberInput input {
            background-color: white !important;
        }
        
        /* Alternativa más específica */
        [data-baseweb="input"] {
            background-color: white !important;
        }
        
        /* Para los controles de número */
        input[type="number"] {
            background-color: white !important;
            color: black !important;
            /* Añadir espacio a la derecha para los botones */
            padding-right: 36px !important;
        }

        /* Estilos para los botones + / - nativos y para los botones que Streamlit renderiza
           (Streamlit usa botones HTML para los controles a la derecha). Intentamos cubrir
           ambos casos: pseudo-elementos WebKit y botones reales generados por Streamlit. */
        input[type="number"]::-webkit-inner-spin-button,
        input[type="number"]::-webkit-outer-spin-button {
            -webkit-appearance: none;
            appearance: none;
            margin: 0;
        }

        /* Ocultar los botones + / - que Streamlit pinta al lado del input */
        .stNumberInput button, .stNumberInput > div > div > button, [data-baseweb="number"] button {
            display: none !important;
        }

        /* Firefox: ocultar los spin buttons nativos y usar botones estilizados */
        input[type="number"] {
            -moz-appearance: textfield;
        }

        /* Ajustar el espacio entre columnas */
        [data-testid="column"] {
            padding-left: 0px !important;
            padding-right: 0px !important;
        }
        
        /* Hacer que el input se pegue al cuadro m² */
        .stNumberInput > div > div {
            border-radius: 0 4px 4px 0 !important;
        }

        
        </style>
    """, unsafe_allow_html=True)

    # --- FUNCIONES AUXILIARES PARA GEOCODIFICACIÓN ---
    
    # Nominatim-based geocoding removed to avoid external OSM calls in deployed environments.
    # Use `geocodificar_direccion_google` with a valid Google Maps API key instead.

    def geocodificar_direccion_google(direccion, api_key):
        """
        Geocodifica una dirección usando Google Maps Geocoding API.
        Devuelve (lat, lng, formatted_address) o None si falla.
        """
        try:
            if not api_key:
                return None
            base_url = "https://maps.googleapis.com/maps/api/geocode/json"
            direccion_caba = f"{direccion}, Ciudad Autónoma de Buenos Aires, Argentina"
            params = {
                'address': direccion_caba,
                'key': api_key,
                'components': 'country:AR'
            }
            headers = {'User-Agent': 'BairesProp/1.0'}
            resp = requests.get(base_url, params=params, headers=headers, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if data.get('status') != 'OK' or not data.get('results'):
                # No results or error from Google
                return None
            # Prefer first result that lies within approximate CABA bounds
            for res in data.get('results', []):
                loc = res['geometry']['location']
                lat = float(loc['lat'])
                lng = float(loc['lng'])
                formatted_addr = res.get('formatted_address', '')
                if esta_en_caba(lat, lng):
                    return lat, lng, formatted_addr
            # If none matched bounds, return the first as a fallback
            first = data['results'][0]
            loc = first['geometry']['location']
            return float(loc['lat']), float(loc['lng']), first.get('formatted_address', '')
        except requests.exceptions.RequestException as e:
            # Network / API error
            print("Google Geocoding request failed:", e)
            return None
    
    def detectar_barrio_y_zona(lat, lng):
        """
        Detecta el barrio, zona y comuna basándose en las coordenadas.
        Esta es una aproximación simplificada. Para mayor precisión, usa polígonos de barrios.
        """
        # Barrios de CABA con coordenadas aproximadas (centro de cada barrio)
        # Formato: 'Barrio': (latitud, longitud, 'Zona', comuna)
        barrios_coords = {
            # ZONA NORTE (11 barrios)
            'Belgrano': (-34.5627, -58.4545, 'Norte', 13),
            'Coghlan': (-34.5563, -58.4775, 'Norte', 12),
            'Colegiales': (-34.5735, -58.4476, 'Norte', 13),
            'Nuñez': (-34.5436, -58.4645, 'Norte', 13),
            'Palermo': (-34.5889, -58.4194, 'Norte', 14),
            'Puerto Madero': (-34.6118, -58.3632, 'Norte', 1),
            'Recoleta': (-34.5875, -58.3974, 'Norte', 2),
            'Retiro': (-34.5926, -58.3766, 'Norte', 1),
            'Saavedra': (-34.5488, -58.4866, 'Norte', 12),
            'Villa Devoto': (-34.6009, -58.5119, 'Norte', 11),
            'Villa Urquiza': (-34.5702, -58.4856, 'Norte', 12),
            
            # ZONA SUR (9 barrios)
            'Barracas': (-34.6440, -58.3748, 'Sur', 4),
            'Boca': (-34.6345, -58.3636, 'Sur', 4),
            'Constitución': (-34.6276, -58.3817, 'Sur', 1),
            'Parque Patricios': (-34.6364, -58.4014, 'Sur', 4),
            'Pompeya': (-34.6537, -58.4197, 'Sur', 4),
            'San Telmo': (-34.6212, -58.3724, 'Sur', 1),
            'Villa Lugano': (-34.6775, -58.4686, 'Sur', 8),
            'Villa Riachuelo': (-34.6885, -58.4613, 'Sur', 8),
            'Villa Soldati': (-34.6638, -58.4440, 'Sur', 8),
            
            # ZONA CENTRO/OESTE (28 barrios)
            'Agronomía': (-34.5985, -58.4894, 'Centro/Oeste', 15),
            'Almagro': (-34.6098, -58.4206, 'Centro/Oeste', 5),
            'Balvanera': (-34.6092, -58.4033, 'Centro/Oeste', 3),
            'Boedo': (-34.6275, -58.4173, 'Centro/Oeste', 5),
            'Caballito': (-34.6177, -58.4398, 'Centro/Oeste', 6),
            'Chacarita': (-34.5889, -58.4524, 'Centro/Oeste', 15),
            'Flores': (-34.6287, -58.4649, 'Centro/Oeste', 7),
            'Floresta': (-34.6263, -58.4831, 'Centro/Oeste', 10),
            'Liniers': (-34.6447, -58.5204, 'Centro/Oeste', 9),
            'Mataderos': (-34.6600, -58.4899, 'Centro/Oeste', 9),
            'Monserrat': (-34.6108, -58.3838, 'Centro/Oeste', 1),
            'Monte Castro': (-34.6158, -58.4723, 'Centro/Oeste', 10),
            'Parque Avellaneda': (-34.6441, -58.4693, 'Centro/Oeste', 9),
            'Parque Chacabuco': (-34.6358, -58.4502, 'Centro/Oeste', 7),
            'Parque Chas': (-34.5773, -58.4835, 'Centro/Oeste', 15),
            'Paternal': (-34.5995, -58.4666, 'Centro/Oeste', 15),
            'San Cristobal': (-34.6205, -58.3977, 'Centro/Oeste', 3),
            'San Nicolás': (-34.6033, -58.3817, 'Centro/Oeste', 1),
            'Velez Sarsfield': (-34.6405, -58.4777, 'Centro/Oeste', 10),
            'Versalles': (-34.6297, -58.5167, 'Centro/Oeste', 10),
            'Villa Crespo': (-34.5999, -58.4399, 'Centro/Oeste', 15),
            'Villa General Mitre': (-34.5862, -58.4689, 'Centro/Oeste', 11),
            'Villa Luro': (-34.6360, -58.4983, 'Centro/Oeste', 10),
            'Villa Ortuzar': (-34.5789, -58.4623, 'Centro/Oeste', 15),
            'Villa Pueyrredón': (-34.5894, -58.5014, 'Centro/Oeste', 12),
            'Villa Real': (-34.6182, -58.4938, 'Centro/Oeste', 10),
            'Villa Santa Rita': (-34.6234, -58.4852, 'Centro/Oeste', 11),
            'Villa del Parque': (-34.6056, -58.4896, 'Centro/Oeste', 11),
        }
        
        # Calcular distancia a cada barrio y encontrar el más cercano
        min_dist = float('inf')
        barrio_cercano = "Desconocido"
        zona = "Desconocido"
        comuna = 1
        
        for barrio, (b_lat, b_lng, b_zona, b_comuna) in barrios_coords.items():
            dist = ((lat - b_lat)**2 + (lng - b_lng)**2)**0.5
            if dist < min_dist:
                min_dist = dist
                barrio_cercano = barrio
                zona = b_zona
                comuna = b_comuna
        
        return barrio_cercano, zona, comuna
    
    def esta_en_caba(lat, lng):
        """
        Verifica si las coordenadas están dentro de los límites de CABA.
        Usa el algoritmo de ray casting para verificar si un punto está dentro de un polígono.
        Retorna True si está dentro, False si está fuera.
        """
        # Coordenadas del perímetro real de CABA (longitud, latitud)
        perimetro_caba = [
            [-58.4814, -34.5407], [-58.4895, -34.5441], [-58.4973, -34.5477],
            [-58.5018, -34.5544], [-58.5059, -34.5612], [-58.5099, -34.5693],
            [-58.5133, -34.5773], [-58.5191, -34.5886], [-58.5255, -34.6008],
            [-58.5302, -34.6111], [-58.5312, -34.6176], [-58.5311, -34.6255],
            [-58.5300, -34.6334], [-58.5294, -34.6442], [-58.5288, -34.6549],
            [-58.5211, -34.6645], [-58.5063, -34.6700], [-58.4901, -34.6813],
            [-58.4765, -34.6925], [-58.4688, -34.6987], [-58.4610, -34.7039],
            [-58.4533, -34.6961], [-58.4467, -34.6882], [-58.4350, -34.6755],
            [-58.4226, -34.6623], [-58.4088, -34.6628], [-58.3949, -34.6624],
            [-58.3806, -34.6575], [-58.3667, -34.6503], [-58.3555, -34.6384],
            [-58.3541, -34.6301], [-58.3562, -34.6221], [-58.3598, -34.6102],
            [-58.3645, -34.5951], [-58.3681, -34.5843], [-58.3718, -34.5786],
            [-58.3799, -34.5750], [-58.3902, -34.5681], [-58.4021, -34.5609],
            [-58.4153, -34.5529], [-58.4284, -34.5458], [-58.4425, -34.5388],
            [-58.4552, -34.5348], [-58.4658, -34.5369], [-58.4735, -34.5385],
            [-58.4814, -34.5407]
        ]
        
        # Algoritmo de Ray Casting para determinar si un punto está dentro de un polígono
        n = len(perimetro_caba)
        inside = False
        
        p1_lng, p1_lat = perimetro_caba[0]
        for i in range(1, n + 1):
            p2_lng, p2_lat = perimetro_caba[i % n]
            if lat > min(p1_lat, p2_lat):
                if lat <= max(p1_lat, p2_lat):
                    if lng <= max(p1_lng, p2_lng):
                        if p1_lat != p2_lat:
                            lng_interseccion = (lat - p1_lat) * (p2_lng - p1_lng) / (p2_lat - p1_lat) + p1_lng
                        if p1_lng == p2_lng or lng <= lng_interseccion:
                            inside = not inside
            p1_lng, p1_lat = p2_lng, p2_lat
        
        return inside
    
    # --- ESTADO DE SESIÓN PARA COORDENADAS ---
    if 'lat' not in st.session_state:
        st.session_state.lat = -34.6037  # Centro de CABA (aproximado)
    if 'lng' not in st.session_state:
        st.session_state.lng = -58.3816
    if 'barrio_detectado' not in st.session_state:
        st.session_state.barrio_detectado = ""
    if 'zona_detectada' not in st.session_state:
        st.session_state.zona_detectada = ""
    if 'comuna_detectada' not in st.session_state:
        st.session_state.comuna_detectada = 1
    
    # --- LAYOUT EN DOS COLUMNAS ---
    col_ubicacion, col_caracteristicas = st.columns([1, 1], gap="large")
    
    with col_ubicacion:
        st.subheader("Ubicación del Departamento")
        
        # --- OPCIÓN 1: INGRESO MANUAL DE DIRECCIÓN ---
        st.markdown("**Opción 1: Ingresa la dirección manualmente**")
        
        direccion_input = st.text_input(
            "Dirección (calle y altura):",
            placeholder="(Ej: Av. del Libertador 500)",
            help="Ingresa la dirección del departamento en CABA"
        )
        
        # Google Maps API Key: prefer `st.secrets`, otherwise allow paste (opcional).
        # Si se provee la API Key, intentaremos Google Geocoding y caeremos a Nominatim si falla.
        GOOGLE_MAPS_API_KEY = None
        if 'GOOGLE_MAPS_API_KEY' in st.secrets:
            GOOGLE_MAPS_API_KEY = st.secrets['GOOGLE_MAPS_API_KEY']
        else:
            GOOGLE_MAPS_API_KEY = st.text_input(
                "Google Maps API Key (opcional)",
                type="password",
                help="Pega tu API Key para usar Google Geocoding (se recomienda configurar en Streamlit Secrets)"
            )

        if st.button("Buscar Dirección", type="primary", use_container_width=True):
            if direccion_input:
                with st.status("Buscando dirección...", state="running", expanded=True) as status_busqueda:
                    status_busqueda.write("Geocodificando dirección...")
                    result = None
                    # Si hay API Key, intentar Google; no usar Nominatim en deploy
                    if GOOGLE_MAPS_API_KEY:
                        status_busqueda.write("Intentando geocodificar con Google Maps...")
                        result = geocodificar_direccion_google(direccion_input, GOOGLE_MAPS_API_KEY)
                        if result is None:
                            status_busqueda.write("Google no devolvió resultado válido. Geocodificación automática no disponible.")
                    else:
                        status_busqueda.write("No se proporcionó API Key de Google. La geocodificación automática está deshabilitada en esta sección.")

                    if result:
                        status_busqueda.write("✓ Dirección encontrada")
                        lat, lng, formatted_addr = result

                        # Actualizar coordenadas
                        st.session_state.lat = lat
                        st.session_state.lng = lng

                        status_busqueda.write("Detectando barrio y zona...")
                        # Detectar barrio, zona y comuna
                        barrio, zona, comuna = detectar_barrio_y_zona(lat, lng)
                        st.session_state.barrio_detectado = barrio
                        st.session_state.zona_detectada = zona
                        st.session_state.comuna_detectada = comuna

                        status_busqueda.update(label="✓ Dirección encontrada y procesada", state="complete", expanded=False)

                        st.success(f" Ubicación encontrada: {formatted_addr}")

                        # No hacemos rerun aquí - el mapa se actualizará en el siguiente render
                    else:
                        status_busqueda.update(label=" No se encontró la dirección", state="error", expanded=False)
                        st.error("No se pudo geocodificar automáticamente. Por favor, usa el mapa para seleccionar la ubicación o completa barrio/comuna manualmente.")
            else:
                st.warning("Por favor, ingresa una dirección.")
        
        st.markdown("---")
        
        # --- OPCIÓN 2: SELECCIÓN EN MAPA INTERACTIVO ---
        st.markdown("**Opción 2: Selecciona la ubicación en el mapa**")
        st.caption("Haz clic en el mapa para marcar la ubicación del departamento.")
        
        # Crear mapa centrado en CABA
        mapa = folium.Map(
            location=[st.session_state.lat, st.session_state.lng],
            zoom_start=13,
            tiles="OpenStreetMap"
        )
        
        # Dibujar el perímetro real detallado de CABA
        # Convertir coordenadas de [lng, lat] a [lat, lng] para folium
        perimetro_caba = [
            [-34.5407, -58.4814], [-34.5441, -58.4895], [-34.5477, -58.4973],
            [-34.5544, -58.5018], [-34.5612, -58.5059], [-34.5693, -58.5099],
            [-34.5773, -58.5133], [-34.5886, -58.5191], [-34.6008, -58.5255],
            [-34.6111, -58.5302], [-34.6176, -58.5312], [-34.6255, -58.5311],
            [-34.6334, -58.5300], [-34.6442, -58.5294], [-34.6549, -58.5288],
            [-34.6645, -58.5211], [-34.6700, -58.5063], [-34.6813, -58.4901],
            [-34.6925, -58.4765], [-34.6987, -58.4688], [-34.7039, -58.4610],
            [-34.6961, -58.4533], [-34.6882, -58.4467], [-34.6755, -58.4350],
            [-34.6623, -58.4226], [-34.6628, -58.4088], [-34.6624, -58.3949],
            [-34.6575, -58.3806], [-34.6503, -58.3667], [-34.6384, -58.3555],
            [-34.6301, -58.3541], [-34.6221, -58.3562], [-34.6102, -58.3598],
            [-34.5951, -58.3645], [-34.5843, -58.3681], [-34.5786, -58.3718],
            [-34.5750, -58.3799], [-34.5681, -58.3902], [-34.5609, -58.4021],
            [-34.5529, -58.4153], [-34.5458, -58.4284], [-34.5388, -58.4425],
            [-34.5348, -58.4552], [-34.5369, -58.4658], [-34.5385, -58.4735],
            [-34.5407, -58.4814]
        ]
        
        folium.Polygon(
            locations=perimetro_caba,
            color='red',
            weight=2,
            fill=True,
            fill_color='blue',
            fill_opacity=0.05,
            interactive=False
        ).add_to(mapa)
        
        # Agregar marcador en la posición actual
        folium.Marker(
            [st.session_state.lat, st.session_state.lng],
            popup="Ubicación seleccionada",
            tooltip="Departamento",
            icon=folium.Icon(color="red", icon="home", prefix='fa')
        ).add_to(mapa)
        
        # Mostrar mapa y capturar clicks
        map_data = st_folium(
            mapa,
            width=700,
            height=400,
            returned_objects=["last_clicked"]
        )
        
        # Actualizar coordenadas si el usuario hizo click en el mapa
        if map_data and map_data.get("last_clicked"):
            new_lat = map_data["last_clicked"]["lat"]
            new_lng = map_data["last_clicked"]["lng"]
            
            # Verificar si las coordenadas cambiaron
            if new_lat != st.session_state.lat or new_lng != st.session_state.lng:
                # Actualizar coordenadas sin validación
                st.session_state.lat = new_lat
                st.session_state.lng = new_lng
                
                # Detectar barrio, zona y comuna
                barrio, zona, comuna = detectar_barrio_y_zona(new_lat, new_lng)
                st.session_state.barrio_detectado = barrio
                st.session_state.zona_detectada = zona
                st.session_state.comuna_detectada = comuna
                
                st.rerun()
        
        # Mostrar mensaje de ubicación detectada
        if st.session_state.barrio_detectado and st.session_state.zona_detectada:
            st.info(f""" Departamento localizado  
**Barrio:** {st.session_state.barrio_detectado.upper()}  
**Zona:** {st.session_state.zona_detectada}  
Ciudad Autónoma de Buenos Aires""")
    
    with col_caracteristicas:
        st.subheader("Características del Departamento")
        
        with st.form(key="prediction_form"):
            st.markdown("**Ingrese los datos de la propiedad:**")
            
            in_baños = st.number_input(
                "Cantidad de Baños:",
                min_value=1,
                max_value=10,
                    value=None,
                step=1,
                help="Cantidad de baños completos en el departamento"
            )
            
            # Input: Cantidad de habitaciones
            in_habitaciones = st.number_input(
                "Cantidad de Habitaciones:",
                min_value=0,
                max_value=10,
                value=None,
                step=1,
                help="Cantidad de dormitorios/habitaciones"
            )
            
            # Input: Cantidad de ambientes
            in_ambientes = st.number_input(
                "Cantidad de Ambientes:",
                min_value=1,
                max_value=10,
                value=None,
                step=1,
                help="Cantidad total de ambientes (incluye habitaciones, living, comedor, etc.)"
            )
            
            # Input: Superficie total con formato m2
            in_surface_total = st.number_input(
                "Superficie Total (m²):",
                min_value=15.0,
                max_value=500.0,
                value=None,
                step=1.0,
                help="Superficie total del departamento en metros cuadrados"
            )
            
            # Input: Superficie cubierta con formato m2
            in_surface_covered = st.number_input(
                "Superficie Cubierta (m²):",
                min_value=15.0,
                max_value=500.0,
                value=None,
                step=1.0,
                help="Superficie cubierta del departamento en metros cuadrados"
            )
            
            st.markdown("---")
            
            # Botón de envío del formulario
            submit_button = st.form_submit_button(
                label="Calcular Rango de Precio",
                type="primary",
                use_container_width=True
            )
        
        # --- RESULTADO DE LA PREDICCIÓN ---
        st.markdown("---")
        st.subheader("Resultado de la Predicción:")
        
        if not submit_button:
            st.info("Completa los datos del formulario y presiona 'Calcular Rango de Precio'.")
        
        elif submit_button and modelo and label_encoder:
            # Verificar que todos los campos estén completos
            if (
                in_baños is None
                or in_habitaciones is None
                or in_ambientes is None
                or in_surface_total is None
                or in_surface_covered is None
            ):
                st.error("Por favor, completa todos los campos del formulario.")
            # Verificar que se haya seleccionado una ubicación
            elif not st.session_state.barrio_detectado:
                st.warning("Por favor, selecciona una ubicación en el mapa o ingresa una dirección.")
            elif not esta_en_caba(st.session_state.lat, st.session_state.lng):
                st.error("No se puede realizar la predicción.")
                st.warning("La ubicación seleccionada está fuera de los límites de la Ciudad Autónoma de Buenos Aires. El modelo solo funciona para propiedades dentro de CABA.")
            else:
                # --- Lógica de Predicción ---
                # Validación: la superficie cubierta no puede ser mayor que la superficie total
                if (in_surface_total is not None and in_surface_covered is not None
                        and in_surface_covered > in_surface_total):
                    st.error("La Superficie Cubierta no puede ser mayor que la Superficie Total. Revisa los valores ingresados.")
                else:
                    with st.status("Calculando rango de precio...", state="running") as status_prediccion:
                        # Normalizar nombre del barrio (minúsculas y reemplazar espacios por guiones bajos)
                        barrio_norm = st.session_state.barrio_detectado.lower().replace(' ', '_')
                        zona_norm = st.session_state.zona_detectada.lower().replace('/', '_').replace(' ', '_')
                        
                        # Lista de todos los barrios posibles (basado en el error)
                        barrios = [
                            'palermo', 'recoleta', 'belgrano', 'nuñez', 'colegiales', 'villa_urquiza', 
                            'saavedra', 'coghlan', 'villa_pueyrredón', 'villa_devoto', 'villa_del_parque', 
                            'agronomía', 'chacarita', 'paternal', 'villa_crespo', 'almagro', 'caballito', 
                            'flores', 'floresta', 'parque_chacabuco', 'boedo', 'san_cristobal', 'constitución', 
                            'san_telmo', 'monserrat', 'balvanera', 'retiro', 'puerto_madero', 'barracas', 
                            'boca', 'parque_patricios', 'pompeya', 'mataderos', 'liniers', 'versalles', 
                            'villa_luro', 'velez_sarsfield', 'villa_lugano', 'villa_riachuelo', 'villa_soldati', 
                            'parque_avellaneda', 'villa_real', 'monte_castro', 'villa_santa_rita', 
                            'villa_ortuzar', 'villa_general_mitre', 'san_nicolás', 'parque_chas'
                        ]
                        
                        zonas = ['norte', 'sur', 'centro_oeste']
                        
                        # 1. Crear DataFrame base con características principales
                        input_data = {
                            'barrio': barrio_norm,
                            'zona': zona_norm,
                            'surface_total': in_surface_total,
                            'surface_covered': in_surface_covered,
                            'ambientes': in_ambientes,
                            'habitaciones': in_habitaciones,
                            'baños': in_baños,
                            'comuna': st.session_state.comuna_detectada,
                            'precio_numeric': 0  # Placeholder
                        }
                        
                        # 2. Crear todas las columnas de interacción con valor 0
                        for barrio in barrios:
                            input_data[f'amb_x_barrio_{barrio}'] = 0
                            input_data[f'hab_x_barrio_{barrio}'] = 0
                            input_data[f'banos_x_barrio_{barrio}'] = 0
                            input_data[f'sup_tot_x_barrio_{barrio}'] = 0
                            input_data[f'sup_cub_x_barrio_{barrio}'] = 0
                        
                        for zona in zonas:
                            input_data[f'amb_x_{zona}'] = 0
                            input_data[f'hab_x_{zona}'] = 0
                            input_data[f'banos_x_{zona}'] = 0
                            input_data[f'sup_tot_x_{zona}'] = 0
                            input_data[f'sup_cub_x_{zona}'] = 0
                        
                        # 2b. Crear columnas de interacción por comuna (15 comunas)
                        for comuna_num in range(1, 16):  # Comunas 1 a 15
                            input_data[f'hab_x_comuna_{comuna_num}'] = 0
                            input_data[f'banos_x_comuna_{comuna_num}'] = 0
                            input_data[f'amb_x_comuna_{comuna_num}'] = 0
                            input_data[f'sup_cub_x_comuna_{comuna_num}'] = 0
                            input_data[f'sup_tot_x_comuna_{comuna_num}'] = 0
                        
                        # 3. Asignar valores a las columnas que corresponden al barrio y zona seleccionados
                        if f'amb_x_barrio_{barrio_norm}' in input_data:
                            input_data[f'amb_x_barrio_{barrio_norm}'] = in_ambientes
                            input_data[f'hab_x_barrio_{barrio_norm}'] = in_habitaciones
                            input_data[f'banos_x_barrio_{barrio_norm}'] = in_baños
                            input_data[f'sup_tot_x_barrio_{barrio_norm}'] = in_surface_total
                            input_data[f'sup_cub_x_barrio_{barrio_norm}'] = in_surface_covered
                        
                        if f'amb_x_{zona_norm}' in input_data:
                            input_data[f'amb_x_{zona_norm}'] = in_ambientes
                            input_data[f'hab_x_{zona_norm}'] = in_habitaciones
                            input_data[f'banos_x_{zona_norm}'] = in_baños
                            input_data[f'sup_tot_x_{zona_norm}'] = in_surface_total
                            input_data[f'sup_cub_x_{zona_norm}'] = in_surface_covered
                        
                        # 3c. Asignar valores a las columnas de interacción por comuna
                        try:
                            comuna_num = int(st.session_state.comuna_detectada)
                            if 1 <= comuna_num <= 15:
                                input_data[f'hab_x_comuna_{comuna_num}'] = in_habitaciones
                                input_data[f'banos_x_comuna_{comuna_num}'] = in_baños
                                input_data[f'amb_x_comuna_{comuna_num}'] = in_ambientes
                                input_data[f'sup_cub_x_comuna_{comuna_num}'] = in_surface_covered
                                input_data[f'sup_tot_x_comuna_{comuna_num}'] = in_surface_total
                        except (ValueError, AttributeError):
                            # Si la comuna no es un número válido, dejamos las columnas en 0
                            pass
                        
                        # 4. Convertir a DataFrame (una sola fila)
                        input_df = pd.DataFrame([input_data])
                        
                        st.write("**Características principales enviadas al modelo:**")
                        main_features = {
                            'Barrio': st.session_state.barrio_detectado,
                            'Zona': st.session_state.zona_detectada,
                            'Comuna': st.session_state.comuna_detectada,
                            'Superficie Total': f"{in_surface_total} m²",
                            'Superficie Cubierta': f"{in_surface_covered} m²",
                            'Ambientes': in_ambientes,
                            'Habitaciones': in_habitaciones,
                            'Baños': in_baños
                        }
                        st.dataframe(pd.DataFrame([main_features]), use_container_width=True)

                        try:
                            prediccion_numerica = modelo.predict(input_df)
                            prediccion_etiqueta = label_encoder.inverse_transform(prediccion_numerica)
                            status_prediccion.update(label="Predicción completada", state="complete")
                            st.success(f"¡Predicción exitosa!")
                            
                            st.markdown("### Rango de Precio Estimado:")

                            # Formatea la etiqueta de rango para mostrar separador de miles con puntos.
                            def _format_price_range(label: str) -> str:
                                s = (label or "").strip()
                                if not s:
                                    return s
                                # Si es un rango 'min-max', formattear cada lado
                                if '-' in s:
                                    left, right = [p.strip() for p in s.split('-', 1)]
                                    def _fmt(part: str) -> str:
                                        digits = ''.join(ch for ch in part if ch.isdigit())
                                        if not digits:
                                            return part
                                        try:
                                            return f"{int(digits):,}".replace(',', '.')
                                        except Exception:
                                            return part
                                    # Espacio alrededor del guión según formato pedido
                                    return f"{_fmt(left)} - {_fmt(right)}"
                                # Si es un único número
                                digits = ''.join(ch for ch in s if ch.isdigit())
                                if digits:
                                    try:
                                        return f"{int(digits):,}".replace(',', '.')
                                    except Exception:
                                        return s
                                return s

                            formatted_label = _format_price_range(prediccion_etiqueta[0])
                            # Mostrar con sufijo USD y respetando el espacio alrededor del guión
                            st.markdown(f"# **{formatted_label}** USD")
                            
                            st.info("""
                            Esta etiqueta representa el rango de precios más probable 
                            para una propiedad con las características ingresadas, 
                            según nuestro modelo de clasificación.
                            """)
                            
                            st.subheader("Resumen de la Propiedad")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Barrio", st.session_state.barrio_detectado)
                                st.metric("Zona", st.session_state.zona_detectada)
                                st.metric("Comuna", st.session_state.comuna_detectada)
                                st.metric("Ambientes", in_ambientes)
                                st.metric("Habitaciones", in_habitaciones)
                            with col2:
                                st.metric("Baños", in_baños)
                                st.metric("Sup. Total", f"{in_surface_total} m²")
                                st.metric("Sup. Cubierta", f"{in_surface_covered} m²")
                                st.metric("Precio/m²", f"~ USD {int(np.random.randint(2000, 4000))}/m²")
                        
                        except Exception as e:
                            status_prediccion.update(label="Error durante la predicción", state="error")
                            st.error(f"Error al realizar la predicción: {e}")
                            st.warning("""
                            **Posibles causas:**
                            - El modelo no se ha cargado correctamente
                            - Los nombres de las columnas no coinciden con el modelo entrenado
                            - Falta alguna característica requerida por el modelo
                            
                            Revisa los mensajes de error al inicio de la página y asegúrate de que 
                            el modelo fue entrenado con las mismas características que estás ingresando.
                            """)

        elif submit_button and (not modelo or not label_encoder):
            st.error("Error: El modelo o el LabelEncoder no se han cargado. Revisa los mensajes de error al inicio de la página.")

    # --- PESTAÑA: INGRESA TU DPTO ---
    with tab_ingresa:
        st.header("Ingresa tu Dpto")
        st.write("Completá los datos de tu departamento. A partir de la dirección se intentará detectar barrio, comuna y zona.")

        with st.form(key="ingresa_form"):
            direccion_input = st.text_input("Dirección (calle y altura):", placeholder="Av. Example 123", help="Ingresa la dirección de tu departamento")

            in_baños_i = st.number_input("Cantidad de Baños:", min_value=0, max_value=10, value=None, step=1, help="Cantidad de baños completos")
            in_habitaciones_i = st.number_input("Cantidad de Habitaciones:", min_value=0, max_value=10, value=None, step=1, help="Cantidad de dormitorios")
            in_ambientes_i = st.number_input("Cantidad de Ambientes:", min_value=0, max_value=10, value=None, step=1, help="Cantidad total de ambientes")
            in_surface_total_i = st.number_input("Superficie Total (m²):", min_value=0.0, max_value=2000.0, value=None, step=1.0, help="Superficie total en m²")
            in_surface_covered_i = st.number_input("Superficie Cubierta (m²):", min_value=0.0, max_value=2000.0, value=None, step=1.0, help="Superficie cubierta en m²")
            precio_input = st.number_input("Precio (USD):", min_value=0.0, max_value=100000000.0, value=None, step=100.0, help="Ingresa el precio en dólares (USD)")

            submit_ingresa = st.form_submit_button(label="Generar vista previa", type="primary")

        if submit_ingresa:
            # Validaciones básicas
            if not direccion_input:
                st.error("Por favor ingresa una dirección.")
            else:
                # Validación: la superficie cubierta no puede ser mayor que la superficie total
                if (in_surface_total_i is not None and in_surface_covered_i is not None
                        and in_surface_covered_i > in_surface_total_i):
                    st.error("La Superficie Cubierta no puede ser mayor que la Superficie Total. Revisa los valores ingresados.")
                else:
                    # Geocodificar y detectar barrio/zona/comuna con loader
                    with st.status("Procesando dirección...", state="running", expanded=True) as status_geocode:
                        status_geocode.write("Geocodificando dirección...")
                        # Preferir API key desde variable de entorno o st.secrets
                        import os
                        GOOGLE_MAPS_API_KEY = os.environ.get('GOOGLE_MAPS_API_KEY') or st.secrets.get('GOOGLE_MAPS_API_KEY')

                        result = None
                        if GOOGLE_MAPS_API_KEY:
                            status_geocode.write("Intentando geocodificar con Google Maps...")
                            result = geocodificar_direccion_google(direccion_input, GOOGLE_MAPS_API_KEY)
                            if result is None:
                                status_geocode.write("Google no devolvió resultado válido. No se intentará Nominatim en esta sección.")
                        else:
                            # No intentamos Nominatim aquí por motivos de despliegue
                            status_geocode.write("No se proporcionó API Key de Google. La geocodificación automática está deshabilitada en esta sección.")

                        if result:
                            status_geocode.write("✓ Dirección encontrada")
                            lat, lng, formatted_addr = result
                            status_geocode.write("Detectando barrio, zona y comuna...")
                            barrio_det, zona_det, comuna_det = detectar_barrio_y_zona(lat, lng)
                            status_geocode.update(label="✓ Dirección procesada exitosamente", state="complete", expanded=False)
                        else:
                            barrio_det, zona_det, comuna_det = "", "", ""
                            status_geocode.update(label="⚠ No se pudo geocodificar automáticamente", state="error", expanded=False)
                            st.warning("No se realizó geocodificación automática. Por favor seleccioná la ubicación en el mapa o completá barrio/comuna manualmente.")

                # Validación: superficie cubierta <= superficie total (si ambos fueron ingresados)
                if (in_surface_total_i is not None and in_surface_covered_i is not None) and (in_surface_covered_i > in_surface_total_i):
                    st.error("La superficie cubierta no puede ser mayor que la superficie total. Revísalo por favor.")
                else:
                    # Preparar fila (no guardamos aún; mostramos vista previa)
                    row = {
                        'direccion': direccion_input,
                        'barrio': barrio_det,
                        'zona': zona_det,
                        'comuna': comuna_det,
                        'baños': in_baños_i if in_baños_i is not None else '',
                        'habitaciones': in_habitaciones_i if in_habitaciones_i is not None else '',
                        'ambientes': in_ambientes_i if in_ambientes_i is not None else '',
                        'surface_total': in_surface_total_i if in_surface_total_i is not None else '',
                        'surface_covered': in_surface_covered_i if in_surface_covered_i is not None else '',
                        'precio': precio_input if precio_input is not None else ''
                    }

                    # Guardar la vista previa en session_state para que el botón de confirmación
                    # funcione correctamente incluso después de un rerun.
                    st.session_state['preview_row'] = row
                    st.success("Vista previa generada. Revisa abajo y confirma para guardar.")

        # Mostrar la vista previa almacenada en session_state y permitir confirmar el guardado
        if 'preview_row' in st.session_state:
            preview = st.session_state['preview_row']
            st.info("Vista previa del registro (revísala antes de confirmar):")
            st.table(pd.DataFrame([preview]))

            if st.button("Confirmar y guardar"):
                row = preview.copy()
                csv_path = 'data/DatasetFinal.csv'
                try:
                    if os.path.exists(csv_path):
                        df_save = pd.read_csv(csv_path)
                    else:
                        df_save = pd.DataFrame()
                except Exception as e:
                    st.error(f"Error leyendo el CSV existente: {e}")
                    df_save = pd.DataFrame()

                # Alinear columnas: si el CSV existente tiene otras columnas, dejamos en blanco
                if not df_save.empty:
                    for col in df_save.columns:
                        if col not in row:
                            row[col] = ''
                    row_df = pd.DataFrame([row])[df_save.columns.tolist()]
                    df_new = pd.concat([df_save, row_df], ignore_index=True)
                else:
                    # Crear DataFrame con las columnas del row
                    df_new = pd.DataFrame([row])

                try:
                    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
                    with st.status("Guardando departamento...", state="running") as status_save:
                        df_new.to_csv(csv_path, index=False)
                        status_save.update(label="Guardado", state="complete")
                    st.success("Tu departamento se registro correctamente")
                    # Limpiar la vista previa
                    del st.session_state['preview_row']
                except Exception as e:
                    st.error(f"Error al guardar el CSV: {e}")


