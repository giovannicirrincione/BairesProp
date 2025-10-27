import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
from geopy.geocoders import Nominatim
import folium
from streamlit_folium import st_folium
# Descomenta la siguiente línea si guardaste tu modelo con joblib
# import joblib 

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="Predicción de Precios de Departamentos en CABA",
    page_icon="🏙️",
    layout="wide",
)

# --- DATOS Y MODELO (SIMULADOS) ---
# Esta sección simula la carga de tus datos y tu modelo.
# DEBERÁS REEMPLAZAR ESTO CON LA CARGA DE TUS ARCHIVOS REALES.

import os
@st.cache_data(show_spinner=False)
def get_barrios_coords_top10(df):
    """
    Devuelve un diccionario con coordenadas para los 10 barrios más frecuentes del dataset.
    Usa el promedio del CSV y no consulta geopy.
    """
    top_barrios = df['barrio'].value_counts().head(10).index.tolist()
    coords = {}
    for barrio in top_barrios:
        group = df[df['barrio'] == barrio]
        valid = group[(~group['latitud'].isnull()) & (~group['longitud'].isnull())]
        if not valid.empty:
            lat = valid['latitud'].astype(float).mean()
            lon = valid['longitud'].astype(float).mean()
            coords[barrio] = (lat, lon)
    return coords

@st.cache_data
def cargar_datos():
    """Carga los datos limpios (reemplaza con tu .csv)"""
    # Simulación de datos
    data = {
        'barrio': np.random.choice(['Palermo', 'Recoleta', 'Belgrano', 'Caballito', 'Almagro', 'Villa Urquiza'], 100),
        'superficie_total': np.random.randint(30, 150, 100),
        'ambientes': np.random.randint(1, 5, 100),
        'baños': np.random.randint(1, 3, 100),
        'precio_usd': np.random.randint(80000, 400000, 100)
    }
    # Ajustamos el precio basado en el barrio para que tenga más sentido
    def ajustar_precio(row):
        if row['barrio'] == 'Palermo':
            return row['precio_usd'] * 1.5
        if row['barrio'] == 'Recoleta':
            return row['precio_usd'] * 1.4
        if row['barrio'] == 'Belgrano':
            return row['precio_usd'] * 1.2
        return row['precio_usd'] * 0.9
    
    df = pd.DataFrame(data)
    df['precio_usd'] = df.apply(ajustar_precio, axis=1).astype(int)
    df['precio_m2_usd'] = df['precio_usd'] / df['superficie_total']
    
    # EN LA VIDA REAL:
    # df = pd.read_csv('data/tu_dataset_limpio.csv')
    return df

@st.cache_resource
def cargar_modelo_y_preprocesador():
    """Carga el modelo y el preprocesador (reemplaza con tu .pkl)"""
    # Simulación de un pipeline (modelo + preprocesador)
    # En tu caso, cargarías tus archivos
    
    # class MockPipeline:
    #     """Clase para simular un pipeline de scikit-learn"""
    #     def __init__(self):
    #         # Simulamos las columnas que el modelo "aprendió"
    #         self.columnas_ = ['superficie_total', 'ambientes', 'baños', 'barrio_Almagro', 'barrio_Belgrano', 'barrio_Caballito', 'barrio_Palermo', 'barrio_Recoleta', 'barrio_Villa Urquiza']
    #         self.barrios_conocidos_ = ['Almagro', 'Belgrano', 'Caballito', 'Palermo', 'Recoleta', 'Villa Urquiza']

    #     def predict(self, X_processed):
    #         # Simulación de predicción
    #         # Una predicción base + un factor por superficie
    #         base_price = 100000 
    #         price = base_price + (X_processed['superficie_total'] * 1500)
            
    #         # Ajuste por barrio (simplificado)
    #         if X_processed['barrio_Palermo'].iloc[0] == 1:
    #             price *= 1.5
    #         elif X_processed['barrio_Recoleta'].iloc[0] == 1:
    #             price *= 1.4
                
    #         return np.array([price.iloc[0]])

    #     def transform(self, X_raw):
    #         """Simula el preprocesamiento (One-Hot Encoding)"""
    #         X_processed = X_raw.copy()
    #         for barrio in self.barrios_conocidos_:
    #             col_name = f'barrio_{barrio}'
    #             X_processed[col_name] = (X_processed['barrio'] == barrio).astype(int)
            
    #         # Reordenar y seleccionar columnas para que coincida con el "entrenamiento"
    #         final_cols = [col for col in self.columnas_ if col in X_processed.columns]
    #         X_processed = X_processed[final_cols]
    #         return X_processed

    # pipeline = MockPipeline()
    
    # EN LA VIDA REAL:
    # pipeline = joblib.load('model/tu_pipeline_completo.pkl') 
    # O cargar modelo y preprocesador por separado:
    # model = joblib.load('model/tu_modelo_entrenado.pkl')
    # preprocessor = joblib.load('model/tu_preprocesador.pkl')
    
    # Por simplicidad para el ejemplo, devolvemos None
    # Deberás implementar la carga real
    
    # NOTA: Para que el ejemplo funcione sin errores, simularemos un modelo simple
    # que no necesita un pipeline complejo.
    class MockModel:
        def predict(self, df_input):
            # Lógica de predicción simulada simple
            precio = 100000 + (df_input['superficie_total'].values[0] * 1500) + (df_input['ambientes'].values[0] * 5000)
            if df_input['barrio'].values[0] == 'Palermo':
                precio *= 1.5
            elif df_input['barrio'].values[0] == 'Recoleta':
                precio *= 1.4
            return np.array([precio])

    model = MockModel()
    
    return model # Reemplaza por 'model' y 'preprocessor' o 'pipeline'

# --- CARGA DE DATOS ---

# Cargar datos reales del CSV
df = pd.read_csv('data/DatasetFinal.csv')
df = df.rename(columns={
    'surface_total': 'superficie_total',
    'precio': 'precio_usd',
    'baños': 'baños',
    'ambientes': 'ambientes',
    'barrio': 'barrio'
})
df['precio_m2_usd'] = df['precio_usd'] / df['superficie_total']
df = df.dropna(subset=['barrio', 'superficie_total', 'precio_usd', 'ambientes', 'baños'])
df['ambientes'] = df['ambientes'].astype(int)
df['baños'] = df['baños'].astype(int)

# Obtener coordenadas de todos los barrios únicos (CSV + geopy fallback, cacheado)
COORDENADAS_BARRIOS = get_barrios_coords_top10(df)

# Aquí deberías cargar tu modelo real
modelo = cargar_modelo_y_preprocesador() 

# --- TÍTULO PRINCIPAL ---
st.title("🏙️ Proyecto: Predicción de Precios de Departamentos en CABA")
st.markdown("Integración final de análisis y modelo predictivo.")

# --- PESTAÑAS DE NAVEGACIÓN ---
tab_inicio, tab_eda, tab_prediccion = st.tabs([
    "🏠 Inicio", 
    "📊 Análisis Exploratorio (EDA)", 
    "🤖 Predictor de Precios"
])

# --- PESTAÑA 1: INICIO ---
with tab_inicio:
    st.header("Bienvenido al Proyecto")
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
    
    col1, col2 = st.columns(2)

    with col1:
        # --- Gráfico 1: Histograma de Precios (Expresivo) ---
        st.subheader("1. Distribución de Precios (USD)")
        
        # FILTROS para Gráfico 1
        with st.expander("🔍 Filtros - Distribución de Precios", expanded=False):
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                precio_min_hist = st.number_input(
                    "Precio mínimo (USD)", 
                    min_value=int(df['precio_usd'].min()), 
                    max_value=int(df['precio_usd'].max()),
                    value=int(df['precio_usd'].min()),
                    step=10000,
                    key="precio_min_hist"
                )
            with col_f2:
                precio_max_hist = st.number_input(
                    "Precio máximo (USD)", 
                    min_value=int(df['precio_usd'].min()), 
                    max_value=int(df['precio_usd'].max()),
                    value=int(df['precio_usd'].max()),
                    step=10000,
                    key="precio_max_hist"
                )
            barrios_hist = st.multiselect(
                "Seleccionar barrios",
                options=sorted(df['barrio'].unique()),
                default=[],
                key="barrios_hist"
            )
            ambientes_hist = st.multiselect(
                "Cantidad de ambientes",
                options=sorted(df['ambientes'].unique()),
                default=[],
                key="ambientes_hist"
            )
        
        # Aplicar filtros
        df_filtered_hist = df[
            (df['precio_usd'] >= precio_min_hist) & 
            (df['precio_usd'] <= precio_max_hist)
        ]
        if barrios_hist:
            df_filtered_hist = df_filtered_hist[df_filtered_hist['barrio'].isin(barrios_hist)]
        if ambientes_hist:
            df_filtered_hist = df_filtered_hist[df_filtered_hist['ambientes'].isin(ambientes_hist)]
        
        st.caption(f"📊 Mostrando {len(df_filtered_hist)} de {len(df)} propiedades")
        
        chart_hist = alt.Chart(df_filtered_hist).mark_bar().encode(
            x=alt.X('precio_usd', bin=alt.Bin(maxbins=30), title='Precio (USD)'),
            y=alt.Y('count()', title='Cantidad de Propiedades'),
            tooltip=[alt.X('precio_usd', bin=alt.Bin(maxbins=30)), 'count()']
        ).properties(
            title='Distribución de los precios de las propiedades'
        ).interactive()
        st.altair_chart(chart_hist, use_container_width=True)

    with col2:
        # --- Gráfico 2: Precio Promedio por Barrio (Comparable) ---
        st.subheader("2. Precio Promedio por Barrio")
        
        # FILTROS para Gráfico 2
        with st.expander("🔍 Filtros - Precio por Barrio", expanded=False):
            col_f3, col_f4 = st.columns(2)
            with col_f3:
                superficie_min_bar = st.number_input(
                    "Superficie mínima (m²)", 
                    min_value=int(df['superficie_total'].min()), 
                    max_value=int(df['superficie_total'].max()),
                    value=int(df['superficie_total'].min()),
                    step=5,
                    key="superficie_min_bar"
                )
            with col_f4:
                superficie_max_bar = st.number_input(
                    "Superficie máxima (m²)", 
                    min_value=int(df['superficie_total'].min()), 
                    max_value=int(df['superficie_total'].max()),
                    value=int(df['superficie_total'].max()),
                    step=5,
                    key="superficie_max_bar"
                )
            ambientes_min_bar = st.select_slider(
                "Ambientes mínimos",
                options=sorted(df['ambientes'].unique()),
                value=sorted(df['ambientes'].unique())[0],
                key="ambientes_min_bar"
            )
            banos_min_bar = st.select_slider(
                "Baños mínimos",
                options=sorted(df['baños'].unique()),
                value=sorted(df['baños'].unique())[0],
                key="banos_min_bar"
            )
        
        # Aplicar filtros
        df_filtered_bar = df[
            (df['superficie_total'] >= superficie_min_bar) & 
            (df['superficie_total'] <= superficie_max_bar) &
            (df['ambientes'] >= ambientes_min_bar) &
            (df['baños'] >= banos_min_bar)
        ]
        
        st.caption(f"📊 Mostrando {len(df_filtered_bar)} de {len(df)} propiedades")
        
        chart_bar = alt.Chart(df_filtered_bar).mark_bar().encode(
            x=alt.X('barrio', sort='-y', title='Barrio'),
            y=alt.Y('mean(precio_usd)', title='Precio Promedio (USD)'),
            color=alt.Color('barrio', legend=None),
            tooltip=['barrio', alt.Tooltip('mean(precio_usd)', format=',.0f')]
        ).properties(
            title='Precio Promedio (USD) por Barrio'
        ).interactive()
        st.altair_chart(chart_bar, use_container_width=True)

    # --- Gráfico 3: Relación Precio vs. Superficie (Interactivo) ---
    st.subheader("3. Relación Precio vs. Superficie Total")
    st.write("Usa el mouse para hacer zoom y panear la visualización.")
    
    # FILTROS para Gráfico 3
    with st.expander("🔍 Filtros - Precio vs Superficie", expanded=False):
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
                min_value=int(df['precio_usd'].min()), 
                max_value=int(df['precio_usd'].max()),
                value=int(df['precio_usd'].min()),
                step=10000,
                key="precio_min_scatter"
            )
            precio_max_scatter = st.number_input(
                "Precio máximo (USD)", 
                min_value=int(df['precio_usd'].min()), 
                max_value=int(df['precio_usd'].max()),
                value=int(df['precio_usd'].max()),
                step=10000,
                key="precio_max_scatter"
            )
        
        with col_f7:
            superficie_min_scatter = st.number_input(
                "Superficie mínima (m²)", 
                min_value=int(df['superficie_total'].min()), 
                max_value=int(df['superficie_total'].max()),
                value=int(df['superficie_total'].min()),
                step=5,
                key="superficie_min_scatter"
            )
            superficie_max_scatter = st.number_input(
                "Superficie máxima (m²)", 
                min_value=int(df['superficie_total'].min()), 
                max_value=int(df['superficie_total'].max()),
                value=int(df['superficie_total'].max()),
                step=5,
                key="superficie_max_scatter"
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
                options=sorted(df['baños'].unique()),
                default=[],
                key="banos_scatter"
            )
    
    # Aplicar filtros
    df_filtered_scatter = df[
        (df['precio_usd'] >= precio_min_scatter) & 
        (df['precio_usd'] <= precio_max_scatter) &
        (df['superficie_total'] >= superficie_min_scatter) & 
        (df['superficie_total'] <= superficie_max_scatter)
    ]
    if barrios_scatter:
        df_filtered_scatter = df_filtered_scatter[df_filtered_scatter['barrio'].isin(barrios_scatter)]
    if ambientes_scatter:
        df_filtered_scatter = df_filtered_scatter[df_filtered_scatter['ambientes'].isin(ambientes_scatter)]
    if banos_scatter:
        df_filtered_scatter = df_filtered_scatter[df_filtered_scatter['baños'].isin(banos_scatter)]
    
    st.caption(f"📊 Mostrando {len(df_filtered_scatter)} de {len(df)} propiedades")
    
    chart_scatter = alt.Chart(df_filtered_scatter).mark_circle(opacity=0.7).encode(
        x=alt.X('superficie_total', title='Superficie Total (m²)'),
        y=alt.Y('precio_usd', title='Precio (USD)', scale=alt.Scale(zero=False)),
        color=alt.Color('barrio', title='Barrio'),
        tooltip=[
            'barrio',
            'superficie_total',
            'ambientes',
            alt.Tooltip('precio_usd', title='Precio (USD)', format=',.0f')
        ]
    ).properties(
        title='Precio vs. Superficie, coloreado por Barrio'
    ).interactive() # <-- La clave para que sea interactivo (zoom/pan)
    
    st.altair_chart(chart_scatter, use_container_width=True)

    # --- Gráfico 4: Mapa Interactivo de CABA ---
    st.subheader("4. 🗺️ Mapa Interactivo de Propiedades en CABA")
    st.write("Explora las propiedades en el mapa de Buenos Aires. Los marcadores están coloreados según el precio.")
    
    # FILTROS para Gráfico 4
    with st.expander("🔍 Filtros - Mapa de CABA", expanded=False):
        col_f10, col_f11 = st.columns(2)
        
        with col_f10:
            precio_min_map = st.number_input(
                "Precio mínimo (USD)", 
                min_value=int(df['precio_usd'].min()), 
                max_value=int(df['precio_usd'].max()),
                value=int(df['precio_usd'].min()),
                step=10000,
                key="precio_min_map"
            )
            precio_max_map = st.number_input(
                "Precio máximo (USD)", 
                min_value=int(df['precio_usd'].min()), 
                max_value=int(df['precio_usd'].max()),
                value=int(df['precio_usd'].max()),
                step=10000,
                key="precio_max_map"
            )
        
        with col_f11:
            superficie_min_map = st.number_input(
                "Superficie mínima (m²)", 
                min_value=int(df['superficie_total'].min()), 
                max_value=int(df['superficie_total'].max()),
                value=int(df['superficie_total'].min()),
                step=5,
                key="superficie_min_map"
            )
            superficie_max_map = st.number_input(
                "Superficie máxima (m²)", 
                min_value=int(df['superficie_total'].min()), 
                max_value=int(df['superficie_total'].max()),
                value=int(df['superficie_total'].max()),
                step=5,
                key="superficie_max_map"
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
        (df['precio_usd'] >= precio_min_map) & 
        (df['precio_usd'] <= precio_max_map) &
        (df['superficie_total'] >= superficie_min_map) & 
        (df['superficie_total'] <= superficie_max_map)
    ]
    if barrios_map:
        df_filtered_map = df_filtered_map[df_filtered_map['barrio'].isin(barrios_map)]
    if ambientes_map:
        df_filtered_map = df_filtered_map[df_filtered_map['ambientes'].isin(ambientes_map)]
    df_filtered_map = df_filtered_map.head(limite_propiedades)
    
    st.caption(f"📊 Mostrando {len(df_filtered_map)} de {len(df)} propiedades en el mapa")
    
    if len(df_filtered_map) == 0:
        st.info("Selecciona filtros para ver propiedades en el mapa.")
    else:
        mapa_caba = folium.Map(
            location=[-34.6037, -58.3816],
            zoom_start=12,
            tiles='OpenStreetMap'
        )
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
        barrios_sin_coord = set()
        barrios_fallback = set()
        for idx, row in df_filtered_map.iterrows():
            barrio = row['barrio']
            coords = COORDENADAS_BARRIOS.get(barrio)
            if coords and not pd.isnull(coords[0]) and not pd.isnull(coords[1]):
                lat, lon = coords
                lat += np.random.uniform(-0.005, 0.005)
                lon += np.random.uniform(-0.005, 0.005)
                popup_html = f"""
                <div style='font-family: Arial; font-size: 12px;'>
                    <b>🏘️ {barrio}</b><br>
                    <b>💰 Precio:</b> ${row['precio_usd']:,.0f} USD<br>
                    <b>📏 Superficie:</b> {row['superficie_total']} m²<br>
                    <b>🚪 Ambientes:</b> {row['ambientes']}<br>
                    <b>🚿 Baños:</b> {row['baños']}<br>
                    <b>📊 Precio/m²:</b> ${row['precio_m2_usd']:,.0f} USD
                </div>
                """
                folium.Marker(
                    location=[lat, lon],
                    popup=folium.Popup(popup_html, max_width=250),
                    tooltip=f"{barrio} - ${row['precio_usd']:,.0f}",
                    icon=folium.Icon(
                        color=get_color_by_price(row['precio_usd']),
                        icon='home',
                        prefix='fa'
                    )
                ).add_to(mapa_caba)
                propiedades_mostradas += 1
            else:
                barrios_sin_coord.add(barrio)

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
                    border-radius: 5px;">
            <p style="margin:0; font-weight:bold;">💰 Leyenda de Precios</p>
            <p style="margin:3px 0;"><i class="fa fa-circle" style="color:green"></i> &lt; $150,000</p>
            <p style="margin:3px 0;"><i class="fa fa-circle" style="color:blue"></i> $150,000 - $250,000</p>
            <p style="margin:3px 0;"><i class="fa fa-circle" style="color:orange"></i> $250,000 - $350,000</p>
            <p style="margin:3px 0;"><i class="fa fa-circle" style="color:red"></i> &gt; $350,000</p>
        </div>
        '''
        mapa_caba.get_root().html.add_child(folium.Element(legend_html))
        st_folium(mapa_caba, width=None, height=500)

    # --- Gráfico 5: Precio por m² vs Superficie (Línea) ---
    st.subheader("5. Precio por m² vs Superficie Total (Gráfico de Líneas)")
    st.write("Analiza cómo varía el precio por metro cuadrado según la superficie y cantidad de ambientes.")

    # FILTROS para Gráfico 5
    with st.expander("🔍 Filtros - Precio por m² vs Superficie", expanded=False):
        col_f14, col_f15 = st.columns(2)
        with col_f14:
            barrios_line = st.multiselect(
                "Seleccionar barrios",
                options=sorted(df['barrio'].unique()),
                default=[],
                key="barrios_line"
            )
            ambientes_line = st.multiselect(
                "Cantidad de ambientes",
                options=sorted(df['ambientes'].unique()),
                default=[],
                key="ambientes_line"
            )
        with col_f15:
            precio_m2_min = st.number_input(
                "Precio/m² mínimo (USD)", 
                min_value=int(df['precio_m2_usd'].min()), 
                max_value=int(df['precio_m2_usd'].max()),
                value=int(df['precio_m2_usd'].min()),
                step=100,
                key="precio_m2_min"
            )
            precio_m2_max = st.number_input(
                "Precio/m² máximo (USD)", 
                min_value=int(df['precio_m2_usd'].min()), 
                max_value=int(df['precio_m2_usd'].max()),
                value=int(df['precio_m2_usd'].max()),
                step=100,
                key="precio_m2_max"
            )
            banos_line = st.multiselect(
                "Cantidad de baños",
                options=sorted(df['baños'].unique()),
                default=sorted(df['baños'].unique()),
                key="banos_line"
            )

    # Aplicar filtros
    df_filtered_line = df.copy()
    if barrios_line:
        df_filtered_line = df_filtered_line[df_filtered_line['barrio'].isin(barrios_line)]
    if ambientes_line:
        df_filtered_line = df_filtered_line[df_filtered_line['ambientes'].isin(ambientes_line)]
    if banos_line:
        df_filtered_line = df_filtered_line[df_filtered_line['baños'].isin(banos_line)]
    df_filtered_line = df_filtered_line[
        (df_filtered_line['precio_m2_usd'] >= precio_m2_min) & 
        (df_filtered_line['precio_m2_usd'] <= precio_m2_max)
    ]

    st.caption(f"📊 Mostrando {len(df_filtered_line)} de {len(df)} propiedades")

    # Crear el gráfico de líneas con puntos
    chart_line = alt.Chart(df_filtered_line).mark_line(point=True).encode(
        x=alt.X('superficie_total:Q', title='Superficie Total (m²)', scale=alt.Scale(zero=False)),
        y=alt.Y('mean(precio_m2_usd):Q', title='Precio Promedio por m² (USD)'),
        color=alt.Color('ambientes:N', title='Cantidad de Ambientes'),
        tooltip=[
            alt.Tooltip('ambientes:N', title='Ambientes'),
            alt.Tooltip('mean(superficie_total)', title='Superficie Promedio', format='.1f'),
            alt.Tooltip('mean(precio_m2_usd)', title='Precio/m² Promedio', format=',.0f'),
            alt.Tooltip('count()', title='Cantidad de Propiedades')
        ]
    ).properties(
        title='Precio por m² vs Superficie, segmentado por Ambientes',
        height=400
    ).interactive()

    st.altair_chart(chart_line, use_container_width=True)


# --- PESTAÑA 3: PREDICTOR DE PRECIOS ---
with tab_prediccion:
    st.header("🤖 Prueba nuestro Modelo Predictivo")
    st.write("Ingresa las características del departamento para obtener una estimación de su precio.")

    col_form, col_result = st.columns([1, 1], gap="large")

    with col_form:
        st.subheader("Ingresa los datos de la propiedad:")
        
        # --- Widgets de Inputs ---
        
        # Opciones para los selectbox (basadas en los datos)
        barrios_opciones = sorted(df['barrio'].unique())
        
        with st.form(key="prediction_form"):
            # Input: Barrio (Selectbox)
            in_barrio = st.selectbox(
                "Barrio:",
                options=barrios_opciones,
                help="Selecciona el barrio donde se encuentra la propiedad."
            )
            
            # Input: Superficie (Number Input)
            in_superficie = st.number_input(
                "Superficie Total (en m²):",
                min_value=15,
                max_value=500,
                value=50,
                step=1,
                help="Ingresa la superficie total en metros cuadrados."
            )
            
            # Input: Ambientes (Slider)
            in_ambientes = st.slider(
                "Cantidad de Ambientes:",
                min_value=1,
                max_value=6,
                value=2,
                help="Selecciona la cantidad de ambientes (1 a 6+)."
            )
            
            # Input: Baños (Slider)
            in_baños = st.slider(
                "Cantidad de Baños:",
                min_value=1,
                max_value=4,
                value=1,
                help="Selecciona la cantidad de baños."
            )
            
            # TODO: Añade aquí más inputs si tu modelo los requiere
            # Ej: st.checkbox("Amenities (Piscina)")
            # Ej: st.checkbox("Balcón")
            
            # Botón de envío del formulario
            submit_button = st.form_submit_button(
                label="Calcular Precio Estimado",
                type="primary"
            )

    with col_result:
        st.subheader("Resultado de la Predicción:")
        
        if not submit_button:
            st.info("Ingresa los datos en el formulario de la izquierda y presiona 'Calcular'.")
            
        if submit_button and modelo:
            # --- Lógica de Predicción ---
            
            # 1. Crear un DataFrame con los inputs del usuario
            #    (Debe tener EXACTAMENTE los mismos nombres de columnas que usaste para entrenar)
            input_data = {
                'barrio': [in_barrio],
                'superficie_total': [in_superficie],
                'ambientes': [in_ambientes],
                'baños': [in_baños]
                # TODO: Añade aquí las otras variables
            }
            input_df = pd.DataFrame(input_data)
            
            st.write("Valores ingresados:")
            st.dataframe(input_df)

            try:
                # 2. Aplicar el preprocesamiento
                #    (Esto depende de cómo hayas guardado tu pipeline)
                
                # --- Opción A: Si tienes un pipeline completo ---
                # prediccion = pipeline.predict(input_df)
                
                # --- Opción B: Si tienes preprocesador y modelo separados ---
                # input_procesado = preprocesador.transform(input_df)
                # prediccion = modelo.predict(input_procesado)
                
                # --- Opción C: Para el EJEMPLO SIMULADO ---
                prediccion = modelo.predict(input_df)
                
                # 3. Mostrar el resultado
                precio_predicho = prediccion[0]
                
                st.success(f"**¡Predicción exitosa!**")
                st.metric(
                    label="Precio Estimado (USD)",
                    value=f"$ {precio_predicho:,.0f}",
                    help="Este es el precio estimado por el modelo de ML."
                )
                
            except Exception as e:
                st.error(f"Error al realizar la predicción: {e}")
                st.warning("""
                Asegúrate de que el modelo y el preprocesador se hayan cargado correctamente
                y que los datos de entrada coincidan con los datos de entrenamiento.
                """)

        elif submit_button and not modelo:
            st.error("Error: El modelo no se ha cargado. Revisa el código `app.py`.")