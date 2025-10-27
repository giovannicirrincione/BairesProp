import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
# Descomenta la siguiente línea si guardaste tu modelo con joblib
# import joblib 

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="Predicción de Precios de Deptos. en CABA",
    page_icon="🏙️",
    layout="wide",
)

# --- DATOS Y MODELO (SIMULADOS) ---
# Esta sección simula la carga de tus datos y tu modelo.
# DEBERÁS REEMPLAZAR ESTO CON LA CARGA DE TUS ARCHIVOS REALES.

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
df = cargar_datos()
# Aquí deberías cargar tu modelo real
modelo = cargar_modelo_y_preprocesador() 
# ej: modelo, preprocesador = cargar_modelo_y_preprocesador()

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
    st.image("https://placehold.co/1200x400/333/FFF?text=Foto+Skyline+CABA", use_column_width=True)
    
    st.subheader("Objetivo del Proyecto")
    st.write("""
    El objetivo de este trabajo es analizar el mercado inmobiliario de la Ciudad Autónoma de Buenos Aires (CABA)
    y desarrollar un modelo de Machine Learning capaz de predecir el precio de venta (en USD) de un departamento
    basado en sus características principales, como la ubicación, superficie, y cantidad de ambientes.
    """)
    
    st.subheader("Integrantes del Grupo")
    st.markdown("""
    * Nombre Alumno 1
    * Nombre Alumno 2
    * Nombre Alumno 3
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
        chart_hist = alt.Chart(df).mark_bar().encode(
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
        chart_bar = alt.Chart(df).mark_bar().encode(
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
    
    chart_scatter = alt.Chart(df).mark_circle(opacity=0.7).encode(
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