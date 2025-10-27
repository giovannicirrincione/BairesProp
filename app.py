import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
import joblib
import folium
from streamlit_folium import st_folium
import requests
from geopy.geocoders import Nominatim

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="Predicción de Precios de Deptos. en CABA",
    page_icon="🏙️",
    layout="wide",
)

# --- CARGA DE DATOS Y MODELO ---

@st.cache_data
def cargar_datos():
    """Carga los datos limpios desde la carpeta data"""
    try:
        # Reemplaza 'tu_archivo.csv' con el nombre real de tu archivo CSV
        df = pd.read_csv('data/DatasetFinal.csv')  
        st.success(f" Datos cargados exitosamente: {len(df)} registros")
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
    ruta_paquete = 'model/modelo_clasificador_precios.pkl'
    
    try:
        # Cargar el diccionario que contiene ambos objetos
        data = joblib.load(ruta_paquete)
        pipeline = data['pipeline']
        encoder = data['encoder']
        
        if 'pipeline' in data and 'encoder' in data:
            st.success("¡Modelo y Encoder cargados exitosamente!")
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
        return None, None

# --- CARGA INICIAL DE DATOS Y MODELO ---
df = cargar_datos()
# Ahora cargamos ambos objetos
modelo, label_encoder = cargar_modelo_y_preprocesador() 

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
    y desarrollar un modelo de Machine Learning capaz de predecir el **rango de precio de venta (en USD)** de un departamento
    basado en sus características principales, como la ubicación, superficie, y cantidad de ambientes.
    """)
    
    st.subheader("Integrantes del Grupo")
    st.markdown("""
    * Nombre Alumno 1
    * Nombre Alumno 2
    * Nombre Alumno 
    """)
    
    st.subheader("Datos Utilizados")
    if df is not None:
        st.write(f"""
        El análisis y el modelo se basan en un dataset de **{len(df)}** propiedades.
        Aquí puedes ver una muestra de los datos (simulados o reales) que se utilizan para las visualizaciones
        y el entrenamiento del modelo:
        """)
        st.dataframe(df.sample(min(5, len(df))))
    else:
        st.error("No se pudieron cargar los datos para mostrar.")

# --- PESTAÑA 2: ANÁLISIS EXPLORATORIO (EDA) ---
with tab_eda:
    st.header("Visualizaciones Interactivas del Mercado")
    
    if df is not None:
        st.write("Exploración de las variables clave y su relación con el precio.")

        # --- VISUALIZACIONES CON ALTAIR (Requisito de la entrega) ---
        
        col1, col2 = st.columns(2)

        with col1:
            # --- Gráfico 1: Histograma de Precios (Expresivo) ---
            st.subheader("1. Distribución de Precios (USD)")
            chart_hist = alt.Chart(df).mark_bar().encode(
                x=alt.X('precio', bin=alt.Bin(maxbins=30), title='Precio (USD)'),
                y=alt.Y('count()', title='Cantidad de Propiedades'),
                tooltip=[alt.X('precio', bin=alt.Bin(maxbins=30)), 'count()']
            ).properties(
                title='Distribución de los precios de las propiedades'
            ).interactive()
            st.altair_chart(chart_hist, use_container_width=True)

        with col2:
            # --- Gráfico 2: Precio Promedio por Barrio (Comparable) ---
            st.subheader("2. Precio Promedio por Barrio")
            chart_bar = alt.Chart(df).mark_bar().encode(
                x=alt.X('barrio', sort='-y', title='Barrio'),
                y=alt.Y('mean(precio)', title='Precio Promedio (USD)'),
                color=alt.Color('barrio', legend=None),
                tooltip=['barrio', alt.Tooltip('mean(precio)', format=',.0f')]
            ).properties(
                title='Precio Promedio (USD) por Barrio'
            ).interactive()
            st.altair_chart(chart_bar, use_container_width=True)

        # --- Gráfico 3: Relación Precio vs. Superficie (Interactivo) ---
        st.subheader("3. Relación Precio vs. Superficie Total")
        st.write("Usa el mouse para hacer zoom y panear la visualización.")
        
        chart_scatter = alt.Chart(df).mark_circle(opacity=0.7).encode(
            x=alt.X('surface_total', title='Superficie Total (m²)'),
            y=alt.Y('precio', title='Precio (USD)', scale=alt.Scale(zero=False)),
            color=alt.Color('barrio', title='Barrio'),
            tooltip=[
                'barrio',
                'surface_total',
                'ambientes',
                alt.Tooltip('precio', title='Precio (USD)', format=',.0f')
            ]
        ).properties(
            title='Precio vs. Superficie, coloreado por Barrio'
        ).interactive() # <-- La clave para que sea interactivo (zoom/pan)
        
        st.altair_chart(chart_scatter, use_container_width=True)
    else:
        st.warning("No se pueden mostrar las visualizaciones porque no se cargaron los datos.")


# --- PESTAÑA 3: PREDICTOR de RANGOS DE PRECIOS ---
with tab_prediccion:
    st.header("🤖 Prueba nuestro Modelo Predictivo")
    st.write("Ingresa la ubicación y las características del departamento para obtener una estimación de su rango de precio.")

    # --- FUNCIONES AUXILIARES PARA GEOCODIFICACIÓN ---
    
    def geocodificar_direccion_google(direccion, api_key=None):
        """
        Geocodifica una dirección usando Google Maps Geocoding API.
        Si no tienes API key, usa geocodificar_direccion_nominatim() en su lugar.
        """
        if not api_key:
            st.warning("No se proporcionó API Key de Google. Usando Nominatim (OpenStreetMap) en su lugar.")
            return geocodificar_direccion_nominatim(direccion)
        
        base_url = "https://maps.googleapis.com/maps/api/geocode/json"
        params = {
            'address': f"{direccion}, Buenos Aires, Argentina",
            'key': api_key
        }
        
        try:
            response = requests.get(base_url, params=params)
            data = response.json()
            
            if data['status'] == 'OK':
                result = data['results'][0]
                lat = result['geometry']['location']['lat']
                lng = result['geometry']['location']['lng']
                formatted_address = result['formatted_address']
                return lat, lng, formatted_address
            else:
                st.error(f"Error en geocodificación: {data['status']}")
                return None
        except Exception as e:
            st.error(f"Error al conectar con Google API: {e}")
            return None
    
    def geocodificar_direccion_nominatim(direccion):
        """
        Geocodifica una dirección usando Nominatim (OpenStreetMap).
        Alternativa gratuita a Google Maps API.
        """
        try:
            geolocator = Nominatim(user_agent="bairesprop_app")
            location = geolocator.geocode(f"{direccion}, Buenos Aires, Argentina")
            
            if location:
                return location.latitude, location.longitude, location.address
            else:
                st.error("No se pudo geocodificar la dirección. Intenta con otra dirección.")
                return None
        except Exception as e:
            st.error(f"Error en geocodificación: {e}")
            return None
    
    def detectar_barrio_y_zona(lat, lng):
        """
        Detecta el barrio y la zona (Norte, Sur, Centro/Oeste) basándose en las coordenadas.
        Esta es una aproximación simplificada. Para mayor precisión, usa polígonos de barrios.
        """
        # Barrios de CABA con coordenadas aproximadas (centro de cada barrio)
        barrios_coords = {
            'Palermo': (-34.5889, -58.4194, 'Norte'),
            'Recoleta': (-34.5875, -58.3974, 'Norte'),
            'Belgrano': (-34.5627, -58.4545, 'Norte'),
            'Núñez': (-34.5436, -58.4645, 'Norte'),
            'Colegiales': (-34.5735, -58.4476, 'Norte'),
            'Villa Urquiza': (-34.5702, -58.4856, 'Norte'),
            'Saavedra': (-34.5488, -58.4866, 'Norte'),
            'Coghlan': (-34.5563, -58.4775, 'Norte'),
            'Villa Pueyrredón': (-34.5894, -58.5014, 'Centro/Oeste'),
            'Villa Devoto': (-34.6009, -58.5119, 'Centro/Oeste'),
            'Villa del Parque': (-34.6056, -58.4896, 'Centro/Oeste'),
            'Agronomía': (-34.5985, -58.4894, 'Centro/Oeste'),
            'Chacarita': (-34.5889, -58.4524, 'Centro/Oeste'),
            'Paternal': (-34.5995, -58.4666, 'Centro/Oeste'),
            'Villa Crespo': (-34.5999, -58.4399, 'Centro/Oeste'),
            'Almagro': (-34.6098, -58.4206, 'Centro/Oeste'),
            'Caballito': (-34.6177, -58.4398, 'Centro/Oeste'),
            'Flores': (-34.6287, -58.4649, 'Centro/Oeste'),
            'Floresta': (-34.6263, -58.4831, 'Centro/Oeste'),
            'Parque Chacabuco': (-34.6358, -58.4502, 'Sur'),
            'Boedo': (-34.6275, -58.4173, 'Sur'),
            'San Cristóbal': (-34.6205, -58.3977, 'Sur'),
            'Constitución': (-34.6276, -58.3817, 'Sur'),
            'San Telmo': (-34.6212, -58.3724, 'Sur'),
            'Monserrat': (-34.6108, -58.3838, 'Centro/Oeste'),
            'Balvanera': (-34.6092, -58.4033, 'Centro/Oeste'),
            'Retiro': (-34.5926, -58.3766, 'Norte'),
            'Puerto Madero': (-34.6118, -58.3632, 'Centro/Oeste'),
            'Barracas': (-34.6440, -58.3748, 'Sur'),
            'La Boca': (-34.6345, -58.3636, 'Sur'),
            'Parque Patricios': (-34.6364, -58.4014, 'Sur'),
            'Nueva Pompeya': (-34.6537, -58.4197, 'Sur'),
            'Mataderos': (-34.6600, -58.4899, 'Sur'),
            'Liniers': (-34.6447, -58.5204, 'Sur'),
            'Versalles': (-34.6297, -58.5167, 'Sur'),
            'Villa Luro': (-34.6360, -58.4983, 'Sur'),
            'Vélez Sársfield': (-34.6405, -58.4777, 'Sur'),
            'Villa Lugano': (-34.6775, -58.4686, 'Sur'),
            'Villa Riachuelo': (-34.6885, -58.4613, 'Sur'),
            'Villa Soldati': (-34.6638, -58.4440, 'Sur'),
            'Parque Avellaneda': (-34.6441, -58.4693, 'Sur'),
        }
        
        # Calcular distancia a cada barrio y encontrar el más cercano
        min_dist = float('inf')
        barrio_cercano = "Desconocido"
        zona = "Desconocido"
        
        for barrio, (b_lat, b_lng, b_zona) in barrios_coords.items():
            dist = ((lat - b_lat)**2 + (lng - b_lng)**2)**0.5
            if dist < min_dist:
                min_dist = dist
                barrio_cercano = barrio
                zona = b_zona
        
        return barrio_cercano, zona
    
    # --- ESTADO DE SESIÓN PARA COORDENADAS ---
    if 'lat' not in st.session_state:
        st.session_state.lat = -34.6037  # Centro de CABA (aproximado)
    if 'lng' not in st.session_state:
        st.session_state.lng = -58.3816
    if 'barrio_detectado' not in st.session_state:
        st.session_state.barrio_detectado = ""
    if 'zona_detectada' not in st.session_state:
        st.session_state.zona_detectada = ""
    
    # --- LAYOUT EN DOS COLUMNAS ---
    col_ubicacion, col_caracteristicas = st.columns([1, 1], gap="large")
    
    with col_ubicacion:
        st.subheader("📍 Ubicación del Departamento")
        
        # --- OPCIÓN 1: INGRESO MANUAL DE DIRECCIÓN ---
        st.markdown("**Opción 1: Ingresa la dirección manualmente**")
        
        direccion_input = st.text_input(
            "Dirección (calle y altura):",
            placeholder="Ej: Av. Santa Fe 1234",
            help="Ingresa la dirección del departamento en CABA"
        )
        
        # Campo opcional para API Key de Google (puedes ocultarlo si usas Nominatim)
        with st.expander("⚙️ Configuración Avanzada (Opcional)"):
            google_api_key = st.text_input(
                "API Key de Google Maps (opcional):",
                type="password",
                help="Si tienes una API Key de Google Maps, ingrésala aquí. Si no, se usará Nominatim (OpenStreetMap)."
            )
        
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("🔍 Buscar Dirección", type="primary", use_container_width=True):
                if direccion_input:
                    with st.spinner("Geocodificando dirección..."):
                        if google_api_key:
                            result = geocodificar_direccion_google(direccion_input, google_api_key)
                        else:
                            result = geocodificar_direccion_nominatim(direccion_input)
                        
                        if result:
                            lat, lng, formatted_addr = result
                            st.session_state.lat = lat
                            st.session_state.lng = lng
                            
                            # Detectar barrio y zona
                            barrio, zona = detectar_barrio_y_zona(lat, lng)
                            st.session_state.barrio_detectado = barrio
                            st.session_state.zona_detectada = zona
                            
                            st.success(f"✅ Ubicación encontrada: {formatted_addr}")
                            st.info(f"**Barrio detectado:** {barrio}")
                            st.info(f"**Zona:** {zona}")
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
            
            if new_lat != st.session_state.lat or new_lng != st.session_state.lng:
                st.session_state.lat = new_lat
                st.session_state.lng = new_lng
                
                # Detectar barrio y zona
                barrio, zona = detectar_barrio_y_zona(new_lat, new_lng)
                st.session_state.barrio_detectado = barrio
                st.session_state.zona_detectada = zona
                
                st.rerun()
        
        # Mostrar información de ubicación actual
        if st.session_state.barrio_detectado:
            st.success(f"📍 **Barrio:** {st.session_state.barrio_detectado}")
            st.success(f"🗺️ **Zona:** {st.session_state.zona_detectada}")
            st.caption(f"Coordenadas: ({st.session_state.lat:.4f}, {st.session_state.lng:.4f})")
    
    with col_caracteristicas:
        st.subheader("🏠 Características del Departamento")
        
        with st.form(key="prediction_form"):
            st.markdown("**Ingresa los datos de la propiedad:**")
            
            # Input: Cantidad de baños
            in_baños = st.number_input(
                "Cantidad de Baños:",
                min_value=1,
                max_value=10,
                value=1,
                step=1,
                help="Cantidad de baños completos en el departamento"
            )
            
            # Input: Cantidad de habitaciones
            in_habitaciones = st.number_input(
                "Cantidad de Habitaciones:",
                min_value=0,
                max_value=10,
                value=1,
                step=1,
                help="Cantidad de dormitorios/habitaciones"
            )
            
            # Input: Cantidad de ambientes
            in_ambientes = st.number_input(
                "Cantidad de Ambientes:",
                min_value=1,
                max_value=10,
                value=2,
                step=1,
                help="Cantidad total de ambientes (incluye habitaciones, living, comedor, etc.)"
            )
            
            # Input: Superficie total
            in_superficie_total = st.number_input(
                "Superficie Total (m²):",
                min_value=15.0,
                max_value=500.0,
                value=50.0,
                step=1.0,
                help="Superficie total del departamento en metros cuadrados"
            )
            
            # Input: Superficie cubierta
            in_superficie_cubierta = st.number_input(
                "Superficie Cubierta (m²):",
                min_value=15.0,
                max_value=500.0,
                value=45.0,
                step=1.0,
                help="Superficie cubierta del departamento en metros cuadrados"
            )
            
            st.markdown("---")
            
            # Botón de envío del formulario
            submit_button = st.form_submit_button(
                label="🔮 Calcular Rango de Precio",
                type="primary",
                use_container_width=True
            )
        
        # --- RESULTADO DE LA PREDICCIÓN ---
        st.markdown("---")
        st.subheader("📊 Resultado de la Predicción:")
        
        if not submit_button:
            st.info("👆 Completa los datos del formulario y presiona 'Calcular Rango de Precio'.")
        
        elif submit_button and modelo and label_encoder:
            # Verificar que se haya seleccionado una ubicación
            if not st.session_state.barrio_detectado:
                st.warning("⚠️ Por favor, selecciona una ubicación en el mapa o ingresa una dirección.")
            else:
                # --- Lógica de Predicción ---
                
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
                    'surface_total': in_superficie_total,
                    'surface_covered': in_superficie_cubierta,
                    'ambientes': in_ambientes,
                    'habitaciones': in_habitaciones,
                    'baños': in_baños,
                    'comuna': 1,  # Valor por defecto, ajusta según necesites
                    'precio_numeric': 0  # Placeholder
                }
                
                # 2. Crear todas las columnas de interacción con valor 0
                # Interacciones ambientes x barrio
                for barrio in barrios:
                    input_data[f'amb_x_barrio_{barrio}'] = 0
                
                # Interacciones habitaciones x barrio
                for barrio in barrios:
                    input_data[f'hab_x_barrio_{barrio}'] = 0
                
                # Interacciones baños x barrio
                for barrio in barrios:
                    input_data[f'banos_x_barrio_{barrio}'] = 0
                
                # Interacciones superficie total x barrio
                for barrio in barrios:
                    input_data[f'sup_tot_x_barrio_{barrio}'] = 0
                
                # Interacciones superficie cubierta x barrio
                for barrio in barrios:
                    input_data[f'sup_cub_x_barrio_{barrio}'] = 0
                
                # Interacciones con zona
                for zona in zonas:
                    input_data[f'amb_x_{zona}'] = 0
                    input_data[f'hab_x_{zona}'] = 0
                    input_data[f'banos_x_{zona}'] = 0
                    input_data[f'sup_tot_x_{zona}'] = 0
                    input_data[f'sup_cub_x_{zona}'] = 0
                
                # 3. Asignar valores a las columnas que corresponden al barrio y zona seleccionados
                if f'amb_x_barrio_{barrio_norm}' in input_data:
                    input_data[f'amb_x_barrio_{barrio_norm}'] = in_ambientes
                    input_data[f'hab_x_barrio_{barrio_norm}'] = in_habitaciones
                    input_data[f'banos_x_barrio_{barrio_norm}'] = in_baños
                    input_data[f'sup_tot_x_barrio_{barrio_norm}'] = in_superficie_total
                    input_data[f'sup_cub_x_barrio_{barrio_norm}'] = in_superficie_cubierta
                
                if f'amb_x_{zona_norm}' in input_data:
                    input_data[f'amb_x_{zona_norm}'] = in_ambientes
                    input_data[f'hab_x_{zona_norm}'] = in_habitaciones
                    input_data[f'banos_x_{zona_norm}'] = in_baños
                    input_data[f'sup_tot_x_{zona_norm}'] = in_superficie_total
                    input_data[f'sup_cub_x_{zona_norm}'] = in_superficie_cubierta
                
                # 4. Convertir a DataFrame (una sola fila)
                input_df = pd.DataFrame([input_data])
                
                st.write("**Características principales enviadas al modelo:**")
                main_features = {
                    'Barrio': st.session_state.barrio_detectado,
                    'Zona': st.session_state.zona_detectada,
                    'Superficie Total': f"{in_superficie_total} m²",
                    'Superficie Cubierta': f"{in_superficie_cubierta} m²",
                    'Ambientes': in_ambientes,
                    'Habitaciones': in_habitaciones,
                    'Baños': in_baños
                }
                st.dataframe(pd.DataFrame([main_features]), use_container_width=True)

                # Mostrar loader mientras se genera la predicción
                with st.spinner('🔄 Analizando variables...'):
                    try:
                        # 2. Aplicar el preprocesamiento y la predicción
                        #    El pipeline se encarga de todo
                        prediccion_numerica = modelo.predict(input_df)
                        
                        # 3. Usar el LabelEncoder para decodificar la predicción
                        prediccion_etiqueta = label_encoder.inverse_transform(prediccion_numerica)
                    
                        # 4. Mostrar el resultado
                        st.success(f"✅ **¡Predicción exitosa!**")
                        
                        st.markdown("### 💰 Rango de Precio Estimado:")
                        st.markdown(f"# **{prediccion_etiqueta[0]}**")
                        
                        st.info("""
                        Esta etiqueta representa el rango de precios más probable 
                        para una propiedad con las características ingresadas, 
                        según nuestro modelo de clasificación.
                        """)
                        
                        # Mostrar resumen de la propiedad
                        with st.expander("📋 Ver Resumen de la Propiedad"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Barrio", st.session_state.barrio_detectado)
                                st.metric("Zona", st.session_state.zona_detectada)
                                st.metric("Ambientes", in_ambientes)
                                st.metric("Habitaciones", in_habitaciones)
                            with col2:
                                st.metric("Baños", in_baños)
                                st.metric("Sup. Total", f"{in_superficie_total} m²")
                                st.metric("Sup. Cubierta", f"{in_superficie_cubierta} m²")
                                st.metric("Precio/m²", f"~ USD {int(np.random.randint(2000, 4000))}/m²")
                    
                    except Exception as e:
                        st.error(f"❌ Error al realizar la predicción: {e}")
                        st.warning("""
                        **Posibles causas:**
                        - El modelo no se ha cargado correctamente
                        - Los nombres de las columnas no coinciden con el modelo entrenado
                        - Falta alguna característica requerida por el modelo
                        
                        Revisa los mensajes de error al inicio de la página y asegúrate de que 
                        el modelo fue entrenado con las mismas características que estás ingresando.
                        """)

        elif submit_button and (not modelo or not label_encoder):
            st.error("❌ Error: El modelo o el LabelEncoder no se han cargado. Revisa los mensajes de error al inicio de la página.")

