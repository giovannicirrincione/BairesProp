import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
import joblib
import folium
from streamlit_folium import st_folium
import requests
from geopy.geocoders import Nominatim
import urllib3

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
        return None, None# --- CARGA INICIAL DE DATOS Y MODELO ---
df = cargar_datos()
# Ahora cargamos ambos objetos
modelo, label_encoder = cargar_modelo_y_preprocesador() 

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
        border-color: #4CAF50;
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
tab_inicio, tab_eda, tab_prediccion = st.tabs([
    " Inicio", 
    " Análisis Exploratorio (EDA)", 
    " ¿Cuánto vale mi Dpto?"
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
            # Usar requests directamente para evitar problemas de SSL
            base_url = "https://nominatim.openstreetmap.org/search"
            params = {
                'q': direccion,
                'format': 'json',
                'limit': 1
            }
            headers = {
                'User-Agent': 'BairesProp/1.0'
            }
            
            response = requests.get(base_url, params=params, headers=headers, verify=False)
            data = response.json()
            
            if data and len(data) > 0:
                result = data[0]
                lat = float(result['lat'])
                lng = float(result['lon'])
                formatted_addr = result['display_name']
                return lat, lng, formatted_addr
            else:
                st.error("No se pudo geocodificar la dirección. Intenta con otra dirección.")
                return None
        except Exception as e:
            st.error(f"Error en geocodificación: {e}")
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
            placeholder="Ingrese dirección (Ej: Av. del Libertador 500)",
            help="Ingresa la dirección del departamento en CABA"
        )
        
        # API Key de Google Maps (proporcionada por el usuario)
        # Si la API Key no funciona, el sistema usará Nominatim automáticamente
        GOOGLE_MAPS_API_KEY = "AIzaSyDIXRaiTX03X4qZuK1d_9xNfa1iWDgXg3Q"
        
        if st.button("Buscar Dirección", type="primary", use_container_width=True):
            if direccion_input:
                with st.spinner("Geocodificando dirección..."):
                    # Intentar primero con Google Maps
                    direccion_completa = f"{direccion_input}, Buenos Aires, Argentina"
                    result = geocodificar_direccion_google(direccion_completa, GOOGLE_MAPS_API_KEY)
                    
                    # Si Google falla, intentar con Nominatim
                    if not result:
                        st.info("Intentando con servicio alternativo (OpenStreetMap)...")
                        result = geocodificar_direccion_nominatim(direccion_completa)
                    
                    if result:
                        lat, lng, formatted_addr = result
                        
                        # Actualizar coordenadas
                        st.session_state.lat = lat
                        st.session_state.lng = lng
                        
                        # Detectar barrio, zona y comuna
                        barrio, zona, comuna = detectar_barrio_y_zona(lat, lng)
                        st.session_state.barrio_detectado = barrio
                        st.session_state.zona_detectada = zona
                        st.session_state.comuna_detectada = comuna
                        
                        st.success(f"Ubicación encontrada: {formatted_addr}")
                        st.info(f"**Barrio detectado:** {barrio}")
                        st.info(f"**Zona:** {zona}")
                        st.info(f"**Comuna:** {comuna}")
                        
                        st.rerun()  # Recargar para actualizar el mapa
                    else:
                        st.error("No se pudo encontrar la dirección. Intenta con otro formato o usa el mapa para seleccionar la ubicación.")
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
        
        # Mostrar información de ubicación actual
        if st.session_state.barrio_detectado:
            st.success(f"Barrio: {st.session_state.barrio_detectado}")
            st.success(f"Zona: {st.session_state.zona_detectada}")
            st.success(f"Comuna: {st.session_state.comuna_detectada}")
            st.caption(f"Coordenadas: ({st.session_state.lat:.4f}, {st.session_state.lng:.4f})")
            st.caption(f"Coordenadas: ({st.session_state.lat:.4f}, {st.session_state.lng:.4f})")
    
    with col_caracteristicas:
        st.subheader("Características del Departamento")
        
        with st.form(key="prediction_form"):
            st.markdown("**Ingrese los datos de la propiedad:**")
            
            # Input: Cantidad de baños
            in_baños = st.number_input(
                "Cantidad de Baños:",
                min_value=1,
                max_value=10,
                value=None,
                step=1,
                help="Cantidad de baños completos en el departamento",
                placeholder="Ingrese la cantidad de Baños"
            )
            
            # Input: Cantidad de habitaciones
            in_habitaciones = st.number_input(
                "Cantidad de Habitaciones:",
                min_value=0,
                max_value=10,
                value=None,
                step=1,
                help="Cantidad de dormitorios/habitaciones",
                placeholder="Ingrese la cantidad de Habitaciones"
            )
            
            # Input: Cantidad de ambientes
            in_ambientes = st.number_input(
                "Cantidad de Ambientes:",
                min_value=1,
                max_value=10,
                value=None,
                step=1,
                help="Cantidad total de ambientes (incluye habitaciones, living, comedor, etc.)",
                placeholder="Ingrese la cantidad de Ambientes"
            )
            
            # Input: Superficie total con formato m2
            st.markdown("**Superficie Total:**")
            col_m2_total, col_input_total = st.columns([0.15, 0.85], gap="small")
            with col_m2_total:
                st.markdown("""
                    <div style='
                        background-color: #f0f2f6;
                        padding: 8px 12px;
                        border-radius: 4px 0 0 4px;
                        border: 1px solid #d4d4d4;
                        border-right: none;
                        height: 38px;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        font-weight: 500;
                        margin-top: 0px;
                    '>m²</div>
                """, unsafe_allow_html=True)
            with col_input_total:
                in_superficie_total = st.number_input(
                    "Superficie Total",
                    min_value=15.0,
                    max_value=500.0,
                    value=None,
                    step=1.0,
                    help="Superficie total del departamento en metros cuadrados",
                    placeholder="Ingrese la Superficie Total",
                    label_visibility="collapsed"
                )
            
            # Input: Superficie cubierta con formato m2
            st.markdown("**Superficie Cubierta:**")
            col_m2_cubierta, col_input_cubierta = st.columns([0.15, 0.85], gap="small")
            with col_m2_cubierta:
                st.markdown("""
                    <div style='
                        background-color: #f0f2f6;
                        padding: 8px 12px;
                        border-radius: 4px 0 0 4px;
                        border: 1px solid #d4d4d4;
                        border-right: none;
                        height: 38px;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        font-weight: 500;
                        margin-top: 0px;
                    '>m²</div>
                """, unsafe_allow_html=True)
            with col_input_cubierta:
                in_superficie_cubierta = st.number_input(
                    "Superficie Cubierta",
                    min_value=15.0,
                    max_value=500.0,
                    value=None,
                    step=1.0,
                    help="Superficie cubierta del departamento en metros cuadrados",
                    placeholder="Ingrese la Superficie Cubierta",
                    label_visibility="collapsed"
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
            if in_baños is None or in_habitaciones is None or in_ambientes is None or in_superficie_total is None or in_superficie_cubierta is None:
                st.error("Por favor, completa todos los campos del formulario.")
            # Verificar que se haya seleccionado una ubicación
            elif not st.session_state.barrio_detectado:
                st.warning("Por favor, selecciona una ubicación en el mapa o ingresa una dirección.")
            elif not esta_en_caba(st.session_state.lat, st.session_state.lng):
                st.error("No se puede realizar la predicción.")
                st.warning("La ubicación seleccionada está fuera de los límites de la Ciudad Autónoma de Buenos Aires. El modelo solo funciona para propiedades dentro de CABA.")
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
                    'comuna': st.session_state.comuna_detectada,
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
                    'Comuna': st.session_state.comuna_detectada,
                    'Superficie Total': f"{in_superficie_total} m²",
                    'Superficie Cubierta': f"{in_superficie_cubierta} m²",
                    'Ambientes': in_ambientes,
                    'Habitaciones': in_habitaciones,
                    'Baños': in_baños
                }
                st.dataframe(pd.DataFrame([main_features]), use_container_width=True)

                # Mostrar loader mientras se genera la predicción
                with st.spinner('Analizando variables...'):
                    try:
                        # 2. Aplicar el preprocesamiento y la predicción
                        #    El pipeline se encarga de todo
                        prediccion_numerica = modelo.predict(input_df)
                        
                        # 3. Usar el LabelEncoder para decodificar la predicción
                        prediccion_etiqueta = label_encoder.inverse_transform(prediccion_numerica)
                    
                        # 4. Mostrar el resultado
                        st.success(f"¡Predicción exitosa!")
                        
                        st.markdown("### Rango de Precio Estimado:")
                        st.markdown(f"# **{prediccion_etiqueta[0]}** USD")
                        
                        st.info("""
                        Esta etiqueta representa el rango de precios más probable 
                        para una propiedad con las características ingresadas, 
                        según nuestro modelo de clasificación.
                        """)
                        
                        # Mostrar resumen de la propiedad
                        with st.expander("Ver Resumen de la Propiedad"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Barrio", st.session_state.barrio_detectado)
                                st.metric("Zona", st.session_state.zona_detectada)
                                st.metric("Comuna", st.session_state.comuna_detectada)
                                st.metric("Ambientes", in_ambientes)
                                st.metric("Habitaciones", in_habitaciones)
                            with col2:
                                st.metric("Baños", in_baños)
                                st.metric("Sup. Total", f"{in_superficie_total} m²")
                                st.metric("Sup. Cubierta", f"{in_superficie_cubierta} m²")
                                st.metric("Precio/m²", f"~ USD {int(np.random.randint(2000, 4000))}/m²")
                    
                    except Exception as e:
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

