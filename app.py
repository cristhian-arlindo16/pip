import streamlit as st
import pandas as pd
import folium
import networkx as nx
import random
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
from geopy.distance import geodesic

# Configurar la página y estilos CSS
st.set_page_config(page_title="Optimizador de Rutas 🚀", layout="wide")
st.markdown(
    """
    <style>
        body {
            background-color: #f5f5f5;
            font-family: 'Arial', sans-serif;
        }
        .stButton button {
            background-color: #FF5733;
            color: white;
            border-radius: 10px;
            font-size: 18px;
            width: 100%;
        }
        .stTextInput>div>div>input {
            font-size: 18px;
        }
        .big-font {
            font-size:20px !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Inicializar geolocalizador
geolocator = Nominatim(user_agent="route_optimizer", timeout=5)

def obtener_coordenadas(direccion):
    try:
        location = geolocator.geocode(direccion, timeout=5)
        return (location.latitude, location.longitude) if location else None
    except:
        return None

# Panel lateral de configuración
st.sidebar.header("📌 Configuración")
opcion_ingreso = st.sidebar.radio("Cómo ingresar ubicaciones:", ["Manual", "Desde CSV"])

if opcion_ingreso == "Manual":
    ubicaciones = st.sidebar.text_area("Ingrese ubicaciones separadas por comas:", "Lima, Cusco, Arequipa")
    ubicaciones = [x.strip() for x in ubicaciones.split(",")]
elif opcion_ingreso == "Desde CSV":
    archivo = st.sidebar.file_uploader("Cargar archivo CSV", type=["csv"])
    if archivo is not None:
        df = pd.read_csv(archivo)
        ubicaciones = df.iloc[:, 0].tolist()
    else:
        ubicaciones = []

if st.sidebar.button("📍 Obtener Coordenadas"):
    coordenadas = {ubi: obtener_coordenadas(ubi) for ubi in ubicaciones if obtener_coordenadas(ubi)}
else:
    coordenadas = {}

if coordenadas:
    st.sidebar.write("🔍 Coordenadas obtenidas:")
    for lugar, coord in coordenadas.items():
        st.sidebar.write(f"{lugar}: {coord}")

# Crear grafo de distancias
def calcular_distancia(nodo1, nodo2):
    return geodesic(nodo1, nodo2).kilometers

def construir_grafo(ciudades):
    G = nx.Graph()
    for i, ciudad1 in enumerate(ciudades):
        for j, ciudad2 in enumerate(ciudades):
            if i != j:
                distancia = calcular_distancia(ciudades[ciudad1], ciudades[ciudad2])
                G.add_edge(ciudad1, ciudad2, weight=distancia)
    return G

if coordenadas:
    G = construir_grafo(coordenadas)

# Algoritmo Genético con DEAP
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("indices", random.sample, list(coordenadas.keys()), len(coordenadas))
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evaluar(individual):
    distancia_total = 0
    for i in range(len(individual) - 1):
        distancia_total += calcular_distancia(coordenadas[individual[i]], coordenadas[individual[i + 1]])
    return (distancia_total,)

toolbox.register("mate", tools.cxOrdered)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluar)

# Ejecutar optimización
if st.sidebar.button("🚀 Optimizar Ruta"):
    pop = toolbox.population(n=50)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)

    pop, log = tools.eaMuPlusLambda(pop, toolbox, mu=50, lambda_=100, cxpb=0.7, mutpb=0.2, ngen=100, stats=stats, halloffame=hof, verbose=True)
    
    mejor_ruta = hof[0]
    mejor_distancia = evaluar(mejor_ruta)[0]
    tiempo_estimado = mejor_distancia / 60  # Aproximando que se viaja a 60 km/h

    # Mostrar resultados
    st.success(f"✅ ¡Ruta Optimizada! Distancia Total: {mejor_distancia:.2f} km")
    st.info(f"🕒 Tiempo Estimado de Viaje: {tiempo_estimado:.2f} horas")

    # Visualización en mapa
    mapa = folium.Map(location=coordenadas[mejor_ruta[0]], zoom_start=6)
    for i in range(len(mejor_ruta) - 1):
        folium.Marker(location=coordenadas[mejor_ruta[i]], popup=mejor_ruta[i]).add_to(mapa)
        folium.PolyLine([coordenadas[mejor_ruta[i]], coordenadas[mejor_ruta[i + 1]]], color="blue", weight=2.5).add_to(mapa)

    st_folium(mapa, width=800, height=500)

    # Mostrar evolución del algoritmo
    generacion = range(len(log))
    fitness_min = [log[i]["min"] for i in generacion]
    plt.figure(figsize=(8, 4))
    plt.plot(generacion, fitness_min, marker="o", linestyle="-", color="red", label="Mejor Fitness")
    plt.xlabel("Generación")
    plt.ylabel("Distancia (km)")
    plt.title("Evolución del Algoritmo Genético")
    plt.legend()
    st.pyplot(plt)

# Botón para reiniciar la aplicación
if st.sidebar.button("🔄 Reiniciar"):
   st.rerun()

