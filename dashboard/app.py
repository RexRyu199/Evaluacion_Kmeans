import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Segmentacion musical con K-Means", layout="wide")

VARIABLES = ["danceability", "energy", "loudness", "speechiness", "acousticness", "valence", "tempo"]

COLORES = {
    "Bailables y alegres": "#2ca02c",
    "Intensas y oscuras": "#1f77b4",
    "Acusticas y tranquilas": "#e6b800",
    "Rap y voces habladas": "#9467bd",
}

DESCRIPCION = {
    "Bailables y alegres": "Canciones con alta bailabilidad (0.74), energia alta (0.74) y la valencia mas alta (0.69), es decir, las de sonido mas positivo. Es el grupo mas grande (35.7%) y el principal de la musica latina (58.5% de las canciones latinas).",
    "Intensas y oscuras": "Canciones con la energia (0.81), el volumen (-5.4 dB) y el tempo (131 BPM) mas altos, pero con baja valencia (0.35) y casi nada de acustica (0.06). Dominan en EDM (60.4%) y rock (50.5%).",
    "Acusticas y tranquilas": "Canciones con baja energia (0.43), poco volumen (-10.5 dB) y la mayor acustica (0.49). Es el grupo mas suave y concentra una tercera parte del R&B (33.2% de las canciones de ese genero).",
    "Rap y voces habladas": "Canciones con la speechiness (palabra hablada) mas alta (0.31, cuatro veces la de los demas grupos) y buena bailabilidad (0.72). Mas de la mitad de este grupo (54.5%) son canciones de rap.",
}

HALLAZGOS = [
    "El genero de la playlist no determina el sonido: todos los clusteres contienen canciones de los seis generos (por ejemplo, 31.8% del cluster acustico es R&B, pero tambien hay 17.6% de rock).",
    "La palabra hablada (speechiness) es la caracteristica que mas separa a un grupo: el cluster Rap y voces habladas tiene 0.31, mientras que los demas tienen entre 0.07 y 0.08.",
    "EDM y rock comparten perfil sonoro: 60.4% del EDM y 50.5% del rock estan en el cluster Intensas y oscuras.",
    "El cluster Intensas y oscuras es el menos popular (37.8 de popularidad promedio contra 41.3 a 42.0 de los demas).",
]

RECOMENDACIONES = [
    "Crear playlists por estado de animo (fiesta, intensidad, relajacion, rap) en lugar de solo por genero, usando los clusteres como base.",
    "Usar el cluster como criterio en un sistema de recomendacion, para sugerir canciones de sonido similar aunque sean de otro genero (por ejemplo, rock a quien escucha EDM).",
    "Revisar la promocion de las canciones del cluster Intensas y oscuras, que tiene la menor popularidad promedio, y evaluar si se pueden complementar con canciones mas alegres para llegar a mas oyentes.",
]


@st.cache_data
def cargar():
    return pd.read_csv("data/spotify_procesado.csv")


df = cargar()

st.sidebar.title("Filtros")
pagina = st.sidebar.radio("Pagina", ["1. Analisis general", "2. Segmentacion K-Means"])

generos = st.sidebar.multiselect("Genero", sorted(df["playlist_genre"].unique()), default=sorted(df["playlist_genre"].unique()))
anios = st.sidebar.slider("Anio de lanzamiento", int(df["anio"].min()), int(df["anio"].max()), (int(df["anio"].min()), int(df["anio"].max())))
popularidad = st.sidebar.slider("Popularidad", 0, 100, (0, 100))

filtro = (
    df["playlist_genre"].isin(generos)
    & df["anio"].between(anios[0], anios[1])
    & df["track_popularity"].between(popularidad[0], popularidad[1])
)

if pagina.startswith("2"):
    clusteres = st.sidebar.multiselect("Cluster", sorted(df["cluster_nombre"].unique()), default=sorted(df["cluster_nombre"].unique()))
    filtro = filtro & df["cluster_nombre"].isin(clusteres)

datos = df[filtro]

st.title("Segmentacion de canciones de Spotify con K-Means")

if datos.empty:
    st.warning("Con los filtros seleccionados no hay canciones.")
    st.stop()

if pagina.startswith("1"):
    st.header("Pagina 1. Analisis general")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Canciones", f"{len(datos):,}")
    c2.metric("Artistas distintos", f"{datos['track_artist'].nunique():,}")
    c3.metric("Popularidad promedio", f"{datos['track_popularity'].mean():.1f}")
    c4.metric("Duracion promedio (min)", f"{datos['duracion_min'].mean():.2f}")

    izq, der = st.columns(2)
    por_genero = datos["playlist_genre"].value_counts().reset_index()
    por_genero.columns = ["Genero", "Canciones"]
    izq.plotly_chart(px.bar(por_genero, x="Genero", y="Canciones", title="Canciones por genero", color_discrete_sequence=["steelblue"]), width="stretch")

    variable = der.selectbox("Variable para la distribucion", VARIABLES, index=1)
    der.plotly_chart(px.histogram(datos, x=variable, nbins=40, title=f"Distribucion de {variable}", color_discrete_sequence=["steelblue"]), width="stretch")

    izq, der = st.columns(2)
    corr = datos[VARIABLES].corr().round(2)
    izq.plotly_chart(px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="Correlacion entre caracteristicas"), width="stretch")
    por_anio = datos.groupby("anio")[["energy", "danceability", "valence"]].mean().reset_index()
    der.plotly_chart(px.line(por_anio, x="anio", y=["energy", "danceability", "valence"], title="Promedio de caracteristicas por anio"), width="stretch")

    st.subheader("Estadisticas descriptivas")
    st.dataframe(datos[VARIABLES + ["track_popularity", "duracion_min"]].describe().T.round(3), width="stretch")

else:
    st.header("Pagina 2. Segmentacion mediante K-Means")

    por_cluster = datos["cluster_nombre"].value_counts()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Clusteres en la seleccion", datos["cluster_nombre"].nunique())
    c2.metric("Cluster mas grande", f"{por_cluster.iloc[0] / len(datos) * 100:.1f}%", por_cluster.index[0], delta_color="off")
    pop = datos.groupby("cluster_nombre")["track_popularity"].mean()
    c3.metric("Cluster mas popular", f"{pop.max():.1f}", pop.idxmax(), delta_color="off")
    c4.metric("Canciones seleccionadas", f"{len(datos):,}")

    izq, der = st.columns(2)
    dist = por_cluster.reset_index()
    dist.columns = ["Cluster", "Canciones"]
    izq.plotly_chart(px.bar(dist, x="Cluster", y="Canciones", color="Cluster", color_discrete_map=COLORES, title="Canciones por cluster"), width="stretch")
    muestra = datos.sample(min(len(datos), 8000), random_state=42)
    der.plotly_chart(px.scatter(muestra, x="pca1", y="pca2", color="cluster_nombre", color_discrete_map=COLORES, hover_data=["track_name", "track_artist"], opacity=0.6, title="Agrupamientos (proyeccion PCA)"), width="stretch")

    medias = datos.groupby("cluster_nombre")[VARIABLES].mean()
    normalizadas = (medias - df[VARIABLES].min()) / (df[VARIABLES].max() - df[VARIABLES].min())
    radar = go.Figure()
    for nombre, fila in normalizadas.iterrows():
        radar.add_trace(go.Scatterpolar(r=list(fila.values) + [fila.values[0]], theta=VARIABLES + [VARIABLES[0]], name=nombre, fill="toself", opacity=0.5, line_color=COLORES[nombre]))
    radar.update_layout(title="Perfil de cada cluster (valores entre 0 y 1)", margin=dict(l=90, r=90))

    izq, der = st.columns(2)
    izq.plotly_chart(radar, width="stretch")
    variable = der.selectbox("Variable para comparar los grupos", VARIABLES, index=1)
    der.plotly_chart(px.box(datos, x="cluster_nombre", y=variable, color="cluster_nombre", color_discrete_map=COLORES, title=f"{variable} por cluster"), width="stretch")

    st.subheader("Estadisticas descriptivas por cluster")
    tabla = datos.groupby("cluster_nombre")[VARIABLES + ["track_popularity"]].agg(["mean", "std"]).round(3)
    tabla.columns = [f"{a} ({b})" for a, b in tabla.columns]
    tabla.insert(0, "canciones", por_cluster)
    st.dataframe(tabla, width="stretch")

    cruce = pd.crosstab(datos["cluster_nombre"], datos["playlist_genre"], normalize="index").mul(100).round(1)
    st.plotly_chart(px.imshow(cruce, text_auto=True, color_continuous_scale="Blues", title="Porcentaje de cada genero dentro de cada cluster"), width="stretch")

    st.subheader("Descripcion de los grupos")
    for nombre, texto in DESCRIPCION.items():
        st.markdown(f"**{nombre}:** {texto}")

    izq, der = st.columns(2)
    izq.subheader("Principales hallazgos")
    for h in HALLAZGOS:
        izq.markdown(f"- {h}")
    der.subheader("Recomendaciones")
    for r in RECOMENDACIONES:
        der.markdown(f"- {r}")
