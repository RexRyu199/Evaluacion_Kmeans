# Segmentacion de canciones de Spotify con K-Means

Proyecto integrador de mineria de datos: se agrupan 26,217 canciones segun sus caracteristicas de audio (bailabilidad, energia, volumen, palabra hablada, acustica, valencia y tempo) usando el algoritmo no supervisado **K-Means**, y los resultados se presentan en un dashboard interactivo hecho con **Streamlit**.

## Dataset

- **Fuente:** [TidyTuesday 2020-01-21, Spotify Songs](https://github.com/rfordatascience/tidytuesday/tree/master/data/2020/2020-01-21) (datos obtenidos con la API de Spotify).
- **Original:** 32,833 canciones y 23 variables. **Despues de la limpieza:** 26,217 canciones.

## Herramientas

Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Plotly y Streamlit.

## Como ejecutar

Se necesita Python 3.10 o superior. Todos los comandos se ejecutan desde la carpeta raiz del repositorio.

```bash
pip install -r requirements.txt

```

## Como ver el dashboard

```bash
streamlit run dashboard/app.py
```

Se abre en `http://localhost:8501`. Tiene dos paginas (se cambia en la barra lateral):

1. **Analisis general:** indicadores, canciones por genero, distribuciones, correlacion y evolucion por anio.
2. **Segmentacion K-Means:** distribucion por cluster, grafica de dispersion (PCA), perfil y comparacion de grupos, tabla de estadisticas, descripcion de los grupos, hallazgos y recomendaciones.

Filtros disponibles: genero, anio de lanzamiento, popularidad y (en la pagina 2) cluster.


