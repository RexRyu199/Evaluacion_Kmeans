import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)

df = pd.read_csv("data/spotify_con_clusters.csv")

nombres = {
    0: "Rap y voces habladas",
    1: "Intensas y oscuras",
    2: "Bailables y alegres",
    3: "Acusticas y tranquilas"
}
df["cluster_nombre"] = df["cluster"].map(nombres)

variables = ["danceability", "energy", "loudness", "speechiness",
             "acousticness", "valence", "tempo"]

print("ESTADISTICAS DESCRIPTIVAS POR CLUSTER (MEDIA)")
medias = df.groupby("cluster_nombre")[variables + ["track_popularity", "duracion_min"]].mean().round(3)
print(medias)

print("\nESTADISTICAS DESCRIPTIVAS POR CLUSTER (MEDIANA)")
print(df.groupby("cluster_nombre")[variables].median().round(3))

print("\nDESVIACION ESTANDAR POR CLUSTER")
print(df.groupby("cluster_nombre")[variables].std().round(3))

print("\nTABLA COMPARATIVA DE LOS GRUPOS")
tabla = medias.copy()
tabla.insert(0, "canciones", df["cluster_nombre"].value_counts())
tabla.insert(1, "porcentaje", (df["cluster_nombre"].value_counts(normalize=True) * 100).round(1))
print(tabla)
tabla.to_csv("outputs/tabla_comparativa.csv")

print("\nGENERO PRINCIPAL DE CADA CLUSTER (% DEL CLUSTER)")
generos = pd.crosstab(df["cluster_nombre"], df["playlist_genre"], normalize="index").mul(100).round(1)
print(generos)

print("\nCLUSTER PRINCIPAL DE CADA GENERO (% DEL GENERO)")
print(pd.crosstab(df["playlist_genre"], df["cluster_nombre"], normalize="index").mul(100).round(1))

print("\nPOPULARIDAD POR CLUSTER")
print(df.groupby("cluster_nombre")["track_popularity"].agg(["mean", "median"]).round(2))

print("\nANIO PROMEDIO POR CLUSTER")
print(df.groupby("cluster_nombre")["anio"].mean().round(1))

zmedias = (medias[variables] - medias[variables].mean()) / medias[variables].std()
plt.figure(figsize=(10, 5))
sns.heatmap(zmedias, annot=medias[variables], fmt=".2f", cmap="coolwarm", center=0)
plt.title("Perfil de cada cluster (color: diferencia respecto al promedio de los grupos)")
plt.xlabel("Caracteristica")
plt.ylabel("Cluster")
plt.tight_layout()
plt.savefig("images/08_perfil_clusters.png", dpi=110)
plt.close()

plt.figure(figsize=(9, 5))
sns.barplot(x=df["cluster_nombre"].value_counts().index, y=df["cluster_nombre"].value_counts().values, color="steelblue", edgecolor="black")
plt.title("Canciones por cluster")
plt.xlabel("Cluster")
plt.ylabel("Canciones")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("images/09_distribucion_clusters.png", dpi=110)
plt.close()

fig, ejes = plt.subplots(2, 2, figsize=(13, 8))
for eje, col in zip(ejes.flatten(), ["energy", "danceability", "acousticness", "speechiness"]):
    sns.boxplot(data=df, x="cluster_nombre", y=col, ax=eje, color="lightblue")
    eje.set_xlabel("")
    eje.tick_params(axis="x", rotation=15)
    eje.set_title(col)
plt.suptitle("Comparacion de caracteristicas por cluster")
plt.tight_layout()
plt.savefig("images/10_boxplots_clusters.png", dpi=110)
plt.close()

plt.figure(figsize=(10, 6))
sns.heatmap(pd.crosstab(df["cluster_nombre"], df["playlist_genre"], normalize="index").mul(100), annot=True, fmt=".1f", cmap="Blues")
plt.title("Porcentaje de cada genero dentro de cada cluster")
plt.xlabel("Genero de la playlist")
plt.ylabel("Cluster")
plt.tight_layout()
plt.savefig("images/11_generos_por_cluster.png", dpi=110)
plt.close()

plt.figure(figsize=(10, 7))
colores = {"Rap y voces habladas": "purple", "Intensas y oscuras": "steelblue", "Bailables y alegres": "green", "Acusticas y tranquilas": "gold"}
for nombre, grupo in df.groupby("cluster_nombre"):
    plt.scatter(grupo["pca1"], grupo["pca2"], s=6, alpha=0.4, color=colores[nombre], label=nombre)
centros = df.groupby("cluster_nombre")[["pca1", "pca2"]].mean()
plt.scatter(centros["pca1"], centros["pca2"], color="red", marker="x", s=250, linewidths=3, label="Centroides")
plt.legend(markerscale=3, loc="upper right")
plt.title("Clusteres de canciones (proyeccion PCA)")
plt.xlabel("Componente principal 1")
plt.ylabel("Componente principal 2")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("images/07_clusters_pca.png", dpi=110)
plt.close()

df.to_csv("data/spotify_procesado.csv", index=False)
