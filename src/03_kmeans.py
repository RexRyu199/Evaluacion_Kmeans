import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

df = pd.read_csv("data/spotify_limpio.csv")

variables = ["danceability", "energy", "loudness", "speechiness",
             "acousticness", "valence", "tempo"]

escalador = StandardScaler()
x = escalador.fit_transform(df[variables])

print("EVALUACION DE DIFERENTES VALORES DE K")
resultados = []
for k in range(2, 9):
    modelo = KMeans(n_clusters=k, random_state=42, n_init=10)
    grupos = modelo.fit_predict(x)
    resultados.append({
        "K": k,
        "Inercia": round(modelo.inertia_, 1),
        "Silhouette": round(silhouette_score(x, grupos, sample_size=10000, random_state=42), 4),
        "Davies_Bouldin": round(davies_bouldin_score(x, grupos), 4)
    })
resultados = pd.DataFrame(resultados)
print(resultados)
resultados.to_csv("outputs/evaluacion_k.csv", index=False)

fig, ejes = plt.subplots(1, 3, figsize=(15, 4.5))
ejes[0].plot(resultados["K"], resultados["Inercia"], marker="o", color="steelblue")
ejes[0].set_title("Metodo del codo (inercia)")
ejes[1].plot(resultados["K"], resultados["Silhouette"], marker="o", color="green")
ejes[1].set_title("Coeficiente de silueta (mayor es mejor)")
ejes[2].plot(resultados["K"], resultados["Davies_Bouldin"], marker="o", color="darkorange")
ejes[2].set_title("Indice Davies-Bouldin (menor es mejor)")
for eje in ejes:
    eje.set_xlabel("Numero de clusteres (K)")
    eje.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("images/06_evaluacion_k.png", dpi=110)
plt.close()

K_FINAL = 4
modelo = KMeans(n_clusters=K_FINAL, random_state=42, n_init=10)

df["cluster"] = modelo.fit_predict(x)

print("\nMODELO FINAL (K=4)")
print("Inercia:", round(modelo.inertia_, 1))
print("Iteraciones:", modelo.n_iter_)

print("\nCENTROIDES (VALORES ESCALADOS)")
centroides_escalados = pd.DataFrame(modelo.cluster_centers_, columns=variables).round(3)
print(centroides_escalados)

print("\nCENTROIDES (VALORES ORIGINALES)")
centroides = pd.DataFrame(escalador.inverse_transform(modelo.cluster_centers_), columns=variables).round(3)
print(centroides)
centroides.to_csv("outputs/centroides.csv", index_label="cluster")

print("\nCANCIONES POR CLUSTER")
print(df["cluster"].value_counts().sort_index())

pca = PCA(n_components=2, random_state=42)
componentes = pca.fit_transform(x)
df["pca1"] = componentes[:, 0]
df["pca2"] = componentes[:, 1]
print("\nVARIANZA EXPLICADA POR PCA")
print(pca.explained_variance_ratio_.round(3))

plt.figure(figsize=(10, 7))
scatter = plt.scatter(df["pca1"], df["pca2"], c=df["cluster"], cmap="viridis", s=6, alpha=0.5)
centros_pca = pca.transform(modelo.cluster_centers_)
plt.scatter(centros_pca[:, 0], centros_pca[:, 1], color="red", marker="x", s=250, linewidths=3, label="Centroides")
plt.legend(*scatter.legend_elements(), title="Cluster", loc="upper right")
plt.title("Clusteres de canciones (proyeccion PCA)")
plt.xlabel("Componente principal 1")
plt.ylabel("Componente principal 2")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("images/07_clusters_pca.png", dpi=110)
plt.close()

print("\nCOMPARACION DE DOS CONFIGURACIONES (K=4 VS K=2)")
modelo2 = KMeans(n_clusters=2, random_state=42, n_init=10)
grupos2 = modelo2.fit_predict(x)
comparacion = pd.DataFrame({
    "Configuracion": ["K=4", "K=2"],
    "Inercia": [round(modelo.inertia_, 1), round(modelo2.inertia_, 1)],
    "Silhouette": [round(silhouette_score(x, df["cluster"], sample_size=10000, random_state=42), 4),
                   round(silhouette_score(x, grupos2, sample_size=10000, random_state=42), 4)],
    "Davies_Bouldin": [round(davies_bouldin_score(x, df["cluster"]), 4),
                       round(davies_bouldin_score(x, grupos2), 4)]
})
print(comparacion)
print("\nCanciones por cluster con K=2:")
print(pd.Series(grupos2).value_counts().sort_index())
print("\nCentroides con K=2 (valores originales):")
print(pd.DataFrame(escalador.inverse_transform(modelo2.cluster_centers_), columns=variables).round(3))
print("\nAdjusted Rand Index entre K=4 y K=2:", round(adjusted_rand_score(df["cluster"], grupos2), 4))
comparacion.to_csv("outputs/comparacion_configuraciones.csv", index=False)

df.to_csv("data/spotify_con_clusters.csv", index=False)
