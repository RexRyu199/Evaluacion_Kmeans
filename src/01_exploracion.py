import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

df = pd.read_csv("data/spotify_songs.csv")

print("PRIMEROS 10 REGISTROS")
print(df.head(10))

print("\nDIMENSIONES")
print(df.shape)

print("\nCOLUMNAS")
print(df.columns)

print("\nTIPOS DE DATOS")
print(df.dtypes)

print("\nESTADISTICAS DESCRIPTIVAS")
print(df.describe().T)

print("\nVALORES NULOS")
print(df.isnull().sum())

print("\nREGISTROS DUPLICADOS (FILA COMPLETA)")
print(df.duplicated().sum())

print("\nCANCIONES REPETIDAS (TRACK_ID)")
print(df["track_id"].duplicated().sum())

print("\nCANCIONES POR GENERO")
print(df["playlist_genre"].value_counts())

variables = ["danceability", "energy", "loudness", "speechiness",
             "acousticness", "instrumentalness", "liveness", "valence", "tempo"]

df[variables].hist(bins=30, figsize=(14, 9), color="steelblue", edgecolor="black")
plt.suptitle("Distribucion de las caracteristicas de audio")
plt.tight_layout()
plt.savefig("images/01_histogramas.png", dpi=110)
plt.close()

plt.figure(figsize=(10, 8))
sns.heatmap(df[variables].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlacion entre caracteristicas de audio")
plt.tight_layout()
plt.savefig("images/02_correlacion.png", dpi=110)
plt.close()

plt.figure(figsize=(10, 6))
sns.countplot(data=df, x="playlist_genre", order=df["playlist_genre"].value_counts().index, color="steelblue", edgecolor="black")
plt.title("Cantidad de canciones por genero")
plt.xlabel("Genero")
plt.ylabel("Canciones")
plt.tight_layout()
plt.savefig("images/03_generos.png", dpi=110)
plt.close()

plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x="playlist_genre", y="energy", order=df["playlist_genre"].value_counts().index, color="lightblue")
plt.title("Energia por genero")
plt.xlabel("Genero")
plt.ylabel("Energia")
plt.tight_layout()
plt.savefig("images/04_energia_genero.png", dpi=110)
plt.close()

plt.figure(figsize=(10, 6))
plt.scatter(df["energy"], df["loudness"], alpha=0.15, s=8, color="steelblue")
plt.title("Energia vs volumen (loudness)")
plt.xlabel("Energia")
plt.ylabel("Loudness (dB)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("images/05_energia_loudness.png", dpi=110)
plt.close()
