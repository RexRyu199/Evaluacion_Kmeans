import pandas as pd
from sklearn.preprocessing import StandardScaler

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

df = pd.read_csv("data/spotify_songs.csv")

print("REGISTROS INICIALES")
print(df.shape)

print("\nNULOS ANTES")
print(df.isnull().sum()[df.isnull().sum() > 0])
df = df.dropna(subset=["track_name", "track_artist", "track_album_name"])

print("\nCANCIONES REPETIDAS POR TRACK_ID")
print(df["track_id"].duplicated().sum())
df = df.sort_values("track_popularity", ascending=False)
df = df.drop_duplicates(subset="track_id")

print("\nCANCIONES REPETIDAS POR NOMBRE Y ARTISTA")
print(df.duplicated(subset=["track_name", "track_artist"]).sum())
df = df.drop_duplicates(subset=["track_name", "track_artist"])

print("\nREGISTROS INVALIDOS")
print("Tempo igual a 0:", (df["tempo"] == 0).sum())
print("Duracion menor a 30 segundos:", (df["duration_ms"] < 30000).sum())
print("Danceability igual a 0:", (df["danceability"] == 0).sum())
print("Loudness menor a -25 dB (casi silencio):", (df["loudness"] < -25).sum())
df = df[(df["tempo"] > 0) & (df["duration_ms"] >= 30000) & (df["danceability"] > 0) & (df["loudness"] >= -25)]

print("\nTIPOS DE DATOS ANTES")
print(df[["track_album_release_date", "duration_ms"]].dtypes)
df["track_album_release_date"] = pd.to_datetime(df["track_album_release_date"], format="mixed")
df["anio"] = df["track_album_release_date"].dt.year
df["duracion_min"] = (df["duration_ms"] / 60000).round(2)
print("\nTIPOS DE DATOS DESPUES")
print(df[["track_album_release_date", "anio", "duracion_min"]].dtypes)

variables = ["danceability", "energy", "loudness", "speechiness",
             "acousticness", "valence", "tempo"]

print("\nVALORES ATIPICOS POR VARIABLE (IQR)")
for col in variables:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    atipicos = ((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum()
    print(col, ":", atipicos)

print("\nCODIFICACION DEL GENERO")
df["genero_codigo"] = df["playlist_genre"].astype("category").cat.codes
print(df[["playlist_genre", "genero_codigo"]].drop_duplicates().sort_values("genero_codigo"))

print("\nESTADISTICAS ANTES DEL ESCALAMIENTO")
print(df[variables].describe().T[["mean", "std", "min", "max"]])

escalador = StandardScaler()
escalados = escalador.fit_transform(df[variables])
df_escalado = pd.DataFrame(escalados, columns=variables, index=df.index)

print("\nESTADISTICAS DESPUES DEL ESCALAMIENTO")
print(df_escalado.describe().T[["mean", "std", "min", "max"]].round(3))

print("\nDATASET LIMPIO")
print(df.shape)
print(df.isnull().sum().sum(), "nulos")

df = df.reset_index(drop=True)
df.to_csv("data/spotify_limpio.csv", index=False)
