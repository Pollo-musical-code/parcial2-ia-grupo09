"""
parcial2.py - Segundo Parcial de Inteligencia Artificial (Grupo 09)
Tema: Educación - estudiantes de Cartago. Objetivo: predecir deserción (1 = desertó, 0 = continúa).
Entradas: asistencia_pct y promedio_notas.

Tareas que cubre este script:
  2. Carga e inspección de datos
  3. Calidad y limpieza con Pandas
  4. Visualización con Seaborn
  5. Perceptrón desde cero con NumPy (sin Scikit-Learn)
  6. Números para la interpretación

Ejecución dentro del contenedor:  cd src && python parcial2.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # sin ventanas: funciona dentro de Docker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

RAIZ = Path(__file__).resolve().parent.parent
DIR_DATOS = RAIZ / "data"
CSV_ORIGINAL = DIR_DATOS / "grupo_09.csv"
CSV_LIMPIO = DIR_DATOS / "grupo_09_limpio.csv"

COL_X = ["asistencia_pct", "promedio_notas"]
COL_Y = "desercion"

TASA_APRENDIZAJE = 0.1
EPOCAS = 500

sns.set_theme(style="whitegrid", context="talk")


def titulo(texto):
    print("\n" + "=" * 70)
    print(texto)
    print("=" * 70)


# ============================================================
# TAREA 2. Carga e inspección
# ============================================================
def cargar_e_inspeccionar():
    titulo("TAREA 2. CARGA E INSPECCIÓN DE DATOS")
    df = pd.read_csv(CSV_ORIGINAL)

    print("\nPrimeras 5 filas:")
    print(df.head())
    print(f"\nForma (shape): {df.shape}")
    print("\nTipos de datos:")
    print(df.dtypes)
    print("\nNulos por columna:")
    print(df.isnull().sum())
    print(f"\nFilas duplicadas: {df.duplicated().sum()}")

    print("\nPREGUNTA CLAVE: ¿alguna columna numérica tiene un tipo que no debería?")
    for col in COL_X:
        if not pd.api.types.is_numeric_dtype(df[col]):
            convertida = pd.to_numeric(df[col], errors="coerce")
            raros = df.loc[convertida.isna() & df[col].notna(), ["estudiante_id", col]]
            print(f"  '{col}' es de tipo {df[col].dtype} (texto) y debería ser numérica.")
            print(f"  Valores que lo causan ({len(raros)}):")
            print(raros.to_string(index=False))
    return df


# ============================================================
# TAREA 3. Calidad y limpieza
# ============================================================
def limpiar(df):
    titulo("TAREA 3. CALIDAD Y LIMPIEZA DE DATOS")
    filas_antes = len(df)
    print(f"Filas antes de la limpieza: {filas_antes}")
    print(f"Clases antes (0 / 1): {df[COL_Y].value_counts().sort_index().to_dict()}")
    d = df.copy()

    # --- Problema 1: texto dentro de una columna numérica ('79%') ---
    print("\n[Problema 1] Texto con '%' dentro de asistencia_pct")
    con_pct = d[d["asistencia_pct"].astype(str).str.contains("%")]
    print("  Detección: d[d['asistencia_pct'].str.contains('%')]")
    print(con_pct.to_string(index=False))
    d["asistencia_pct"] = pd.to_numeric(d["asistencia_pct"].astype(str).str.replace("%", "", regex=False).str.strip())
    print(f"  Decisión: quitar el '%' y convertir a número. Tipo ahora: {d['asistencia_pct'].dtype}")

    # --- Problema 2: filas duplicadas ---
    print("\n[Problema 2] Filas duplicadas")
    dup = d[d.duplicated(keep=False)].sort_values("estudiante_id")
    print("  Detección: d[d.duplicated(keep=False)]")
    print(dup.to_string(index=False))
    n_dup = d.duplicated().sum()
    d = d.drop_duplicates()
    print(f"  Decisión: eliminar {n_dup} filas repetidas (se conserva la primera aparición).")

    # --- Problema 3: valor imposible en promedio_notas (escala 0 a 5) ---
    print("\n[Problema 3] Valor imposible en promedio_notas (la escala es 0 a 5)")
    imposibles = d[(d["promedio_notas"] > 5) | (d["promedio_notas"] < 0)]
    print("  Detección: d[(d['promedio_notas'] > 5) | (d['promedio_notas'] < 0)]")
    print(imposibles.to_string(index=False))
    sin = d.drop(imposibles.index)
    print(f"  Media de promedio_notas ANTES de corregir:   {d['promedio_notas'].mean():.4f}  (desviación {d['promedio_notas'].std(ddof=0):.4f})")
    print(f"  Media de promedio_notas DESPUÉS de corregir: {sin['promedio_notas'].mean():.4f}  (desviación {sin['promedio_notas'].std(ddof=0):.4f})")
    valor = imposibles["promedio_notas"].iloc[0]
    z_antes = (valor - d["promedio_notas"].mean()) / d["promedio_notas"].std(ddof=0)
    print(f"  Con el valor sucio, su puntaje estandarizado sería {z_antes:.2f} desviaciones.")
    print("  Decisión: eliminar la fila (no se puede saber cuál era la nota real).")
    d = sin

    # --- Problema 4: valores faltantes ---
    print("\n[Problema 4] Valores faltantes")
    faltantes = d[d.isnull().any(axis=1)]
    print("  Detección: d[d.isnull().any(axis=1)]")
    print(faltantes.to_string(index=False))
    d = d.dropna()
    print(f"  Decisión: eliminar {len(faltantes)} filas sin promedio_notas.")

    # --- Categorías ---
    print(f"\n[Revisión] Categorías de 'programa' (sin variantes de escritura): {sorted(d['programa'].unique())}")

    d = d.reset_index(drop=True)
    print(f"\nFilas antes: {filas_antes} | Filas después: {len(d)}")
    print(f"Clases después (0 / 1): {d[COL_Y].value_counts().sort_index().to_dict()}")
    d.to_csv(CSV_LIMPIO, index=False)
    print(f"Dataset limpio guardado: {CSV_LIMPIO.relative_to(RAIZ)}")
    return d


# ============================================================
# TAREA 4. Visualización con Seaborn
# ============================================================
def visualizar(d):
    titulo("TAREA 4. VISUALIZACIÓN CON SEABORN")
    # Gráfico 1: dispersión de las dos variables, coloreada por deserción
    plt.figure(figsize=(9, 6))
    sns.scatterplot(data=d, x="asistencia_pct", y="promedio_notas", hue=COL_Y,
                    palette={0: "tab:green", 1: "tab:red"}, s=90)
    plt.title("Asistencia vs promedio de notas según deserción")
    plt.xlabel("Asistencia (%)")
    plt.ylabel("Promedio de notas (0 a 5)")
    plt.legend(title="Deserción (1 = desertó)", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(DIR_DATOS / "grafico1_dispersion.png", dpi=150)
    plt.close()

    # Gráfico 2: boxplot del promedio de notas por clase
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=d, x=COL_Y, y="promedio_notas", hue=COL_Y,
                palette={0: "tab:green", 1: "tab:red"}, legend=False)
    plt.title("Promedio de notas según deserción")
    plt.xlabel("Deserción (0 = continúa, 1 = desertó)")
    plt.ylabel("Promedio de notas (0 a 5)")
    plt.tight_layout()
    plt.savefig(DIR_DATOS / "grafico2_boxplot_notas.png", dpi=150)
    plt.close()

    print("Gráficos guardados: grafico1_dispersion.png y grafico2_boxplot_notas.png (en data/)")
    print("\nPromedio por clase:")
    print(d.groupby(COL_Y)[COL_X].agg(["mean", "median"]).round(2))


# ============================================================
# TAREA 5. Perceptrón desde cero (NumPy)
# ============================================================
def sigmoide(z):
    """Convierte cualquier z en un valor entre 0 y 1."""
    return 1 / (1 + np.exp(-z))


def predecir_probabilidad(X, w, b):
    """Probabilidad de clase 1: sigmoide(X @ w + b)."""
    return sigmoide(X @ w + b)


def calcular_perdida(y_real, y_prob):
    """Pérdida logarítmica (entropía cruzada binaria)."""
    y_prob = np.clip(y_prob, 1e-9, 1 - 1e-9)
    return -np.mean(y_real * np.log(y_prob) + (1 - y_real) * np.log(1 - y_prob))


def calcular_exactitud(y_real, y_prob):
    return np.mean((y_prob >= 0.5).astype(int) == y_real)


def entrenar(X_esc, y):
    """Descenso del gradiente. Retorna (w, b, historial_perdida, historial_exactitud)."""
    w = np.zeros(X_esc.shape[1])
    b = 0.0
    print(f"Pesos iniciales: w1={w[0]:.4f}, w2={w[1]:.4f}, b={b:.4f}")
    hist_perdida, hist_exactitud = [], []

    for epoca in range(1, EPOCAS + 1):
        y_prob = predecir_probabilidad(X_esc, w, b)
        error = y_prob - y
        gradiente_w = X_esc.T @ error / len(y)
        gradiente_b = np.mean(error)

        w = w - TASA_APRENDIZAJE * gradiente_w
        b = b - TASA_APRENDIZAJE * gradiente_b

        hist_perdida.append(calcular_perdida(y, y_prob))
        hist_exactitud.append(calcular_exactitud(y, y_prob))
        if epoca == 1 or epoca % 100 == 0:
            print(f"Época {epoca:>3} | Pérdida: {hist_perdida[-1]:.4f} | Exactitud: {hist_exactitud[-1]:.2%}")
    return w, b, hist_perdida, hist_exactitud


def perceptron(d):
    titulo("TAREA 5. PERCEPTRÓN DESDE CERO CON NUMPY")
    X = d[COL_X].to_numpy(dtype=float)
    y = d[COL_Y].to_numpy()
    print(f"X: {X.shape} | y: {y.shape} | clase 1: {y.sum()} de {len(y)}")

    media = X.mean(axis=0)
    desviacion = X.std(axis=0)
    X_esc = (X - media) / desviacion
    print(f"Media usada: {np.round(media, 4)} | Desviación usada: {np.round(desviacion, 4)}\n")

    w, b, hist_perdida, hist_exactitud = entrenar(X_esc, y)

    y_prob = predecir_probabilidad(X_esc, w, b)
    print("\nRESULTADOS")
    print(f"  Pesos finales: w1 (asistencia_pct) = {w[0]:.4f} | w2 (promedio_notas) = {w[1]:.4f} | b = {b:.4f}")
    print(f"  Pérdida inicial: {hist_perdida[0]:.4f} | Pérdida final: {calcular_perdida(y, y_prob):.4f}")
    print(f"  Exactitud final: {calcular_exactitud(y, y_prob):.2%}")

    # Curva de aprendizaje
    fig, (eje1, eje2) = plt.subplots(1, 2, figsize=(12, 4.5))
    eje1.plot(hist_perdida, color="red")
    eje1.set_title("Pérdida por época")
    eje1.set_xlabel("Época")
    eje1.set_ylabel("Pérdida logarítmica")
    eje1.grid(True, alpha=0.3)
    eje2.plot(hist_exactitud, color="green")
    eje2.set_title("Exactitud por época")
    eje2.set_xlabel("Época")
    eje2.set_ylim(0, 1.05)
    eje2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(DIR_DATOS / "curva_aprendizaje.png", dpi=150)
    plt.close()

    # Frontera de decisión en unidades originales: w1*x1 + w2*x2 + b = 0
    x1_orig = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    x1_esc = (x1_orig - media[0]) / desviacion[0]
    x2_esc = -(w[0] * x1_esc + b) / w[1]
    x2_orig = x2_esc * desviacion[1] + media[1]

    plt.figure(figsize=(9, 6))
    plt.scatter(X[y == 1, 0], X[y == 1, 1], color="red", edgecolors="black", s=80, label="Desertó (1)")
    plt.scatter(X[y == 0, 0], X[y == 0, 1], color="green", marker="s", edgecolors="black", s=80, label="Continúa (0)")
    plt.plot(x1_orig, x2_orig, color="blue", linewidth=2.5, label="Frontera de decisión")
    plt.ylim(X[:, 1].min() - 0.3, X[:, 1].max() + 0.3)
    plt.title("Frontera de decisión – Deserción estudiantil, Cartago")
    plt.xlabel("Asistencia (%)")
    plt.ylabel("Promedio de notas (0 a 5)")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(DIR_DATOS / "frontera_decision.png", dpi=150)
    plt.close()
    print("\nGráficos guardados: curva_aprendizaje.png y frontera_decision.png (en data/)")
    return X, y, w, b, media, desviacion, y_prob


# ============================================================
# TAREA 6. Números para la interpretación
# ============================================================
def interpretar(d, X, y, w, b, media, desviacion, y_prob):
    titulo("TAREA 6. NÚMEROS PARA LA INTERPRETACIÓN")
    pred = (y_prob >= 0.5).astype(int)
    errados = d[pred != y].copy()
    errados["prob_desercion"] = y_prob[pred != y].round(3)
    print(f"Registros del lado equivocado de la frontera: {len(errados)} de {len(y)}")
    print(f"  Desertores no detectados (y=1, predicho 0): {int(((y == 1) & (pred == 0)).sum())}")
    print(f"  Falsas alarmas (y=0, predicho 1):           {int(((y == 0) & (pred == 1)).sum())}")
    if len(errados):
        print(errados.to_string(index=False))

    print("\nDos registros nuevos (se escalan con la media y desviación del entrenamiento):")
    nuevos = [
        ("Esperado 1 (desertor)", 50.0, 2.0),
        ("Esperado 0 (continúa)", 90.0, 4.0),
    ]
    for nombre, asistencia, notas in nuevos:
        x_esc = (np.array([[asistencia, notas]]) - media) / desviacion
        p = predecir_probabilidad(x_esc, w, b)[0]
        print(f"  {nombre}: asistencia={asistencia}%, notas={notas} -> P(deserción)={p:.2%} -> predice {int(p >= 0.5)}")


def main():
    df = cargar_e_inspeccionar()
    d = limpiar(df)
    visualizar(d)
    X, y, w, b, media, desviacion, y_prob = perceptron(d)
    interpretar(d, X, y, w, b, media, desviacion, y_prob)
    print("\nProceso completado.")


if __name__ == "__main__":
    main()