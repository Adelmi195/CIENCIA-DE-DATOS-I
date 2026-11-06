

import json
import os
import time
import warnings
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd


MOSTRAR_GRAFICOS = os.environ.get("MOSTRAR_GRAFICOS", "1") != "0"
if not MOSTRAR_GRAFICOS:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")


# Carpeta del proyecto (se puede cambiar con la variable de entorno CARPETA_PROYECTO)
CARPETA = Path(os.environ.get("CARPETA_PROYECTO", r"C:\FRED\lizzypredictor"))
CARPETA.mkdir(parents=True, exist_ok=True)
GRAFICOS = CARPETA / "graficos"
GRAFICOS.mkdir(exist_ok=True)

RUTA_CSV = CARPETA / "202608-PDE-REP-00204.csv"
SALIDA_CSV = CARPETA / "dataset_limpio.csv"
SALIDA_XLSX = CARPETA / "dataset_limpio.xlsx"

SEMILLA = 42
# Eliminar duplicados exactos (el dataset no tiene ID ni dia: ver advertencia en seccion 3)
ELIMINAR_DUPLICADOS = True
# Umbral de la variable objetivo: ALTO_IMPACTO = 1 si NRO_AFECTADOS >= UMBRAL_AFECTADOS
UMBRAL_AFECTADOS = 3
# Para acelerar el benchmark de lazypredict se usa a lo sumo esta cantidad de filas de
# entrenamiento (muestra estratificada). El modelo final se reentrena con TODO el train.
MAX_FILAS_BENCHMARK = 20000
# Segundos maximos por modelo en lazypredict (solo se aplica si la version lo soporta)
TIMEOUT_MODELO = 180
# Cantidad de modelos de lazypredict que pasan a validacion cruzada
TOP_CANDIDATOS = 5
# Modelos excluidos del benchmark: LabelPropagation y LabelSpreading construyen una
# matriz n x n densa (con 20 000 filas necesitan > 3 GB de RAM y cierran el programa);
# SVC y NuSVC (nucleo no lineal) son demasiado lentos a esta escala.
MODELOS_EXCLUIDOS = ["LabelPropagation", "LabelSpreading", "SVC", "NuSVC"]

print("Los archivos se guardaran en:", CARPETA)


def guardar_grafico(nombre):
    """Guarda la figura actual en la carpeta de graficos y la muestra si corresponde."""
    plt.tight_layout()
    plt.savefig(GRAFICOS / nombre, dpi=200, bbox_inches="tight")
    if MOSTRAR_GRAFICOS:
        plt.show()
    plt.close()


df = pd.read_csv(RUTA_CSV, sep=";", encoding="utf-8-sig")

print("\n========== DATASET ORIGINAL ==========")
print("Filas y columnas:", df.shape)
print(df.head())
print(df.info())

resumen = {"filas_originales": int(df.shape[0]), "columnas_originales": int(df.shape[1])}

# ANALISIS DE VALORES FALTANTES


print("\n========== VALORES FALTANTES ==========")

faltantes = pd.DataFrame(
    {
        "Cantidad": df.isnull().sum(),
        "Porcentaje": (df.isnull().mean() * 100).round(2),
    }
)
faltantes_pos = faltantes[faltantes["Cantidad"] > 0]
print(faltantes_pos)
resumen["faltantes_originales"] = {
    k: int(v) for k, v in faltantes_pos["Cantidad"].items()
}

if len(faltantes_pos) > 0:
    plt.figure(figsize=(9, 5))
    ax = faltantes_pos["Porcentaje"].plot(kind="bar", color="#c0504d")
    for p in ax.patches:
        ax.annotate(
            f"{p.get_height():.1f}%",
            (p.get_x() + p.get_width() / 2, p.get_height()),
            ha="center",
            va="bottom",
            fontsize=9,
        )
    plt.title("Valores faltantes por variable (dataset original)")
    plt.xlabel("Variable")
    plt.ylabel("% de valores faltantes")
    plt.xticks(rotation=30, ha="right")
    guardar_grafico("01_valores_faltantes.png")


# Se normaliza ANTES de buscar duplicados: "'AUTO" y "AUTO" son el mismo valor.

columnas_categoricas = list(df.select_dtypes(exclude="number").columns)

PLACEHOLDERS = {
    "", "-", "--", "NAN", "NA", "N/A", "N.P.", "NP", "SIN INFO", "SIN INFORMACION",
    "NO PRECISA", "NO INDICA", "NO ESPECIFICA", "NO ESPECIFICADO", "SIN ESPECIFICAR",
    "NO REGISTRADO",
}


def normalizar_texto(serie, quitar_ordinal=False):
    """Mayusculas, sin tildes, sin caracteres corruptos y con placeholders unificados."""
    s = serie.astype("string")
    s = (
        s.str.replace("\u00a0", " ", regex=False)       # espacio no separable
        .str.replace("\u00bf", "", regex=False)         # '¿' = tilde corrupta (INVASIO¿N)
        .str.normalize("NFKD")
        .str.replace(r"[\u0300-\u036f]", "", regex=True)  # quitar tildes
        .str.replace(r"^'+", "", regex=True)            # comilla inicial ('AUTO)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .str.upper()
    )
    if quitar_ordinal:  # "AUTO 2" -> "AUTO" (numero de vehiculo)
        s = s.str.replace(r"\s+\d$", "", regex=True)
    s = s.fillna("NO REGISTRADO")
    s = s.where(~s.isin(PLACEHOLDERS), "NO REGISTRADO")
    return s.astype(object)


for col in columnas_categoricas:
    df[col] = normalizar_texto(
        df[col], quitar_ordinal=col in ("TIPO_VEHICULO", "TIPO_TRANSPORTE")
    )

print("\nVariables categoricas normalizadas.")

print("\n========== REGISTROS DUPLICADOS ==========")
n_dup = int(df.duplicated().sum())
print("Duplicados exactos encontrados:", n_dup)
resumen["duplicados_exactos"] = n_dup

# Guardar duplicados para revision
df[df.duplicated(keep=False)].to_csv(
    CARPETA / "duplicados_revisar.csv", index=False, encoding="utf-8-sig"
)

# ADVERTENCIA: cada fila es un accidente (NRO_ACCIDENTES = 1) y el dataset NO tiene
# ID ni dia. Dos accidentes distintos del mismo mes y tramo pueden verse identicos.
# Se eliminan para evitar que un mismo registro caiga en train y test (fuga de datos),
# pero puede ponerse ELIMINAR_DUPLICADOS = False para conservarlos.
if ELIMINAR_DUPLICADOS:
    df = df.drop_duplicates().reset_index(drop=True)
print("Registros despues del tratamiento de duplicados:", len(df))
resumen["filas_tras_duplicados"] = int(len(df))

#  CONVERTIR VARIABLES NUMERICAS


columnas_numericas = [
    "ANIO",
    "MES",
    "NRO_VEHINVOLUCRADOS",
    "NRO_AFECTADOS",
    "NRO_ACCIDENTES",
]

for col in columnas_numericas:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

print("\n========== RESUMEN NUMERICO ==========")
print(df[columnas_numericas].describe())

#  VERIFICAR VALORES INCORRECTOS


print("\n========== VALIDACION DE DATOS ==========")
meses_mal = int((~df["MES"].between(1, 12)).sum())
print("Meses incorrectos:", meses_mal)
for col in columnas_numericas:
    print(col, "- Valores negativos:", int((df[col] < 0).sum()),
          "- Nulos:", int(df[col].isnull().sum()))

# Accidentes con 0 vehiculos: se conservan pero se marcan
df["VEHICULOS_CERO"] = (df["NRO_VEHINVOLUCRADOS"] == 0).astype(int)
print("Registros con 0 vehiculos involucrados:", int(df["VEHICULOS_CERO"].sum()))
resumen["meses_incorrectos"] = meses_mal
resumen["registros_vehiculos_cero"] = int(df["VEHICULOS_CERO"].sum())
resumen["afectados_max"] = int(df["NRO_AFECTADOS"].max())
resumen["afectados_p99"] = float(df["NRO_AFECTADOS"].quantile(0.99))

#  IDENTIFICAR VARIABLES CONSTANTES

print("\n========== VARIABLES CONSTANTES ==========")
constantes = [c for c in df.columns if df[c].nunique(dropna=False) == 1]
for c in constantes:
    print(c, "es constante")
resumen["variables_constantes"] = constantes
# No se eliminan del archivo general; se excluyen al preparar el modelo.

# CONSOLIDACION DE CATEGORIAS (INGENIERIA DE VARIABLES)


def agrupar(serie, reglas, defecto="OTRO"):
    out = pd.Series(defecto, index=serie.index, dtype=object)
    asignado = pd.Series(False, index=serie.index)
    for etiqueta, patron in reglas:
        m = serie.str.contains(patron, regex=True, na=False) & ~asignado
        out[m] = etiqueta
        asignado |= m
    return out


REGLAS_TIPO_ACCIDENTE = [
    # Las clases A/B/C ya describen la gravedad (fuga de la variable objetivo):
    ("NO ESPECIFICADO", r"^[ABC]:"),
    ("NO ESPECIFICADO", r"^NO REGISTRADO$"),
    ("DESPISTE Y VOLCADURA", r"(DESPIST|DESPITE).*VOLCAD|VOLCAD.*(DESPIST|DESPITE)"),
    ("DESPISTE Y CHOQUE", r"(DESPIST|DESPITE).*(CHOQUE|COLISION)|(CHOQUE|COLISION).*(DESPIST|DESPITE)"),
    ("DESPISTE", r"DESPIST|DESPITE"),
    ("CHOQUE Y VOLCADURA", r"(CHOQUE|COLISION).*VOLCAD"),
    ("VOLCADURA", r"VOLCAD"),
    ("CHOQUE POR ALCANCE", r"ALCANCE|FRENTE/POSTERIOR|EMBISTE"),
    ("CHOQUE FRONTAL", r"FRONTAL"),
    ("CHOQUE LATERAL", r"LATERAL"),
    ("CHOQUE MULTIPLE", r"MULTIPLE"),
    ("ATROPELLO", r"ATROPELL"),
    ("INCENDIO", r"INCENDIO"),
    ("OTRO CHOQUE", r"CHOQUE|COLISION"),
]

REGLAS_CAUSA = [
    ("SIN DETERMINAR", r"^(NO REGISTRADO|EN INVESTIGACION|POR DETERMINAR|PENDIENTE)"),
    ("EXCESO DE VELOCIDAD", r"VELOCIDAD"),
    ("CONDUCTOR EBRIO O DROGADO", r"EBRI|DROGAD|ALCOHOL"),
    ("CANSANCIO O FATIGA", r"CANSAD|CANSANCIO|FATIGA|EXHAUST|SUENO|DORMID|SOMNOLEN"),
    ("INVASION DE CARRIL", r"INVASION"),
    ("FALLA MECANICA", r"FALLA|MECANIC|NEUMATIC|LLANTA|FRENO|AVERIA|SOBRECARGA|REVENTAD"),
    ("FACTORES AMBIENTALES", r"AMBIENTAL|MOJAD|LLUVIA|NEBLINA|NATURAL|DESLIZ|HUAYCO|DERRUMB|NIEVE|HELADA|RESBAL|GRANIZ|VIENTO|VISIBILIDAD"),
    ("ANIMALES EN LA VIA", r"ANIMAL"),
    ("IMPRUDENCIA DE PEATON", r"PEATON"),
    ("IMPRUDENCIA DEL CONDUCTOR", r"IMPRUDEN|TEMERARI|DISTRAC|MANIOBRA|EVASI|DESPISTE|CONDUCCION|ADELANT|NO RESPET|INADECUAD"),
    ("OBSTACULO EN LA VIA", r"OBSTACULO|HUECO|SENALIZ"),
]

REGLAS_CLIMA = [
    ("NO REGISTRADO", r"^NO REGISTRADO$"),
    ("LLUVIOSO", r"LLUV|LLOVIZNA|LLUCIOSO"),
    ("NIEVE O GRANIZO", r"NIEVE|NEVADA|GRANIZ"),
    ("NEBLINA", r"NEBLINA"),
    ("NOCHE U OSCURO", r"NOCHE|OSCURO"),
    ("TEMPERATURA BAJO CERO", r"BAJO CERO|HELADA"),
    ("NUBLADO", r"NUBLAD|CIELO CUBIERTO"),
    ("DESPEJADO O SOLEADO", r"DESPEJAD|SOLEAD|BUEN|NORMAL|BUENA"),
]

REGLAS_TIPO_VIA = [
    ("NO REGISTRADO", r"^(NO REGISTRADO|1|2)$"),
    ("AUTOPISTA", r"AUTOPISTA"),
    ("CARRETERA", r"^CARRETERA$"),
    ("MULTICARRIL", r"MULTICARRIL|MULTIPLE"),
    ("DOBLE CARRIL", r"DOBLE CARRIL"),
]

REGLAS_VEHICULO = [
    ("NO REGISTRADO", r"^NO REGISTRADO$"),
    ("SEMITRAILER O TRAILER", r"SEMI|TRAILER|TRAYLER|REMOLC|TRACTO|CABEZAL"),
    ("BUS U OMNIBUS", r"\bBUS\b|AUTOBUS|OMNIBUS|MICROBUS|MINIBUS|COASTER|CUSTER"),
    ("CAMION", r"CAMION(?!ETA)|VOLQUETE|CISTERNA|FURGON|GRUA"),
    ("MOTOCICLETA", r"MOTO"),
    ("CAMIONETA", r"CAMIONETA|PICK|RURAL|SUV|STATION|PANEL"),
    ("MINIVAN O COMBI", r"MINIVAN|COMBI|\bVAN\b"),
    ("AUTO", r"AUTO|TAXI|SEDAN|HATCHBACK|PARTICULAR"),
]


def grupo_transporte(serie):
    tiene_carga = serie.str.contains("CARGA", na=False)
    tiene_pas = serie.str.contains("PASAJEROS|PARTICULAR", na=False)
    tiene_pub = serie.str.contains("PUBLICO", na=False)
    out = pd.Series("OTRO", index=serie.index, dtype=object)
    out[serie == "NO REGISTRADO"] = "NO REGISTRADO"
    out[tiene_pas & ~tiene_carga] = "PASAJEROS PRIVADO O PARTICULAR"
    out[tiene_pub & ~tiene_carga] = "PASAJEROS PUBLICO"
    out[tiene_carga & ~tiene_pas] = "CARGA"
    out[tiene_carga & tiene_pas] = "MIXTO (CARGA Y PASAJEROS)"
    return out


df["TIPO_ACCIDENTE_GRUPO"] = agrupar(df["TIPO_ACCIDENTE"], REGLAS_TIPO_ACCIDENTE)
df["CAUSA_GRUPO"] = agrupar(df["CAUSA_ACCIDENTE"], REGLAS_CAUSA)
df["CLIMA_GRUPO"] = agrupar(df["CLIMA"], REGLAS_CLIMA)
df["TIPO_VIA_GRUPO"] = agrupar(df["TIPO_VIA"], REGLAS_TIPO_VIA)
df["TIPO_VEHICULO_GRUPO"] = agrupar(df["TIPO_VEHICULO"], REGLAS_VEHICULO)
df["TIPO_TRANSPORTE_GRUPO"] = grupo_transporte(df["TIPO_TRANSPORTE"])

# Clasificacion de gravedad ya reportada (A/B/C): se conserva como informacion,
# pero NO se usa como predictor porque revela la gravedad.
df["CLASIFICACION_REPORTADA"] = (
    df["TIPO_ACCIDENTE"].str.extract(r"^([ABC]):", expand=False).fillna("NO APLICA")
)

# Variables temporales y objetivo
df["TRIMESTRE"] = ((df["MES"] - 1) // 3 + 1).astype(int)
df["PERIODO"] = df["ANIO"].astype(int).astype(str) + "-" + df["MES"].astype(int).astype(str).str.zfill(2)
df["ALTO_IMPACTO"] = (df["NRO_AFECTADOS"] >= UMBRAL_AFECTADOS).astype(int)

print("\n========== CATEGORIAS ANTES -> DESPUES ==========")
reduccion = {}
for original, grupo in [
    ("TIPO_ACCIDENTE", "TIPO_ACCIDENTE_GRUPO"),
    ("CAUSA_ACCIDENTE", "CAUSA_GRUPO"),
    ("CLIMA", "CLIMA_GRUPO"),
    ("TIPO_VIA", "TIPO_VIA_GRUPO"),
    ("TIPO_VEHICULO", "TIPO_VEHICULO_GRUPO"),
    ("TIPO_TRANSPORTE", "TIPO_TRANSPORTE_GRUPO"),
]:
    reduccion[original] = [int(df[original].nunique()), int(df[grupo].nunique())]
    print(f"{original}: {reduccion[original][0]} -> {reduccion[original][1]} categorias")
resumen["reduccion_categorias"] = reduccion
resumen["prevalencia_alto_impacto"] = float(df["ALTO_IMPACTO"].mean())

# ==========================================================
# 8. ANALISIS EXPLORATORIO (EDA)
# ==========================================================

# GRAFICO 2: Accidentes por anio
plt.figure(figsize=(9, 5))
sns.countplot(data=df, x="ANIO", color="#4f81bd")
plt.title("Cantidad de accidentes registrados por anio (2026: solo hasta agosto)")
plt.xlabel("Anio")
plt.ylabel("Cantidad de accidentes")
guardar_grafico("02_accidentes_por_anio.png")

# GRAFICO 3: Tipos de accidente (agrupados)
plt.figure(figsize=(9, 6))
df["TIPO_ACCIDENTE_GRUPO"].value_counts().sort_values().plot(kind="barh", color="#4f81bd")
plt.title("Tipos de accidente (categorias consolidadas)")
plt.xlabel("Frecuencia")
plt.ylabel("")
guardar_grafico("03_tipos_accidentes.png")

# GRAFICO 4: Causas (agrupadas)
plt.figure(figsize=(9, 6))
df["CAUSA_GRUPO"].value_counts().sort_values().plot(kind="barh", color="#9bbb59")
plt.title("Causas de accidente (categorias consolidadas)")
plt.xlabel("Frecuencia")
plt.ylabel("")
guardar_grafico("04_causas_accidentes.png")

# GRAFICO 5: Distribucion de afectados
plt.figure(figsize=(8, 5))
sns.histplot(df["NRO_AFECTADOS"].clip(upper=15), bins=16, discrete=True, color="#c0504d")
plt.yscale("log")
plt.axvline(UMBRAL_AFECTADOS - 0.5, color="black", linestyle="--",
            label=f"Umbral de alto impacto (>= {UMBRAL_AFECTADOS})")
plt.title("Distribucion del numero de afectados (valores > 15 agrupados en 15)")
plt.xlabel("Numero de afectados")
plt.ylabel("Frecuencia (escala log)")
plt.legend()
guardar_grafico("05_distribucion_afectados.png")

# GRAFICO 6: Correlacion entre variables numericas (sin constantes)
numericas = df[["ANIO", "MES", "NRO_VEHINVOLUCRADOS", "NRO_AFECTADOS", "ALTO_IMPACTO"]]
plt.figure(figsize=(7, 5.5))
sns.heatmap(numericas.corr(), annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
plt.title("Matriz de correlacion")
guardar_grafico("06_correlaciones.png")

# GRAFICO 7: Tasa de alto impacto por tipo de accidente
tasa_tipo = (
    df[df["TIPO_ACCIDENTE_GRUPO"] != "NO ESPECIFICADO"]
    .groupby("TIPO_ACCIDENTE_GRUPO")["ALTO_IMPACTO"].mean().sort_values() * 100
)
plt.figure(figsize=(9, 6))
tasa_tipo.plot(kind="barh", color="#8064a2", label="Tasa por tipo")
plt.axvline(df["ALTO_IMPACTO"].mean() * 100, color="black", linestyle="--", label="Promedio general")
plt.title("% de accidentes de alto impacto por tipo de accidente")
plt.xlabel("% con 3 o mas afectados")
plt.ylabel("")
plt.legend()
guardar_grafico("07_alto_impacto_por_tipo.png")

# GRAFICO 8: Tasa de alto impacto por numero de vehiculos
vehi = df["NRO_VEHINVOLUCRADOS"].clip(upper=5)
tasa_veh = df.groupby(vehi)["ALTO_IMPACTO"].mean() * 100
plt.figure(figsize=(7, 5))
tasa_veh.plot(kind="bar", color="#4bacc6")
plt.title("% de alto impacto segun vehiculos involucrados (5 = 5 o mas)")
plt.xlabel("Vehiculos involucrados")
plt.ylabel("% con 3 o mas afectados")
plt.xticks(rotation=0)
guardar_grafico("08_alto_impacto_por_vehiculos.png")

# ==========================================================
# 9. RESULTADO FINAL DE LA LIMPIEZA
# ==========================================================

print("\n========== RESULTADO FINAL ==========")
print("Filas:", df.shape[0])
print("Columnas:", df.shape[1])
print("Valores nulos:", int(df.isnull().sum().sum()))
print("Duplicados:", int(df.duplicated().sum()))
resumen["filas_finales"] = int(df.shape[0])
resumen["columnas_finales"] = int(df.shape[1])
resumen["nulos_finales"] = int(df.isnull().sum().sum())

# ==========================================================
# 10. GUARDAR DATASET LIMPIO (CSV Y EXCEL)
# ==========================================================

df.to_csv(SALIDA_CSV, index=False, encoding="utf-8-sig")

DICCIONARIO = [
    ("ANIO", "Anio del reporte", "Original"),
    ("MES", "Mes del reporte (1-12)", "Original"),
    ("ENTIDAD_PRESTADORA", "Empresa concesionaria", "Original normalizada"),
    ("CONCESION", "Nombre de la concesion", "Original normalizada"),
    ("SIGLAS", "Siglas de la concesion", "Original normalizada"),
    ("TIPO_ACCIDENTE", "Tipo de accidente (texto libre)", "Original normalizada"),
    ("CAUSA_ACCIDENTE", "Causa reportada (texto libre)", "Original normalizada"),
    ("TIPO_TRANSPORTE", "Tipo de transporte (texto libre)", "Original normalizada"),
    ("TIPO_VEHICULO", "Tipo de vehiculo (texto libre)", "Original normalizada"),
    ("CLIMA", "Condicion climatica (texto libre)", "Original normalizada"),
    ("TIPO_VIA", "Tipo de via", "Original normalizada"),
    ("RUTA", "Ruta", "Original normalizada"),
    ("TRAMO", "Tramo", "Original normalizada"),
    ("NRO_VEHINVOLUCRADOS", "Numero de vehiculos involucrados", "Original"),
    ("NRO_AFECTADOS", "Numero de personas afectadas", "Original"),
    ("NRO_ACCIDENTES", "Accidentes por fila (siempre 1: constante)", "Original"),
    ("VEHICULOS_CERO", "1 si NRO_VEHINVOLUCRADOS = 0", "Creada"),
    ("TIPO_ACCIDENTE_GRUPO", "Tipo de accidente consolidado (clases A/B/C = NO ESPECIFICADO)", "Creada"),
    ("CAUSA_GRUPO", "Causa consolidada (prioridad por orden de reglas)", "Creada"),
    ("CLIMA_GRUPO", "Clima consolidado", "Creada"),
    ("TIPO_VIA_GRUPO", "Tipo de via consolidado", "Creada"),
    ("TIPO_VEHICULO_GRUPO", "Vehiculo consolidado (si hay varios, el de mayor tamano)", "Creada"),
    ("TIPO_TRANSPORTE_GRUPO", "Tipo de transporte consolidado", "Creada"),
    ("CLASIFICACION_REPORTADA", "Clase A/B/C reportada; no se usa como predictor", "Creada"),
    ("TRIMESTRE", "Trimestre del anio (1-4)", "Creada"),
    ("PERIODO", "Periodo AAAA-MM", "Creada"),
    ("ALTO_IMPACTO", f"Variable objetivo: 1 si NRO_AFECTADOS >= {UMBRAL_AFECTADOS}", "Creada"),
]
df_dic = pd.DataFrame(DICCIONARIO, columns=["Variable", "Descripcion", "Origen"])

df_res = pd.DataFrame(
    [
        ("Filas originales", resumen["filas_originales"]),
        ("Duplicados exactos (tras normalizar)", resumen["duplicados_exactos"]),
        ("Duplicados eliminados", resumen["duplicados_exactos"] if ELIMINAR_DUPLICADOS else 0),
        ("Filas finales", resumen["filas_finales"]),
        ("Columnas finales", resumen["columnas_finales"]),
        ("Nulos finales", resumen["nulos_finales"]),
        ("Registros con 0 vehiculos (marcados)", resumen["registros_vehiculos_cero"]),
        ("Prevalencia de ALTO_IMPACTO", round(resumen["prevalencia_alto_impacto"], 4)),
    ],
    columns=["Indicador", "Valor"],
)

with pd.ExcelWriter(SALIDA_XLSX, engine="openpyxl") as escritor:
    from openpyxl.styles import Alignment, Font, PatternFill

    libro = escritor.book
    try:  # fuente por defecto del libro
        libro._named_styles["Normal"].font = Font(name="Arial", size=10)
    except Exception:
        pass

    df.to_excel(escritor, sheet_name="Datos_limpios", index=False)
    df_dic.to_excel(escritor, sheet_name="Diccionario", index=False)
    df_res.to_excel(escritor, sheet_name="Resumen_limpieza", index=False)

    for hoja in escritor.sheets.values():
        for celda in hoja[1]:
            celda.font = Font(name="Arial", size=10, bold=True, color="FFFFFF")
            celda.fill = PatternFill("solid", fgColor="1F4E78")
            celda.alignment = Alignment(horizontal="center", vertical="center")
        hoja.freeze_panes = "A2"
        hoja.auto_filter.ref = hoja.dimensions
        for columna in hoja.columns:
            ancho = max(len(str(c.value)) if c.value is not None else 0
                        for c in list(columna)[:200])
            hoja.column_dimensions[columna[0].column_letter].width = min(max(ancho + 2, 10), 45)

print("\n====================================")
print("LIMPIEZA Y EDA FINALIZADOS")
print("CSV limpio  :", SALIDA_CSV)
print("Excel limpio:", SALIDA_XLSX)
print("====================================")

# ==========================================================
# 11. PREPARACION PARA EL MODELO
# ==========================================================

print("\n========== PREPARACION DEL MODELO ==========")

VARIABLES_NUM = ["ANIO", "MES", "NRO_VEHINVOLUCRADOS"]
VARIABLES_CAT = [
    "SIGLAS", "TIPO_VIA_GRUPO", "TIPO_ACCIDENTE_GRUPO", "CAUSA_GRUPO",
    "CLIMA_GRUPO", "TIPO_VEHICULO_GRUPO", "TIPO_TRANSPORTE_GRUPO",
]
# Excluidas a proposito: NRO_AFECTADOS y CLASIFICACION_REPORTADA (fuga de la
# variable objetivo), NRO_ACCIDENTES (constante), RUTA/TRAMO (alta cardinalidad).

X = df[VARIABLES_NUM + VARIABLES_CAT].copy()
X[VARIABLES_CAT] = X[VARIABLES_CAT].astype(object)
y = df["ALTO_IMPACTO"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=SEMILLA
)
print("Train:", X_train.shape, "| Test:", X_test.shape)
print("Prevalencia train/test:", round(y_train.mean(), 4), "/", round(y_test.mean(), 4))

# Muestra de entrenamiento para el benchmark de lazypredict
if len(X_train) > MAX_FILAS_BENCHMARK:
    X_bench, _, y_bench, _ = train_test_split(
        X_train, y_train, train_size=MAX_FILAS_BENCHMARK,
        stratify=y_train, random_state=SEMILLA,
    )
else:
    X_bench, y_bench = X_train, y_train
print("Filas usadas en el benchmark:", len(X_bench))

#  BENCHMARK CON LAZYPREDICT


try:
    from lazypredict.Supervised import CLASSIFIERS, LazyClassifier
except ImportError as e:
    raise SystemExit("Instala lazypredict con: pip install lazypredict") from e

clasificadores = [cls for nombre, cls in CLASSIFIERS if nombre not in MODELOS_EXCLUIDOS]
print("Clasificadores a evaluar:", len(clasificadores))

print("\n========== LAZYPREDICT (LazyClassifier) ==========")
t0 = time.time()
argumentos = dict(verbose=0, ignore_warnings=True, custom_metric=None,
                  random_state=SEMILLA, classifiers=clasificadores)
try:
    lazy = LazyClassifier(timeout=TIMEOUT_MODELO, **argumentos)
except TypeError:  # versiones antiguas de lazypredict no tienen "timeout"
    lazy = LazyClassifier(**argumentos)

resultado = lazy.fit(X_bench, X_test, y_bench, y_test)
modelos_lazy = resultado[0] if isinstance(resultado, tuple) else resultado
pipelines_lazy = lazy.provide_models(X_bench, X_test, y_bench, y_test)
tiempo_lazy = time.time() - t0

modelos_lazy = modelos_lazy.sort_values("ROC AUC", ascending=False)
modelos_lazy.to_csv(CARPETA / "lazypredict_resultados.csv", encoding="utf-8-sig")
print(f"Modelos evaluados: {len(modelos_lazy)} en {tiempo_lazy/60:.1f} min")
print(modelos_lazy.head(15).round(4))

top15 = modelos_lazy.head(15)[::-1]
plt.figure(figsize=(9, 6))
plt.barh(top15.index, top15["ROC AUC"], color="#4f81bd")
for i, v in enumerate(top15["ROC AUC"]):
    plt.text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=8)
plt.xlim(0.5, top15["ROC AUC"].max() + 0.05)
plt.title("LazyPredict: 15 mejores modelos por ROC AUC (conjunto de prueba)")
plt.xlabel("ROC AUC")
guardar_grafico("09_lazypredict_top15.png")

#  VALIDACION CRUZADA DE LOS MEJORES CANDIDATOS


print("\n========== VALIDACION CRUZADA (solo train) ==========")
candidatos = [m for m in modelos_lazy.index if "Dummy" not in m][:TOP_CANDIDATOS]
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA)
metricas_cv = {
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
    "f1_positiva": "f1",
    "balanced_acc": "balanced_accuracy",
}
filas_cv = []
for nombre in candidatos:
    r = cross_validate(clone(pipelines_lazy[nombre]), X_train, y_train, cv=cv,
                       scoring=metricas_cv, n_jobs=-1)
    fila = {"Modelo": nombre}
    for k in metricas_cv:
        fila[f"{k}_media"] = r[f"test_{k}"].mean()
        fila[f"{k}_std"] = r[f"test_{k}"].std()
    filas_cv.append(fila)
    print(f"{nombre:28s} ROC AUC = {fila['roc_auc_media']:.4f} +/- {fila['roc_auc_std']:.4f}")

tabla_cv = pd.DataFrame(filas_cv).sort_values("roc_auc_media", ascending=False)
tabla_cv.to_csv(CARPETA / "validacion_cruzada_candidatos.csv", index=False, encoding="utf-8-sig")
mejor_nombre = tabla_cv.iloc[0]["Modelo"]
print("\nModelo seleccionado:", mejor_nombre)


#  MODELO FINAL, UMBRAL Y EVALUACION EN TEST


def puntaje(modelo, datos):
    """Probabilidad de la clase positiva (o decision_function si no hay proba)."""
    if hasattr(modelo, "predict_proba"):
        return modelo.predict_proba(datos)[:, 1]
    return modelo.decision_function(datos)


modelo_final = clone(pipelines_lazy[mejor_nombre])

# Umbral que maximiza F1 usando predicciones fuera de muestra (solo train)
metodo = "predict_proba" if hasattr(modelo_final, "predict_proba") else "decision_function"
oof = cross_val_predict(clone(modelo_final), X_train, y_train, cv=cv, method=metodo, n_jobs=-1)
oof = oof[:, 1] if oof.ndim == 2 else oof
prec, rec, umbrales = precision_recall_curve(y_train, oof)
f1s = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
umbral_opt = float(umbrales[np.argmax(f1s)])

modelo_final.fit(X_train, y_train)  # entrenado con TODO el train
s_test = puntaje(modelo_final, X_test)
pred_05 = modelo_final.predict(X_test)
pred_opt = (s_test >= umbral_opt).astype(int)


def metricas(y_real, y_pred):
    return {
        "accuracy": accuracy_score(y_real, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_real, y_pred),
        "precision": precision_score(y_real, y_pred, zero_division=0),
        "recall": recall_score(y_real, y_pred),
        "f1": f1_score(y_real, y_pred),
    }


resultados = {
    "modelo": mejor_nombre,
    "n_train": int(len(X_train)),
    "n_test": int(len(X_test)),
    "prevalencia_test": float(y_test.mean()),
    "roc_auc_test": float(roc_auc_score(y_test, s_test)),
    "pr_auc_test": float(average_precision_score(y_test, s_test)),
    "umbral_optimo_f1": umbral_opt,
    "umbral_0.5": metricas(y_test, pred_05),
    "umbral_optimo": metricas(y_test, pred_opt),
}
with open(CARPETA / "metricas_modelo_final.json", "w", encoding="utf-8") as f:
    json.dump({"limpieza": resumen, "modelo": resultados, "tiempo_lazypredict_min": tiempo_lazy / 60},
              f, indent=2, ensure_ascii=False, default=float)

print("\n========== EVALUACION EN TEST ==========")
print(f"ROC AUC = {resultados['roc_auc_test']:.4f} | PR AUC = {resultados['pr_auc_test']:.4f} "
      f"(linea base = {resultados['prevalencia_test']:.4f})")
print(f"\nUmbral 0.50:\n{classification_report(y_test, pred_05, digits=3)}")
print(f"Umbral optimo ({umbral_opt:.3f}):\n{classification_report(y_test, pred_opt, digits=3)}")

# Matriz de confusion (umbral optimo)
fig, ejes = plt.subplots(1, 2, figsize=(10, 4.5))
for eje, pred, titulo in [(ejes[0], pred_05, "Umbral 0.50"),
                          (ejes[1], pred_opt, f"Umbral optimo ({umbral_opt:.2f})")]:
    ConfusionMatrixDisplay.from_predictions(
        y_test, pred, display_labels=["Bajo impacto", "Alto impacto"],
        cmap="Blues", ax=eje, colorbar=False)
    eje.set_title(titulo)
    eje.set_xlabel("Clase predicha")
    eje.set_ylabel("Clase real")
    eje.grid(False)
guardar_grafico("10_matriz_confusion.png")

# Curvas ROC y Precision-Recall
fig, ejes = plt.subplots(1, 2, figsize=(10, 4.5))
RocCurveDisplay.from_predictions(y_test, s_test, ax=ejes[0], name=mejor_nombre)
ejes[0].plot([0, 1], [0, 1], "k--", label="Azar")
ejes[0].set_title("Curva ROC")
ejes[0].set_xlabel("Tasa de falsos positivos")
ejes[0].set_ylabel("Tasa de verdaderos positivos")
ejes[0].legend()
PrecisionRecallDisplay.from_predictions(y_test, s_test, ax=ejes[1], name=mejor_nombre)
ejes[1].axhline(y_test.mean(), color="k", linestyle="--", label="Linea base")
ejes[1].set_title("Curva Precision-Recall")
ejes[1].set_xlabel("Recall (sensibilidad)")
ejes[1].set_ylabel("Precision")
ejes[1].legend()
guardar_grafico("11_curvas_roc_pr.png")

# Importancia por permutacion (sobre las variables originales)
perm = permutation_importance(modelo_final, X_test, y_test, scoring="roc_auc",
                              n_repeats=5, random_state=SEMILLA, n_jobs=-1)
imp = pd.DataFrame({"Variable": X_test.columns,
                    "Caida_ROC_AUC": perm.importances_mean,
                    "Desv": perm.importances_std}).sort_values("Caida_ROC_AUC")
imp.sort_values("Caida_ROC_AUC", ascending=False).to_csv(
    CARPETA / "importancia_permutacion.csv", index=False, encoding="utf-8-sig")
plt.figure(figsize=(8, 5))
plt.barh(imp["Variable"], imp["Caida_ROC_AUC"], xerr=imp["Desv"], color="#9bbb59")
plt.title("Importancia por permutacion (caida del ROC AUC en test)")
plt.xlabel("Caida del ROC AUC")
guardar_grafico("12_importancia_permutacion.png")

joblib.dump(modelo_final, CARPETA / "modelo_final.joblib")

print("\n====================================")
print("PROCESO COMPLETO")
print("Modelo guardado en:", CARPETA / "modelo_final.joblib")
print("====================================")
