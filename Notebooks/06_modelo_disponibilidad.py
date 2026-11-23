# Databricks notebook source
# MAGIC %md
# MAGIC # Modelo de disponibilidad — ¿habrá bicicletas, y cuántas?
# MAGIC
# MAGIC **Proyecto:** Predicción de demanda y rebalanceo de MiBici
# MAGIC **Equipo:** César Roberto Murillo Figueroa y Andrés Muñoz Rubalcava
# MAGIC
# MAGIC ## 1. Quién pregunta, y qué
# MAGIC
# MAGIC El modelo de `04_modelo_demanda` responde al **operador**: dónde mandar la camioneta.
# MAGIC Este responde al **usuario**: *«voy a esta estación el viernes a las 6 de la tarde,
# MAGIC ¿va a haber bicicleta?»*.
# MAGIC
# MAGIC Son dos preguntas distintas y necesitan dos salidas distintas:
# MAGIC
# MAGIC | | Tarea | Variable objetivo | Por qué |
# MAGIC |---|---|---|---|
# MAGIC | **A** | Regresión | `bicis_inicio` | «cuántas bicicletas» pide un número |
# MAGIC | **B** | Clasificación | `se_vacio` | «¿habrá o no?» pide una probabilidad |
# MAGIC
# MAGIC No basta con una: un promedio de 2 bicicletas puede venir de «casi siempre hay 2» o de
# MAGIC «la mitad de los días hay 4 y la otra mitad ninguna». Al usuario que se va a parar frente
# MAGIC al anclaje le importa muchísimo la diferencia.
# MAGIC
# MAGIC ## 2. Las métricas, y por qué aquí sí hay F1
# MAGIC
# MAGIC En `04_modelo_demanda` argumentamos que **F1 no aplicaba**: no había clases, había conteos.
# MAGIC Aquí la situación es la contraria y conviene decirlo explícitamente en la exposición.
# MAGIC
# MAGIC - **Modelo A (regresión):** RMSE y MAE, en bicicletas. Misma lógica que antes.
# MAGIC - **Modelo B (clasificación):** la clase «se vació» es **minoritaria**. Con un 5 % de casos
# MAGIC   positivos, un modelo que siempre diga «sí habrá bicis» acierta el 95 % de las veces y es
# MAGIC   completamente inútil. Por eso:
# MAGIC   - **AUC-PR** (precisión-exhaustividad promedio) como métrica principal, que es la
# MAGIC     correcta cuando la clase positiva es rara;
# MAGIC   - **F1** en el umbral de operación;
# MAGIC   - **Brier** para saber si las probabilidades están calibradas, no solo ordenadas.
# MAGIC
# MAGIC ## 3. El costo no es simétrico, y el umbral no lo arregla
# MAGIC
# MAGIC Equivocarse hacia los dos lados no cuesta igual:
# MAGIC
# MAGIC - decir **«sí hay bici»** y que no la haya → el usuario camina hasta allá en vano;
# MAGIC - decir **«no hay bici»** y que sí la hubiera → el usuario toma otra ruta sin necesidad.
# MAGIC
# MAGIC El primero es peor. Conviene ser explícito sobre lo que hacemos con eso, porque es fácil
# MAGIC decir «el costo es asimétrico» y después optimizar algo que no lo es.
# MAGIC
# MAGIC **El umbral se elige maximizando F1, que es un criterio balanceado.** F1 pesa igual la
# MAGIC precisión y la exhaustividad, así que *no* codifica la asimetría. Si quisiéramos favorecerla
# MAGIC de verdad, el criterio correcto sería F-beta con beta > 1, y el umbral quedaría más abajo.
# MAGIC
# MAGIC Lo que sí hacemos con la asimetría es otra cosa, y es la decisión de diseño importante:
# MAGIC **el API devuelve la probabilidad cruda, no solo el sí/no.** El campo `hay_bicis` es una
# MAGIC comodidad; `probabilidad_sin_bicis` es el dato. Quien consuma el endpoint y pese distinto
# MAGIC los dos errores puede poner su propio corte sin volver a entrenar nada, y la página web
# MAGIC muestra tres niveles en lugar de un binario por la misma razón.
# MAGIC
# MAGIC ## 4. Contra qué se compara
# MAGIC
# MAGIC Contra la **frecuencia histórica** de esa estación, ese día y esa hora en las cuatro
# MAGIC semanas previas (`vacia_freq_4sem`), y contra la **ocupación media** histórica
# MAGIC (`bicis_inicio_media_4sem`). Es lo que haría cualquiera sin modelo. Mismo criterio que en
# MAGIC el notebook 04: un modelo que no le gane a eso no justifica existir.

# COMMAND ----------

CATALOG, SCHEMA = "workspace", "mibici"
TBL_DISP = f"{CATALOG}.{SCHEMA}.gold_disponibilidad_estacion_hora"

# Misma partición temporal y misma ventana que 04_modelo_demanda, para que los dos modelos
# del API se puedan comparar entre sí sin asteriscos.
CORTE_ENTRENAMIENTO = "2026-08-31"
DESDE = "2024-11-01"

HORA_MIN, HORA_MAX = 5, 23
ARRANQUE_SUPUESTO = 0.50          # el mismo que usó 05_gold_ocupacion

RUTA_ARTEFACTO = "/Volumes/workspace/mibici/raw/modelo"
VERSION_MODELO = "1.0.0"

FESTIVOS_ML = [f"{a}-{d}" for a in (2023, 2024, 2025, 2026, 2027)
               for d in ("01-01", "02-05", "03-21", "05-01", "09-16", "11-20", "12-25")]

import json
import numpy as np
import pandas as pd
from pyspark.sql import functions as F

spark.conf.set("spark.sql.ansi.enabled", "false")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Los datos
# MAGIC
# MAGIC Las variables de rezago solo miran al pasado, igual que en el notebook 04. Las primeras
# MAGIC cuatro semanas de cada estación vienen nulas porque no hay historia que promediar, y esas
# MAGIC filas se descartan: no se entrena con una variable que no existe.

# COMMAND ----------

FEATURES = ["hora", "dia_semana", "mes", "es_fin_semana", "es_festivo",
            "capacidad_anclajes",
            "salidas_media_4sem", "llegadas_media_4sem", "salidas_max_4sem",
            "bicis_inicio_media_4sem", "bicis_inicio_min_4sem", "vacia_freq_4sem"]

datos = (spark.table(TBL_DISP)
         .filter(f"fecha >= '{DESDE}'")
         .filter("salidas_media_4sem IS NOT NULL")
         .filter("bicis_inicio_media_4sem IS NOT NULL")
         .filter("vacia_freq_4sem IS NOT NULL")
         .select("estacion_id", "fecha", *FEATURES, "bicis_inicio", "se_vacio"))

print(f"Filas utilizables: {datos.count():,}")
display(datos.limit(5))

# COMMAND ----------

pdf = datos.toPandas()
for c in ("es_fin_semana", "es_festivo"):
    pdf[c] = pdf[c].astype(int)
pdf["se_vacio"] = pdf["se_vacio"].astype(int)
pdf["fecha"] = pd.to_datetime(pdf["fecha"])

entrena = pdf[pdf.fecha <= CORTE_ENTRENAMIENTO]
prueba  = pdf[pdf.fecha >  CORTE_ENTRENAMIENTO]

print(f"Entrenamiento: {len(entrena):>10,} filas  "
      f"({entrena.fecha.min():%Y-%m-%d} a {entrena.fecha.max():%Y-%m-%d})")
print(f"Prueba       : {len(prueba):>10,} filas  "
      f"({prueba.fecha.min():%Y-%m-%d} a {prueba.fecha.max():%Y-%m-%d})")
assert len(prueba) > 0, "La partición de prueba quedó vacía: revisa CORTE_ENTRENAMIENTO"

tasa_ent = entrena.se_vacio.mean()
tasa_pru = prueba.se_vacio.mean()
print(f"\nHoras con estación vacía — entrenamiento: {tasa_ent:.2%}")
print(f"Horas con estación vacía — prueba       : {tasa_pru:.2%}")
print(f"\nDesbalance de {1/max(tasa_pru, 1e-9):.0f} a 1. Aquí F1 y AUC-PR sí son las métricas.")

X_ent = entrena[FEATURES].to_numpy(dtype="float32")
X_pru = prueba[FEATURES].to_numpy(dtype="float32")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Las líneas base
# MAGIC
# MAGIC Están literalmente en el panel: usar la columna histórica como predicción.

# COMMAND ----------

from sklearn.metrics import (average_precision_score, roc_auc_score, brier_score_loss,
                             f1_score, precision_score, recall_score, confusion_matrix)


def rmse(y, yhat):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(yhat)) ** 2)))


def mae(y, yhat):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(yhat))))


y_bicis = prueba["bicis_inicio"].to_numpy()
y_vacia = prueba["se_vacio"].to_numpy()

base_bicis = prueba["bicis_inicio_media_4sem"].to_numpy()
base_prob  = np.clip(prueba["vacia_freq_4sem"].to_numpy(), 0.0, 1.0)

base_metricas = {
    "bicis": {"rmse": rmse(y_bicis, base_bicis), "mae": mae(y_bicis, base_bicis)},
    "vacia": {"auc_pr": float(average_precision_score(y_vacia, base_prob)),
              "roc_auc": float(roc_auc_score(y_vacia, base_prob)),
              "brier": float(brier_score_loss(y_vacia, base_prob))},
}

print(f"Línea base — bicis   RMSE {base_metricas['bicis']['rmse']:.4f}   "
      f"MAE {base_metricas['bicis']['mae']:.4f}")
print(f"Línea base — vacía   AUC-PR {base_metricas['vacia']['auc_pr']:.4f}   "
      f"ROC-AUC {base_metricas['vacia']['roc_auc']:.4f}   "
      f"Brier {base_metricas['vacia']['brier']:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Modelo A — cuántas bicicletas

# COMMAND ----------

import mlflow
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier

mlflow.set_experiment(
    f"/Users/{spark.sql('SELECT current_user()').first()[0]}/mibici_disponibilidad")

metricas = {}

with mlflow.start_run(run_name=f"gbt_disponibilidad_v{VERSION_MODELO}") as run:
    mlflow.log_params({
        "features": ",".join(FEATURES),
        "corte_entrenamiento": CORTE_ENTRENAMIENTO,
        "filas_entrenamiento": len(entrena),
        "filas_prueba": len(prueba),
        "particion": "temporal",
        "arranque_supuesto": ARRANQUE_SUPUESTO,
        "tasa_positivos": float(tasa_ent),
    })

    modelo_bicis = HistGradientBoostingRegressor(
        loss="squared_error", max_iter=400, learning_rate=0.08,
        max_leaf_nodes=63, min_samples_leaf=40, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=25,
        random_state=42,
    )
    modelo_bicis.fit(X_ent, entrena["bicis_inicio"].to_numpy())

    # La predicción se recorta al rango físico, igual que la reconstrucción: el modelo no
    # tiene por qué saber que una estación de 19 anclajes no puede tener 21 bicicletas.
    pred_bicis = np.clip(modelo_bicis.predict(X_pru), 0, prueba["capacidad_anclajes"].to_numpy())

    metricas["bicis"] = {"rmse": rmse(y_bicis, pred_bicis), "mae": mae(y_bicis, pred_bicis),
                         "iteraciones": int(modelo_bicis.n_iter_)}
    mlflow.log_metrics({"bicis_rmse": metricas["bicis"]["rmse"],
                        "bicis_mae": metricas["bicis"]["mae"]})

    print(f"Modelo    — bicis   RMSE {metricas['bicis']['rmse']:.4f}   "
          f"MAE {metricas['bicis']['mae']:.4f}   ({modelo_bicis.n_iter_} árboles)")
    mejora = 1 - metricas["bicis"]["rmse"] / base_metricas["bicis"]["rmse"]
    print(f"Línea base— bicis   RMSE {base_metricas['bicis']['rmse']:.4f}")
    print(f"Mejora sobre la línea base: {mejora:+.2%}")

    # ------------------------------------------------------------------ Modelo B
    modelo_vacia = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.08,
        max_leaf_nodes=63, min_samples_leaf=40, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=25,
        random_state=42,
    )
    modelo_vacia.fit(X_ent, entrena["se_vacio"].to_numpy())
    prob_vacia = modelo_vacia.predict_proba(X_pru)[:, 1]

    metricas["vacia"] = {
        "auc_pr": float(average_precision_score(y_vacia, prob_vacia)),
        "roc_auc": float(roc_auc_score(y_vacia, prob_vacia)),
        "brier": float(brier_score_loss(y_vacia, prob_vacia)),
        "iteraciones": int(modelo_vacia.n_iter_),
    }
    mlflow.log_metrics({"vacia_auc_pr": metricas["vacia"]["auc_pr"],
                        "vacia_roc_auc": metricas["vacia"]["roc_auc"],
                        "vacia_brier": metricas["vacia"]["brier"]})
    run_id = run.info.run_id

print(f"\nModelo    — vacía   AUC-PR {metricas['vacia']['auc_pr']:.4f}   "
      f"ROC-AUC {metricas['vacia']['roc_auc']:.4f}   Brier {metricas['vacia']['brier']:.4f}")
print(f"Línea base— vacía   AUC-PR {base_metricas['vacia']['auc_pr']:.4f}   "
      f"ROC-AUC {base_metricas['vacia']['roc_auc']:.4f}   "
      f"Brier {base_metricas['vacia']['brier']:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. El umbral de operación
# MAGIC
# MAGIC El modelo entrega una probabilidad; la aplicación tiene que dar una respuesta. El umbral
# MAGIC que las separa es una **decisión de producto**, no del algoritmo, y por eso se elige
# MAGIC explícitamente en lugar de dejar el 0.5 por omisión.
# MAGIC
# MAGIC Se barre el rango y se escoge el que maximiza F1 sobre la clase «se vació».

# COMMAND ----------

umbrales = np.arange(0.05, 0.96, 0.01)
f1s = [f1_score(y_vacia, (prob_vacia >= u).astype(int), zero_division=0) for u in umbrales]
UMBRAL = float(umbrales[int(np.argmax(f1s))])

pred_vacia = (prob_vacia >= UMBRAL).astype(int)
metricas["vacia"].update({
    "umbral": UMBRAL,
    "f1": float(f1_score(y_vacia, pred_vacia, zero_division=0)),
    "precision": float(precision_score(y_vacia, pred_vacia, zero_division=0)),
    "recall": float(recall_score(y_vacia, pred_vacia, zero_division=0)),
})

base_pred = (base_prob >= UMBRAL).astype(int)
base_metricas["vacia"].update({
    "umbral": UMBRAL,
    "f1": float(f1_score(y_vacia, base_pred, zero_division=0)),
    "precision": float(precision_score(y_vacia, base_pred, zero_division=0)),
    "recall": float(recall_score(y_vacia, base_pred, zero_division=0)),
})

print(f"Umbral elegido: {UMBRAL:.2f}\n")
print(f"{'':12s} {'F1':>8s} {'precisión':>10s} {'exhaust.':>10s}")
print(f"{'modelo':12s} {metricas['vacia']['f1']:>8.4f} "
      f"{metricas['vacia']['precision']:>10.4f} {metricas['vacia']['recall']:>10.4f}")
print(f"{'línea base':12s} {base_metricas['vacia']['f1']:>8.4f} "
      f"{base_metricas['vacia']['precision']:>10.4f} {base_metricas['vacia']['recall']:>10.4f}")

vn, fp, fn, vp = confusion_matrix(y_vacia, pred_vacia).ravel()
print(f"\nMatriz de confusión sobre {len(y_vacia):,} horas de prueba")
print(f"  verdaderos negativos (dijimos «hay» y había)      : {vn:>9,}")
print(f"  falsos positivos     (dijimos «no hay» y sí había): {fp:>9,}  ← ruta alterna de más")
print(f"  falsos negativos     (dijimos «hay» y no había)   : {fn:>9,}  ← el error que duele")
print(f"  verdaderos positivos (dijimos «no hay» y no había): {vp:>9,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. ¿Están calibradas las probabilidades?
# MAGIC
# MAGIC Que el modelo **ordene** bien no significa que sus probabilidades **signifiquen** algo. Si
# MAGIC la aplicación va a mostrar «65 % de probabilidad de que no haya bici», ese 65 % tiene que
# MAGIC cumplirse aproximadamente 65 de cada 100 veces.
# MAGIC
# MAGIC Se agrupan las predicciones en deciles y se compara lo prometido contra lo observado.

# COMMAND ----------

cal = pd.DataFrame({"p": prob_vacia, "y": y_vacia})
cal["bin"] = pd.qcut(cal.p, 10, labels=False, duplicates="drop")
tabla_cal = (cal.groupby("bin")
             .agg(horas=("y", "size"), prometido=("p", "mean"), observado=("y", "mean"))
             .round(4).reset_index(drop=True))
tabla_cal["desvio"] = (tabla_cal.observado - tabla_cal.prometido).round(4)

print("Calibración por deciles de probabilidad predicha\n")
print(f"{'horas':>10s} {'prometido':>11s} {'observado':>11s} {'desvío':>9s}")
for _, r in tabla_cal.iterrows():
    print(f"{int(r.horas):>10,} {r.prometido:>11.3f} {r.observado:>11.3f} {r.desvio:>+9.3f}")

peor_desvio = float(tabla_cal.desvio.abs().max())
metricas["vacia"]["peor_desvio_calibracion"] = peor_desvio
print(f"\nPeor desvío de calibración: {peor_desvio:.3f}")
print("Si está por debajo de ~0.05, la probabilidad se puede enseñar tal cual en la aplicación.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Qué variables pesan
# MAGIC
# MAGIC Importancia por permutación sobre una muestra de la partición de prueba: se revuelve una
# MAGIC columna y se mide cuánto empeora el AUC-PR. Es más honesta que la importancia interna del
# MAGIC árbol, que infla las variables con muchos valores distintos.

# COMMAND ----------

from sklearn.inspection import permutation_importance

muestra = np.random.default_rng(42).choice(len(X_pru), size=min(60_000, len(X_pru)),
                                           replace=False)
imp = permutation_importance(modelo_vacia, X_pru[muestra], y_vacia[muestra],
                             scoring="average_precision", n_repeats=3, random_state=42)
orden = np.argsort(imp.importances_mean)[::-1]
print("Caída de AUC-PR al revolver cada variable\n")
for i in orden:
    print(f"  {FEATURES[i]:<26s} {imp.importances_mean[i]:+.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. La tabla de rezagos que viaja al API
# MAGIC
# MAGIC Mismo mecanismo que en el notebook 04, y por la misma razón: el API tiene que construir
# MAGIC **exactamente** las mismas variables con las que se entrenó. Si el entrenamiento usa el
# MAGIC promedio de cuatro semanas y el API usa otra cosa, el modelo recibe números con una
# MAGIC semántica distinta de la que aprendió y falla en silencio.
# MAGIC
# MAGIC Por eso los rezagos no se recalculan en el API: se congelan aquí y viajan dentro del
# MAGIC artefacto.

# COMMAND ----------

ultimo_dia = pdf.fecha.max()
ventana = pdf[pdf.fecha > ultimo_dia - pd.Timedelta(days=28)]

rezagos = (ventana.groupby(["estacion_id", "dia_semana", "hora"])
           .agg(salidas_media_4sem=("salidas_media_4sem", "mean"),
                llegadas_media_4sem=("llegadas_media_4sem", "mean"),
                salidas_max_4sem=("salidas_max_4sem", "max"),
                bicis_inicio_media_4sem=("bicis_inicio_media_4sem", "mean"),
                bicis_inicio_min_4sem=("bicis_inicio_min_4sem", "min"),
                vacia_freq_4sem=("vacia_freq_4sem", "mean"))
           .round(4).reset_index())

capacidades = (spark.table(TBL_DISP)
               .groupBy("estacion_id")
               .agg(F.max("capacidad_anclajes").alias("capacidad_anclajes"))
               .toPandas())

print(f"Rezagos: {len(rezagos):,} combinaciones de estación, día y hora")
print(f"Ventana: {ultimo_dia - pd.Timedelta(days=28):%Y-%m-%d} a {ultimo_dia:%Y-%m-%d}")
print(f"Capacidades: {len(capacidades):,} estaciones")

# COMMAND ----------

import joblib, os

os.makedirs(RUTA_ARTEFACTO, exist_ok=True)
destino = f"{RUTA_ARTEFACTO}/modelo_disponibilidad.joblib"

joblib.dump({
    "version": VERSION_MODELO,
    "tipo": "hist_gradient_boosting",
    "features": FEATURES,
    "modelo_bicis": modelo_bicis,
    "modelo_vacia": modelo_vacia,
    "umbral": UMBRAL,
    "horario_servicio": [HORA_MIN, HORA_MAX],
    "arranque_supuesto": ARRANQUE_SUPUESTO,
    # llave (estacion_id, dia_semana, hora) -> las seis variables de rezago
    "rezagos": {(int(r.estacion_id), int(r.dia_semana), int(r.hora)):
                (float(r.salidas_media_4sem), float(r.llegadas_media_4sem),
                 float(r.salidas_max_4sem), float(r.bicis_inicio_media_4sem),
                 float(r.bicis_inicio_min_4sem), float(r.vacia_freq_4sem))
                for r in rezagos.itertuples()},
    "capacidades": {int(r.estacion_id): int(r.capacidad_anclajes)
                    for r in capacidades.itertuples()
                    if r.capacidad_anclajes is not None},
    "rezagos_ventana": f"{ultimo_dia - pd.Timedelta(days=28):%Y-%m-%d} a {ultimo_dia:%Y-%m-%d}",
    "festivos": FESTIVOS_ML,
    "metricas": metricas,
    "baseline": base_metricas,
    "entrenado_con": (f"{len(entrena):,} filas del panel de disponibilidad, "
                      f"{entrena.fecha.min():%Y-%m-%d} a {entrena.fecha.max():%Y-%m-%d}"),
    "evaluado_con": (f"{len(prueba):,} filas, "
                     f"{prueba.fecha.min():%Y-%m-%d} a {prueba.fecha.max():%Y-%m-%d}"),
    "mlflow_run_id": run_id,
}, destino, compress=3)

print(f"Artefacto: {destino}  ({os.path.getsize(destino)/2**20:.1f} MB)")
print("\nDescárgalo del Volume y déjalo en api/datos/modelo_disponibilidad.joblib")
print("El API lo detecta solo al arrancar y habilita el endpoint /disponibilidad.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Resumen para la exposición

# COMMAND ----------

print("MODELO DE DISPONIBILIDAD — resumen")
print("=" * 64)
print(f"Entrenado con : {len(entrena):,} filas  "
      f"({entrena.fecha.min():%Y-%m-%d} a {entrena.fecha.max():%Y-%m-%d})")
print(f"Evaluado con  : {len(prueba):,} filas  "
      f"({prueba.fecha.min():%Y-%m-%d} a {prueba.fecha.max():%Y-%m-%d})")
print(f"Partición     : temporal, corte {CORTE_ENTRENAMIENTO}")
print(f"Desbalance    : {tasa_pru:.2%} de horas con estación vacía")
print()
print("A · ¿Cuántas bicicletas?           modelo      línea base      mejora")
m, b = metricas["bicis"]["rmse"], base_metricas["bicis"]["rmse"]
print(f"   RMSE (bicicletas)            {m:>8.4f}      {b:>8.4f}     {1-m/b:>+7.2%}")
m, b = metricas["bicis"]["mae"], base_metricas["bicis"]["mae"]
print(f"   MAE  (bicicletas)            {m:>8.4f}      {b:>8.4f}     {1-m/b:>+7.2%}")
print()
print("B · ¿Habrá bicicletas?             modelo      línea base      mejora")
def _mejora(m, b, menor_es_mejor=False):
    # La linea base puede dar 0 exacto (por ejemplo, si nunca cruza el umbral).
    if b == 0:
        return "    n/d"
    razon = (1 - m / b) if menor_es_mejor else (m / b - 1)
    return f"{razon:>+7.2%}"

for k, et in (("auc_pr", "AUC-PR"), ("roc_auc", "ROC-AUC"), ("f1", "F1")):
    m, b = metricas["vacia"][k], base_metricas["vacia"][k]
    print(f"   {et:<28s} {m:>8.4f}      {b:>8.4f}     {_mejora(m, b)}")
m, b = metricas["vacia"]["brier"], base_metricas["vacia"]["brier"]
print(f"   {'Brier (menor es mejor)':<28s} {m:>8.4f}      {b:>8.4f}     "
      f"{_mejora(m, b, menor_es_mejor=True)}")
print()
print(f"Umbral de operación : {UMBRAL:.2f}")
print(f"Peor desvío de calib: {peor_desvio:.3f}")
print()
print("Supuesto que hay que declarar: los datos abiertos de MiBici no publican el")
print(f"inventario. La ocupación se reconstruye por balance de flujos arrancando cada")
print(f"día a {ARRANQUE_SUPUESTO:.0%} de la capacidad. El notebook 05 mide qué tanto")
print("cambia la etiqueta si se arranca a 25 % o a 75 %.")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ### Siguiente paso
# MAGIC Descargar `modelo_disponibilidad.joblib` del Volume y dejarlo en `api/datos/`. El API
# MAGIC expone el endpoint `/disponibilidad` y la página web añade la pestaña «¿Habrá bici?».
