# Databricks notebook source
# MAGIC %md
# MAGIC # Modelo de demanda — entrenamiento y evaluación
# MAGIC
# MAGIC **Proyecto:** Predicción de demanda y rebalanceo de MiBici
# MAGIC **Equipo:** César Roberto Murillo Figueroa y Andrés Muñoz Rubalcava
# MAGIC
# MAGIC ## 1. El problema
# MAGIC
# MAGIC **Tarea: regresión.** Dada una estación y una hora, predecir cuántas bicicletas van a salir
# MAGIC y cuántas van a llegar. No es clasificación: la respuesta es un conteo, y la diferencia
# MAGIC entre que falten 3 bicis o 30 es justamente la que importa para decidir si vale la pena
# MAGIC mandar una camioneta.
# MAGIC
# MAGIC Son dos modelos, uno por cada variable, porque el operador necesita las dos por separado:
# MAGIC el déficit es `salidas − llegadas`, y una estación puede tener mucho movimiento y estar en
# MAGIC equilibrio.
# MAGIC
# MAGIC ## 2. La métrica: RMSE, en bicicletas
# MAGIC
# MAGIC **RMSE** como métrica principal, por tres razones:
# MAGIC
# MAGIC 1. **Está en la unidad de la decisión.** Un RMSE de 2.4 significa "el modelo se equivoca
# MAGIC    por unas 2.4 bicicletas". Eso se puede comparar contra la capacidad de la estación y
# MAGIC    contra el costo de mandar un camión. Un R² de 0.87 no le dice nada al operador.
# MAGIC 2. **Castiga más los errores grandes**, y aquí eso es lo correcto: equivocarse por 20 bicis
# MAGIC    en la estación de Vallarta a las 6pm deja gente a pie; equivocarse por 1 bici en una
# MAGIC    estación tranquila no le importa a nadie.
# MAGIC 3. Se reporta **MAE** junto a él como control. Si RMSE ≫ MAE, el error se concentra en
# MAGIC    pocas horas extremas, y eso cambia cómo se lee el modelo.
# MAGIC
# MAGIC *Por qué no F1 ni accuracy:* no hay clases. El profesor menciona F1 para clases
# MAGIC desbalanceadas, y ese caso aquí no aplica.
# MAGIC
# MAGIC ## 3. Contra qué se compara
# MAGIC
# MAGIC **El promedio histórico de esa estación, ese día de la semana y esa hora.** Es la regla que
# MAGIC cualquiera aplicaría sin modelo, y es exactamente lo que responde el API hoy. Un GBT que no
# MAGIC le gane no justifica su existencia, y decirlo es parte del trabajo.

# COMMAND ----------

CATALOG, SCHEMA = "workspace", "mibici"
TBL_AGG = f"{CATALOG}.{SCHEMA}.gold_agg_demanda_estacion_hora"

# Partición temporal, NO aleatoria: el modelo predice el futuro, así que se entrena con el
# pasado y se evalúa con meses que nunca vio. Una partición aleatoria dejaría filtrarse el
# lunes siguiente al entrenamiento y daría una métrica inflada que no se sostiene en producción.
CORTE_ENTRENAMIENTO = "2026-08-31"

RUTA_ARTEFACTO = "/Volumes/workspace/mibici/raw/modelo"
VERSION_MODELO = "1.0.0"

# Los mismos festivos que usó gold_dim_tiempo. Viajan dentro del artefacto para que el API
# calcule `es_festivo` igual que el entrenamiento, sin tener que consultar nada.
FESTIVOS_ML = [f"{a}-{d}" for a in (2023, 2024, 2025, 2026, 2027)
               for d in ("01-01", "02-05", "03-21", "05-01", "09-16", "11-20", "12-25")]

import json
import numpy as np
import pandas as pd
from pyspark.sql import functions as F

spark.conf.set("spark.sql.ansi.enabled", "false")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Datos de entrenamiento
# MAGIC
# MAGIC Se usan los últimos 24 meses. Más atrás la red era otra: pasó de 300 a 470 estaciones
# MAGIC entre 2023 y 2026, y los patrones de 2023 ya no describen la ciudad de hoy.
# MAGIC
# MAGIC Las variables de rezago (`salidas_media_4sem`) vienen nulas en las primeras cuatro semanas
# MAGIC de cada estación, porque no hay historia suficiente. Esas filas se descartan: no se puede
# MAGIC entrenar con una feature que no existe.

# COMMAND ----------

FEATURES = ["hora", "dia_semana", "mes", "es_fin_semana", "es_festivo",
            "capacidad_anclajes", "salidas_media_4sem", "llegadas_media_4sem",
            "salidas_max_4sem"]
OBJETIVOS = ["salidas", "llegadas"]

datos = (spark.table(TBL_AGG)
         .filter("fecha >= '2024-11-01'")
         .filter("salidas_media_4sem IS NOT NULL AND llegadas_media_4sem IS NOT NULL")
         .filter("capacidad_anclajes IS NOT NULL")
         .select("estacion_id", "fecha", *FEATURES, *OBJETIVOS))

print(f"Filas utilizables: {datos.count():,}")
display(datos.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Por qué el entrenamiento sale de Spark
# MAGIC
# MAGIC Spark hizo el trabajo de volumen: 17,468,665 viajes se convirtieron en un panel agregado.
# MAGIC Ese panel ya es chico — unos pocos millones de filas y nueve columnas — y cabe de sobra en
# MAGIC la memoria del driver.
# MAGIC
# MAGIC Se entrena con **scikit-learn**, no con Spark MLlib, por una razón de despliegue: un modelo
# MAGIC de MLlib solo se puede cargar desde Spark, así que servirlo obligaría a meter una JVM y
# MAGIC Spark completo dentro del contenedor del API. El artefacto de scikit-learn pesa unos pocos
# MAGIC megabytes y arranca en milisegundos, que es lo que un servicio HTTP necesita.
# MAGIC
# MAGIC Es la división de trabajo normal: **Spark para los datos, scikit-learn para el modelo que
# MAGIC se sirve.**

# COMMAND ----------

pdf = datos.toPandas()
pdf["es_fin_semana"] = pdf["es_fin_semana"].astype(int)
pdf["es_festivo"] = pdf["es_festivo"].astype(int)
pdf["fecha"] = pd.to_datetime(pdf["fecha"])

entrena = pdf[pdf.fecha <= CORTE_ENTRENAMIENTO]
prueba  = pdf[pdf.fecha >  CORTE_ENTRENAMIENTO]

print(f"Entrenamiento: {len(entrena):>10,} filas  ({entrena.fecha.min():%Y-%m-%d} a {entrena.fecha.max():%Y-%m-%d})")
print(f"Prueba       : {len(prueba):>10,} filas  ({prueba.fecha.min():%Y-%m-%d} a {prueba.fecha.max():%Y-%m-%d})")
assert len(prueba) > 0, "La partición de prueba quedó vacía: revisa CORTE_ENTRENAMIENTO"

X_ent = entrena[FEATURES].to_numpy(dtype="float32")
X_pru = prueba[FEATURES].to_numpy(dtype="float32")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. La línea base
# MAGIC
# MAGIC El promedio histórico de las cuatro semanas previas para esa estación, ese día y esa hora.
# MAGIC Ya está calculado en el panel como `salidas_media_4sem`, así que la línea base es,
# MAGIC literalmente, usar esa columna como predicción.

# COMMAND ----------

def rmse(y, yhat):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(yhat)) ** 2)))

def mae(y, yhat):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(yhat))))

base_metricas = {}
for obj in OBJETIVOS:
    y = prueba[obj].to_numpy()
    yhat = prueba[f"{obj}_media_4sem"].to_numpy()
    base_metricas[obj] = {"rmse": rmse(y, yhat), "mae": mae(y, yhat)}
    print(f"Línea base — {obj:9s} RMSE {base_metricas[obj]['rmse']:.4f}   "
          f"MAE {base_metricas[obj]['mae']:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. El modelo
# MAGIC
# MAGIC `HistGradientBoostingRegressor`: árboles con boosting por histogramas. Se eligió porque los
# MAGIC datos son tabulares con interacciones fuertes y no lineales —la hora importa distinto entre
# MAGIC semana que en fin de semana, y distinto en cada estación—, que es justo donde los árboles
# MAGIC con boosting ganan sin necesidad de escalar ni normalizar nada.
# MAGIC
# MAGIC `early_stopping` con validación interna: el número de árboles lo decide el propio
# MAGIC entrenamiento, no una corazonada.

# COMMAND ----------

import mlflow
import mlflow.sklearn
from sklearn.ensemble import HistGradientBoostingRegressor

mlflow.set_experiment(f"/Users/{spark.sql('SELECT current_user()').first()[0]}/mibici_demanda")

modelos, metricas = {}, {}

with mlflow.start_run(run_name=f"gbt_demanda_v{VERSION_MODELO}") as run:
    mlflow.log_params({
        "algoritmo": "HistGradientBoostingRegressor",
        "features": ",".join(FEATURES),
        "corte_entrenamiento": CORTE_ENTRENAMIENTO,
        "filas_entrenamiento": len(entrena),
        "filas_prueba": len(prueba),
        "particion": "temporal",
    })

    for obj in OBJETIVOS:
        m = HistGradientBoostingRegressor(
            loss="squared_error",
            max_iter=400,
            learning_rate=0.08,
            max_depth=None,
            max_leaf_nodes=63,
            min_samples_leaf=40,
            l2_regularization=1.0,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=25,
            random_state=42,
        )
        m.fit(X_ent, entrena[obj].to_numpy())
        pred = np.clip(m.predict(X_pru), 0, None)        # no existen salidas negativas

        metricas[obj] = {
            "rmse": rmse(prueba[obj], pred),
            "mae": mae(prueba[obj], pred),
            "arboles": int(m.n_iter_),
        }
        modelos[obj] = m

        mlflow.log_metrics({
            f"{obj}_rmse": metricas[obj]["rmse"],
            f"{obj}_mae": metricas[obj]["mae"],
            f"{obj}_rmse_baseline": base_metricas[obj]["rmse"],
            f"{obj}_mejora_pct": 100 * (1 - metricas[obj]["rmse"] / base_metricas[obj]["rmse"]),
        })
        print(f"{obj:9s} RMSE {metricas[obj]['rmse']:.4f}  MAE {metricas[obj]['mae']:.4f}  "
              f"({m.n_iter_} árboles)")

    run_id = run.info.run_id

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. ¿Le ganó a la línea base?
# MAGIC
# MAGIC La tabla que decide si este modelo merece existir.

# COMMAND ----------

filas = []
for obj in OBJETIVOS:
    b, m = base_metricas[obj], metricas[obj]
    mejora = 100 * (1 - m["rmse"] / b["rmse"])
    filas.append({
        "objetivo": obj,
        "RMSE linea base": round(b["rmse"], 4),
        "RMSE modelo": round(m["rmse"], 4),
        "mejora %": round(mejora, 2),
        "MAE linea base": round(b["mae"], 4),
        "MAE modelo": round(m["mae"], 4),
        "veredicto": "el modelo gana" if mejora > 0 else "NO justifica su existencia",
    })
comparacion = pd.DataFrame(filas)
display(spark.createDataFrame(comparacion))

gana = all(metricas[o]["rmse"] < base_metricas[o]["rmse"] for o in OBJETIVOS)
if not gana:
    print("\nEl modelo NO le gana a la línea base. Hay que decirlo en la exposición y quedarse\n"
          "con la línea base, que es más simple y más barata de operar.")
else:
    print("\nEl modelo mejora la línea base en ambos objetivos.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cuánto margen había realmente
# MAGIC
# MAGIC Un RMSE de 1.88 contra una línea base de 1.91 parece una mejora pobre. Pero medir la mejora
# MAGIC contra cero es la comparación equivocada, porque **cero error es imposible**.
# MAGIC
# MAGIC La demanda de bicicletas es un conteo. Si una estación recibe en promedio λ viajes en una
# MAGIC hora, el número exacto varía por azar aunque conozcas λ con precisión perfecta. Para un
# MAGIC proceso de Poisson —el modelo estándar de conteos de eventos independientes— la varianza
# MAGIC es igual a la media, así que el error cuadrático medio mínimo alcanzable es **√λ**.
# MAGIC
# MAGIC Ese es el piso. La pregunta correcta no es "¿cuánto baja el RMSE?" sino **"¿qué fracción
# MAGIC del margen alcanzable captura el modelo?"**

# COMMAND ----------

piso = {o: float(np.sqrt(prueba[o].mean())) for o in OBJETIVOS}

print(f"{'objetivo':<10}{'piso Poisson':>14}{'linea base':>12}{'modelo':>10}"
      f"{'margen':>9}{'capturado':>11}")
print("-" * 68)
for o in OBJETIVOS:
    b, m, p = base_metricas[o]["rmse"], metricas[o]["rmse"], piso[o]
    margen = b - p                       # lo máximo que se le podía ganar a la línea base
    ganado = b - m                       # lo que efectivamente se le ganó
    pct = 100 * ganado / margen if margen > 0 else float("nan")
    print(f"{o:<10}{p:>14.3f}{b:>12.3f}{m:>10.3f}{margen:>9.3f}{pct:>10.1f}%")

print(f"""
Lectura: la línea base ya está a {base_metricas['salidas']['rmse'] - piso['salidas']:.2f} bicicletas
del piso teórico. El modelo cierra una fracción de esa distancia, no de la distancia a cero.
El resto es ruido irreducible: dos martes a las 8 de la mañana en la misma estación no
registran el mismo número de viajes ni con la mejor predicción posible.""")

# COMMAND ----------

# MAGIC ### Dónde se equivoca
# MAGIC
# MAGIC El promedio esconde lo importante. Lo que interesa es si falla justo en las horas pico,
# MAGIC que es cuando la predicción se usa.

# COMMAND ----------

ev = prueba[["hora", "dia_semana", "salidas"]].copy()
ev["pred"] = np.clip(modelos["salidas"].predict(X_pru), 0, None)
ev["base"] = prueba["salidas_media_4sem"].to_numpy()
ev["err_modelo"] = (ev.salidas - ev.pred) ** 2
ev["err_base"] = (ev.salidas - ev.base) ** 2

por_hora = (ev.groupby("hora")
            .agg(demanda_media=("salidas", "mean"),
                 rmse_modelo=("err_modelo", lambda s: float(np.sqrt(s.mean()))),
                 rmse_base=("err_base", lambda s: float(np.sqrt(s.mean()))))
            .round(3).reset_index())
por_hora["mejora_pct"] = (100 * (1 - por_hora.rmse_modelo / por_hora.rmse_base)).round(1)
display(spark.createDataFrame(por_hora))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Qué variables usa
# MAGIC
# MAGIC Importancia por permutación sobre la partición de prueba: cuánto empeora el RMSE si se
# MAGIC desordena esa columna. Es más honesta que la importancia interna del árbol, que tiende a
# MAGIC favorecer a las variables con muchos valores distintos.

# COMMAND ----------

from sklearn.inspection import permutation_importance

muestra = np.random.default_rng(42).choice(len(X_pru), size=min(40_000, len(X_pru)), replace=False)
imp = permutation_importance(modelos["salidas"], X_pru[muestra],
                             prueba["salidas"].to_numpy()[muestra],
                             n_repeats=5, random_state=42, scoring="neg_root_mean_squared_error")
display(spark.createDataFrame(
    pd.DataFrame({"feature": FEATURES,
                  "importancia": imp.importances_mean.round(4),
                  "desv": imp.importances_std.round(4)})
    .sort_values("importancia", ascending=False)))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Exportar el artefacto para el API
# MAGIC
# MAGIC El API no tiene Spark ni acceso a Unity Catalog, así que se le entrega un solo archivo con
# MAGIC todo lo que necesita: los dos modelos, el orden exacto de las features y los metadatos que
# MAGIC `/modelo` va a publicar.
# MAGIC
# MAGIC **El desajuste de las features de rezago, y cómo se resuelve.** Al predecir un viernes
# MAGIC futuro no existe el promedio de las cuatro semanas previas a *esa* fecha. La tentación es
# MAGIC usar el promedio histórico de 12 meses que ya tiene el API, pero sería un error: el modelo
# MAGIC se entrenó con una media de 4 observaciones (ruidosa) y recibiría una media de 52 (suave).
# MAGIC Es la misma columna con otro significado, y el modelo se degradaría sin avisar.
# MAGIC
# MAGIC La solución es que el artefacto **cargue su propia tabla de rezagos**, calculada sobre las
# MAGIC últimas cuatro semanas de datos — exactamente la misma ventana y la misma semántica con la
# MAGIC que se entrenó, y exactamente lo que estaría disponible en producción al predecir la
# MAGIC semana entrante.

# COMMAND ----------

ultimo_dia = pdf.fecha.max()
ventana = pdf[pdf.fecha > ultimo_dia - pd.Timedelta(days=28)]

rezagos = (ventana.groupby(["estacion_id", "dia_semana", "hora"])
           .agg(salidas_media_4sem=("salidas", "mean"),
                llegadas_media_4sem=("llegadas", "mean"),
                salidas_max_4sem=("salidas", "max"))
           .round(4).reset_index())

print(f"Tabla de rezagos: {len(rezagos):,} combinaciones de estación, día y hora")
print(f"Ventana: {ultimo_dia - pd.Timedelta(days=28):%Y-%m-%d} a {ultimo_dia:%Y-%m-%d}")

# COMMAND ----------

import joblib, os

os.makedirs(RUTA_ARTEFACTO, exist_ok=True)
destino = f"{RUTA_ARTEFACTO}/modelo_demanda.joblib"

joblib.dump({
    "version": VERSION_MODELO,
    "tipo": "hist_gradient_boosting",
    "features": FEATURES,
    "modelos": {o: modelos[o] for o in OBJETIVOS},
    # llave (estacion_id, dia_semana, hora) -> (media_salidas, media_llegadas, max_salidas)
    "rezagos": {(int(r.estacion_id), int(r.dia_semana), int(r.hora)):
                (float(r.salidas_media_4sem), float(r.llegadas_media_4sem),
                 float(r.salidas_max_4sem))
                for r in rezagos.itertuples()},
    "rezagos_ventana": f"{ultimo_dia - pd.Timedelta(days=28):%Y-%m-%d} a {ultimo_dia:%Y-%m-%d}",
    "festivos": FESTIVOS_ML,
    "metricas": metricas,
    "baseline": base_metricas,
    "entrenado_con": (f"{len(entrena):,} filas del panel, "
                      f"{entrena.fecha.min():%Y-%m-%d} a {entrena.fecha.max():%Y-%m-%d}"),
    "evaluado_con": (f"{len(prueba):,} filas, "
                     f"{prueba.fecha.min():%Y-%m-%d} a {prueba.fecha.max():%Y-%m-%d}"),
    "mlflow_run_id": run_id,
}, destino, compress=3)

print(f"Artefacto: {destino}  ({os.path.getsize(destino)/2**20:.1f} MB)")
print("\nDescárgalo del Volume y déjalo en api/datos/modelo_demanda.joblib")
print("El API lo detecta solo al arrancar y /salud pasa de 'degradado' a 'ok'.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Resumen para la exposición

# COMMAND ----------

print(f"""
TAREA        regresión — cuántas bicis salen y llegan por estación y hora
MÉTRICA      RMSE en bicicletas (MAE como control)
PARTICIÓN    temporal: entrena hasta {CORTE_ENTRENAMIENTO}, evalúa con lo posterior
MODELO       HistGradientBoostingRegressor, dos objetivos
""")
for o in OBJETIVOS:
    b, m = base_metricas[o], metricas[o]
    print(f"{o:9s}  linea base RMSE {b['rmse']:.3f}  ->  modelo RMSE {m['rmse']:.3f}  "
          f"({100*(1-m['rmse']/b['rmse']):+.1f}%)")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ### Siguiente paso
# MAGIC Bajar `modelo_demanda.joblib` del Volume, ponerlo en `api/datos/` y desplegar el API.
