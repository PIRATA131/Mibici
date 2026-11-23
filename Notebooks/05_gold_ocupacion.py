# Databricks notebook source
# MAGIC %md
# MAGIC # Gold — reconstrucción de la ocupación por estación y hora
# MAGIC
# MAGIC **Proyecto:** Predicción de demanda y rebalanceo de MiBici
# MAGIC **Equipo:** César Roberto Murillo Figueroa y Andrés Muñoz Rubalcava
# MAGIC
# MAGIC ## El problema que resuelve este notebook
# MAGIC
# MAGIC El usuario de MiBici quiere saber algo muy concreto: *«voy a ir a esta estación el viernes
# MAGIC a las 6, ¿va a haber bicicletas?»*. Para responderlo hace falta el **inventario** de la
# MAGIC estación: cuántas bicicletas hay ancladas en ese momento.
# MAGIC
# MAGIC **Y los datos abiertos de MiBici no publican el inventario.** Publican viajes: quién salió,
# MAGIC de dónde, a qué hora y a dónde llegó. Nunca vemos cuántas bicicletas había paradas.
# MAGIC
# MAGIC Entonces la ocupación no se puede *medir*. Se puede **reconstruir**, y eso es lo que hace
# MAGIC este notebook.
# MAGIC
# MAGIC ## Cómo se reconstruye
# MAGIC
# MAGIC Un balance de flujos, hora por hora, acotado por la capacidad física de la estación:
# MAGIC
# MAGIC ```
# MAGIC bicis_fin(h) = recortar( bicis_inicio(h) + llegadas(h) − salidas(h),  0,  capacidad )
# MAGIC bicis_inicio(h+1) = bicis_fin(h)
# MAGIC ```
# MAGIC
# MAGIC El recorte no es un detalle cosmético: es lo que convierte una cuenta aritmética en un
# MAGIC modelo físico. Una estación no puede tener −4 bicicletas ni 30 bicicletas en 19 anclajes.
# MAGIC
# MAGIC ## Las dos cosas que hay que saber defender
# MAGIC
# MAGIC **1. El supuesto de arranque.** Cada día empieza con un inventario que no observamos.
# MAGIC Suponemos que MiBici rebalancea de noche y que cada estación amanece a media capacidad.
# MAGIC Para no quedarnos con el supuesto a ciegas, la reconstrucción se corre **tres veces en la
# MAGIC misma pasada**, arrancando a 25 %, 50 % y 75 % de la capacidad, y medimos en qué porcentaje
# MAGIC de las horas las tres trayectorias dan la misma respuesta a «¿se quedó sin bicis?».
# MAGIC Si el número es alto, el supuesto importa poco y podemos decirlo con cifras.
# MAGIC
# MAGIC **2. La demanda censurada.** Cuando una estación se vacía, la gente que llega y no
# MAGIC encuentra bici *no deja rastro en los datos*: no hay viaje que registrar. Así que las
# MAGIC salidas observadas son un piso de la demanda real, no la demanda real. Eso tiene una
# MAGIC consecuencia honesta que hay que decir: el modelo puede afirmar **que no habrá bicis**,
# MAGIC pero no puede decir **a cuánta gente dejó a pie**.
# MAGIC
# MAGIC ## Qué produce
# MAGIC
# MAGIC `gold_disponibilidad_estacion_hora` — el panel de demanda ya existente, más la ocupación
# MAGIC reconstruida y las variables de rezago que el modelo de disponibilidad va a consumir.

# COMMAND ----------

CATALOG, SCHEMA = "workspace", "mibici"
TBL_AGG  = f"{CATALOG}.{SCHEMA}.gold_agg_demanda_estacion_hora"
TBL_DISP = f"{CATALOG}.{SCHEMA}.gold_disponibilidad_estacion_hora"

# El mismo horario de servicio que usó 03_gold_estrella.
HORA_MIN, HORA_MAX = 5, 23

# Fracciones de capacidad con las que arranca cada día. La primera es la oficial; las otras
# dos existen solo para medir cuánto depende el resultado de ese supuesto.
ARRANQUES = (0.50, 0.25, 0.75)

import numpy as np
import pandas as pd
from pyspark.sql import functions as F, types as T, Window

spark.conf.set("spark.sql.ansi.enabled", "false")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. El insumo
# MAGIC
# MAGIC Solo las estaciones con capacidad conocida: sin capacidad no hay nada que recortar y el
# MAGIC balance pierde el sentido físico. Se cuenta cuántas quedan fuera, para reportarlo.

# COMMAND ----------

panel = spark.table(TBL_AGG)

n_total = panel.count()
n_sin_cap = panel.filter("capacidad_anclajes IS NULL OR capacidad_anclajes <= 0").count()
est_sin_cap = (panel.filter("capacidad_anclajes IS NULL OR capacidad_anclajes <= 0")
               .select("estacion_id").distinct().count())

print(f"Panel completo           : {n_total:,} filas")
print(f"Sin capacidad conocida   : {n_sin_cap:,} filas  ({est_sin_cap} estaciones)")
print(f"Entran a la reconstrucción: {n_total - n_sin_cap:,} filas")

base = (panel
        .filter("capacidad_anclajes IS NOT NULL AND capacidad_anclajes > 0")
        .select("estacion_id", "clave_estacion", "fecha", "hora", "anio", "mes",
                "dia_semana", "es_fin_semana", "es_festivo", "capacidad_anclajes",
                "salidas", "llegadas",
                "salidas_media_4sem", "llegadas_media_4sem", "salidas_max_4sem"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. La reconstrucción
# MAGIC
# MAGIC El balance es **secuencial**: la ocupación de las 10 depende de la de las 9. Eso no se
# MAGIC puede expresar con una función de ventana, porque el recorte a `[0, capacidad]` rompe la
# MAGIC acumulación — una vez que la estación toca cero, lo que pasó antes deja de sumarse.
# MAGIC
# MAGIC Así que se agrupa por estación y día (19 filas por grupo, una por hora de servicio) y cada
# MAGIC grupo se recorre con `applyInPandas`. Spark reparte los grupos entre los ejecutores; cada
# MAGIC uno hace un bucle de 19 pasos.
# MAGIC
# MAGIC Las tres trayectorias se calculan **en el mismo bucle**. Medir la sensibilidad al supuesto
# MAGIC no cuesta una pasada extra, cuesta dos variables más.

# COMMAND ----------

ESQUEMA = T.StructType([
    T.StructField("estacion_id", T.IntegerType()),
    T.StructField("fecha", T.DateType()),
    T.StructField("hora", T.IntegerType()),
    T.StructField("bicis_inicio", T.DoubleType()),
    T.StructField("bicis_fin", T.DoubleType()),
    T.StructField("se_vacio", T.BooleanType()),
    T.StructField("se_lleno", T.BooleanType()),
    T.StructField("se_vacio_a25", T.BooleanType()),
    T.StructField("se_vacio_a75", T.BooleanType()),
])


def reconstruir_dia(g: pd.DataFrame) -> pd.DataFrame:
    """Balance de flujos de un día de una estación. Devuelve una fila por hora."""
    g = g.sort_values("hora")
    cap = float(g["capacidad_anclajes"].iloc[0])
    flujo = (g["llegadas"] - g["salidas"]).to_numpy(dtype="float64")

    # Una trayectoria por supuesto de arranque. La primera es la oficial.
    x = [cap * a for a in ARRANQUES]
    inicio = np.empty(len(flujo))
    fin = np.empty(len(flujo))
    vacio = np.empty(len(flujo), dtype=bool)
    lleno = np.empty(len(flujo), dtype=bool)
    vacio25 = np.empty(len(flujo), dtype=bool)
    vacio75 = np.empty(len(flujo), dtype=bool)

    for i, f in enumerate(flujo):
        inicio[i] = x[0]
        for k in range(len(x)):
            x[k] = min(cap, max(0.0, x[k] + f))
        fin[i] = x[0]
        vacio[i] = x[0] <= 0.0
        lleno[i] = x[0] >= cap
        vacio25[i] = x[1] <= 0.0
        vacio75[i] = x[2] <= 0.0

    return pd.DataFrame({
        "estacion_id": g["estacion_id"].to_numpy(),
        "fecha": g["fecha"].to_numpy(),
        "hora": g["hora"].to_numpy(dtype="int32"),
        "bicis_inicio": inicio,
        "bicis_fin": fin,
        "se_vacio": vacio,
        "se_lleno": lleno,
        "se_vacio_a25": vacio25,
        "se_vacio_a75": vacio75,
    })


ocupacion = base.groupBy("estacion_id", "fecha").applyInPandas(reconstruir_dia, ESQUEMA)
ocupacion.cache()
n_ocupacion = ocupacion.count()
print(f"Ocupación reconstruida: {n_ocupacion:,} filas")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Comprobaciones
# MAGIC
# MAGIC Lo que *tiene* que ser cierto se afirma; lo que estamos *midiendo* se reporta.

# COMMAND ----------

# 3.1 — No se perdió ni se duplicó ninguna fila.
n_base = base.count()
print(f"Filas de entrada : {n_base:,}")
print(f"Filas de salida  : {n_ocupacion:,}")
assert n_base == n_ocupacion, (
    f"La reconstrucción cambió el número de filas en {n_ocupacion - n_base:+,}")
print("La reconstrucción conserva una fila por estación, día y hora.")

# COMMAND ----------

# 3.2 — La ocupación nunca sale de [0, capacidad]. Si saliera, el recorte estaría mal escrito
# y toda la tabla sería basura con aspecto razonable.
limites = (ocupacion.join(base.select("estacion_id", "fecha", "hora", "capacidad_anclajes"),
                          ["estacion_id", "fecha", "hora"])
           .agg(F.min("bicis_fin").alias("minimo"),
                F.max(F.col("bicis_fin") - F.col("capacidad_anclajes")).alias("exceso"))
           .first())
print(f"Ocupación mínima          : {limites['minimo']:.2f}")
print(f"Exceso máximo sobre la cap: {limites['exceso']:.2f}")
assert limites["minimo"] >= 0, "Hay ocupaciones negativas: el recorte inferior no funcionó"
assert limites["exceso"] <= 0, "Hay ocupaciones sobre la capacidad: el recorte superior no funcionó"
print("La ocupación se mantiene dentro de los límites físicos de la estación.")

# COMMAND ----------

# 3.3 — Conservación del flujo donde no hubo recorte.
# En una hora que no tocó ningún límite, bicis_fin − bicis_inicio tiene que ser exactamente
# llegadas − salidas. Si no cuadra, el balance está mal planteado.
sin_recorte = (ocupacion
               .join(base.select("estacion_id", "fecha", "hora", "salidas", "llegadas"),
                     ["estacion_id", "fecha", "hora"])
               .filter("NOT se_vacio AND NOT se_lleno")
               .withColumn("desajuste",
                           F.abs((F.col("bicis_fin") - F.col("bicis_inicio"))
                                 - (F.col("llegadas") - F.col("salidas")))))
peor = sin_recorte.agg(F.max("desajuste")).first()[0]
n_sin_recorte = sin_recorte.count()
print(f"Horas sin tocar ningún límite: {n_sin_recorte:,}")
print(f"Peor desajuste del balance   : {peor:.10f}")
assert peor < 1e-9, "El balance de flujos no se conserva donde no hubo recorte"
print("Donde no hay recorte, el balance cuadra exactamente.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.4 — ¿Cuánto importa el supuesto de arranque?
# MAGIC
# MAGIC Esta es la medición que vuelve defendible toda la reconstrucción. Comparamos la etiqueta
# MAGIC «se quedó sin bicis» de las tres trayectorias. Si arrancar a 25 %, 50 % o 75 % de la
# MAGIC capacidad da casi siempre la misma respuesta, el supuesto es casi irrelevante y la
# MAGIC etiqueta no es un artefacto de lo que decidimos suponer.

# COMMAND ----------

acuerdo = ocupacion.select(
    F.avg(F.when(F.col("se_vacio") == F.col("se_vacio_a25"), 1.0).otherwise(0.0)).alias("vs_25"),
    F.avg(F.when(F.col("se_vacio") == F.col("se_vacio_a75"), 1.0).otherwise(0.0)).alias("vs_75"),
    F.avg(F.when((F.col("se_vacio") == F.col("se_vacio_a25")) &
                 (F.col("se_vacio") == F.col("se_vacio_a75")), 1.0).otherwise(0.0)).alias("las_tres"),
    F.avg(F.col("se_vacio").cast("double")).alias("tasa_vacio"),
).first()

print(f"Arrancar a 50 % vs 25 %  : coinciden en {acuerdo['vs_25']:.2%} de las horas")
print(f"Arrancar a 50 % vs 75 %  : coinciden en {acuerdo['vs_75']:.2%} de las horas")
print(f"Las tres de acuerdo      : {acuerdo['las_tres']:.2%}")
print(f"\nHoras con estación vacía : {acuerdo['tasa_vacio']:.2%} del panel")
print("\nEste número va al README y a la exposición: dice qué tan frágil es el supuesto.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Variables de rezago
# MAGIC
# MAGIC Mismo criterio que en `03_gold_estrella`: solo miran al pasado, para que en producción
# MAGIC estén realmente disponibles. El usuario pregunta hoy por el viernes que viene; lo único
# MAGIC que sabemos entonces son los viernes anteriores.
# MAGIC
# MAGIC Dos variables nuevas, las que de verdad informan la disponibilidad:
# MAGIC
# MAGIC - `bicis_inicio_media_4sem` — cuántas bicicletas solía haber en esa franja.
# MAGIC - `vacia_freq_4sem` — con qué frecuencia se vació esa franja.
# MAGIC
# MAGIC La segunda es, además, **la línea base contra la que se mide el clasificador**: predecir la
# MAGIC frecuencia histórica es lo que haría cualquiera sin modelo.

# COMMAND ----------

completo = ocupacion.join(base, ["estacion_id", "fecha", "hora"])

v = (Window.partitionBy("estacion_id", "dia_semana", "hora")
     .orderBy(F.col("fecha").cast("timestamp").cast("long"))
     .rangeBetween(-4 * 7 * 86400, -7 * 86400))     # de 4 semanas atrás a 1 semana atrás

disponibilidad = (completo
                  .withColumn("bicis_inicio_media_4sem",
                              F.round(F.avg("bicis_inicio").over(v), 3))
                  .withColumn("bicis_inicio_min_4sem",
                              F.round(F.min("bicis_inicio").over(v), 3))
                  .withColumn("vacia_freq_4sem",
                              F.round(F.avg(F.col("se_vacio").cast("double")).over(v), 4))
                  .withColumn("ocupacion_relativa",
                              F.round(F.col("bicis_inicio") / F.col("capacidad_anclajes"), 4))
                  .select("estacion_id", "clave_estacion", "fecha", "anio", "mes", "hora",
                          "dia_semana", "es_fin_semana", "es_festivo", "capacidad_anclajes",
                          "salidas", "llegadas",
                          "bicis_inicio", "bicis_fin", "ocupacion_relativa",
                          "se_vacio", "se_lleno",
                          "salidas_media_4sem", "llegadas_media_4sem", "salidas_max_4sem",
                          "bicis_inicio_media_4sem", "bicis_inicio_min_4sem", "vacia_freq_4sem"))

disponibilidad.write.mode("overwrite").option("overwriteSchema", "true") \
              .partitionBy("anio").saveAsTable(TBL_DISP)

n_disp = spark.table(TBL_DISP).count()
print(f"{TBL_DISP}: {n_disp:,} filas")
assert n_disp == n_ocupacion, "La tabla final no tiene las mismas filas que la reconstrucción"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Qué quedó

# COMMAND ----------

display(spark.sql(f"""
    SELECT anio,
           COUNT(*)                                   AS filas,
           COUNT(DISTINCT estacion_id)                AS estaciones,
           ROUND(AVG(bicis_inicio), 2)                AS bicis_medias,
           ROUND(AVG(ocupacion_relativa), 3)          AS ocupacion_media,
           ROUND(AVG(CAST(se_vacio AS DOUBLE)), 4)    AS frac_horas_vacia,
           ROUND(AVG(CAST(se_lleno AS DOUBLE)), 4)    AS frac_horas_llena
    FROM {TBL_DISP}
    GROUP BY anio ORDER BY anio
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Las franjas donde de verdad no vas a encontrar bici
# MAGIC
# MAGIC Esto es lo que la aplicación va a responder, y conviene mirarlo antes de modelar nada:
# MAGIC si la tabla no tiene sentido para alguien que conoce Guadalajara, el modelo tampoco lo
# MAGIC va a tener.

# COMMAND ----------

display(spark.sql(f"""
    SELECT clave_estacion,
           CASE dia_semana WHEN 0 THEN 'lunes' WHEN 1 THEN 'martes' WHEN 2 THEN 'miércoles'
                           WHEN 3 THEN 'jueves' WHEN 4 THEN 'viernes' WHEN 5 THEN 'sábado'
                           ELSE 'domingo' END         AS dia,
           hora,
           capacidad_anclajes,
           COUNT(*)                                   AS dias_observados,
           ROUND(AVG(CAST(se_vacio AS DOUBLE)), 3)    AS prob_sin_bicis,
           ROUND(AVG(bicis_inicio), 1)                AS bicis_medias
    FROM {TBL_DISP}
    WHERE anio = 2026
    GROUP BY clave_estacion, dia_semana, hora, capacidad_anclajes
    HAVING COUNT(*) >= 20
    ORDER BY prob_sin_bicis DESC, dias_observados DESC
    LIMIT 15
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ### Siguiente paso
# MAGIC `06_modelo_disponibilidad.py` — entrenar los dos modelos que responden «¿habrá bicis y
# MAGIC cuántas?», medirlos contra la línea base histórica y exportar el segundo artefacto del API.
