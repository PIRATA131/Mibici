# Databricks notebook source
# MAGIC %md
# MAGIC # Capa Gold — Modelo analítico en esquema estrella
# MAGIC
# MAGIC **Proyecto:** Predicción de demanda y rebalanceo de MiBici
# MAGIC **Equipo:** César Roberto Murillo Figueroa y Andrés Muñoz Rubalcava
# MAGIC
# MAGIC ## Qué produce
# MAGIC
# MAGIC | Tabla | Tipo | Contenido |
# MAGIC |---|---|---|
# MAGIC | `gold_fact_viajes` | Hechos | Un renglón por viaje válido |
# MAGIC | `gold_dim_estacion` | Dimensión | 475 estaciones: nombre, municipio, coordenadas y capacidad |
# MAGIC | `gold_dim_tiempo` | Dimensión | Un renglón por hora del periodo |
# MAGIC | `gold_agg_demanda_estacion_hora` | Agregada | **La tabla que entrena el modelo**: salidas y llegadas por estación y hora |
# MAGIC
# MAGIC ## Por qué solo los válidos llegan a los hechos
# MAGIC
# MAGIC Silver conserva todo. Gold es la capa de consumo analítico, y ahí un viaje con duración
# MAGIC negativa o sin estación contaminaría cualquier agregado. Los inválidos no se pierden: siguen
# MAGIC en `silver_viajes` y en `silver_log_errores`, que es exactamente para lo que existen.

# COMMAND ----------

CATALOG, SCHEMA = "workspace", "mibici"
TBL_SILVER     = f"{CATALOG}.{SCHEMA}.silver_viajes"
TBL_ESTACIONES = f"{CATALOG}.{SCHEMA}.bronze_estaciones"

TBL_FACT    = f"{CATALOG}.{SCHEMA}.gold_fact_viajes"
TBL_DIM_EST = f"{CATALOG}.{SCHEMA}.gold_dim_estacion"
TBL_DIM_TMP = f"{CATALOG}.{SCHEMA}.gold_dim_tiempo"
TBL_AGG     = f"{CATALOG}.{SCHEMA}.gold_agg_demanda_estacion_hora"

# Horario de servicio de MiBici. Fuera de esta franja la actividad es residual (el perfilado
# encontró menos de 30 viajes en toda la red entre las 2 y las 4 de la mañana), así que incluir
# esas horas en la tabla de entrenamiento solo agregaría ceros que el modelo aprende de memoria.
HORA_MIN, HORA_MAX = 5, 23

from pyspark.sql import functions as F, Window

# Igual que en Silver: los casts del catálogo (latitud, longitud, capacidad) tienen que
# devolver NULL ante un valor raro, no lanzar excepción.
spark.conf.set("spark.sql.ansi.enabled", "false")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. `gold_dim_estacion`
# MAGIC
# MAGIC El catálogo viene con el nombre en un solo campo: `(GDL-001) C. Epigmenio Glez./ Av. 16 de Sept.`
# MAGIC De ahí se extraen la clave, el municipio y el nombre de calle.
# MAGIC
# MAGIC **Las cinco estaciones sin ficha.** Cinco identificadores aparecen en los viajes pero no en el
# MAGIC catálogo vigente de octubre 2026: las 240 y 310 se dieron de baja, y las 124, 405 y 407 son
# MAGIC altas recientes que el catálogo todavía no publica. Entre las cinco tocan 14,047 viajes.
# MAGIC Se agregan como *unknown members* en lugar de descartar esos viajes, para que ningún hecho
# MAGIC quede huérfano de dimensión.

# COMMAND ----------

MUNICIPIOS = {"GDL": "Guadalajara", "ZPN": "Zapopan", "TLQ": "Tlaquepaque",
              "SPT": "San Pedro Tlaquepaque", "TON": "Tonala", "TLJ": "Tlajomulco"}

mapa_municipio = F.create_map(*[x for k, v in MUNICIPIOS.items()
                                for x in (F.lit(k), F.lit(v))])

catalogo = (
    spark.table(TBL_ESTACIONES)
    .withColumn("estacion_id", F.col("estacion_id").cast("int"))
    .withColumn("clave_municipio", F.regexp_extract("nombre_completo", r"^\((\w{3})-(\d+)\)", 1))
    .withColumn("num_estacion",    F.regexp_extract("nombre_completo", r"^\((\w{3})-(\d+)\)", 2).cast("int"))
    .withColumn("nombre_estacion",
                F.trim(F.regexp_replace(F.regexp_replace("nombre_completo", r"^\(\w{3}-\d+\)", ""),
                                        r"\s+", " ")))
    .withColumn("clave_estacion", F.concat_ws("-", "clave_municipio",
                                              F.lpad(F.col("num_estacion").cast("string"), 3, "0")))
    .withColumn("municipio", F.coalesce(mapa_municipio[F.col("clave_municipio")],
                                        F.lit("Sin catalogar")))
    .withColumn("latitud",  F.round(F.col("latitud").cast("double"), 6))
    .withColumn("longitud", F.round(F.col("longitud").cast("double"), 6))
    .withColumn("capacidad_anclajes", F.col("capacidad_anclajes").cast("int"))
    .select("estacion_id", "clave_estacion", "clave_municipio", "municipio", "num_estacion",
            "nombre_estacion", "latitud", "longitud", "capacidad_anclajes", "nombre_completo")
)

# Estaciones que los viajes usan pero el catálogo no tiene
usadas = (spark.table(TBL_SILVER).filter("registro_valido")
          .select(F.col("origen_id").alias("estacion_id"))
          .union(spark.table(TBL_SILVER).filter("registro_valido")
                 .select(F.col("destino_id").alias("estacion_id")))
          .distinct().filter("estacion_id IS NOT NULL"))

huerfanas = (usadas.join(catalogo.select("estacion_id"), "estacion_id", "left_anti")
             .withColumn("clave_estacion", F.concat(F.lit("SIN-FICHA-"), F.col("estacion_id")))
             .withColumn("clave_municipio", F.lit(None).cast("string"))
             .withColumn("municipio", F.lit("Sin catalogar"))
             .withColumn("num_estacion", F.lit(None).cast("int"))
             .withColumn("nombre_estacion",
                         F.concat(F.lit("Estacion "), F.col("estacion_id"),
                                  F.lit(" sin ficha en el catalogo vigente")))
             .withColumn("latitud", F.lit(None).cast("double"))
             .withColumn("longitud", F.lit(None).cast("double"))
             .withColumn("capacidad_anclajes", F.lit(None).cast("int"))
             .withColumn("nombre_completo", F.lit(None).cast("string"))
             .select(*catalogo.columns))

dim_estacion = catalogo.unionByName(huerfanas).orderBy("estacion_id")
dim_estacion.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(TBL_DIM_EST)

n_cat = catalogo.count()
n_hue = huerfanas.count()
print(f"gold_dim_estacion: {n_cat} del catálogo + {n_hue} sin ficha = {n_cat + n_hue}")
assert n_cat == 470, f"Se esperaban 470 estaciones en el catálogo, hay {n_cat}"
display(spark.table(TBL_DIM_EST).groupBy("municipio").count().orderBy(F.desc("count")))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. `gold_fact_viajes`
# MAGIC
# MAGIC Solo los válidos, con las llaves hacia las dimensiones y las medidas del hecho.

# COMMAND ----------

fact = (spark.table(TBL_SILVER)
        .filter("registro_valido")
        .select(
            "viaje_sk", "viaje_id", "usuario_id",
            F.col("origen_id").alias("estacion_origen_id"),
            F.col("destino_id").alias("estacion_destino_id"),
            F.date_trunc("hour", "inicio_viaje").alias("tiempo_sk"),
            "inicio_viaje", "fin_viaje", "duracion_min",
            "genero", "anio_nacimiento", "edad_usuario",
            "anio", "mes", "dia", "hora", "dia_semana", "es_fin_semana",
            F.when(F.col("origen_id") == F.col("destino_id"), True)
             .otherwise(False).alias("es_viaje_circular"),
        ))

fact.write.mode("overwrite").option("overwriteSchema", "true") \
    .partitionBy("anio").saveAsTable(TBL_FACT)

# Los hechos tienen que ser EXACTAMENTE los válidos de Silver: ni uno más, ni uno menos.
# Se compara contra la tabla, no contra un número escrito a mano, para que la aserción
# siga siendo cierta si una regla de Silver cambia.
n_validos_silver = spark.table(TBL_SILVER).filter("registro_valido").count()
n_fact = spark.table(TBL_FACT).count()
print(f"gold_fact_viajes : {n_fact:,} viajes")
print(f"validos en Silver: {n_validos_silver:,}")
assert n_fact == n_validos_silver, (
    f"Gold tiene {n_fact:,} hechos pero Silver tiene {n_validos_silver:,} válidos")

# viaje_sk es la llave del hecho: tiene que ser única aquí sin excepción.
n_sk = spark.table(TBL_FACT).select("viaje_sk").distinct().count()
assert n_sk == n_fact, f"viaje_sk repetido en los hechos: {n_fact - n_sk:,} colisiones"
print(f"viaje_sk unico   : {n_sk:,}  OK")

# integridad referencial: ningún hecho puede apuntar a una estación que no exista
huerfanos = spark.sql(f"""
    SELECT COUNT(*) AS n FROM {TBL_FACT} f
    LEFT ANTI JOIN {TBL_DIM_EST} d ON f.estacion_origen_id = d.estacion_id
""").first()["n"]
print(f"Hechos sin dimensión de origen: {huerfanos}")
assert huerfanos == 0, "Hay hechos huérfanos: la dimensión está incompleta"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. `gold_dim_tiempo`
# MAGIC
# MAGIC Una fila por hora del periodo observado. Incluye la marca de festivo nacional, porque un
# MAGIC 16 de septiembre no se comporta como un martes cualquiera y el modelo necesita poder
# MAGIC distinguirlos.

# COMMAND ----------

rango = spark.table(TBL_FACT).select(
    F.min(F.to_date("inicio_viaje")).alias("desde"),
    F.max(F.to_date("inicio_viaje")).alias("hasta")).first()
print(f"Periodo: {rango['desde']} a {rango['hasta']}")

# Festivos oficiales de México en el periodo (los de ley, no los puentes bancarios)
FESTIVOS = [f"{a}-{d}" for a in (2023, 2024, 2025, 2026)
            for d in ("01-01", "02-05", "03-21", "05-01", "09-16", "11-20", "12-25")]

dim_tiempo = (
    spark.sql(f"SELECT explode(sequence(to_timestamp('{rango['desde']} 00:00:00'), "
              f"to_timestamp('{rango['hasta']} 23:00:00'), interval 1 hour)) AS tiempo_sk")
    .withColumn("fecha",        F.to_date("tiempo_sk"))
    .withColumn("anio",         F.year("tiempo_sk"))
    .withColumn("mes",          F.month("tiempo_sk"))
    .withColumn("dia",          F.dayofmonth("tiempo_sk"))
    .withColumn("hora",         F.hour("tiempo_sk"))
    .withColumn("dia_semana",   (F.dayofweek("tiempo_sk") + 5) % 7)
    .withColumn("nombre_dia",   F.date_format("tiempo_sk", "EEEE"))
    .withColumn("es_fin_semana", F.col("dia_semana").isin(5, 6))
    .withColumn("es_festivo",   F.date_format("fecha", "yyyy-MM-dd").isin(FESTIVOS))
    .withColumn("semana_anio",  F.weekofyear("tiempo_sk"))
    .withColumn("en_horario_servicio", F.col("hora").between(HORA_MIN, HORA_MAX))
)

dim_tiempo.write.mode("overwrite").option("overwriteSchema", "true") \
    .partitionBy("anio").saveAsTable(TBL_DIM_TMP)
print(f"gold_dim_tiempo: {spark.table(TBL_DIM_TMP).count():,} horas")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. `gold_agg_demanda_estacion_hora` — la tabla de entrenamiento
# MAGIC
# MAGIC Aquí está la decisión de diseño que hay que saber defender: **el panel tiene que incluir los
# MAGIC ceros**. Si solo se agregan las horas con viajes, el modelo nunca ve una hora tranquila y
# MAGIC sobreestima la demanda en todas partes.
# MAGIC
# MAGIC Pero densificar a lo bruto también está mal: la red creció de 300 a 470 estaciones entre 2023
# MAGIC y 2026, y una estación que se inauguró en 2026 no tuvo "cero demanda" en 2023 — simplemente
# MAGIC no existía. Por eso cada estación se densifica **solo dentro de su propia vida útil**, del
# MAGIC primer al último día en que registró movimiento.

# COMMAND ----------

f = spark.table(TBL_FACT)

salidas = (f.groupBy(F.col("estacion_origen_id").alias("estacion_id"), "tiempo_sk")
            .agg(F.count("*").alias("salidas"),
                 F.round(F.avg("duracion_min"), 2).alias("duracion_media_salida")))

llegadas = (f.groupBy(F.col("estacion_destino_id").alias("estacion_id"), "tiempo_sk")
             .agg(F.count("*").alias("llegadas")))

observado = salidas.join(llegadas, ["estacion_id", "tiempo_sk"], "outer")

# vida útil de cada estación
vida = (observado.groupBy("estacion_id")
        .agg(F.min(F.to_date("tiempo_sk")).alias("primer_dia"),
             F.max(F.to_date("tiempo_sk")).alias("ultimo_dia")))

# panel denso: cada estación x cada hora de servicio dentro de su vida útil
panel = (vida
         .withColumn("fecha", F.explode(F.expr("sequence(primer_dia, ultimo_dia, interval 1 day)")))
         .withColumn("hora", F.explode(F.array(*[F.lit(h) for h in range(HORA_MIN, HORA_MAX + 1)])))
         .withColumn("tiempo_sk", F.expr("fecha + make_interval(0,0,0,0,hora,0,0)"))
         .select("estacion_id", "tiempo_sk"))

agg = (panel.join(observado, ["estacion_id", "tiempo_sk"], "left")
       .na.fill({"salidas": 0, "llegadas": 0})
       .join(spark.table(TBL_DIM_TMP).select(
                 "tiempo_sk", "fecha", "anio", "mes", "dia", "hora",
                 "dia_semana", "es_fin_semana", "es_festivo", "semana_anio"),
             "tiempo_sk", "left")
       .join(spark.table(TBL_DIM_EST).select(
                 "estacion_id", "clave_estacion", "municipio", "capacidad_anclajes"),
             "estacion_id", "left")
       .withColumn("balance", F.col("llegadas") - F.col("salidas"))
       .withColumn("deficit_bicis", F.greatest(F.lit(0), -F.col("balance")))
       .withColumn("saturacion",
                   F.when(F.col("capacidad_anclajes") > 0,
                          F.round(F.col("deficit_bicis") / F.col("capacidad_anclajes"), 4))))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Variables de rezago
# MAGIC
# MAGIC La demanda de una estación a las 8 de la mañana del martes se parece muchísimo a la de los
# MAGIC martes anteriores. Darle al modelo el promedio de las cuatro semanas previas en esa misma
# MAGIC franja le ahorra tener que redescubrirlo, y es información que en producción sí estaría
# MAGIC disponible: solo mira al pasado.

# COMMAND ----------

v = (Window.partitionBy("estacion_id", "dia_semana", "hora")
     .orderBy(F.col("fecha").cast("long"))
     .rangeBetween(-4 * 7 * 86400, -7 * 86400))     # de 4 semanas atrás a 1 semana atrás

agg_final = (agg
             .withColumn("salidas_media_4sem",  F.round(F.avg("salidas").over(v), 3))
             .withColumn("llegadas_media_4sem", F.round(F.avg("llegadas").over(v), 3))
             .withColumn("salidas_max_4sem",    F.max("salidas").over(v))
             .select("estacion_id", "clave_estacion", "municipio", "capacidad_anclajes",
                     "tiempo_sk", "fecha", "anio", "mes", "dia", "hora", "dia_semana",
                     "es_fin_semana", "es_festivo", "semana_anio",
                     "salidas", "llegadas", "balance", "deficit_bicis", "saturacion",
                     "duracion_media_salida",
                     "salidas_media_4sem", "llegadas_media_4sem", "salidas_max_4sem"))

agg_final.write.mode("overwrite").option("overwriteSchema", "true") \
         .partitionBy("anio").saveAsTable(TBL_AGG)

n_agg = spark.table(TBL_AGG).count()
print(f"gold_agg_demanda_estacion_hora: {n_agg:,} filas")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Comprobaciones

# COMMAND ----------

# Los viajes del panel tienen que sumar lo mismo que los hechos, dentro del horario de servicio.
en_servicio = spark.table(TBL_FACT).filter(F.col("hora").between(HORA_MIN, HORA_MAX)).count()
suma_panel = spark.table(TBL_AGG).agg(F.sum("salidas")).first()[0]
print(f"Viajes en horario de servicio: {en_servicio:,}")
print(f"Suma de salidas en el panel  : {suma_panel:,}")
print(f"Diferencia                   : {en_servicio - suma_panel:+,}")
assert en_servicio == suma_panel, "El panel perdió o duplicó viajes al densificar"

pct_cero = spark.table(TBL_AGG).filter("salidas = 0").count() / n_agg
print(f"\nProporción de horas sin salidas: {pct_cero:.1%}  "
      f"(son reales: estaciones tranquilas en horas tranquilas)")

display(spark.sql(f"""
    SELECT anio, COUNT(*) AS filas, COUNT(DISTINCT estacion_id) AS estaciones,
           ROUND(AVG(salidas), 2) AS salidas_media,
           MAX(salidas) AS salidas_max,
           ROUND(AVG(deficit_bicis), 2) AS deficit_medio
    FROM {TBL_AGG} GROUP BY anio ORDER BY anio
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. El problema, visto en los datos
# MAGIC
# MAGIC Las diez combinaciones de estación y franja con mayor déficit medio. Esto es lo que el
# MAGIC modelo tiene que aprender a anticipar, y lo que la página web va a mostrar en el mapa.

# COMMAND ----------

display(spark.sql(f"""
    SELECT e.clave_estacion, e.nombre_estacion, a.dia_semana, a.hora,
           ROUND(AVG(a.salidas), 1)       AS salidas_medias,
           ROUND(AVG(a.llegadas), 1)      AS llegadas_medias,
           ROUND(AVG(a.deficit_bicis), 1) AS deficit_medio,
           e.capacidad_anclajes,
           ROUND(AVG(a.deficit_bicis) / e.capacidad_anclajes, 2) AS veces_que_se_vacia
    FROM {TBL_AGG} a
    JOIN {TBL_DIM_EST} e USING (estacion_id)
    WHERE a.anio = 2026 AND NOT a.es_fin_semana AND e.capacidad_anclajes > 0
    GROUP BY e.clave_estacion, e.nombre_estacion, a.dia_semana, a.hora, e.capacidad_anclajes
    ORDER BY deficit_medio DESC
    LIMIT 10
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ### Siguiente paso
# MAGIC `04_modelo_demanda.py` — entrenar el GBT, medirlo contra la línea base histórica y exportar
# MAGIC el artefacto que consume el API.
