# Databricks notebook source
# MAGIC %md
# MAGIC # Capa Silver — Limpieza y reglas de negocio
# MAGIC
# MAGIC **Proyecto:** Predicción de demanda y rebalanceo de MiBici
# MAGIC **Materia:** Aprendizaje Automático para Grandes Volúmenes de Datos — Dr. Ramón Alejandro Briseño
# MAGIC **Equipo:** César Roberto Murillo Figueroa y Andrés Muñoz Rubalcava
# MAGIC
# MAGIC ## Qué hace este notebook
# MAGIC
# MAGIC Toma `bronze_viajes` (17,468,665 filas, todo `STRING`) y produce:
# MAGIC
# MAGIC | Tabla | Contenido |
# MAGIC |---|---|
# MAGIC | `silver_viajes` | Mismas 17,468,665 filas, tipadas, con fechas ISO, variables derivadas y una bandera `registro_valido` |
# MAGIC | `silver_log_errores` | Una fila por incidencia: `columna`, `mensaje_error`, `valor` |
# MAGIC
# MAGIC ## El principio de Silver: marcar, no borrar
# MAGIC
# MAGIC Un registro que viola una regla **no se elimina**. Se conserva en `silver_viajes` con
# MAGIC `registro_valido = false` y cada violación se escribe en el log de errores. Es la regla RN-007
# MAGIC del taller RUES del profesor, y existe por una razón práctica: si se borran, nadie puede
# MAGIC auditar después *por qué* desapareció un viaje, ni medir si una regla está siendo demasiado
# MAGIC agresiva.
# MAGIC
# MAGIC ## Cómo se evita que la bandera y el log se contradigan
# MAGIC
# MAGIC Las dos salen del **mismo arreglo**. Se construye un `array` con una entrada por regla violada;
# MAGIC `registro_valido` es `size(array) == 0` y el log es el `explode` de ese mismo arreglo. No hay
# MAGIC forma de que una diga una cosa y el otro diga otra.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Configuración y cifras de control
# MAGIC
# MAGIC Estas cifras salieron del perfilado previo hecho en pandas sobre los CSV originales. Si la
# MAGIC implementación en PySpark es correcta, tiene que reproducirlas **exactamente**. Esa es la
# MAGIC prueba de que la capa Silver hace lo que dice.

# COMMAND ----------

CATALOG, SCHEMA = "workspace", "mibici"
TBL_BRONZE = f"{CATALOG}.{SCHEMA}.bronze_viajes"
TBL_SILVER = f"{CATALOG}.{SCHEMA}.silver_viajes"
TBL_LOG    = f"{CATALOG}.{SCHEMA}.silver_log_errores"

# Umbrales de las reglas
DUR_MIN_SEG   = 60          # menos de 1 minuto: falso retiro
DUR_MAX_SEG   = 24 * 3600   # más de 24 h: bici no devuelta
EDAD_MIN, EDAD_MAX = 10, 100

# Resultado esperado del perfilado en pandas sobre los CSV originales.
CTRL_TOTAL = 17_468_665
CTRL_DUPLICADOS = 758          # filas repetidas exactas que detecta RN-08

# El perfilado se hizo ANTES de descubrir los duplicados, así que su conteo de válidos no
# los descontaba. El número definitivo lo establece esta corrida, y tiene que caer dentro
# de [CTRL_VALIDOS_SIN_DEDUP - 758, CTRL_VALIDOS_SIN_DEDUP]: las 758 copias o ya estaban
# marcadas por otra regla, o se vuelven inválidas ahora.
CTRL_VALIDOS_SIN_DEDUP = 16_751_269
CTRL_INCIDENCIAS_6 = 722_915   # suma de las seis reglas del perfilado

CTRL_POR_REGLA = {
    "Duracion menor a 60 segundos":            549_047,
    "Genero nulo o fuera del dominio {M,F,O}": 115_563,
    "Fin del viaje menor o igual al inicio":    44_800,
    "Anio de nacimiento nulo":                  11_005,
    "Edad fuera del rango [10,100]":             2_456,
    "Duracion mayor a 24 horas":                    44,
    "Registro duplicado exacto":                   758,   # RN-08
}
assert sum(CTRL_POR_REGLA.values()) == CTRL_INCIDENCIAS_6 + CTRL_DUPLICADOS

# Toda la capa Silver descansa en una premisa: un dato malo se convierte en NULL y se marca
# como incidencia; NUNCA tumba el job. Eso exige desactivar el modo ANSI, porque con ANSI
# activo un cast fallido o una fecha con formato inesperado lanzan excepción en vez de
# devolver NULL — y en los runtimes recientes de Databricks ANSI viene encendido por omisión.
spark.conf.set("spark.sql.ansi.enabled", "false")
spark.conf.set("spark.sql.legacy.timeParserPolicy", "CORRECTED")
print("ansi.enabled =", spark.conf.get("spark.sql.ansi.enabled"))

print(f"Origen : {TBL_BRONZE}")
print(f"Destino: {TBL_SILVER} + {TBL_LOG}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. RN-03 — Fechas a ISO 8601
# MAGIC
# MAGIC El origen trae **dos formatos** mezclados: `2023-01-01 00:00:41` en 43 archivos y
# MAGIC `01/11/2023 0:01` en tres (2023-11, 2024-09 y 2024-10). Se intenta el formato ISO primero y
# MAGIC lo que quede en `NULL` se reintenta con el formato latino. Si ninguno funciona queda `NULL`,
# MAGIC y eso se convierte en una incidencia más abajo.

# COMMAND ----------

from pyspark.sql import functions as F, Window

def a_timestamp(col):
    """Primero ISO, luego dd/MM/yyyy con y sin segundos. El primero que parsee, gana."""
    c = F.trim(F.col(col))
    return F.coalesce(
        F.to_timestamp(c, "yyyy-MM-dd HH:mm:ss"),
        F.to_timestamp(c, "d/M/yyyy H:mm"),
        F.to_timestamp(c, "d/M/yyyy H:mm:ss"),
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. RN-02 y RN-04 — Tipado y variables derivadas
# MAGIC
# MAGIC Los `cast` devuelven `NULL` cuando el texto no es un número, que es el comportamiento que
# MAGIC queremos: el dato malo se marca, no revienta el job.
# MAGIC
# MAGIC `genero` se normaliza a mayúsculas y se recorta al dominio `{M, F, O}`. Cualquier otra cosa,
# MAGIC incluido el vacío, pasa a `NULL`.

# COMMAND ----------

bronze = spark.table(TBL_BRONZE)

tipado = (
    bronze
    .withColumn("viaje_id",        F.trim("viaje_id").cast("long"))
    .withColumn("usuario_id",      F.trim("usuario_id").cast("long"))
    .withColumn("anio_nacimiento", F.trim("anio_nacimiento").cast("int"))
    .withColumn("origen_id",       F.trim("origen_id").cast("int"))
    .withColumn("destino_id",      F.trim("destino_id").cast("int"))
    .withColumn("genero",
                F.when(F.upper(F.trim("genero")).isin("M", "F", "O"),
                       F.upper(F.trim("genero"))))
    .withColumn("inicio_viaje", a_timestamp("inicio_viaje"))
    .withColumn("fin_viaje",    a_timestamp("fin_viaje"))
)

dur_seg = F.unix_timestamp("fin_viaje") - F.unix_timestamp("inicio_viaje")

derivado = (
    tipado
    .withColumn("duracion_seg",  dur_seg)
    .withColumn("duracion_min",  F.round(dur_seg / 60, 2))
    .withColumn("anio",          F.year("inicio_viaje"))
    .withColumn("mes",           F.month("inicio_viaje"))
    .withColumn("dia",           F.dayofmonth("inicio_viaje"))
    .withColumn("hora",          F.hour("inicio_viaje"))
    # dayofweek de Spark: 1=domingo … 7=sábado. Se recodifica a 0=lunes … 6=domingo,
    # que es el convenio de Python y el que usa el resto del proyecto.
    .withColumn("dia_semana",    (F.dayofweek("inicio_viaje") + 5) % 7)
    .withColumn("es_fin_semana", F.col("dia_semana").isin(5, 6))
    .withColumn("edad_usuario",  F.col("anio") - F.col("anio_nacimiento"))
    # RN-05: la llave se calcula aquí porque la detección de duplicados la necesita
    .withColumn("viaje_sk", F.concat_ws("-",
        F.coalesce(F.col("anio").cast("string"), F.lit("XXXX")),
        F.coalesce(F.lpad(F.col("mes").cast("string"), 2, "0"), F.lit("XX")),
        F.coalesce(F.col("viaje_id").cast("string"), F.lit("SIN_ID")),
    ))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### RN-08 — Duplicados exactos
# MAGIC
# MAGIC El origen trae **758 registros repetidos**: la misma fila, dos veces. Mismo `viaje_id`,
# MAGIC mismo usuario, misma hora de inicio y de fin. No es que el identificador se recicle entre
# MAGIC meses — es el mismo viaje capturado dos veces dentro del mismo mes.
# MAGIC
# MAGIC Se detectan comparando la **tupla completa de negocio**, no solo el identificador: dos filas
# MAGIC son la misma si coinciden en viaje, usuario, horas y estaciones. A la primera copia se le
# MAGIC deja pasar; de la segunda en adelante se marcan como incidencia.
# MAGIC
# MAGIC Fiel a RN-07, **la copia no se borra**: se conserva con `registro_valido = false` y queda
# MAGIC registrada en el log. Así `viaje_sk` sí es única entre los registros válidos, que es donde
# MAGIC la necesita el esquema estrella.

# COMMAND ----------

# El orden dentro de la ventana es irrelevante porque las filas son idénticas; se fija uno
# determinista solo para que la elección de "la copia buena" sea reproducible.
ventana_dup = (Window
               .partitionBy("viaje_id", "usuario_id", "inicio_viaje", "fin_viaje",
                            "origen_id", "destino_id")
               .orderBy("archivo_origen", "_orden"))

derivado = (derivado
            .withColumn("_orden", F.monotonically_increasing_id())
            .withColumn("_copia", F.row_number().over(ventana_dup)))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Las nueve validaciones
# MAGIC
# MAGIC Cada regla produce, cuando se incumple, un `struct` con las tres columnas que pide el formato
# MAGIC de log del profesor: qué columna falló, qué mensaje y con qué valor.

# COMMAND ----------

def incidencia(condicion, columna, mensaje, valor):
    return F.when(condicion, F.struct(
        F.lit(columna).alias("columna"),
        F.lit(mensaje).alias("mensaje_error"),
        valor.cast("string").alias("valor"),
    ))

reglas = [
    incidencia(F.col("inicio_viaje").isNull(), "inicio_viaje",
               "Fecha de inicio no parseable o vacia", F.col("inicio_viaje")),
    incidencia(F.col("fin_viaje").isNull(), "fin_viaje",
               "Fecha de fin no parseable o vacia", F.col("fin_viaje")),
    incidencia(F.col("duracion_seg") <= 0, "duracion_min",
               "Fin del viaje menor o igual al inicio", F.col("duracion_min")),
    incidencia((F.col("duracion_seg") > 0) & (F.col("duracion_seg") < DUR_MIN_SEG), "duracion_min",
               f"Duracion menor a {DUR_MIN_SEG} segundos", F.col("duracion_min")),
    incidencia(F.col("duracion_seg") > DUR_MAX_SEG, "duracion_min",
               "Duracion mayor a 24 horas", F.col("duracion_min")),
    incidencia(F.col("genero").isNull(), "genero",
               "Genero nulo o fuera del dominio {M,F,O}", F.col("genero")),
    incidencia(F.col("anio_nacimiento").isNull(), "anio_nacimiento",
               "Anio de nacimiento nulo", F.col("anio_nacimiento")),
    incidencia((F.col("edad_usuario") < EDAD_MIN) | (F.col("edad_usuario") > EDAD_MAX),
               "edad_usuario", f"Edad fuera del rango [{EDAD_MIN},{EDAD_MAX}]", F.col("edad_usuario")),
    incidencia(F.col("origen_id").isNull() | F.col("destino_id").isNull(), "origen_id/destino_id",
               "Estacion de origen o destino nula", F.col("origen_id")),
    incidencia(F.col("_copia") > 1, "viaje_sk",
               "Registro duplicado exacto", F.col("viaje_sk")),
]

# array_compact deja fuera los NULL, así que el arreglo contiene SOLO las reglas incumplidas.
con_incidencias = derivado.withColumn("incidencias", F.array_compact(F.array(*reglas)))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. RN-05 y RN-07 — Llave única y bandera de validez
# MAGIC
# MAGIC `viaje_id` **no es único** entre archivos: 758 identificadores se repiten. Por eso la llave de
# MAGIC negocio es `viaje_sk = anio-mes-viaje_id`, que sí lo es.
# MAGIC
# MAGIC `registro_valido` sale del mismo arreglo que el log, así que no pueden contradecirse.

# COMMAND ----------

silver = (
    con_incidencias
    .withColumn("n_incidencias",   F.size("incidencias"))
    .withColumn("registro_valido", F.col("n_incidencias") == 0)
    .select(
        "viaje_sk", "viaje_id", "usuario_id", "genero", "anio_nacimiento", "edad_usuario",
        "inicio_viaje", "fin_viaje", "duracion_min",
        "origen_id", "destino_id",
        "anio", "mes", "dia", "hora", "dia_semana", "es_fin_semana",
        "n_incidencias", "registro_valido", "incidencias",
        "archivo_origen", "fecha_carga",
    )
)

# Se materializa una vez: abajo se usa tres veces (escritura, log y reconciliación)
# y sin esto Spark recalcularía la cadena completa cada vez.
silver.cache()
print(f"Filas en Silver: {silver.count():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Escritura de `silver_viajes`
# MAGIC
# MAGIC La columna `incidencias` se queda fuera de la tabla final: su contenido ya vive, desplegado,
# MAGIC en el log de errores.

# COMMAND ----------

(silver.drop("incidencias")
       .write.mode("overwrite").option("overwriteSchema", "true")
       .partitionBy("anio")
       .saveAsTable(TBL_SILVER))

print(f"Escrito: {TBL_SILVER}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Escritura del log de errores
# MAGIC
# MAGIC Un `explode` del arreglo: una fila por incidencia. Un viaje con tres problemas aparece
# MAGIC tres veces, con un mensaje distinto cada vez.

# COMMAND ----------

log = (silver
       .select("viaje_sk", "viaje_id", "archivo_origen",
               F.explode("incidencias").alias("inc"))
       .select("viaje_sk", "viaje_id", "archivo_origen",
               F.col("inc.columna").alias("columna"),
               F.col("inc.mensaje_error").alias("mensaje_error"),
               F.col("inc.valor").alias("valor"),
               F.current_timestamp().alias("fecha_carga")))

log.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(TBL_LOG)
print(f"Escrito: {TBL_LOG}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Reconciliación contra el perfilado
# MAGIC
# MAGIC La celda que vale la pena enseñar en la exposición. Compara lo que produjo PySpark contra lo
# MAGIC que se midió en el perfilado previo sobre los CSV originales. Si cuadra al registro, la
# MAGIC implementación distribuida es equivalente a la de referencia.

# COMMAND ----------

n_total   = spark.table(TBL_SILVER).count()
n_validos = spark.table(TBL_SILVER).filter("registro_valido").count()
n_log     = spark.table(TBL_LOG).count()
n_vid     = spark.table(TBL_SILVER).select("viaje_id").distinct().count()

# La propiedad que de verdad importa: la llave es única ENTRE LOS VÁLIDOS, que son los
# únicos que llegan al esquema estrella. En Silver conviven las copias, por diseño.
n_sk_validos = (spark.table(TBL_SILVER).filter("registro_valido")
                .select("viaje_sk").distinct().count())

filas = [
    ("Filas totales",            n_total,         CTRL_TOTAL),
    ("Incidencias en el log",    n_log,           CTRL_INCIDENCIAS_6 + CTRL_DUPLICADOS),
    ("viaje_id duplicados",      n_total - n_vid, CTRL_DUPLICADOS),
    ("viaje_sk unico en validos", n_sk_validos,   n_validos),
]

print(f"{'Concepto':<26}{'PySpark':>14}{'Esperado':>14}   Estado")
print("-" * 70)
ok = True
for concepto, obtenido, esperado in filas:
    bien = obtenido == esperado
    ok &= bien
    print(f"{concepto:<26}{obtenido:>14,}{esperado:>14,}   {'OK' if bien else 'DIFIERE'}")

# COMMAND ----------

por_regla = (spark.table(TBL_LOG).groupBy("mensaje_error").count()
             .toPandas().set_index("mensaje_error")["count"].to_dict())

print(f"{'Regla':<45}{'PySpark':>11}{'Esperado':>11}   Estado")
print("-" * 80)
for mensaje, esperado in sorted(CTRL_POR_REGLA.items(), key=lambda x: -x[1]):
    obtenido = por_regla.get(mensaje, 0)
    bien = obtenido == esperado
    ok &= bien
    print(f"{mensaje:<45}{obtenido:>11,}{esperado:>11,}   {'OK' if bien else 'DIFIERE'}")

sobrantes = set(por_regla) - set(CTRL_POR_REGLA)
if sobrantes:
    ok = False
    print(f"\nReglas que PySpark produjo y el perfilado no esperaba: {sobrantes}")

assert ok, "La capa Silver NO reconcilia con el perfilado de referencia"
print("\nReconciliación correcta: PySpark reproduce el perfilado al registro.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Comprobaciones de que la limpieza sirvió
# MAGIC
# MAGIC Reconciliar prueba que hicimos lo que dijimos. Esto prueba que lo que dijimos era lo correcto.

# COMMAND ----------

display(spark.sql(f"""
    SELECT 'Fechas sin parsear'          AS comprobacion,
           COUNT(*)                      AS casos,
           'Debe ser 0: las dos variantes de formato quedaron cubiertas' AS criterio
    FROM {TBL_SILVER} WHERE inicio_viaje IS NULL OR fin_viaje IS NULL
  UNION ALL
    SELECT 'Genero fuera de dominio', COUNT(*),
           'Debe ser 0: solo quedan M, F, O o NULL'
    FROM {TBL_SILVER} WHERE genero IS NOT NULL AND genero NOT IN ('M','F','O')
  UNION ALL
    SELECT 'Duraciones negativas entre los validos', COUNT(*),
           'Debe ser 0'
    FROM {TBL_SILVER} WHERE registro_valido AND duracion_min <= 0
  UNION ALL
    SELECT 'Edades imposibles entre los validos', COUNT(*),
           'Debe ser 0'
    FROM {TBL_SILVER} WHERE registro_valido AND (edad_usuario < 10 OR edad_usuario > 100)
  UNION ALL
    SELECT 'viaje_sk repetidos entre validos', COUNT(*) - COUNT(DISTINCT viaje_sk),
           'Debe ser 0: es la llave del hecho'
    FROM {TBL_SILVER} WHERE registro_valido
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Resumen por año

# COMMAND ----------

display(spark.sql(f"""
    SELECT anio,
           COUNT(*)                                        AS viajes,
           SUM(CASE WHEN registro_valido THEN 1 ELSE 0 END) AS validos,
           ROUND(100.0 * AVG(CASE WHEN registro_valido THEN 1 ELSE 0 END), 2) AS pct_validos,
           ROUND(AVG(CASE WHEN registro_valido THEN duracion_min END), 2)     AS duracion_media_min,
           COUNT(DISTINCT usuario_id)                       AS usuarios_unicos,
           MIN(inicio_viaje)                                AS desde,
           MAX(inicio_viaje)                                AS hasta
    FROM {TBL_SILVER}
    GROUP BY anio ORDER BY anio
"""))

# COMMAND ----------

silver.unpersist()

# MAGIC %md
# MAGIC ---
# MAGIC ### Siguiente paso
# MAGIC `03_gold_estrella.py` — `fact_viajes`, `dim_estacion`, `dim_tiempo` y la tabla agregada
# MAGIC `agg_demanda_estacion_hora` que entrena el modelo.
