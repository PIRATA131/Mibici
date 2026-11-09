# Databricks notebook source
# MAGIC %md
# MAGIC # Capa Bronze — Ingesta cruda de MiBici
# MAGIC
# MAGIC **Proyecto:** Predicción de demanda y rebalanceo de MiBici
# MAGIC **Materia:** Aprendizaje Automático para Grandes Volúmenes de Datos — Dr. Ramón Alejandro Briseño
# MAGIC **Alumno:** César Roberto Murillo Figueroa
# MAGIC
# MAGIC ## Qué hace este notebook
# MAGIC
# MAGIC Ingesta los 46 CSV mensuales de [datos abiertos de MiBici](https://www.mibici.net/es/datos-abiertos/)
# MAGIC (enero 2023 – octubre 2026) y el catálogo de estaciones, y los deja en tablas Delta **sin transformar**.
# MAGIC
# MAGIC ## El principio de Bronze
# MAGIC
# MAGIC Bronze conserva el dato **tal como llegó de la fuente**: todo se lee como `STRING`, no se castea,
# MAGIC no se limpia y no se descarta nada. Si mañana descubro que una regla de la capa Silver estaba mal,
# MAGIC puedo recalcularla desde aquí sin volver a descargar 1.4 GB del portal.
# MAGIC
# MAGIC Lo único que se agrega son **columnas de auditoría**: de qué archivo vino cada fila y cuándo se cargó.
# MAGIC
# MAGIC ## Las dos decisiones de diseño que hay que justificar
# MAGIC
# MAGIC **1. Por qué se impone un esquema en lugar de inferirlo.**
# MAGIC Los 46 archivos NO son homogéneos. El perfilado previo encontró tres ejes de variación:
# MAGIC
# MAGIC | Variación | Detalle |
# MAGIC |---|---|
# MAGIC | Encoding | 27 archivos en `windows-1252` y 19 en `UTF-8 con BOM` |
# MAGIC | Comillas y saltos de línea | 43 archivos entrecomillados con `LF`; 3 sin comillas y con `CRLF` |
# MAGIC | Encabezado | `Año_de_nacimiento` aparece con 3 secuencias de bytes distintas, 2 de ellas corruptas |
# MAGIC
# MAGIC Con `inferSchema=True` cada archivo produciría un esquema distinto y la unión fallaría.
# MAGIC Al declarar el esquema por **posición** y pedir `header=True`, Spark descarta la primera línea de
# MAGIC *cada* archivo —encabezado corrupto incluido— y aplica nombres estables. Es la unificación
# MAGIC mínima e inevitable; todo lo demás se deja para Silver.
# MAGIC
# MAGIC **2. Por qué se lee todo como UTF-8 pese a que 27 archivos son windows-1252.**
# MAGIC Se verificó byte por byte que **ninguna fila de datos contiene bytes fuera de ASCII**: los ids, las
# MAGIC fechas y el género son ASCII puro. El único carácter no-ASCII (`ñ`) vive exclusivamente en el
# MAGIC encabezado, que se descarta. Por eso una sola lectura UTF-8 es segura y no hace falta partir la
# MAGIC ingesta en dos grupos por encoding.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Configuración

# COMMAND ----------

CATALOG = "workspace"
SCHEMA  = "mibici"
VOLUME  = "raw"

RUTA_VIAJES   = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/bronze/datos_abiertos_*.csv.gz"
RUTA_ESTACION = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/bronze/nomenclatura_2026_10.csv"

TBL_VIAJES   = f"{CATALOG}.{SCHEMA}.bronze_viajes"
TBL_ESTACION = f"{CATALOG}.{SCHEMA}.bronze_estaciones"

# El catálogo `workspace` ya existe en Databricks Free Edition y la edición gratuita no
# concede permiso para crear catálogos, así que no se intenta. El esquema y el volumen sí
# se crean aquí (idempotente: si ya existen, no pasa nada).
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.{VOLUME}")

# Silver y Gold dependen de que un dato malo se vuelva NULL en vez de lanzar excepción.
spark.sql("SET spark.sql.ansi.enabled = false")

print(f"Destino: {TBL_VIAJES}")
print(f"Fuente : {RUTA_VIAJES}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Conteos de control
# MAGIC
# MAGIC Filas por archivo contadas en el origen, antes de subir nada. Sirven para probar al final que la
# MAGIC ingesta no perdió ni duplicó un solo registro.

# COMMAND ----------

ESPERADO = {
    "datos_abiertos_2023_01.csv.gz": 366_018,
    "datos_abiertos_2023_02.csv.gz": 355_291,
    "datos_abiertos_2023_03.csv.gz": 402_460,
    "datos_abiertos_2023_04.csv.gz": 333_211,
    "datos_abiertos_2023_05.csv.gz": 364_184,
    "datos_abiertos_2023_06.csv.gz": 339_628,
    "datos_abiertos_2023_07.csv.gz": 331_787,
    "datos_abiertos_2023_08.csv.gz": 354_810,
    "datos_abiertos_2023_09.csv.gz": 363_859,
    "datos_abiertos_2023_10.csv.gz": 383_239,
    "datos_abiertos_2023_11.csv.gz": 384_464,
    "datos_abiertos_2023_12.csv.gz": 314_164,
    "datos_abiertos_2024_01.csv.gz": 391_591,
    "datos_abiertos_2024_02.csv.gz": 383_732,
    "datos_abiertos_2024_03.csv.gz": 390_513,
    "datos_abiertos_2024_04.csv.gz": 405_308,
    "datos_abiertos_2024_05.csv.gz": 410_806,
    "datos_abiertos_2024_06.csv.gz": 365_000,
    "datos_abiertos_2024_07.csv.gz": 364_177,
    "datos_abiertos_2024_08.csv.gz": 394_894,
    "datos_abiertos_2024_09.csv.gz": 390_072,
    "datos_abiertos_2024_10.csv.gz": 418_690,
    "datos_abiertos_2024_11.csv.gz": 391_089,
    "datos_abiertos_2024_12.csv.gz": 340_049,
    "datos_abiertos_2025_01.csv.gz": 380_184,
    "datos_abiertos_2025_02.csv.gz": 372_529,
    "datos_abiertos_2025_03.csv.gz": 405_110,
    "datos_abiertos_2025_04.csv.gz": 366_193,
    "datos_abiertos_2025_05.csv.gz": 395_891,
    "datos_abiertos_2025_06.csv.gz": 356_611,
    "datos_abiertos_2025_07.csv.gz": 368_186,
    "datos_abiertos_2025_08.csv.gz": 374_727,
    "datos_abiertos_2025_09.csv.gz": 376_789,
    "datos_abiertos_2025_10.csv.gz": 420_064,
    "datos_abiertos_2025_11.csv.gz": 380_328,
    "datos_abiertos_2025_12.csv.gz": 335_420,
    "datos_abiertos_2026_01.csv.gz": 357_180,
    "datos_abiertos_2026_02.csv.gz": 333_756,
    "datos_abiertos_2026_03.csv.gz": 376_243,
    "datos_abiertos_2026_04.csv.gz": 351_851,
    "datos_abiertos_2026_05.csv.gz": 315_895,
    "datos_abiertos_2026_06.csv.gz": 355_303,
    "datos_abiertos_2026_07.csv.gz": 388_257,
    "datos_abiertos_2026_08.csv.gz": 393_524,
    "datos_abiertos_2026_09.csv.gz": 490_056,
    "datos_abiertos_2026_10.csv.gz": 635_532
}

TOTAL_ESPERADO = 17_468_665
assert sum(ESPERADO.values()) == TOTAL_ESPERADO
print(f"{len(ESPERADO)} archivos | {TOTAL_ESPERADO:,} filas esperadas")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Subir los archivos al Volume
# MAGIC
# MAGIC Antes de seguir, los archivos de la carpeta local `Upload_Databricks/bronze/` tienen que estar en el
# MAGIC Volume. En el menú lateral: **Catalog → {CATALOG} → {SCHEMA} → Volumes → {VOLUME} →
# MAGIC Create folder `bronze` → Upload**.
# MAGIC
# MAGIC Son 46 archivos `.csv.gz` (302 MB en total) más `nomenclatura_2026_10.csv`.
# MAGIC Spark lee el gzip directo, no hay que descomprimir nada.
# MAGIC
# MAGIC La celda siguiente verifica que estén todos antes de intentar leerlos.

# COMMAND ----------

archivos = dbutils.fs.ls(f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/bronze/")
gz = [a for a in archivos if a.name.endswith(".csv.gz")]
mb = sum(a.size for a in gz) / 2**20

print(f"Archivos .csv.gz en el Volume: {len(gz)} de 46")
print(f"Tamaño total: {mb:,.0f} MB")

faltantes = set(ESPERADO) - {a.name for a in gz}
assert not faltantes, f"Faltan {len(faltantes)} archivos por subir: {sorted(faltantes)[:5]}"
assert any(a.name == "nomenclatura_2026_10.csv" for a in archivos), "Falta la nomenclatura de estaciones"
print("\nTodos los archivos están en su lugar.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Esquema de lectura
# MAGIC
# MAGIC Todo `STRING`: en Bronze no se castea. El orden de las columnas es el del CSV de origen, que sí es
# MAGIC estable en los 46 archivos aunque los nombres no lo sean.

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType

esquema = StructType([
    StructField("viaje_id",        StringType(), True),
    StructField("usuario_id",      StringType(), True),
    StructField("genero",          StringType(), True),
    StructField("anio_nacimiento", StringType(), True),   # "Año_de_nacimiento" en el origen
    StructField("inicio_viaje",    StringType(), True),
    StructField("fin_viaje",       StringType(), True),
    StructField("origen_id",       StringType(), True),
    StructField("destino_id",      StringType(), True),
])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Lectura de los viajes
# MAGIC
# MAGIC `header=True` con esquema explícito hace que Spark **salte la primera línea de cada archivo** y use
# MAGIC los nombres declarados arriba. Es lo que neutraliza los encabezados corruptos.
# MAGIC
# MAGIC Las columnas de auditoría salen de `_metadata`, que Databricks expone en toda lectura de archivos.

# COMMAND ----------

crudo = (
    spark.read
         .option("header", True)          # descarta la 1a línea de CADA archivo
         .option("quote", '"')            # 43 archivos entrecomillados, 3 sin comillas: funciona para ambos
         .option("escape", '"')
         .option("multiLine", False)
         .option("mode", "PERMISSIVE")    # nada se descarta en Bronze
         .schema(esquema)
         .csv(RUTA_VIAJES)
)

bronze = (
    crudo
    .withColumn("archivo_origen", F.col("_metadata.file_name"))
    .withColumn("fecha_carga",    F.current_timestamp())
    # el nombre del archivo es datos_abiertos_AAAA_MM.csv.gz -> de ahí salen año y mes de partición
    .withColumn("anio_archivo",  F.regexp_extract("archivo_origen", r"(\d{4})_(\d{2})", 1).cast("int"))
    .withColumn("mes_archivo",   F.regexp_extract("archivo_origen", r"(\d{4})_(\d{2})", 2).cast("int"))
)

display(bronze.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Escritura a Delta
# MAGIC
# MAGIC Particionado por `anio_archivo`: 4 particiones de tamaño parejo. No se particiona por mes porque
# MAGIC 46 particiones sobre este volumen generarían archivos demasiado chicos (*small file problem*).

# COMMAND ----------

(bronze.write
       .mode("overwrite")
       .option("overwriteSchema", "true")
       .partitionBy("anio_archivo")
       .saveAsTable(TBL_VIAJES))

print(f"Escrito: {TBL_VIAJES}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Reconciliación
# MAGIC
# MAGIC La prueba que hay que poder enseñar: **cada archivo aportó exactamente las filas que tenía en el
# MAGIC origen**. Si esta celda pasa, la ingesta es fiel.

# COMMAND ----------

real = (spark.table(TBL_VIAJES)
             .groupBy("archivo_origen").count()
             .toPandas().set_index("archivo_origen")["count"].to_dict())

difs = []
for archivo, esperado in sorted(ESPERADO.items()):
    obtenido = real.get(archivo, 0)
    if obtenido != esperado:
        difs.append((archivo, esperado, obtenido, obtenido - esperado))

total_real = spark.table(TBL_VIAJES).count()
print(f"Filas esperadas: {TOTAL_ESPERADO:,}")
print(f"Filas cargadas : {total_real:,}")
print(f"Diferencia     : {total_real - TOTAL_ESPERADO:+,}")

if difs:
    print(f"\n{len(difs)} archivos NO reconcilian:")
    for d in difs[:20]:
        print(f"  {d[0]}: esperado {d[1]:,} | obtenido {d[2]:,} | dif {d[3]:+,}")
    raise AssertionError("La ingesta Bronze no reconcilia con el origen")

print("\nReconciliación correcta: los 46 archivos coinciden al registro.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Catálogo de estaciones
# MAGIC
# MAGIC Archivo aparte, también crudo. Trae `dpcapacity`, la capacidad de anclajes de cada estación, que es
# MAGIC la que permite convertir una predicción de viajes en una predicción de **déficit de bicis**.
# MAGIC
# MAGIC Este sí va en `windows-1252`: los nombres de las calles tienen acentos y `ñ` (*Niños Héroes*,
# MAGIC *Ramón Corona*), y aquí el dato no ASCII está en el contenido, no solo en el encabezado.

# COMMAND ----------

estaciones = (
    spark.read
         .option("header", True)
         .option("encoding", "windows-1252")     # a diferencia de los viajes, aquí sí importa
         .option("inferSchema", False)
         .csv(RUTA_ESTACION)
         .withColumnRenamed("id", "estacion_id")
         .withColumnRenamed("name", "nombre_completo")
         .withColumnRenamed("latitude", "latitud")
         .withColumnRenamed("longitude", "longitud")
         .withColumnRenamed("dpcapacity", "capacidad_anclajes")
         .withColumn("archivo_origen", F.col("_metadata.file_name"))
         .withColumn("fecha_carga", F.current_timestamp())
)

estaciones.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(TBL_ESTACION)

n = spark.table(TBL_ESTACION).count()
print(f"Escrito: {TBL_ESTACION} — {n} estaciones (esperadas 470)")
assert n == 470, f"Se esperaban 470 estaciones, llegaron {n}"
display(spark.table(TBL_ESTACION).limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Resumen de la capa

# COMMAND ----------

display(spark.sql(f"""
    SELECT anio_archivo            AS anio,
           COUNT(DISTINCT archivo_origen) AS archivos,
           COUNT(*)                AS viajes,
           MIN(inicio_viaje)       AS primer_valor_crudo,
           MAX(inicio_viaje)       AS ultimo_valor_crudo
    FROM {TBL_VIAJES}
    GROUP BY anio_archivo
    ORDER BY anio_archivo
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC Fíjate en `primer_valor_crudo` y `ultimo_valor_crudo`: salen como texto y en 2023 y 2024 vas a ver
# MAGIC fechas en formato `dd/mm/aaaa` mezcladas con `aaaa-mm-dd`. Eso es justo lo que Bronze **no** arregla.
# MAGIC Se arregla en el siguiente notebook.
# MAGIC
# MAGIC ---
# MAGIC ### Siguiente paso
# MAGIC `02_silver_limpieza.py` — las 7 reglas de negocio, el log de errores y el casteo de tipos.
