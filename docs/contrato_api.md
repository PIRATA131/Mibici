# Contrato del API — predicción de demanda y rebalanceo de MiBici

Este documento fija qué recibe y qué devuelve el servicio. Es el punto de acuerdo entre las tres
piezas del proyecto: la tabla Gold produce las features, el API las valida y responde, y la página
web pinta la respuesta. Mientras el contrato no cambie, las tres se pueden trabajar en paralelo.

**Base URL (local):** `http://localhost:8000`
**Documentación interactiva:** `/docs` (Swagger UI) y `/redoc`. El esquema OpenAPI crudo vive en
`/openapi.json` y lo genera FastAPI a partir de los modelos Pydantic — no se escribe a mano.

---

## El problema en una frase

Dada una estación y una hora, ¿cuántas bicicletas van a salir y llegar, y cuántas habría que
trasladar para que la estación no se quede sin servicio?

---

## Endpoints

| Método | Ruta | Para qué |
|---|---|---|
| `GET` | `/salud` | ¿Está arriba el servicio y qué modelo tiene cargado? |
| `GET` | `/modelo` | Metadatos del modelo activo: tipo, versión, métrica, features |
| `GET` | `/estaciones` | Catálogo para pintar el mapa |
| `GET` | `/estaciones/{id}` | Ficha de una estación |
| `POST` | `/predecir` | Predicción para una estación y una hora |
| `POST` | `/predecir/lote` | Predicción para muchas estaciones a la misma hora |

### `POST /predecir`

```json
{ "estacion_id": 194, "fecha_hora": "2026-11-20T18:00:00" }
```

```json
{
  "estacion_id": 194,
  "clave_estacion": "GDL-127",
  "nombre": "Av. Vallarta / Av. de Los Arcos",
  "municipio": "Guadalajara",
  "latitud": 20.67438,
  "longitud": -103.38488,
  "capacidad_anclajes": 19,
  "fecha_hora": "2026-11-20T18:00:00",
  "dia_semana": 4,
  "hora": 18,
  "salidas_estimadas": 35.558,
  "llegadas_estimadas": 3.192,
  "balance_estimado": -32.366,
  "deficit_bicis": 32.366,
  "saturacion": 1.703,
  "modelo_tipo": "baseline_historico",
  "modelo_version": "1.0.0"
}
```

**Cómo se leen los tres campos que importan:**

- `balance_estimado` = llegadas − salidas. Negativo significa que la estación se vacía.
- `deficit_bicis` = `max(0, −balance)`. Es cuántas bicis habría que llevar.
- `saturacion` = déficit ÷ capacidad. `1.703` significa que esa estación se vacía **1.7 veces**
  durante esa hora: los 19 anclajes no alcanzan ni de cerca.

### `POST /predecir/lote`

Omite `estaciones` para pedir todas las ubicables. Es lo que usa el mapa: una sola llamada
devuelve las 470 estaciones en ~10 ms.

```json
{ "fecha_hora": "2026-11-20T18:00:00" }
```

```json
{ "fecha_hora": "2026-11-20T18:00:00", "total": 470,
  "modelo_tipo": "...", "modelo_version": "...", "predicciones": [ /* ... */ ] }
```

---

## Validación de entradas

FastAPI rechaza la petición **antes** de que llegue al modelo. Todo error trae
`{"detalle": "...", "codigo": "..."}`.

| Caso | Código HTTP | `codigo` |
|---|---|---|
| `estacion_id` fuera de `[1, 99999]` | 422 | `ENTRADA_INVALIDA` |
| `fecha_hora` no es ISO 8601 válida | 422 | `ENTRADA_INVALIDA` |
| `fecha_hora` fuera de 2014-01-01 … 2030-12-31 | 422 | `ENTRADA_INVALIDA` |
| Falta un campo obligatorio | 422 | `ENTRADA_INVALIDA` |
| Lote vacío, con repetidos o de más de 500 | 422 | `ENTRADA_INVALIDA` |
| La estación no está en el catálogo | 404 | `NO_ENCONTRADO` |
| El servicio aún no arranca | 503 | `NO_DISPONIBLE` |

---

## Qué tiene que manejar la página web

El criterio 4 de la rúbrica pide explícitamente que la aplicación maneje errores, tiempos de espera
y respuestas inválidas. En concreto:

1. **Consumir el endpoint por HTTP.** Nunca cargar el modelo ni la tabla baseline en el navegador:
   eso anularía el criterio.
2. **Timeout propio** de unos 10 s en cada `fetch`, con mensaje al usuario si se vence.
3. **Distinguir los errores:** 422 y 404 son culpa de la petición y se le explican al usuario;
   5xx y los timeouts son del servicio y ofrecen reintentar.
4. **Verificar `/salud` al cargar.** Si `modelo_cargado` es `false`, mostrar un aviso de que se
   está respondiendo con la línea base histórica y no con el modelo entrenado.
5. **No asumir que los campos vienen.** `latitud`, `longitud`, `capacidad_anclajes` y `saturacion`
   pueden ser `null` en las 5 estaciones sin ficha en el catálogo vigente.

---

## El modelo detrás del endpoint

El API declara siempre cuál de los dos está respondiendo, en `/salud` y en `/modelo`:

| `modelo_tipo` | Qué es | Estado |
|---|---|---|
| `baseline_historico` | Promedio de salidas y llegadas por estación, día de la semana y hora, sobre los 4,505,844 viajes válidos de nov-2025 a oct-2026 | **Activo hoy** |
| `gbt_regressor` | Gradient Boosted Trees sobre la tabla Gold | Pendiente — notebook 04 |

La ventana se limita a 12 meses a propósito: la red creció de 300 a 470 estaciones entre 2023 y
2026, y promediar los cuatro años subestimaría las estaciones nuevas.

El baseline no es relleno: es el **punto de comparación obligatorio** del criterio 2. Un modelo que
no le gane en RMSE no justifica su existencia, y eso hay que medirlo y reportarlo.

En cuanto exista `datos/modelo_demanda.joblib`, el API lo carga solo y `modelo_tipo` cambia. El
contrato no se mueve, así que la página web no se entera.

---

## Correrlo

```bash
cd api
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Con Docker:

```bash
docker build -t mibici-api ./api
docker run -p 8000:8000 mibici-api
```
