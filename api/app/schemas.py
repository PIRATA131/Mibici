# -*- coding: utf-8 -*-
"""Contratos de entrada y salida del API. Pydantic valida cada campo y FastAPI
publica estos esquemas en /docs (OpenAPI) sin trabajo extra."""
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

# La red de MiBici arrancó en 2014 y el modelo no tiene sentido más allá de un
# horizonte corto. Fuera de este rango la petición se rechaza con 422.
FECHA_MIN = datetime(2014, 1, 1)
FECHA_MAX = datetime(2030, 12, 31, 23, 59, 59)

MAX_LOTE = 500


class PeticionPrediccion(BaseModel):
    """Una predicción: qué va a pasar en una estación, en una hora concreta."""

    estacion_id: int = Field(
        ..., ge=1, le=99_999,
        description="Identificador de la estación, tal como viene en el catálogo de MiBici.",
        examples=[194],
    )
    fecha_hora: datetime = Field(
        ...,
        description="Momento a predecir, en ISO 8601. Los minutos y segundos se ignoran: "
                    "el modelo trabaja a nivel de hora.",
        examples=["2026-11-20T18:00:00"],
    )

    @field_validator("fecha_hora")
    @classmethod
    def rango_razonable(cls, v: datetime) -> datetime:
        if v.tzinfo is not None:
            v = v.replace(tzinfo=None)
        if not (FECHA_MIN <= v <= FECHA_MAX):
            raise ValueError(
                f"fecha_hora debe estar entre {FECHA_MIN:%Y-%m-%d} y {FECHA_MAX:%Y-%m-%d}"
            )
        return v

    model_config = {
        "json_schema_extra": {
            "examples": [{"estacion_id": 194, "fecha_hora": "2026-11-20T18:00:00"}]
        }
    }


class PeticionLote(BaseModel):
    """Predicción para varias estaciones a la misma hora. Es lo que usa el mapa:
    una sola llamada pinta la ciudad completa."""

    estaciones: list[int] | None = Field(
        None,
        description=f"Lista de hasta {MAX_LOTE} estaciones. Si se omite, se predicen todas.",
        examples=[[194, 51, 271]],
    )
    fecha_hora: datetime = Field(..., examples=["2026-11-20T18:00:00"])

    @field_validator("estaciones")
    @classmethod
    def lote_acotado(cls, v):
        if v is None:
            return v
        if not v:
            raise ValueError("Si se envía 'estaciones', no puede ir vacía. Omítela para pedir todas.")
        if len(v) > MAX_LOTE:
            raise ValueError(f"Máximo {MAX_LOTE} estaciones por petición, llegaron {len(v)}")
        if len(set(v)) != len(v):
            raise ValueError("La lista de estaciones tiene identificadores repetidos")
        return v

    _rango = field_validator("fecha_hora")(PeticionPrediccion.rango_razonable.__func__)


class Prediccion(BaseModel):
    """Lo que devuelve el modelo para una estación y una hora."""

    estacion_id: int
    clave_estacion: str = Field(description="Clave oficial, por ejemplo GDL-127.")
    nombre: str
    municipio: str
    latitud: float | None
    longitud: float | None
    capacidad_anclajes: int | None = Field(description="Anclajes físicos de la estación.")

    fecha_hora: datetime
    dia_semana: int = Field(ge=0, le=6, description="0 = lunes … 6 = domingo.")
    hora: int = Field(ge=0, le=23)

    salidas_estimadas: float = Field(description="Bicis que se espera que salgan en esa hora.")
    llegadas_estimadas: float = Field(description="Bicis que se espera que lleguen en esa hora.")
    balance_estimado: float = Field(
        description="llegadas - salidas. Negativo = la estación se vacía."
    )
    deficit_bicis: float = Field(
        ge=0,
        description="Bicis que habría que llevar a la estación para que no se quede sin servicio. "
                    "Es max(0, -balance). Cero si la estación acumula.",
    )
    saturacion: float | None = Field(
        description="Déficit como proporción de la capacidad. 1.0 significa que la estación se "
                    "vacía exactamente una vez en esa hora; 2.5 que se vacía dos veces y media."
    )

    modelo_tipo: str = Field(description="Qué produjo esta predicción.")
    modelo_version: str


class RespuestaLote(BaseModel):
    fecha_hora: datetime
    total: int
    modelo_tipo: str
    modelo_version: str
    predicciones: list[Prediccion]


# ─────────────────────────────────────────────────────────── disponibilidad
# MiBici presta servicio de 5:00 a 23:59. Preguntar por las 3 de la mañana no es un
# caso raro que haya que manejar: es una pregunta sin respuesta posible, y el contrato
# la rechaza con 422 en lugar de inventar un número.
HORA_MIN_SERVICIO = 5
HORA_MAX_SERVICIO = 23

DIAS = ("lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo")


class PeticionDisponibilidad(BaseModel):
    """«Voy a esta estación el viernes a las 6, ¿va a haber bicicleta?»

    El usuario piensa en día de la semana y hora, no en una fecha ISO. El API resuelve
    por su cuenta a qué fecha corresponde —la próxima vez que caiga ese día— y la
    devuelve en la respuesta, para que no quede ambiguo qué se predijo.
    """

    estacion_id: int = Field(
        ..., ge=1, le=99_999,
        description="Identificador de la estación, tal como viene en el catálogo.",
        examples=[194],
    )
    dia_semana: int = Field(
        ..., ge=0, le=6,
        description="0 = lunes, 1 = martes … 6 = domingo.",
        examples=[4],
    )
    hora: int = Field(
        ..., ge=HORA_MIN_SERVICIO, le=HORA_MAX_SERVICIO,
        description=f"Hora del día, de {HORA_MIN_SERVICIO} a {HORA_MAX_SERVICIO}. "
                    f"Fuera del horario de servicio de MiBici la petición se rechaza.",
        examples=[18],
    )

    model_config = {
        "json_schema_extra": {
            "examples": [{"estacion_id": 194, "dia_semana": 4, "hora": 18}]
        }
    }


class PeticionDisponibilidadLote(BaseModel):
    """La misma pregunta para muchas estaciones: alimenta el mapa de disponibilidad."""

    estaciones: list[int] | None = Field(
        None,
        description=f"Hasta {MAX_LOTE} estaciones. Si se omite, se consultan todas.",
        examples=[[194, 51, 271]],
    )
    dia_semana: int = Field(..., ge=0, le=6, examples=[4])
    hora: int = Field(..., ge=HORA_MIN_SERVICIO, le=HORA_MAX_SERVICIO, examples=[18])

    _lote = field_validator("estaciones")(PeticionLote.lote_acotado.__func__)


class Disponibilidad(BaseModel):
    """La respuesta que ve el usuario de MiBici."""

    estacion_id: int
    clave_estacion: str
    nombre: str
    municipio: str
    latitud: float | None
    longitud: float | None
    capacidad_anclajes: int | None

    dia_semana: int = Field(ge=0, le=6)
    nombre_dia: str
    hora: int
    fecha_resuelta: str = Field(
        description="Fecha concreta que se predijo: la próxima vez que cae ese día de la "
                    "semana. El mes y los festivos son variables del modelo, así que la "
                    "respuesta necesita una fecha, no solo un día."
    )

    hay_bicis: bool = Field(
        description="Resultado de comparar la probabilidad contra el umbral de operación."
    )
    bicis_estimadas: float = Field(
        ge=0, description="Bicicletas que se espera encontrar ancladas al inicio de esa hora."
    )
    probabilidad_sin_bicis: float = Field(
        ge=0, le=1,
        description="Probabilidad de que la estación se quede sin ninguna bicicleta "
                    "durante esa hora.",
    )
    nivel: Literal["alta", "media", "baja"] = Field(
        description="Disponibilidad esperada. Los cortes cuelgan del umbral medido en el "
                    "entrenamiento, no de un 0.5 arbitrario."
    )
    mensaje: str = Field(description="La respuesta en una frase, lista para mostrar.")

    umbral: float = Field(description="Umbral de operación del clasificador.")
    advertencia: str = Field(
        description="El supuesto sobre el que descansa esta estimación. Se devuelve en cada "
                    "respuesta a propósito: el inventario no se mide, se reconstruye."
    )
    modelo_tipo: str
    modelo_version: str


class RespuestaDisponibilidadLote(BaseModel):
    dia_semana: int
    nombre_dia: str
    hora: int
    fecha_resuelta: str
    total: int
    modelo_tipo: str
    modelo_version: str
    umbral: float
    advertencia: str
    disponibilidades: list[Disponibilidad]


class Estacion(BaseModel):
    estacion_id: int
    clave_estacion: str
    nombre: str
    municipio: str
    latitud: float | None
    longitud: float | None
    capacidad_anclajes: int | None


class Salud(BaseModel):
    estado: Literal["ok", "degradado"]
    modelo_cargado: bool
    modelo_tipo: str
    modelo_version: str
    estaciones_en_catalogo: int
    detalle: str | None = None

    # El segundo modelo se reporta aparte: el API puede estar sirviendo la demanda
    # perfectamente y no tener todavía el de disponibilidad. Mezclar los dos estados
    # en una sola bandera escondería exactamente lo que hay que ver.
    disponibilidad_cargada: bool = False
    disponibilidad_version: str | None = None
    disponibilidad_detalle: str | None = None

    servidor_hora: str = Field(
        description="Hora UTC en que el servidor atendió esta consulta. Sirve para "
                    "comprobar desde fuera que la respuesta es de ahora y no de una caché."
    )


class InfoModelo(BaseModel):
    tipo: str
    version: str
    descripcion: str
    entrenado_con: str
    metrica_principal: str
    valor_metrica: float | None
    baseline_rmse: float | None
    features: list[str]


class ErrorRespuesta(BaseModel):
    detalle: str
    codigo: str
