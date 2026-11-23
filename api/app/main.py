# -*- coding: utf-8 -*-
"""API de predicción de demanda y rebalanceo de MiBici.

Expone el modelo como servicio consultable (criterio 3 de la rúbrica). La página web
del proyecto consume estos endpoints por HTTP; no carga el modelo localmente.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

from fastapi import FastAPI, HTTPException, Path as PathParam, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from dataclasses import asdict
from pathlib import Path

from .catalogo import Catalogo, FichaEstacion
from .predictor import cargar_disponibilidad, cargar_predictor
from .schemas import (DIAS, Disponibilidad, Estacion, ErrorRespuesta, InfoModelo,
                      PeticionDisponibilidad, PeticionDisponibilidadLote, PeticionLote,
                      PeticionPrediccion, Prediccion, RespuestaDisponibilidadLote,
                      RespuestaLote, Salud)

ESTADO: dict = {}


@asynccontextmanager
async def ciclo_vida(app: FastAPI):
    """Catálogo y predictores se cargan una sola vez, al arrancar: cada petición
    después es una consulta en memoria."""
    ESTADO["catalogo"] = Catalogo()
    ESTADO["predictor"], ESTADO["aviso"] = cargar_predictor()
    ESTADO["disponibilidad"], ESTADO["aviso_disponibilidad"] = cargar_disponibilidad()
    yield
    ESTADO.clear()


app = FastAPI(
    title="API de demanda y rebalanceo de MiBici",
    version="1.0.0",
    lifespan=ciclo_vida,
    description=(
        "Predice cuántas bicicletas salen y llegan a cada estación de MiBici en una hora dada, "
        "y cuántas habría que trasladar para que la estación no se quede sin servicio.\n\n"
        "**Proyecto final** — Aprendizaje Automático para Grandes Volúmenes de Datos, "
        "Universidad Panamericana campus Guadalajara.\n\n"
        "**Equipo:** César Roberto Murillo Figueroa y Andrés Muñoz Rubalcava.\n\n"
        "Entrenado sobre 17,468,665 viajes de los datos abiertos de MiBici (ene-2023 a oct-2026)."
    ),
    contact={"name": "Equipo MiBici"},
    license_info={"name": "Datos abiertos del Instituto de Movilidad de Jalisco"},
)

# La página web se sirve desde otro origen, así que necesita CORS.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _armar(ficha: FichaEstacion, momento: datetime,
           salidas: float, llegadas: float) -> Prediccion:
    """Arma la respuesta a partir de una predicción ya calculada. Separar el cálculo del
    armado es lo que permite que el lote haga una sola pasada por el modelo."""
    pred = ESTADO["predictor"]
    balance = round(llegadas - salidas, 3)
    deficit = round(max(0.0, -balance), 3)
    cap = ficha.capacidad_anclajes
    return Prediccion(
        estacion_id=ficha.estacion_id,
        clave_estacion=ficha.clave_estacion,
        nombre=ficha.nombre,
        municipio=ficha.municipio,
        latitud=ficha.latitud,
        longitud=ficha.longitud,
        capacidad_anclajes=cap,
        fecha_hora=momento.replace(minute=0, second=0, microsecond=0),
        dia_semana=momento.weekday(),
        hora=momento.hour,
        salidas_estimadas=round(salidas, 3),
        llegadas_estimadas=round(llegadas, 3),
        balance_estimado=balance,
        deficit_bicis=deficit,
        saturacion=round(deficit / cap, 3) if cap else None,
        modelo_tipo=pred.tipo,
        modelo_version=pred.version,
    )


@app.get("/salud", response_model=Salud, tags=["Operación"],
         summary="Estado del servicio")
def salud() -> Salud:
    """Para monitoreo y para que la página web sepa si el API está listo antes de pedir nada."""
    pred = ESTADO.get("predictor")
    cat = ESTADO.get("catalogo")
    if pred is None or cat is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail="El servicio todavía no termina de arrancar")
    aviso = ESTADO.get("aviso")
    disp = ESTADO.get("disponibilidad")
    return Salud(
        estado="ok" if pred.tipo != "baseline_historico" else "degradado",
        modelo_cargado=pred.tipo != "baseline_historico",
        modelo_tipo=pred.tipo,
        modelo_version=pred.version,
        estaciones_en_catalogo=len(cat),
        detalle=aviso,
        disponibilidad_cargada=disp is not None,
        disponibilidad_version=disp.version if disp else None,
        disponibilidad_detalle=ESTADO.get("aviso_disponibilidad"),
        servidor_hora=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )


@app.get("/modelo", response_model=InfoModelo, tags=["Operación"],
         summary="Qué modelo está respondiendo")
def info_modelo() -> InfoModelo:
    p = ESTADO["predictor"]
    return InfoModelo(
        tipo=p.tipo, version=p.version, descripcion=p.descripcion,
        entrenado_con=p.entrenado_con, metrica_principal=p.metrica_principal,
        valor_metrica=p.valor_metrica,
        baseline_rmse=getattr(p, "baseline_rmse", None),
        features=list(p.features),
    )


@app.get("/estaciones", response_model=list[Estacion], tags=["Catálogo"],
         summary="Catálogo de estaciones")
def estaciones(solo_ubicables: bool = True) -> list[Estacion]:
    """El mapa pide esto una vez al cargar. Por defecto devuelve solo las estaciones con
    coordenadas; `solo_ubicables=false` incluye las 5 que aparecen en viajes pero ya no
    están en el catálogo vigente."""
    cat = ESTADO["catalogo"]
    fichas = cat.ubicables() if solo_ubicables else cat.todas()
    return [Estacion(**asdict(f)) for f in fichas]


@app.post("/predecir", response_model=Prediccion, tags=["Predicción"],
          summary="Predicción para una estación",
          responses={404: {"model": ErrorRespuesta, "description": "La estación no existe"}})
def predecir(peticion: PeticionPrediccion) -> Prediccion:
    ficha = ESTADO["catalogo"].obtener(peticion.estacion_id)
    if ficha is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            detail=f"La estación {peticion.estacion_id} no está en el catálogo. "
                   f"Consulta /estaciones para ver las disponibles.",
        )
    salidas, llegadas = ESTADO["predictor"].predecir(
        peticion.estacion_id, peticion.fecha_hora, ficha.capacidad_anclajes)
    return _armar(ficha, peticion.fecha_hora, salidas, llegadas)


@app.post("/predecir/lote", response_model=RespuestaLote, tags=["Predicción"],
          summary="Predicción para varias estaciones a la misma hora",
          responses={404: {"model": ErrorRespuesta, "description": "Alguna estación no existe"}})
def predecir_lote(peticion: PeticionLote) -> RespuestaLote:
    """Una sola llamada alimenta el mapa completo de la ciudad."""
    cat = ESTADO["catalogo"]
    if peticion.estaciones is None:
        fichas = cat.ubicables()
    else:
        faltantes = [e for e in peticion.estaciones if e not in cat]
        if faltantes:
            raise HTTPException(
                status.HTTP_404_NOT_FOUND,
                detail=f"Estaciones fuera del catálogo: {faltantes[:10]}"
                       f"{'…' if len(faltantes) > 10 else ''}",
            )
        fichas = [cat.obtener(e) for e in peticion.estaciones]

    p = ESTADO["predictor"]
    estimados = p.predecir_muchos(
        [(f.estacion_id, f.capacidad_anclajes) for f in fichas], peticion.fecha_hora)
    preds = [_armar(f, peticion.fecha_hora, s, l)
             for f, (s, l) in zip(fichas, estimados)]
    return RespuestaLote(
        fecha_hora=peticion.fecha_hora.replace(minute=0, second=0, microsecond=0),
        total=len(preds), modelo_tipo=p.tipo, modelo_version=p.version, predicciones=preds,
    )


# ──────────────────────────────────────────────────────────── disponibilidad

def _proxima_fecha(dia_semana: int, hora: int) -> datetime:
    """La próxima vez que caiga ese día de la semana a esa hora.

    El usuario pregunta por «el viernes», pero el modelo usa el mes y si es festivo como
    variables. Alguien tiene que convertir «el viernes» en una fecha concreta, y es mejor
    que lo haga el API y lo declare en la respuesta, a que cada cliente lo resuelva por su
    cuenta y cada uno prediga un viernes distinto.
    """
    ahora = datetime.now()
    objetivo = (ahora.replace(minute=0, second=0, microsecond=0, hour=hora)
                + timedelta(days=(dia_semana - ahora.weekday()) % 7))
    if objetivo <= ahora:
        objetivo += timedelta(days=7)
    return objetivo


def _disponibilidad_o_503():
    disp = ESTADO.get("disponibilidad")
    if disp is None:
        # Sin artefacto no hay reconstrucción de ocupación, y cualquier número que
        # devolviéramos sería inventado. Mejor decir que no se puede.
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=ESTADO.get("aviso_disponibilidad")
                   or "El modelo de disponibilidad no está cargado.",
        )
    return disp


def _frase(nivel: str, bicis: float, prob: float) -> str:
    if nivel == "alta":
        return (f"Sí vas a encontrar bicicleta: se esperan unas {bicis:.0f} ancladas "
                f"al inicio de esa hora.")
    if nivel == "media":
        return (f"Probablemente sí, pero con poco margen: se esperan unas {bicis:.0f} "
                f"bicicletas y hay {prob:.0%} de probabilidad de que se agoten.")
    return (f"Mejor considera otra estación: hay {prob:.0%} de probabilidad de que no "
            f"quede ninguna bicicleta en esa hora.")


def _armar_disponibilidad(ficha: FichaEstacion, momento: datetime,
                          bicis: float, prob: float) -> Disponibilidad:
    disp = ESTADO["disponibilidad"]
    nivel = disp.nivel(prob)
    return Disponibilidad(
        estacion_id=ficha.estacion_id,
        clave_estacion=ficha.clave_estacion,
        nombre=ficha.nombre,
        municipio=ficha.municipio,
        latitud=ficha.latitud,
        longitud=ficha.longitud,
        capacidad_anclajes=ficha.capacidad_anclajes,
        dia_semana=momento.weekday(),
        nombre_dia=DIAS[momento.weekday()],
        hora=momento.hour,
        fecha_resuelta=momento.strftime("%Y-%m-%d"),
        hay_bicis=prob < disp.umbral,
        bicis_estimadas=round(bicis, 2),
        probabilidad_sin_bicis=round(prob, 4),
        nivel=nivel,
        mensaje=_frase(nivel, bicis, prob),
        umbral=round(disp.umbral, 4),
        advertencia=disp.advertencia,
        modelo_tipo=disp.tipo,
        modelo_version=disp.version,
    )


@app.post("/disponibilidad", response_model=Disponibilidad, tags=["Disponibilidad"],
          summary="¿Habrá bicicletas en esta estación?",
          responses={404: {"model": ErrorRespuesta, "description": "La estación no existe"},
                     503: {"model": ErrorRespuesta,
                           "description": "El modelo de disponibilidad no está cargado"}})
def disponibilidad(peticion: PeticionDisponibilidad) -> Disponibilidad:
    """Responde la pregunta del usuario de MiBici: *«voy a esta estación el viernes a las
    6, ¿va a haber bicicleta?»*

    Devuelve las dos mitades de la respuesta —cuántas bicicletas y qué tan probable es que
    no quede ninguna— porque una media de 2 bicicletas puede venir de «casi siempre hay 2»
    o de «la mitad de los días no hay ninguna», y no son lo mismo para quien va a caminar
    hasta allá.
    """
    disp = _disponibilidad_o_503()
    ficha = ESTADO["catalogo"].obtener(peticion.estacion_id)
    if ficha is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            detail=f"La estación {peticion.estacion_id} no está en el catálogo. "
                   f"Consulta /estaciones para ver las disponibles.",
        )
    momento = _proxima_fecha(peticion.dia_semana, peticion.hora)
    bicis, prob = disp.predecir(peticion.estacion_id, momento, ficha.capacidad_anclajes)
    return _armar_disponibilidad(ficha, momento, bicis, prob)


@app.post("/disponibilidad/lote", response_model=RespuestaDisponibilidadLote,
          tags=["Disponibilidad"],
          summary="Disponibilidad de varias estaciones en la misma franja",
          responses={404: {"model": ErrorRespuesta, "description": "Alguna estación no existe"},
                     503: {"model": ErrorRespuesta,
                           "description": "El modelo de disponibilidad no está cargado"}})
def disponibilidad_lote(peticion: PeticionDisponibilidadLote) -> RespuestaDisponibilidadLote:
    """Alimenta el mapa de disponibilidad: una sola llamada pinta la ciudad completa."""
    disp = _disponibilidad_o_503()
    cat = ESTADO["catalogo"]
    if peticion.estaciones is None:
        fichas = cat.ubicables()
    else:
        faltantes = [e for e in peticion.estaciones if e not in cat]
        if faltantes:
            raise HTTPException(
                status.HTTP_404_NOT_FOUND,
                detail=f"Estaciones fuera del catálogo: {faltantes[:10]}"
                       f"{'…' if len(faltantes) > 10 else ''}",
            )
        fichas = [cat.obtener(e) for e in peticion.estaciones]

    momento = _proxima_fecha(peticion.dia_semana, peticion.hora)
    estimados = disp.predecir_muchos(
        [(f.estacion_id, f.capacidad_anclajes) for f in fichas], momento)
    filas = [_armar_disponibilidad(f, momento, b, p)
             for f, (b, p) in zip(fichas, estimados)]
    return RespuestaDisponibilidadLote(
        dia_semana=momento.weekday(),
        nombre_dia=DIAS[momento.weekday()],
        hora=momento.hour,
        fecha_resuelta=momento.strftime("%Y-%m-%d"),
        total=len(filas),
        modelo_tipo=disp.tipo,
        modelo_version=disp.version,
        umbral=round(disp.umbral, 4),
        advertencia=disp.advertencia,
        disponibilidades=filas,
    )


@app.get("/modelo/disponibilidad", response_model=InfoModelo, tags=["Operación"],
         summary="Ficha del modelo de disponibilidad",
         responses={503: {"model": ErrorRespuesta,
                          "description": "El modelo de disponibilidad no está cargado"}})
def info_disponibilidad() -> InfoModelo:
    d = _disponibilidad_o_503()
    return InfoModelo(
        tipo=d.tipo, version=d.version, descripcion=d.descripcion,
        entrenado_con=d.entrenado_con, metrica_principal=d.metrica_principal,
        valor_metrica=d.valor_metrica, baseline_rmse=d.baseline_rmse,
        features=list(d.features),
    )


@app.get("/estaciones/{estacion_id}", response_model=Estacion, tags=["Catálogo"],
         summary="Ficha de una estación",
         responses={404: {"model": ErrorRespuesta, "description": "La estación no existe"}})
def estacion(estacion_id: int = PathParam(ge=1, le=99_999)) -> Estacion:
    ficha = ESTADO["catalogo"].obtener(estacion_id)
    if ficha is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND,
                            detail=f"La estación {estacion_id} no está en el catálogo")
    return Estacion(**asdict(ficha))


# La página web se sirve desde el mismo contenedor: un solo despliegue y una sola URL
# para el profesor. Sigue consumiendo el API por HTTP, solo que contra el mismo origen.
_WEB = Path(__file__).resolve().parent.parent.parent / "web"
if _WEB.is_dir():
    app.mount("/app", StaticFiles(directory=str(_WEB), html=True), name="web")


@app.get("/", include_in_schema=False)
def raiz():
    """La raíz lleva al mapa si está disponible, y si no a la documentación del API."""
    return RedirectResponse("/app/" if _WEB.is_dir() else "/docs")


@app.exception_handler(HTTPException)
async def http_error(request, exc: HTTPException):
    """Respuesta de error uniforme, para que la página web siempre pueda leer `detalle`."""
    codigos = {404: "NO_ENCONTRADO", 422: "ENTRADA_INVALIDA", 503: "NO_DISPONIBLE"}
    return JSONResponse(
        status_code=exc.status_code,
        content={"detalle": exc.detail, "codigo": codigos.get(exc.status_code, "ERROR")},
    )
