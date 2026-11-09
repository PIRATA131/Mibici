# -*- coding: utf-8 -*-
"""Capa de predicción.

Hay dos implementaciones detrás de la misma interfaz:

  PredictorBaseline — promedio histórico por (estación, día de la semana, hora), calculado
                      sobre los últimos 12 meses. Es el punto de comparación obligatorio: un
                      modelo de ML que no le gane a esto no aporta nada.
  PredictorML       — el modelo entrenado. Se activa en cuanto el artefacto existe.

El API reporta en /modelo cuál está activo, así que nunca presenta un promedio como si fuera
una predicción del modelo.
"""
from __future__ import annotations

import csv
import os
from datetime import datetime
from pathlib import Path

DIR_DATOS = Path(os.getenv("DIR_DATOS", Path(__file__).resolve().parent.parent / "datos"))
RUTA_BASELINE = DIR_DATOS / "baseline_estacion_dia_hora.csv"
RUTA_MODELO = DIR_DATOS / "modelo_demanda.joblib"


class PredictorBaseline:
    """Promedio histórico. Sin parámetros aprendidos: es una tabla de consulta."""

    tipo = "baseline_historico"
    version = "1.0.0"
    descripcion = (
        "Promedio de salidas y llegadas por estación, día de la semana y hora, calculado sobre "
        "los 4,505,844 viajes válidos de los últimos 12 meses (nov-2025 a oct-2026). Se limita a "
        "12 meses porque la red creció de 300 a 470 estaciones entre 2023 y 2026 y promediar los "
        "cuatro años subestimaría las estaciones nuevas."
    )
    entrenado_con = "nov-2025 a oct-2026 — 4,505,844 viajes válidos"
    metrica_principal = "RMSE (viajes por estación-hora)"
    valor_metrica = None          # se mide en el notebook de evaluación
    baseline_rmse = None
    features = ["estacion_id", "dia_semana", "hora"]

    def __init__(self) -> None:
        self.tabla: dict[tuple[int, int, int], tuple[float, float]] = {}
        if not RUTA_BASELINE.exists():
            raise FileNotFoundError(f"No se encontró la tabla baseline en {RUTA_BASELINE}")
        with RUTA_BASELINE.open(encoding="utf-8") as fh:
            for fila in csv.DictReader(fh):
                clave = (int(fila["estacion_id"]), int(fila["dia_semana"]), int(fila["hora"]))
                self.tabla[clave] = (float(fila["salidas_promedio"]),
                                     float(fila["llegadas_promedio"]))

    def predecir(self, estacion_id: int, momento: datetime,
                 capacidad: float | None = None) -> tuple[float, float]:
        """Devuelve (salidas_estimadas, llegadas_estimadas). `capacidad` se ignora aquí:
        existe para que las dos implementaciones compartan firma.

        Una combinación ausente significa que esa estación nunca registró movimiento en ese
        día y esa hora: la predicción correcta es cero, no un error.
        """
        clave = (estacion_id, momento.weekday(), momento.hour)
        return self.tabla.get(clave, (0.0, 0.0))

    def predecir_muchos(self, peticiones: list[tuple[int, float | None]],
                        momento: datetime) -> list[tuple[float, float]]:
        """Aquí no hay nada que vectorizar: son búsquedas en un diccionario."""
        return [self.predecir(e, momento, cap) for e, cap in peticiones]


class PredictorML:
    """El modelo entrenado en el notebook 04.

    El artefacto trae todo lo necesario para reproducir en producción las mismas features
    con las que se entrenó: los dos modelos, el orden exacto de las columnas, la tabla de
    rezagos de las últimas 4 semanas y la lista de festivos.
    """

    tipo = "hist_gradient_boosting"
    metrica_principal = "RMSE (bicicletas por estación-hora)"

    def __init__(self, ruta: Path = RUTA_MODELO) -> None:
        import joblib                      # solo se importa si hay modelo que cargar

        a = joblib.load(ruta)
        faltan = {"modelos", "features", "rezagos", "festivos", "version"} - set(a)
        if faltan:
            raise ValueError(f"El artefacto no trae {sorted(faltan)}")

        self.version = a["version"]
        self.features = list(a["features"])
        self._modelos = a["modelos"]
        self._rezagos = a["rezagos"]
        self._festivos = set(a["festivos"])
        self.entrenado_con = a.get("entrenado_con", "")
        self._metricas = a.get("metricas", {})
        self._base = a.get("baseline", {})
        self.valor_metrica = self._metricas.get("salidas", {}).get("rmse")
        self.baseline_rmse = self._base.get("salidas", {}).get("rmse")

        mejora = ""
        if self.valor_metrica and self.baseline_rmse:
            mejora = (f" Mejora {100 * (1 - self.valor_metrica / self.baseline_rmse):.1f}% "
                      f"en RMSE sobre el promedio histórico.")
        self.descripcion = (
            f"Gradient boosting por histogramas, un modelo por objetivo (salidas y llegadas). "
            f"Entrenado sobre el panel Gold de demanda por estación y hora con partición "
            f"temporal.{mejora} Rezagos de la ventana {a.get('rezagos_ventana', 'n/d')}."
        )

    def _vector(self, estacion_id: int, momento: datetime, capacidad: float) -> list[float]:
        """Arma la fila de features en el MISMO orden en que se entrenó."""
        dia_semana = momento.weekday()
        # Sin historia para esa combinación, el supuesto correcto es cero movimiento,
        # que es lo que significa una estación-hora que nunca registró viajes.
        med_s, med_l, max_s = self._rezagos.get((estacion_id, dia_semana, momento.hour),
                                                (0.0, 0.0, 0.0))
        valores = {
            "hora": momento.hour,
            "dia_semana": dia_semana,
            "mes": momento.month,
            "es_fin_semana": 1 if dia_semana >= 5 else 0,
            "es_festivo": 1 if momento.strftime("%Y-%m-%d") in self._festivos else 0,
            "capacidad_anclajes": capacidad,
            "salidas_media_4sem": med_s,
            "llegadas_media_4sem": med_l,
            "salidas_max_4sem": max_s,
        }
        return [float(valores[f]) for f in self.features]

    def predecir(self, estacion_id: int, momento: datetime,
                 capacidad: float | None = None) -> tuple[float, float]:
        return self.predecir_muchos([(estacion_id, capacidad)], momento)[0]

    def predecir_muchos(self, peticiones: list[tuple[int, float | None]],
                        momento: datetime) -> list[tuple[float, float]]:
        """Una sola llamada a scikit-learn con la matriz completa.

        Importa de verdad: el mapa pide las 470 estaciones de un golpe, y 470 llamadas
        sueltas a `predict` tardan dos órdenes de magnitud más que una con 470 renglones.
        En una instancia de 0.1 CPU esa diferencia se ve a simple vista.
        """
        if not peticiones:
            return []
        X = [self._vector(e, momento, cap if cap is not None else 0.0)
             for e, cap in peticiones]
        salidas = self._modelos["salidas"].predict(X)
        llegadas = self._modelos["llegadas"].predict(X)
        # No existen conteos negativos: el modelo puede predecirlos, la realidad no.
        return [(max(0.0, float(s)), max(0.0, float(l)))
                for s, l in zip(salidas, llegadas)]


def cargar_predictor():
    """Prefiere el modelo entrenado; si no existe todavía, cae al baseline.

    La degradación es explícita y visible en /salud y /modelo: el API nunca miente sobre
    qué está respondiendo.
    """
    if RUTA_MODELO.exists():
        try:
            return PredictorML(), None
        except Exception as e:                                    # noqa: BLE001
            return PredictorBaseline(), f"El modelo existe pero no cargó ({e}); se usa el baseline."
    return PredictorBaseline(), "Todavía no hay modelo entrenado; se responde con el baseline histórico."
