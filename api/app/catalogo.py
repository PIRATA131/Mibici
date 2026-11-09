# -*- coding: utf-8 -*-
"""Catálogo de estaciones en memoria: 475 renglones, se carga una vez al arrancar."""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path

DIR_DATOS = Path(os.getenv("DIR_DATOS", Path(__file__).resolve().parent.parent / "datos"))
RUTA = DIR_DATOS / "dim_estacion.csv"


def _num(v: str):
    v = (v or "").strip()
    return None if v == "" else float(v)


@dataclass(frozen=True, slots=True)
class FichaEstacion:
    estacion_id: int
    clave_estacion: str
    nombre: str
    municipio: str
    latitud: float | None
    longitud: float | None
    capacidad_anclajes: int | None


class Catalogo:
    def __init__(self) -> None:
        if not RUTA.exists():
            raise FileNotFoundError(f"No se encontró el catálogo de estaciones en {RUTA}")
        self._fichas: dict[int, FichaEstacion] = {}
        with RUTA.open(encoding="utf-8") as fh:
            for f in csv.DictReader(fh):
                eid = int(f["estacion_id"])
                cap = _num(f["capacidad_anclajes"])
                self._fichas[eid] = FichaEstacion(
                    estacion_id=eid,
                    clave_estacion=f["clave_estacion"],
                    nombre=f["nombre_estacion"],
                    municipio=f["municipio"],
                    latitud=_num(f["latitud"]),
                    longitud=_num(f["longitud"]),
                    capacidad_anclajes=int(cap) if cap is not None else None,
                )

    def __len__(self) -> int:
        return len(self._fichas)

    def __contains__(self, estacion_id: int) -> bool:
        return estacion_id in self._fichas

    def obtener(self, estacion_id: int) -> FichaEstacion | None:
        return self._fichas.get(estacion_id)

    def todas(self) -> list[FichaEstacion]:
        return sorted(self._fichas.values(), key=lambda f: f.estacion_id)

    def ubicables(self) -> list[FichaEstacion]:
        """Solo las que tienen coordenadas: son las que el mapa puede pintar.
        Quedan fuera los 5 'unknown members' sin ficha en el catálogo vigente."""
        return [f for f in self.todas() if f.latitud is not None and f.longitud is not None]
