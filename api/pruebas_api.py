# -*- coding: utf-8 -*-
"""Pruebas del API. Se corren sin servidor: TestClient levanta la app en proceso.

    cd api && python pruebas_api.py

Cubren las tres cosas que la rúbrica pide del servicio: que prediga, que valide las
entradas y que los errores salgan con el código correcto.
"""
import sys
import time
from datetime import datetime

from fastapi.testclient import TestClient

from app.main import app

fallos: list[str] = []


def check(nombre: str, condicion: bool, detalle: str = "") -> None:
    print(f"  {'OK   ' if condicion else 'FALLA'} {nombre}{'  ' + detalle if detalle else ''}")
    if not condicion:
        fallos.append(nombre)


with TestClient(app) as c:
    print("Estado del servicio")
    s = c.get("/salud").json()
    check("responde /salud", s["estaciones_en_catalogo"] == 475,
          f"{s['estaciones_en_catalogo']} estaciones, modelo {s['modelo_tipo']}")
    check("declara qué modelo usa", s["modelo_cargado"] == (s["modelo_tipo"] != "baseline_historico"))

    print("\nPredicción")
    r = c.post("/predecir", json={"estacion_id": 194, "fecha_hora": "2026-11-20T18:00:00"}).json()
    check("devuelve la ficha de la estación", r["clave_estacion"] == "GDL-127")
    check("déficit = max(0, salidas - llegadas)",
          abs(r["deficit_bicis"] - max(0, r["salidas_estimadas"] - r["llegadas_estimadas"])) < 1e-6,
          f"déficit {r['deficit_bicis']}")
    check("saturación = déficit / capacidad",
          abs(r["saturacion"] - r["deficit_bicis"] / r["capacidad_anclajes"]) < 1e-3,
          f"{r['saturacion']}")

    t = time.perf_counter()
    d = c.post("/predecir/lote", json={"fecha_hora": "2026-11-20T18:00:00"}).json()
    ms = (time.perf_counter() - t) * 1000
    check("el lote cubre las estaciones ubicables", d["total"] == 470, f"{ms:.0f} ms")

    p194 = next(p for p in d["predicciones"] if p["estacion_id"] == 194)
    check("el lote coincide con la llamada individual",
          p194["salidas_estimadas"] == r["salidas_estimadas"]
          and p194["deficit_bicis"] == r["deficit_bicis"])

    print("\nValidación de entradas")
    casos = [
        ("estación inexistente",   "/predecir",      {"estacion_id": 99998, "fecha_hora": "2026-11-20T18:00:00"}, 404),
        ("estacion_id negativo",   "/predecir",      {"estacion_id": -5, "fecha_hora": "2026-11-20T18:00:00"}, 422),
        ("fecha fuera de rango",   "/predecir",      {"estacion_id": 194, "fecha_hora": "1999-01-01T10:00:00"}, 422),
        ("fecha no parseable",     "/predecir",      {"estacion_id": 194, "fecha_hora": "xx"}, 422),
        ("campo obligatorio falta","/predecir",      {"estacion_id": 194}, 422),
        ("lote vacío",             "/predecir/lote", {"estaciones": [], "fecha_hora": "2026-11-20T18:00:00"}, 422),
        ("lote con repetidos",     "/predecir/lote", {"estaciones": [1, 1], "fecha_hora": "2026-11-20T18:00:00"}, 422),
        ("lote de más de 500",     "/predecir/lote", {"estaciones": list(range(1, 502)), "fecha_hora": "2026-11-20T18:00:00"}, 422),
        ("lote con id inexistente","/predecir/lote", {"estaciones": [194, 99998], "fecha_hora": "2026-11-20T18:00:00"}, 404),
    ]
    for nombre, ruta, payload, esperado in casos:
        resp = c.post(ruta, json=payload)
        cuerpo = resp.json()
        tiene_forma = "detalle" in cuerpo or "detail" in cuerpo
        check(nombre, resp.status_code == esperado and tiene_forma, f"-> {resp.status_code}")

    print("\nDisponibilidad — ¿habrá bicicletas?")
    if not s["disponibilidad_cargada"]:
        # Sin artefacto el endpoint tiene que negarse, no inventar un número.
        r = c.post("/disponibilidad", json={"estacion_id": 194, "dia_semana": 4, "hora": 18})
        check("sin artefacto responde 503 y lo explica",
              r.status_code == 503 and "detalle" in r.json(),
              "corre el notebook 06 y deja el artefacto en api/datos/")
    else:
        r = c.post("/disponibilidad", json={"estacion_id": 194, "dia_semana": 4, "hora": 18})
        d = r.json()
        check("responde la pregunta del usuario", r.status_code == 200,
              f"{d.get('bicis_estimadas')} bicis, "
              f"p(sin bicis)={d.get('probabilidad_sin_bicis')}, nivel={d.get('nivel')}")
        check("resuelve el día pedido a una fecha concreta",
              datetime.strptime(d["fecha_resuelta"], "%Y-%m-%d").weekday() == 4,
              f"{d['nombre_dia']} {d['fecha_resuelta']}")
        check("la estimación respeta la capacidad física",
              d["bicis_estimadas"] <= d["capacidad_anclajes"],
              f"{d['bicis_estimadas']} de {d['capacidad_anclajes']} anclajes")
        check("hay_bicis se deriva del umbral medido",
              d["hay_bicis"] == (d["probabilidad_sin_bicis"] < d["umbral"]),
              f"umbral {d['umbral']}")
        check("declara el supuesto de la reconstrucción", bool(d["advertencia"]))

        # El horario de servicio es parte del contrato, no una comprobación interna.
        fuera = [h for h in (0, 3, 4, 24)
                 if c.post("/disponibilidad",
                           json={"estacion_id": 194, "dia_semana": 4, "hora": h}
                           ).status_code != 422]
        check("rechaza horas fuera del servicio (5–23)", not fuera, str(fuera))

        dentro = [h for h in range(5, 24)
                  if c.post("/disponibilidad",
                            json={"estacion_id": 194, "dia_semana": 4, "hora": h}
                            ).status_code != 200]
        check("acepta las 19 horas de servicio", not dentro, str(dentro))

        t = time.perf_counter()
        L = c.post("/disponibilidad/lote", json={"dia_semana": 4, "hora": 18}).json()
        ms = (time.perf_counter() - t) * 1000
        check("el lote cubre la ciudad", L["total"] == 470, f"{ms:.0f} ms")
        uno = next(x for x in L["disponibilidades"] if x["estacion_id"] == 194)
        check("el lote coincide con la llamada individual",
              abs(uno["bicis_estimadas"] - d["bicis_estimadas"]) < 1e-6
              and abs(uno["probabilidad_sin_bicis"] - d["probabilidad_sin_bicis"]) < 1e-6)
        check("la ficha del modelo declara sus 12 variables",
              len(c.get("/modelo/disponibilidad").json()["features"]) == 12)

    print("\nCatálogo y documentación")
    check("solo estaciones ubicables", len(c.get("/estaciones").json()) == 470)
    check("catálogo completo",         len(c.get("/estaciones?solo_ubicables=false").json()) == 475)
    check("ficha individual",          c.get("/estaciones/194").json()["clave_estacion"] == "GDL-127")
    check("404 en ficha inexistente",  c.get("/estaciones/99998").status_code == 404)
    rutas = set(c.get("/openapi.json").json()["paths"])
    esperadas = {"/salud", "/modelo", "/modelo/disponibilidad", "/estaciones",
                 "/estaciones/{estacion_id}", "/predecir", "/predecir/lote",
                 "/disponibilidad", "/disponibilidad/lote"}
    check("OpenAPI publica las 9 rutas", rutas == esperadas,
          str(sorted(esperadas ^ rutas)) if rutas != esperadas else "")
    check("Swagger UI disponible",     c.get("/docs").status_code == 200)
    check("la raíz lleva al mapa",
          c.get("/", follow_redirects=False).headers.get("location") == "/app/")
    check("la página web se sirve",    c.get("/app/index.html").status_code == 200)

print()
if fallos:
    print(f"FALLAN {len(fallos)}: {', '.join(fallos)}")
    sys.exit(1)
print("Todas las pruebas pasan.")
