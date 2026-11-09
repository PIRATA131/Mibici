# -*- coding: utf-8 -*-
"""Pruebas del API. Se corren sin servidor: TestClient levanta la app en proceso.

    cd api && python pruebas_api.py

Cubren las tres cosas que la rúbrica pide del servicio: que prediga, que valide las
entradas y que los errores salgan con el código correcto.
"""
import sys
import time

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

    print("\nCatálogo y documentación")
    check("solo estaciones ubicables", len(c.get("/estaciones").json()) == 470)
    check("catálogo completo",         len(c.get("/estaciones?solo_ubicables=false").json()) == 475)
    check("ficha individual",          c.get("/estaciones/194").json()["clave_estacion"] == "GDL-127")
    check("404 en ficha inexistente",  c.get("/estaciones/99998").status_code == 404)
    check("OpenAPI publica 6 rutas",   len(c.get("/openapi.json").json()["paths"]) == 6)
    check("Swagger UI disponible",     c.get("/docs").status_code == 200)
    check("la raíz lleva al mapa",
          c.get("/", follow_redirects=False).headers.get("location") == "/app/")
    check("la página web se sirve",    c.get("/app/index.html").status_code == 200)

print()
if fallos:
    print(f"FALLAN {len(fallos)}: {', '.join(fallos)}")
    sys.exit(1)
print("Todas las pruebas pasan.")
