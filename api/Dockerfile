# Se construye desde la RAÍZ del proyecto, porque la imagen necesita api/ y web/:
#   docker build -t mibici-api -f api/Dockerfile .
#   docker run -p 8000:8000 mibici-api
#
# En Render el build lo hace la plataforma; no hace falta Docker en la máquina local.
FROM python:3.12-slim

WORKDIR /srv

# Las dependencias primero: esta capa se reutiliza mientras requirements.txt no cambie,
# lo que importa porque scikit-learn y sus dependencias son la parte lenta del build.
COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api/app/   ./api/app/
COPY api/datos/ ./api/datos/
COPY web/       ./web/

ENV DIR_DATOS=/srv/api/datos \
    PYTHONPATH=/srv/api \
    PYTHONUNBUFFERED=1 \
    PORT=8000

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import os,urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen(f\"http://localhost:{os.environ.get('PORT','8000')}/salud\").status==200 else 1)"

# Forma shell para que $PORT se expanda: Render inyecta el puerto en tiempo de ejecución
# y el contenedor tiene que escuchar exactamente ahí.
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
