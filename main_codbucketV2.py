from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import os
import boto3
import tempfile

# Importaciones de lógica de negocio
# Nombres de archivo corregidos según tu estructura.
from model_bucketV2 import generar_forecast 
from qa_V2 import GeminiAssistant

# Importaciones de carga de modelos
from tensorflow.keras.models import load_model
from joblib import load

# --- Configuración de S3 ---
S3_BUCKET_NAME = "aemet-raw"
S3_MODEL_PREFIX = "data/modelos-pesos/"

# --- Variables Globales ---
# Modelos para el pronóstico
model = None
label_encoder = None
scaler_X = None
scaler_y = None

# Instancia del asistente de preguntas
assistant = GeminiAssistant()

# Inicializar cliente S3
s3_client = boto3.client('s3')

# --- Función de Lifespan para cargar recursos al inicio ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Función de lifespan para FastAPI.
    Se ejecuta al iniciar la aplicación para cargar los modelos desde S3.
    """
    global model, label_encoder, scaler_X, scaler_y
    print("Iniciando carga de modelo y scalers desde S3...")

    temp_files = {}
    try:
        # Cargar el modelo Keras
        model_key = f"{S3_MODEL_PREFIX}2025-Encoder-Decoder.keras"
        print(f"Descargando modelo Keras: s3://{S3_BUCKET_NAME}/{model_key}")
        with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as temp_file:
            s3_client.download_fileobj(S3_BUCKET_NAME, model_key, temp_file)
            temp_files['model'] = temp_file.name
        model = load_model(temp_files['model'])
        print("Modelo Keras cargado exitosamente.")

        # Cargar los scalers y el label encoder
        files_to_load = {
            'le': ("label_encoder_global_all_stations.joblib", "LabelEncoder"),
            'scaler_X': ("scaler_X_base.joblib", "Scaler_X"),
            'scaler_y': ("scaler_y_base.joblib", "Scaler_y")
        }

        for key, (filename, name) in files_to_load.items():
            file_key = f"{S3_MODEL_PREFIX}{filename}"
            print(f"Descargando {name}: s3://{S3_BUCKET_NAME}/{file_key}")
            with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as temp_file:
                s3_client.download_fileobj(S3_BUCKET_NAME, file_key, temp_file)
                temp_files[key] = temp_file.name
            
            if key == 'le':
                label_encoder = load(temp_files[key])
            elif key == 'scaler_X':
                scaler_X = load(temp_files[key])
            elif key == 'scaler_y':
                scaler_y = load(temp_files[key])
            print(f"{name} cargado exitosamente.")

        print("Todos los recursos de ML cargados exitosamente desde S3.")
    except Exception as e:
        print(f"ERROR FATAL: No se pudieron cargar los recursos desde S3. Error: {e}")
        raise RuntimeError(f"Fallo al cargar modelos desde S3: {e}")
    finally:
        # Aseguramos que los archivos temporales se eliminen
        for path in temp_files.values():
            if os.path.exists(path):
                os.remove(path)

    yield # La aplicación está lista para recibir solicitudes
    
    print("Cerrando la aplicación y liberando recursos.")


# --- Inicialización de la App FastAPI ---
app = FastAPI(title="API de Predicción y Preguntas", lifespan=lifespan)

# --- Modelos Pydantic para las solicitudes ---
class ForecastRequest(BaseModel):
    ubicacion: str
    dias: int

class AskRequest(BaseModel):
    pregunta: str

# --- Endpoints de la API ---
@app.post("/ask")
async def ask(request: AskRequest):
    """Endpoint para hacer preguntas en lenguaje natural."""
    try:
        pregunta = request.pregunta
        print(f"Recibida pregunta: '{pregunta}'")

        # 1. Generar SQL desde la pregunta
        sql = assistant.generar_sql_desde_pregunta(pregunta)

        # 2. Ejecutar la consulta
        datos = assistant.ejecutar_sql(sql)

        # 3. Verificar errores en los datos
        if isinstance(datos, dict) and "error" in datos:
            # Devuelve el error específico para facilitar la depuración
            raise HTTPException(status_code=400, detail=f"Error al ejecutar SQL: {datos['error']}")

        # 4. Generar respuesta en lenguaje natural
        respuesta = assistant.responder_pregunta(pregunta, datos)
        return {"respuesta": respuesta, "sql_generada": sql}
    except Exception as e:
        # Captura cualquier otro error inesperado
        raise HTTPException(status_code=500, detail=f"Ocurrió un error interno: {e}")


@app.post("/forecast")
async def forecast(request: ForecastRequest):
    """Endpoint para generar un pronóstico del tiempo."""
    try:
        print(f"Solicitud de pronóstico para ubicación: {request.ubicacion}, días: {request.dias}")

        # Llamar a la función de pronóstico, pasando los modelos cargados globalmente
        forecast_results = generar_forecast(
            ubicacion=request.ubicacion.upper(),
            dias_a_predecir=request.dias,
            model=model, 
            scaler_X=scaler_X, 
            scaler_y=scaler_y, 
            label_encoder=label_encoder
        )

        if isinstance(forecast_results, dict) and "error" in forecast_results:
            raise HTTPException(status_code=400, detail=forecast_results["error"])

        return {"ubicacion": request.ubicacion, "pronostico": forecast_results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ocurrió un error interno en el pronóstico: {e}")
