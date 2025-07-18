from fastapi import FastAPI, Query
from pydantic import BaseModel
from model import generar_forecast
from qa import GeminiAssistant

app = FastAPI(title="API de Predicción y Preguntas")

assistant = GeminiAssistant() 

class ForecastRequest(BaseModel):
    ubicacion: str
    dias: int

class AskRequest(BaseModel):
    pregunta: str

@app.post("/ask")
async def ask(request: AskRequest):
    pregunta = request.pregunta

    # 1. Generar SQL desde la pregunta
    sql = assistant.generar_sql_desde_pregunta(pregunta)

    # 2. Ejecutar la consulta
    datos = assistant.ejecutar_sql(sql)

    # 3. Verificar errores
    if isinstance(datos, dict) and "error" in datos:
        return {"error": datos["error"], "sql": sql}

    # 4. Generar respuesta en lenguaje natural
    respuesta = assistant.responder_pregunta(pregunta, datos)
    return {"respuesta": respuesta, "sql_generada": sql}

@app.post("/forecast")
async def forecast(request: ForecastRequest):
    prediccion = generar_forecast(request.ubicacion, request.dias)
    return {
        "ubicacion": request.ubicacion,
        "dias": request.dias,
        "prediccion": prediccion
    }