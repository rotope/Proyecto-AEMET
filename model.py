import tensorflow as tf
import numpy as np
import datetime
import os
import psycopg2
import boto3

# Parámetros de S3
BUCKET_NAME = "proyectofinalhab"
OBJECT_KEY = "carpeta/nombre_modelo.keras"  # ruta dentro del bucket
LOCAL_MODEL_PATH = "/tmp/nombre_modelo.keras"


s3 = boto3.client("s3")

# Descargar el modelo desde S3
s3.download_file(BUCKET_NAME, OBJECT_KEY, LOCAL_MODEL_PATH)

# Cargar el modelo desde el archivo descargado
modelo = tf.keras.models.load_model(LOCAL_MODEL_PATH)

def obtener_datos_historicos(ubicacion: str, dias: int = 20):
    try:
        conn = psycopg2.connect(
            host=os.getenv("PG_HOST"),
            port=os.getenv("PG_PORT"),
            user=os.getenv("PG_USER"),
            password=os.getenv("PG_PASSWORD"),
            dbname=os.getenv("PG_DATABASE"),
        )
        cur = conn.cursor()
        query = f"""
            SELECT temperatura
            FROM temperaturas
            WHERE ubicacion = %s
            ORDER BY fecha DESC
            LIMIT %s
        """
        cur.execute(query, (ubicacion, dias))
        resultados = cur.fetchall()
        # Devuelve en orden cronológico (el modelo necesita de más antiguo a más reciente)
        return [r[0] for r in resultados[::-1]]
    except Exception as e:
        print("Error al obtener datos históricos:", e)
        return []
    finally:
        if cur: cur.close()
        if conn: conn.close()

def generar_forecast(ubicacion: str, dias: int):
    ventana = obtener_datos_historicos(ubicacion, dias=20)
    
    if len(ventana) < 20:
        return [{"error": f"No hay suficientes datos históricos para {ubicacion}"}]

    predicciones = []
    for _ in range(dias):
            entrada = np.array(ventana[-20:]).reshape(1, 20, 1)
            temp_pred = modelo.predict(entrada)[0][0]
            predicciones.append(temp_pred)
            ventana.append(temp_pred)

    fechas = [(datetime.date.today() + datetime.timedelta(days=i)).isoformat() for i in range(dias)]
    return [{"fecha": fechas[i], "temperatura": round(predicciones[i], 2)} for i in range(dias)]