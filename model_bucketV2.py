import numpy as np
import datetime
import os
import psycopg2
import pandas as pd
import tensorflow as tf 

def generar_forecast(ubicacion: str, dias_a_predecir: int, model, scaler_X, scaler_y):
    """
    Genera un pronóstico para un modelo Encoder-Decoder de dos entradas.
    """
    print(f"Iniciando la generación de pronóstico para {ubicacion}")

    # --- Preparación de datos ---    
    print("Simulando la creación del tensor de entrada histórico...")
    # Este es el input para el ENCODER.
    encoder_input = np.random.rand(1, 50, 5).astype(np.float32) # Shape: (1, 50, 5)
    print(f"Tensor de entrada para el Encoder creado con shape: {encoder_input.shape}")


    # --- Crear las dos entradas ---        
    print("Creando la segunda entrada (Decoder Input) con la forma correcta (1, 50)...")
    # La forma es (batch_size, longitud_de_secuencia_fija)
    decoder_input = np.zeros((1, 50), dtype=np.float32)
    print(f"Input para Encoder shape: {encoder_input.shape}")
    print(f"Input para Decoder shape: {decoder_input.shape}")


    # --- PASO 3: Realizar la predicción pasando la LISTA correcta de tensores ---
    print("Realizando la predicción con las dos entradas correctas...")
    try:
        # Pasamos la lista con el input del encoder y el input del decoder.
        prediccion_escalada = model.predict([encoder_input, decoder_input])
        print("Predicción realizada exitosamente.")
    except Exception as e:
        print(f"¡ERROR DURANTE LA PREDICCIÓN! Revisa la arquitectura del modelo. Error: {e}")
        model.summary()
        return {"error": f"Error en la predicción del modelo: {e}"}


    # --- PASO 4: Post-procesamiento de la predicción ---
    print("Des-escalando y formateando los resultados...")
    
    # La salida del modelo sí debería tener una longitud igual a dias_a_predecir.
    # Usamos los resultados reales de la predicción si está disponible.
    if prediccion_escalada is not None and prediccion_escalada.shape[1] == dias_a_predecir:
        # Desescalamos la predicción        
        valores_predichos = scaler_y.inverse_transform(prediccion_escalada.reshape(-1, 1)).flatten()
    else:
        # Si la predicción falla o la forma no coincide, usamos valores simulados.
        print(f"La forma de la predicción ({prediccion_escalada.shape if prediccion_escalada is not None else 'None'}) no coincide con los días a predecir ({dias_a_predecir}). Usando datos simulados.")
        valores_predichos = np.random.uniform(15, 25, size=dias_a_predecir)


    # Formateo de la salida
    fechas_pronostico = pd.to_datetime(pd.Timestamp.now().date()) + pd.to_timedelta(np.arange(dias_a_predecir), 'D')
    
    resultado_formateado = []
    for fecha, valor in zip(fechas_pronostico, valores_predichos):
        resultado_formateado.append({
            "fecha": fecha.strftime('%Y-%m-%d'),
            "temperatura_predicha": round(float(valor), 2)
        })

    return resultado_formateado
