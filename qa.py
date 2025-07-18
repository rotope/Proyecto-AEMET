import google.generativeai as genai
import os
from dotenv import load_dotenv
import psycopg2

load_dotenv()

class GeminiAssistant:
    def __init__(self):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("La variable de entorno 'GOOGLE_API_KEY' no está configurada. Crea un archivo .env con tu clave.")

        genai.configure(api_key=api_key) 
        self.model = genai.GenerativeModel('models/gemini-2.5-flash') 

        print(f"Modelo {self.model.model_name} configurado y listo para usar.") 

        self.db_params={
        "host": os.getenv("PG_HOST"),
        "port": os.getenv("PG_PORT"),
        "user": os.getenv("PG_USER"),
        "password": os.getenv("PG_PASSWORD"),
        "dbname": os.getenv("PG_DATABASE"),
        }

  
    def generar_sql_desde_pregunta(self, pregunta: str, temperature=0.3):
        prompt = f"""
    Eres un asistente que convierte preguntas en lenguaje natural a SQL para PostgreSQL.

    Tabla: `temperaturas`
    Columnas:
    - fecha (DATE)
    - ubicacion (TEXT)
    - temperatura (REAL)

    Devuelve solo una consulta SQL `SELECT` que devuelva las temperaturas para una ubicación y un número de días.

    Pregunta del usuario:
    \"\"\"{pregunta}\"\"\"
    """
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(temperature=temperature)
            )
            return response.text.strip()
        except Exception as e:
            return f"Error generando SQL: {e}"  
        

      
    def ejecutar_sql(self, sql: str):
        try:
            conn = psycopg2.connect(**self.db_params)
            cur = conn.cursor()
            cur.execute(sql)
            rows = cur.fetchall()
            colnames = [desc[0] for desc in cur.description]
            return [dict(zip(colnames, row)) for row in rows]
        except Exception as e:
            return {"error": str(e)}
        finally:
            if 'cur' in locals():
                cur.close()
            if 'conn' in locals():
                conn.close()

    def responder_pregunta(self, pregunta_original: str, datos: list):
        prompt = f"""
    Eres un asistente meteorológico.

    Pregunta del usuario:
    \"\"\"{pregunta_original}\"\"\"

    Datos obtenidos de la base:
    {datos}

    Redacta una respuesta breve en lenguaje natural para el usuario, mencionando fechas, ubicación y temperaturas. No muestres SQL.
    """

        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"Error generando respuesta final: {e}"




assistant = GeminiAssistant()

if __name__ == '__main__':
   
    pregunta = "¿Qué temperatura hizo en Madrid los últimos 3 días?"

    print(" Generando SQL...")
    sql = assistant.generar_sql_desde_pregunta(pregunta)
    if not sql.lower().strip().startswith("select"):
        print(" Gemini no devolvió SQL válida:", sql)
        exit()    
    print(" SQL generado:\n", sql)

    print("\nEjecutando SQL...")
    datos = assistant.ejecutar_sql(sql)
    print(" Datos devueltos:\n", datos)

    if isinstance(datos, dict) and "error" in datos:
        print(" Error al ejecutar SQL:", datos["error"])
    else:
        print("\n Redactando respuesta final...")
        respuesta = assistant.responder_pregunta(pregunta, datos)
        print(" Respuesta final:")
        print(respuesta)

    





    """OLLAMA EN LOCAL 
    import requests

def responder_pregunta(pregunta: str) -> str:
    try:
        response = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": "llama3",
                "messages": [
                    {"role": "system", "content": "Eres un experto en meteorología."},
                    {"role": "user", "content": pregunta}
                ]
            }
        )
        data = response.json()
        return data.get("message", {}).get("content", "").strip()
    except Exception as e:
        return f"Error al consultar LLaMA 3: {str(e)}"
    
    
    """