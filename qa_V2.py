import os
import google.generativeai as genai
import psycopg2
import pandas as pd

# Es una buena práctica cargar las variables de entorno al inicio.
# Si usas un archivo .env, puedes usar la librería python-dotenv.
# from dotenv import load_dotenv
# load_dotenv()

class GeminiAssistant:
    # CORRECCIÓN: El constructor debe llamarse __init__ (con dobles guiones bajos).
    # Un error común es escribir _init_ (simple), lo que causa que no se ejecute
    # al crear el objeto, provocando el error "'GeminiAssistant' object has no attribute 'model'".
    def __init__(self):
        """
        Inicializa el asistente, configurando la API de Gemini
        y los parámetros de conexión a la base de datos PostgreSQL.
        """
        # --- Configuración de Gemini ---
        try:
            api_key = os.getenv("GOOGLE_API_KEY") # Asegúrate que esta variable de entorno coincida
            if not api_key:
                raise ValueError("La variable de entorno 'GOOGLE_API_KEY' no está configurada.")
            
            genai.configure(api_key=api_key)
            # Usamos 'gemini-1.5-flash-latest' que es el alias recomendado para la última versión.
            self.model = genai.GenerativeModel('gemini-1.5-flash-latest')
            print(f"Asistente Gemini con modelo '{self.model.model_name}' inicializado.")

        except Exception as e:
            print(f"Error fatal durante la inicialización de Gemini: {e}")
            self.model = None

        # --- Configuración de la Base de Datos ---
        self.db_params = {
            "host": os.getenv("PG_HOST"),
            "port": os.getenv("PG_PORT"),
            "user": os.getenv("PG_USER"),
            "password": os.getenv("PG_PASSWORD"),
            "dbname": os.getenv("PG_DATABASE"),
        }
        # Verificamos que todos los parámetros de la BD estén presentes
        if not all(self.db_params.values()):
            print("ADVERTENCIA: Faltan una o más variables de entorno para la conexión a PostgreSQL (PG_HOST, PG_PORT, etc.).")

        # Definimos el esquema de la tabla para que el modelo sepa cómo consultarla.
        self.schema = """
        La base de datos PostgreSQL contiene una tabla llamada 'datos_clima' con el siguiente esquema:
        
        -fecha (DATE): La fecha del registro.
        -indicativo (VARCHAR): El código identificador de la estación meteorológica.
        -nombre (VARCHAR): El nombre de la estación (ej. 'PUERTO DE NAVACERRADA').
        -provincia (VARCHAR): La provincia donde se encuentra la estación.
        -altitud (INTEGER): La altitud de la estación en metros.
        -tmed (NUMERIC): La temperatura media registrada para ese día en grados Celsius.
        """

    def generar_sql_desde_pregunta(self, pregunta: str) -> str:
        """
        Convierte una pregunta en lenguaje natural a una consulta SQL para PostgreSQL.
        """
        if not self.model:
            return "ERROR: El modelo Gemini no está inicializado."

        prompt = f"""
        {self.schema}

        Basándote en el esquema anterior, convierte la siguiente pregunta del usuario en una consulta SQL para PostgreSQL.
        La fecha actual es {pd.Timestamp.now().strftime('%Y-%m-%d')}.
        Solo devuelve el código SQL, sin explicaciones ni formato extra, solo el SQL.

        Pregunta: "{pregunta}"
        SQL:
        """
        
        print(f"Generando SQL para la pregunta: '{pregunta}'")
        try:
            response = self.model.generate_content(prompt)
            sql_query = response.text.strip().replace("```sql", "").replace("```", "").strip()
            print(f"SQL generado: {sql_query}")
            return sql_query
        except Exception as e:
            print(f"Error al generar SQL con Gemini: {e}")
            return f"ERROR: No se pudo generar SQL. {e}"

    def ejecutar_sql(self, sql: str):
        """
        Ejecuta una consulta SQL en la base de datos PostgreSQL y devuelve los resultados.
        """
        if sql.startswith("ERROR"):
            return {"error": sql}
        
        print(f"Ejecutando SQL: {sql}")
        try:
            with psycopg2.connect(**self.db_params) as conn:
                with conn.cursor() as cur:
                    cur.execute(sql)
                    if cur.description: # Verificar si la consulta devuelve resultados
                        rows = cur.fetchall()
                        colnames = [desc[0] for desc in cur.description]
                        print(f"Consulta exitosa, se obtuvieron {len(rows)} filas.")
                        return [dict(zip(colnames, row)) for row in rows]
                    else: # Para consultas como INSERT o UPDATE que no devuelven filas
                        print("Consulta ejecutada exitosamente, no se devolvieron filas.")
                        return {"status": "success", "rows_affected": cur.rowcount}
        except Exception as e:
            print(f"Error al ejecutar SQL: {e}")
            return {"error": f"La consulta SQL generada falló: {e}. SQL intentado: {sql}"}

    def responder_pregunta(self, pregunta_original: str, datos: list):
        """
        Usa los datos de la base de datos para responder la pregunta original en lenguaje natural.
        """
        if not self.model:
            return "ERROR: El modelo Gemini no está inicializado."
        
        if isinstance(datos, dict) and "error" in datos:
            return f"No pude responder a tu pregunta porque hubo un error al consultar la base de datos: {datos['error']}"
        
        if not datos:
            return "No encontré datos para tu consulta. ¿Podrías reformular la pregunta o intentarlo con otras fechas/ubicaciones?"

        datos_str = pd.DataFrame(datos).to_string()

        prompt = f"""
        Eres un asistente del clima amigable y servicial.
        Basándote ESTRICTAMENTE en los siguientes datos, responde a la pregunta original del usuario de forma clara y concisa.

        Pregunta original: "{pregunta_original}"

        Datos obtenidos de la base de datos:
        {datos_str}

        Respuesta:
        """
        
        print("Generando respuesta en lenguaje natural...")
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"Pude obtener los datos, pero ocurrió un error al formular la respuesta: {e}"

# --- Bloque de prueba ---
if __name__ == '__main__':
    try:
        assistant = GeminiAssistant()
        
        # Solo ejecutamos la prueba si el asistente y la conexión a la BD se inicializaron bien
        if assistant.model and all(assistant.db_params.values()):
            pregunta = "¿Qué temperatura hizo en Madrid los últimos 3 días?"

            print("\n--- PASO 1: Generando SQL ---")
            sql = assistant.generar_sql_desde_pregunta(pregunta)
            
            if "ERROR" in sql:
                print(sql)
            else:
                print("\n--- PASO 2: Ejecutando SQL ---")
                datos = assistant.ejecutar_sql(sql)
                
                print("\n--- PASO 3: Redactando respuesta final ---")
                respuesta = assistant.responder_pregunta(pregunta, datos)
                print("\n======================")
                print("Respuesta Final:")
                print(respuesta)
                print("======================")
        else:
            print("\nPrueba no ejecutada debido a errores de inicialización.")

    except Exception as e:
        print(f"Ocurrió un error en el script principal: {e}")
    

    """
