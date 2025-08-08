🌍 API de Predicción de Temperatura y Asistente Gemini sobre AEMET 

Bienvenido al repositorio de la API de Predicción de Temperatura y Asistente Gemini. Este proyecto es un servicio web basado en FastAPI que combina un modelo de Machine Learning de tipo Encoder-Decoder para predecir la temperatura y un asistente virtual integrado para consultas de datos históricos.

La API se despliega en una instancia de AWS EC2 y utiliza PostgreSQL como base de datos para almacenar datos meteorológicos históricos. Los modelos de ML se cargan desde un bucket de S3 para una gestión de recursos eficiente.

🚀 Características Principales

    Predicción de Temperatura: Utiliza un modelo de Encoder-Decoder entrenado con TensorFlow para generar pronósticos de temperatura a futuro.

    Asistente Inteligente: Integra un asistente conversacional (basado en Gemini) que puede responder a preguntas sobre los datos históricos disponibles en la base de datos.

    Arquitectura Robusta: Los modelos de ML se cargan al inicio de la aplicación desde AWS S3, garantizando que el servicio esté siempre listo para responder sin demoras.

    API RESTful: Ofrece endpoints RESTful claros para la predicción de pronósticos y para interactuar con el asistente.


🛠️ Prerrequisitos

Para ejecutar este proyecto, necesitas tener instalados los siguientes componentes:

    -Python 3.9+

    -PostgreSQL (con una base de datos configurada y datos históricos)

    -Acceso a AWS (con credenciales configuradas para S3)



⚙️ Instalación y Configuración

Sigue estos pasos para poner el proyecto en marcha en tu entorno local o en una instancia de EC2.


1. Clonar el Repositorio
	git clone <URL_DE_TU_REPOSITORIO>
	cd <nombre_de_tu_repositorio>

2. Configurar el Entorno Virtual
   
	python3 -m venv venv

	source venv/bin/activate

	pip install -r requirements.txt

4. Configurar las Variables de Entorno
	El proyecto usa variables de entorno para conectarse a la base de datos PostgreSQL. Debes definirlas en tu terminal antes de iniciar la aplicación.
	Reemplaza los valores de ejemplo con tus credenciales reales:

		export PG_HOST="<tu_host_de_postgresql>"
		export PG_PORT="5432" # o el puerto que uses
		export PG_USER="<tu_usuario_de_postgresql>"
		export PG_PASSWORD="<tu_contraseña_de_postgresql>"
		export PG_DATABASE="<tu_base_de_datos_de_postgresql>"

   		Nota: Para un entorno de producción, se recomienda usar un método más seguro para gestionar estas variables, como AWS Secrets Manager o un archivo .env cargado de forma segura.

5. Iniciar la Aplicación

	Una vez que las variables de entorno están configuradas, puedes iniciar el servidor Uvicorn. Es importante usar el flag --workers 1 para la carga de modelos.
	
	uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
	
	#Si todo está configurado correctamente, verás en la consola que los modelos se cargan desde S3 y que la aplicación se inicia sin errores.

🧪 Uso de la API

Una vez que la API está funcionando, puedes interactuar con ella a través de dos endpoints principales.

	Streamlit APP:
		(Provisional) https://proyecto-aemet-mun7ecniqswapp7s9ljznnz.streamlit.app/  
	
	Desde Consola:
		Endpoint 1: Predicción de Temperatura
	
		Este endpoint toma una ubicación y un número de días para predecir la temperatura.
	
	   			Ruta: /forecast
	
	    		Método: POST
	
	    		Payload: JSON con los campos ubicacion (string) y dias (integer).
	
		Ejemplo de uso con curl:
	
		curl -X POST "http://<IP_PUBLICA_EC2>:8000/forecast" \
		-H "Content-Type: application/json" \
		-d '{"ubicacion": "Madrid", "dias": 5}'
	
		Respuesta esperada (JSON):
	
		{
	  	"ubicacion": "Madrid",
	  	"pronostico": [
	   	 { "fecha": "2025-08-08", "temperatura": 25.5 },
	   	 { "fecha": "2025-08-09", "temperatura": 26.1 },
	   	 { "fecha": "2025-08-10", "temperatura": 25.9 },
	    	 { "fecha": "2025-08-11", "temperatura": 26.3 },
	         { "fecha": "2025-08-12", "temperatura": 26.8 }
	  							]
									}
	
		Endpoint 2: Asistente de Consultas
	
		Este endpoint te permite hacer preguntas sobre los datos históricos.
	
	   		Ruta: /ask
	
	    		Método: POST
	
	   		Payload: JSON con el campo question (string).
	
		Ejemplo de uso con curl:
	
		curl -X POST "http://<IP_PUBLICA_EC2>:8000/ask" \
		-H "Content-Type: application/json" \
		-d '{"question": "Cuál fue la temperatura media en Madrid en mayo de 2024?"}'
	
		Respuesta esperada (JSON):
	
		{
	 	 "respuesta": "La temperatura media en Madrid en mayo de 2024 fue de 21.5 grados Celsius.",
	  	"sql_generada": "SELECT AVG(tmed) FROM datos_clima WHERE nombre = 'Madrid' AND fecha BETWEEN '2024-05-01' AND '2024-05-31'"
		}


## 🤝 Contribuciones

Las contribuciones son bienvenidas. Si tienes ideas para mejorar, abre un *issue* o crea un *pull request*.

---

## 📝 Licencia

Este proyecto está bajo la licencia MIT.
