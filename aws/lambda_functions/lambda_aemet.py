import json
import boto3
import requests
import time
import os
from datetime import datetime, timedelta

def lambda_handler(event, context):
    """
    Lambda handler para extracción diaria de datos AEMET
    """
    
    # Variables de entorno
    AEMET_API_KEY = os.environ.get('AEMET_API_KEY')
    AEMET_BASE_URL = "https://opendata.aemet.es/opendata/api"
    S3_BUCKET_NAME = "hab-bucket-prueba"
    S3_REGION = "eu-north-1"
    S3_OUTPUT_DIRECTORY = "data/raw/aemet-diaria-2025"
    
    print(f"Lambda iniciada: {datetime.now()}")
    
    if not AEMET_API_KEY:
        error_msg = "AEMET_API_KEY no configurada en variables de entorno"
        print(error_msg)
        return {
            'statusCode': 400,
            'body': json.dumps({'error': error_msg})
        }
    
    # Obtener fecha de ayer para extracción diaria
    fecha_ayer = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    print(f"Procesando datos para: {fecha_ayer}")
    
    try:
        # Ejecutar extracción
        datos = extractorAEMET_Diario_Lambda(
            fecha_ayer, 
            fecha_ayer, 
            AEMET_API_KEY, 
            AEMET_BASE_URL,
            S3_BUCKET_NAME, 
            S3_OUTPUT_DIRECTORY, 
            S3_REGION
        )
        
        resultado = {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Extracción completada exitosamente',
                'fecha_procesada': fecha_ayer,
                'registros_obtenidos': len(datos),
                'bucket_s3': S3_BUCKET_NAME,
                'timestamp': datetime.now().isoformat()
            })
        }
        
        print(f"Lambda completada: {len(datos)} registros procesados")
        return resultado
        
    except Exception as e:
        error_msg = f"Error en Lambda: {str(e)}"
        print(error_msg)
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': error_msg,
                'fecha_procesada': fecha_ayer,
                'timestamp': datetime.now().isoformat()
            })
        }

def upload_to_s3_direct(data, s3_key, bucket_name, region):
    """
    Subir datos JSON directamente a S3
    """
    try:
        s3 = boto3.client("s3", region_name=region)
        
        # Convertir datos a JSON
        json_content = json.dumps(data, indent=4, ensure_ascii=False)
        
        # Subir directamente a S3
        s3.put_object(
            Bucket=bucket_name,
            Key=s3_key,
            Body=json_content.encode('utf-8'),
            ContentType='application/json'
        )
        
        print(f"Archivo subido a S3: s3://{bucket_name}/{s3_key}")
        return True
        
    except Exception as e:
        print(f"Error subiendo a S3: {e}")
        return False

def extractorAEMET_Diario_Lambda(fecha_inicio, fecha_fin, api_key, base_url, bucket, s3_dir, region):
    """
    Extrae datos climatológicos diarios de AEMET y los guarda en S3
    """
    
    fecha_ini_api_format = f"{fecha_inicio}T00:00:00UTC"
    fecha_fin_api_format = f"{fecha_fin}T23:59:59UTC"
    endpoint_url = (
        f"{base_url}/valores/climatologicos/diarios/datos/"
        f"fechaini/{fecha_ini_api_format}/fechafin/{fecha_fin_api_format}/todasestaciones"
    )
    
    print(f"Solicitando datos AEMET: {fecha_inicio} a {fecha_fin}")
    
    try:
        # Solicitud inicial a AEMET
        response_inicial = requests.get(
            endpoint_url,
            params={'api_key': api_key},
            headers={'Cache-Control': 'no-cache'},
            timeout=60
        )
        response_inicial.raise_for_status()
        data_inicial = response_inicial.json()
        
        if response_inicial.status_code != 200 or 'datos' not in data_inicial:
            print(f"Error en solicitud inicial: {response_inicial.status_code}")
            return []
            
        datos_url = data_inicial.get('datos')
        if not datos_url:
            print("No se encontró URL de datos en respuesta AEMET")
            return []
            
        # Esperar 1 segundo (requerimiento AEMET)
        time.sleep(1.0)
        
        # Obtener datos reales
        response_datos = requests.get(datos_url, timeout=60)
        response_datos.raise_for_status()
        batch_data = response_datos.json()
        
        if isinstance(batch_data, list) and batch_data:
            print(f"Datos obtenidos de AEMET: {len(batch_data)} registros")
            
            # Limitar a 10,000 registros por seguridad
            batch_data = batch_data[:10000]
            
            # Subir a S3
            output_filename = f"{fecha_inicio}_{fecha_fin}.json"
            s3_key = f"{s3_dir}/{output_filename}"
            
            success = upload_to_s3_direct(batch_data, s3_key, bucket, region)
            
            if success:
                print(f"Datos guardados en S3: {len(batch_data)} registros")
            else:
                print("Error guardando en S3")
                
            return batch_data
        else:
            print("Respuesta de AEMET vacía o inválida")
            return []
            
    except requests.exceptions.RequestException as e:
        print(f"Error de conexión con AEMET: {e}")
        return []
    except ValueError as e:
        print(f"Error parseando JSON de AEMET: {e}")
        return []
    except Exception as e:
        print(f"Error inesperado: {e}")
        return []