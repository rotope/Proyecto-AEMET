import json
import boto3
import psycopg2
import os

s3 = boto3.client('s3')

def lambda_handler(event, context):
    # Obtener información del archivo cargado
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = event['Records'][0]['s3']['object']['key']

    try:
        # Descargar y leer el archivo desde S3
        response = s3.get_object(Bucket=bucket, Key=key)
        data = json.loads(response['Body'].read().decode('utf-8'))

        # Conectar a la base de datos PostgreSQL
        conn = psycopg2.connect(
            host=os.environ['PG_HOST'],
            database=os.environ['PG_DB'],
            user=os.environ['PG_USER'],
            password=os.environ['PG_PASSWORD'],
            port=os.environ.get('PG_PORT', 5432)
        )
        cursor = conn.cursor()

        # Insertar datos
        for item in data:
            cursor.execute("""
                INSERT INTO productos (id, nombre, precio, fecha)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    nombre = EXCLUDED.nombre,
                    precio = EXCLUDED.precio,
                    fecha = EXCLUDED.fecha;
            """, (item['id'], item['nombre'], item['precio'], item['fecha']))

        conn.commit()
        cursor.close()
        conn.close()

        return {
            'statusCode': 200,
            'body': f'{len(data)} registros insertados correctamente.'
        }

    except Exception as e:
        print(f"Error: {str(e)}")
        return {
            'statusCode': 500,
            'body': f'Error procesando archivo {key} en {bucket}: {str(e)}'
        }
