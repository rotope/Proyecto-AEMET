import pandas as pd
import unicodedata
import numpy as np




def clean_and_interpolate_dataframe(dataframe_name, df_data):
    """
    Elimina columnas específicas, luego limpia las columnas numéricas de tipo 'object',
    convierte a numérico, e interpola los valores nulos en todas las columnas numéricas.

    Args:
        dataframe_name (str): El nombre de la variable del DataFrame (para impresión).
        df_data (pd.DataFrame): El DataFrame a procesar.

    Returns:
        pd.DataFrame: El DataFrame procesado.
    """
    print(f"\n--- Procesando DataFrame: {dataframe_name} ---")

    # --- PASO 1: Eliminar columnas específicas ---
    columns_to_drop = [
        'presMax', 'horaPresMax', 'presMin', 'horaPresMin',
        'sol', 'dir', 'velmedia', 'racha', 'horaracha'
    ]
    # Usar errors='ignore' para evitar que el código falle si alguna columna no existe
    # Capturamos las columnas que realmente existían y se eliminaron para el mensaje
    dropped_cols_actual = [col for col in columns_to_drop if col in df_data.columns]
    df_data.drop(columns=columns_to_drop, inplace=True, errors='ignore')

    if dropped_cols_actual:
        print(f"Columnas eliminadas en {dataframe_name}: {', '.join(dropped_cols_actual)}")
    else:
        print(f"No se encontraron columnas a eliminar de la lista en {dataframe_name}.")


    # --- PASO 2: Procesar la columna 'fecha' ---
    if 'fecha' in df_data.columns and df_data['fecha'].dtype != 'datetime64[ns]':
        df_data['fecha'] = pd.to_datetime(df_data['fecha'], errors='coerce')
        print(f"Columna 'fecha' convertida a datetime en {dataframe_name}.")
    elif 'fecha' not in df_data.columns:
        print(f"Advertencia: La columna 'fecha' no se encontró en {dataframe_name}.")


    # --- PASO 3: Limpiar y convertir columnas 'object' a numéricas ---
    # Columnas que sabemos que son numéricas pero están como 'object'
    numeric_cols_to_clean_and_convert = ['tmed', 'prec', 'tmin', 'tmax']

    for col in numeric_cols_to_clean_and_convert:
        if col in df_data.columns:
            df_data[col] = df_data[col].astype(str).str.strip()
            df_data[col] = df_data[col].str.replace(',', '.', regex=False)
            df_data[col] = df_data[col].replace(['null', 'None', '-'], np.nan)
            df_data[col] = pd.to_numeric(df_data[col], errors='coerce')
            # Pequeña verificación para ver si todo se volvió NaN
            if df_data[col].isnull().all() and df_data[col].shape[0] > 0:
                print(f"Advertencia: La columna '{col}' en {dataframe_name} se convirtió completamente a NaN después de la limpieza. Verifique los datos originales.")
        else:
            print(f"Advertencia: La columna '{col}' no se encontró en {dataframe_name}. Saltando limpieza para esta columna.")


    # --- PASO 4: Aplicar interpolación a todas las columnas numéricas ---
    numeric_columns = df_data.select_dtypes(include=np.number).columns

    if not numeric_columns.empty:
        df_data[numeric_columns] = df_data[numeric_columns].interpolate()
        print(f"Interpolación aplicada a columnas numéricas en {dataframe_name}.")
    else:
        print(f"No se encontraron columnas numéricas para interpolar en {dataframe_name}.")


    # (Opcional) Puedes añadir aquí el drop de las columnas de hora remanentes si lo deseas automatizar:
    # columns_to_drop_hora = ['horatmin', 'horatmax', 'horaHrMax', 'horaHrMin']
    # df_data.drop(columns=columns_to_drop_hora, inplace=True, errors='ignore')
    # if columns_to_drop_hora:
    #     print(f"Columnas de hora eliminadas en {dataframe_name}: {', '.join([col for col in columns_to_drop_hora if col not in df_data.columns])}")


    #normalizar texto eliminando tildes y ñ
   

    def normalizar_texto(df_data):
        if pd.isnull(df_data):
            return df_data
        # Elimina tildes y cambia ñ → n
        df_data = unicodedata.normalize('NFKD', df_data).encode('ASCII', 'ignore').decode('utf-8')
        # Elimina caracteres no deseados si es necesario (opcional)
        return df_data

    # Aplicar a columnas 'nombre' y 'provincia'
    df_data['nombre'] = df_data['nombre'].apply(normalizar_texto)
    df_data['provincia'] = df_data['provincia'].apply(normalizar_texto)


    # --- PASO FINAL: Verificar el resultado ---
    print(f"--- Info de {dataframe_name} después de todo el procesamiento ---")
    df_data.info()
    print(f"\n--- Primeras filas de {dataframe_name} ---")
    print(df_data.head())

    return df_data