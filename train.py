"""
Script de entrenamiento del clasificador de texto.
Entrena un modelo de clasificación usando TfidfVectorizer y LogisticRegression.
"""

import os
import glob
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib


def buscar_csv_en_data():
    
    ruta_data = os.path.join(os.path.dirname(__file__), 'data')
    
    if not os.path.exists(ruta_data):
        raise FileNotFoundError(f"La carpeta 'data' no existe en: {ruta_data}")
    
    archivos_csv = glob.glob(os.path.join(ruta_data, '*.csv'))
    
    if len(archivos_csv) == 0:
        raise FileNotFoundError(f"No se encontró ningún archivo CSV en la carpeta 'data': {ruta_data}")
    
    if len(archivos_csv) > 1:
        print(f"Advertencia: Se encontraron múltiples CSV. Usando: {archivos_csv[0]}")
    
    return archivos_csv[0]


def limpiar_datos(df):
    """
    Aplica limpieza básica al dataset:
    - Elimina filas con valores nulos
    - Convierte texto a minúsculas
    - Elimina espacios extra
    - Elimina duplicados
    """
    print("\n=== LIMPIEZA DE DATOS ===")
    print(f"Filas iniciales: {len(df)}")
    
    # Eliminar valores nulos
    df = df.dropna(subset=['texto', 'categoria'])
    print(f"Filas después de eliminar nulos: {len(df)}")
    
    # Convertir texto a minúsculas
    df['texto'] = df['texto'].str.lower()
    
    # Eliminar espacios extra
    df['texto'] = df['texto'].str.strip()
    
    # Eliminar duplicados
    df = df.drop_duplicates(subset=['texto'], keep='first')
    print(f"Filas después de eliminar duplicados: {len(df)}")
    
    return df


def cargar_y_preparar_datos():
    """
    Carga el CSV, limpia los datos y retorna X (textos) y y (etiquetas).
    """
    print("Buscando archivo CSV...")
    ruta_csv = buscar_csv_en_data()
    print(f"Archivo encontrado: {ruta_csv}")
    
    # Cargar datos
    df = pd.read_csv(ruta_csv)
    print(f"\nDataset cargado con {len(df)} filas")
    print(f"Columnas: {list(df.columns)}")
    
    # Limpiar datos
    df = limpiar_datos(df)
    
    # Verificar balanceo de clases
    print("\n=== DISTRIBUCIÓN DE CLASES ===")
    print(df['categoria'].value_counts())
    
    # Separar características y etiquetas
    X = df['texto']
    y = df['categoria']
    
    return X, y


def entrenar_modelo(X_train, X_test, y_train, y_test):
    """
    Entrena el modelo usando Pipeline con TfidfVectorizer y LogisticRegression.
    Retorna el modelo entrenado y sus métricas.
    """
    print("\n=== ENTRENAMIENTO DEL MODELO ===")
    
    # Crear pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            lowercase=True,
            max_features=5000
        )),
        ('clasificador', LogisticRegression(
            class_weight='balanced',
            max_iter=2000,
            random_state=42,
            solver='lbfgs'
        ))
    ])
    
    # Entrenar modelo
    print("Entrenando modelo...")
    pipeline.fit(X_train, y_train)
    print("Modelo entrenado exitosamente")
    
    # Evaluar modelo
    print("\n=== EVALUACIÓN DEL MODELO ===")
    
    # Predicciones
    y_pred = pipeline.predict(X_test)
    
    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")
    
    # Classification Report
    print("\nReporte de Clasificación:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    # Matriz de Confusión
    print("Matriz de Confusión:")
    print(confusion_matrix(y_test, y_pred))
    
    return pipeline


def guardar_modelo(modelo, ruta_modelo):
    """
    Guarda el modelo entrenado usando joblib.
    """
    os.makedirs(os.path.dirname(ruta_modelo), exist_ok=True)
    joblib.dump(modelo, ruta_modelo)
    print(f"\nModelo guardado en: {ruta_modelo}")


def main():
    
    try:
        print("="*60)
        print("CLASIFICADOR DE TEXTO - ENTRENAMIENTO")
        print("="*60)
        
       
        X, y = cargar_y_preparar_datos()
        
       
        print("\n=== DIVISIÓN TRAIN/TEST ===")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )
        print(f"Conjunto de entrenamiento: {len(X_train)} muestras")
        print(f"Conjunto de prueba: {len(X_test)} muestras")
        
        # Entrenar modelo
        modelo = entrenar_modelo(X_train, X_test, y_train, y_test)
        
        # Guardar modelo
        ruta_modelo = os.path.join(
            os.path.dirname(__file__),
            'models',
            'modelo_clasificador.pkl'
        )
        guardar_modelo(modelo, ruta_modelo)
        
        print("\n" + "="*60)
        print("ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
        print("="*60)
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Por favor, verifica que el archivo CSV esté en la carpeta 'data/'")
    except Exception as e:
        print(f"\nError inesperado: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
