"""
Script de predicción usando el modelo entrenado.
"""

import os
import joblib
import sys


def cargar_modelo(ruta_modelo):
    """
    Carga el modelo guardado desde la ruta especificada.
    Retorna el modelo si existe, sino muestra error.
    """
    if not os.path.exists(ruta_modelo):
        print(f"Error: El modelo no existe en {ruta_modelo}")
        print("Debes entrenar el modelo primero ejecutando: python train.py")
        return None
    
    try:
        modelo = joblib.load(ruta_modelo)
        return modelo
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        return None


def obtener_probabilidad_maxima(probabilidades, clases):
    """
    Obtiene la probabilidad máxima de la predicción.
    """
    indice_max = probabilidades.argmax()
    confianza = probabilidades[indice_max]
    return confianza


def predecir_texto(modelo, texto):
    """
    Realiza una predicción sobre el texto ingresado.
    Retorna la categoría y la confianza.
    Si la confianza es menor al 60%, clasifica como 'otros'.
    """
    # Validar entrada vacía
    if not texto or texto.strip() == '':
        print("Error: Por favor ingresa un texto válido")
        return None, None
    
    # Limpiar el texto (minúsculas y espacios)
    texto_limpio = texto.strip().lower()
    
    # Hacer predicción
    try:
        categoria = modelo.predict([texto_limpio])[0]
        probabilidades = modelo.predict_proba([texto_limpio])[0]
        confianza = obtener_probabilidad_maxima(probabilidades, modelo.classes_)
        
        # Aplicar umbral de confianza: si es menor al 50%, clasificar como 'otros'
        if confianza < 0.50:
            categoria = 'otros'
        
        return categoria, confianza
    except Exception as e:
        print(f"Error durante la predicción: {e}")
        return None, None


def mostrar_resultado(texto, categoria, confianza):
    """
    Muestra los resultados de la predicción de forma clara.
    """
    print("\n" + "="*60)
    print("RESULTADO DE LA PREDICCIÓN")
    print("="*60)
    print(f"Texto ingresado: {texto}")
    print(f"Categoría predicha: {categoria}")
    print(f"Confianza: {confianza*100:.2f}%")
    print("="*60 + "\n")


def main():
    """
    Función principal del script de predicción.
    """
    # Ruta del modelo
    ruta_modelo = os.path.join(
        os.path.dirname(__file__),
        'models',
        'modelo_clasificador.pkl'
    )
    
    # Cargar modelo
    print("="*60)
    print("CLASIFICADOR DE TEXTO - PREDICCIÓN")
    print("="*60)
    print("Cargando modelo...")
    
    modelo = cargar_modelo(ruta_modelo)
    if modelo is None:
        sys.exit(1)
    
    print("Modelo cargado correctamente")
    print("\nCategorías disponibles:")
    print("  - amenaza")
    print("  - discurso_odio")
    print("  - hecho_delictivo")
    print("  - otros")
    print("\nEscribe 'salir' para terminar\n")
    
    # Loop de predicciones
    while True:
        try:
            # Obtener entrada del usuario
            texto = input("Ingresa un texto para clasificar: ").strip()
            
            # Verificar comando de salida
            if texto.lower() == 'salir':
                print("\n¡Hasta luego!")
                break
            
            # Realizar predicción
            categoria, confianza = predecir_texto(modelo, texto)
            
            if categoria is not None and confianza is not None:
                mostrar_resultado(texto, categoria, confianza)
            
        except KeyboardInterrupt:
            print("\n\n¡Programa interrumpido!")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
