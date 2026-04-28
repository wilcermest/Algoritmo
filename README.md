# Clasificador de Texto con scikit-learn

Proyecto simple y funcional para clasificación de textos en 4 categorías usando técnicas clásicas de NLP.

## Descripción

Este proyecto implementa un clasificador de texto capaz de detectar y categorizar textos en una de estas 4 categorías:

- **amenaza**: textos que contienen amenazas o coerción
- **discurso_odio**: textos con contenido de odio o discriminación
- **hecho_delictivo**: textos que describen actividades delictivas
- **otros**: quejas, comentarios generales u otros contenidos

## Estructura del Proyecto

```
.
├── data/
│   └── datos_entrenamiento.csv    # Dataset de entrenamiento
├── models/
│   └── modelo_clasificador.pkl    # Modelo entrenado (generado después de train.py)
├── train.py                        # Script de entrenamiento
├── predict.py                      # Script de predicción
├── requirements.txt                # Dependencias de Python
└── README.md                       # Este archivo
```

## Requisitos

- Python 3.7 o superior
- pip (gestor de paquetes de Python)

## Instalación

### Paso 1: Instalar dependencias

```bash
pip install -r requirements.txt
```

Las librerías requeridas son:
- pandas (>= 1.3.0)
- scikit-learn (>= 1.0.0)
- joblib (>= 1.1.0)

## Dataset

### Ubicación
El archivo CSV debe estar en la carpeta `data/` en el mismo directorio que los scripts.

### Formato esperado

El archivo CSV debe tener exactamente estas columnas:

| Columna | Tipo | Descripción |
|---------|------|-------------|
| id | entero | Identificador único del registro |
| texto | string | Texto libre a clasificar |
| categoria | string | Etiqueta de la categoría (amenaza, discurso_odio, hecho_delictivo, otros) |

### Ejemplo de datos

```
id,texto,categoria
1,No vuelvas a hacer eso o te arrepentirás,amenaza
2,Esos gusanos no merecen vivir,discurso_odio
3,Vendimos 50 kilogramos de cocaína el viernes,hecho_delictivo
4,Me gustaría que me devuelvan el dinero de mi compra,otros
```

## Uso

### 1. Entrenar el modelo

Para entrenar el modelo, ejecuta:

```bash
python train.py
```

**Proceso:**
1. Busca automáticamente el CSV en la carpeta `data/`
2. Limpia los datos (minúsculas, espacios, duplicados, nulos)
3. Divide datos en 80% entrenamiento y 20% prueba
4. Entrena un modelo usando:
   - Vectorización TF-IDF (n-gramas 1-2)
   - Regresión Logística con pesos balanceados
5. Muestra métricas de evaluación (accuracy, precision, recall, F1)
6. Guarda el modelo en `models/modelo_clasificador.pkl`

**Salida esperada EJEMPLO:**
```
============================================================
CLASIFICADOR DE TEXTO - ENTRENAMIENTO
============================================================
Buscando archivo CSV...
Archivo encontrado: ./data/datos_entrenamiento.csv


Columnas: ['id', 'texto', 'categoria']

=== LIMPIEZA DE DATOS ===
Filas iniciales: 400
Filas después de eliminar nulos: 400
Filas después de eliminar duplicados: 398

=== DISTRIBUCIÓN DE CLASES ===
categoria
otros               105
amenaza              98
discurso_odio        95
hecho_delictivo      100

=== DIVISIÓN TRAIN/TEST ===
Conjunto de entrenamiento: 318 muestras
Conjunto de prueba: 80 muestras

=== ENTRENAMIENTO DEL MODELO ===
Entrenando modelo...
Modelo entrenado exitosamente

=== EVALUACIÓN DEL MODELO ===
Accuracy: 0.8750

Reporte de Clasificación:
              precision    recall  f1-score   support
...

Modelo guardado en: ./models/modelo_clasificador.pkl

============================================================
ENTRENAMIENTO COMPLETADO EXITOSAMENTE
============================================================
```

### 2. Hacer predicciones

Para usar el modelo entrenado y hacer predicciones, ejecuta:

```bash
python predict.py
```

**Proceso:**
1. Carga el modelo entrenado desde `models/modelo_clasificador.pkl`
2. Permite ingresar textos interactivamente
3. Predice la categoría y muestra la confianza en porcentaje
4. Repite hasta que escribas `salir`

**Ejemplo de uso:**

```
============================================================
CLASIFICADOR DE TEXTO - PREDICCIÓN
============================================================
Cargando modelo...
Modelo cargado correctamente

Categorías disponibles:
  - amenaza
  - discurso_odio
  - hecho_delictivo
  - otros

Escribe 'salir' para terminar

Ingresa un texto para clasificar: no hagas eso o te mataremos

============================================================
RESULTADO DE LA PREDICCIÓN
============================================================
Texto ingresado: no hagas eso o te mataremos
Categoría predicha: amenaza
Confianza: 92.45%
============================================================

Ingresa un texto para clasificar: quiero reclamar sobre mi pedido

============================================================
RESULTADO DE LA PREDICCIÓN
============================================================
Texto ingresado: quiero reclamar sobre mi pedido
Categoría predicha: otros
Confianza: 78.32%
============================================================

Ingresa un texto para clasificar: salir

¡Hasta luego!
```

## Configuración del Modelo

### TfidfVectorizer
- **ngram_range=(1,2)**: Considera palabras individuales y pares de palabras
- **min_df=2**: Ignora términos que aparecen en menos de 2 documentos
- **max_df=0.95**: Ignora términos que aparecen en más del 95% de los documentos
- **max_features=5000**: Limita a 5000 características

### LogisticRegression
- **class_weight='balanced'**: Pondera automáticamente según la frecuencia de clases
- **max_iter=2000**: Número máximo de iteraciones de optimización
- **solver='lbfgs'**: Algoritmo de optimización (recomendado para datasets pequeños)

## Limpieza de Datos

El script de entrenamiento aplicará automáticamente:

1. **Eliminar valores nulos**: Remueve filas con texto o categoría vacíos
2. **Convertir a minúsculas**: Normaliza el texto
3. **Eliminar espacios extra**: Limpia espacios al inicio y final
4. **Eliminar duplicados**: Mantiene solo la primera ocurrencia de textos duplicados

## Manejo de Errores

### Error: "La carpeta 'data' no existe"
Verifica que exista la carpeta `data/` en el mismo directorio que los scripts.

### Error: "No se encontró ningún archivo CSV"
Coloca el archivo CSV en la carpeta `data/`.

### Error: "El modelo no existe"
Asegúrate de haber ejecutado `python train.py` primero para generar el modelo.

## Tecnologías Utilizadas

- **pandas**: Lectura y manipulación de datos
- **scikit-learn**: Pipeline, TfidfVectorizer, LogisticRegression, métricas
- **joblib**: Serialización y carga del modelo

## Notas Importantes

- El modelo se guarda como archivo pickle (`.pkl`). Mantén este archivo para hacer predicciones futuras.
- La primera ejecución de `train.py` puede tomar algunos segundos.
- Las predicciones son determinísticas (mismo texto siempre da mismo resultado).
- El modelo está optimizado para datasets pequeños a medianos (~400 muestras).

## Solución de Problemas

**P: ¿Puedo usar diferentes formatos de datos?**
R: No, el proyecto está diseñado específicamente para CSV con las columnas especificadas.

**P: ¿Cómo puedo mejorar la precisión del modelo?**
R: Necesitarías más datos de entrenamiento o ajustar los hiperparámetros en `train.py`.

**P: ¿Puedo entrenar el modelo nuevamente sin perder el anterior?**
R: Sí, pero el nuevo modelo sobrescribirá el anterior. Guarda una copia si es importante.

## Autor

Proyecto de clasificación de texto con scikit-learn - 2026

## Licencia

Libre para uso educativo y comercial.
