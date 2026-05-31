# Modelo de Clasificación de Textos - Fiscalía General de la República de Cuba

Clasificador automático de textos basado en Regresión Logística y TF-IDF para la identificación de contenidos de alto riesgo (amenazas, discurso de odio, hechos delictivos) en plataformas digitales.

## 🎓 Información Académica

Este proyecto es resultado de la tesis de diploma:

**"Modelo de clasificación de textos aplicado a la Fiscalía General de la República de Cuba"**

- **Autor**: Wilbert Cereijo Mestre
- **Tutores**: Ing. Pedro Alejandro Cabrera Enríquez, DrC. Yamil Sánchez Castellanos
- **Cotutor**: Ing. Lisandra Valdivia Torres
- **Institución**: Universidad de las Ciencias Informáticas (UCI)
- **Facultad**: Informática Organizacional
- **Fecha**: Mayo 2026

## 📌 Descripción

Este clasificador identifica y categoriza textos provenientes de plataformas digitales en **4 categorías**:

- **amenaza**: Textos que contienen amenazas o coerción directa
- **discurso_odio**: Contenido discriminatorio o de rechazo hacia grupos
- **hecho_delictivo**: Textos que describen o reportan actividades potencialmente delictivas
- **inofensivo**: Planteamientos ciudadanos generales, consultas, sugerencias

### 📊 Resultados Experimentales

El modelo alcanzó un **desempeño excelente** en validación:

| Métrica | Valor |
|---------|-------|
| **Exactitud (Accuracy)** | **95.78%** |
| **Precisión (promedio)** | **0.96** |
| **Exhaustividad (Recall promedio)** | **0.96** |
| **F1-Score (promedio)** | **0.96** |

**Por categoría:**

| Categoría | Precisión | Exhaustividad | F1-Score | Soporte |
|-----------|-----------|---------------|----------|---------|
| amenaza | 0.96 | 0.97 | 0.97 | 159 |
| discurso_odio | 0.95 | 0.96 | 0.96 | 255 |
| hecho_delictivo | 0.97 | 0.97 | 0.97 | 247 |
| inofensivo | 0.95 | 0.93 | 0.94 | 240 |

**Total**: 901 muestras de prueba

## 📂 Estructura del Proyecto

```
.
├── data/
│   └── datos_entrenamiento.csv      # Dataset de 4502 muestras etiquetadas
├── models/
│   └── modelo_clasificador.pkl      # Modelo entrenado (generado por train.py)
├── train.py                         # Script de entrenamiento
├── predict.py                       # Script de predicción interactiva
├── requirements.txt                 # Dependencias Python
└── README.md                        # Este archivo
```

## 🔧 Requisitos

- Python 3.13.3 o superior
- pip (gestor de paquetes de Python)

## 📦 Instalación

### Paso 1: Instalar dependencias

```bash
pip install -r requirements.txt
```

**Librerías requeridas:**
- pandas (>= 1.3.0) - Manipulación de datos
- scikit-learn (>= 1.0.0) - Aprendizaje automático y NLP
- joblib (>= 1.1.0) - Persistencia de modelos

## 📊 Dataset

### Ubicación
El archivo CSV debe estar en la carpeta `data/` con el nombre `datos_entrenamiento.csv`.

### Formato esperado

| Columna | Tipo | Descripción |
|---------|------|-------------|
| id | int | Identificador único del registro |
| texto | string | Contenido textual a clasificar |
| categoria | string | Etiqueta: amenaza, discurso_odio, hecho_delictivo, inofensivo |

### Ejemplo de datos

```csv
id,texto,categoria
1,te voy a hacer daño,amenaza
2,esa gente de tu raza no sirve para nada,discurso_odio
3,me hackearon la cuenta de correo,hecho_delictivo
4,necesito información sobre un trámite,inofensivo
```

### Estadísticas del Dataset Entrenamiento

- **Total de muestras**: 4502 (después de limpieza)
- **División**: 80% entrenamiento (3601), 20% prueba (901)
- **Distribución de clases**: Balanceada

## 🚀 Uso

### 1. Entrenar el modelo

```bash
python train.py
```

**Proceso automático:**
1. ✅ Busca el CSV en `data/`
2. ✅ Limpia datos (minúsculas, espacios, duplicados, nulos)
3. ✅ Aplica preprocesamiento de NLP
4. ✅ Vectoriza textos con TF-IDF (unigramas + bigramas)
5. ✅ Entrena modelo con Regresión Logística
6. ✅ Evalúa con métricas estándar
7. ✅ Guarda modelo en `models/modelo_clasificador.pkl`

**Salida esperada:**

```
============================================================
CLASIFICADOR DE TEXTO - ENTRENAMIENTO
============================================================
Buscando archivo CSV...
Archivo encontrado: ./data/datos_entrenamiento.csv

Columnas: ['id', 'texto', 'categoria']

=== LIMPIEZA DE DATOS ===
Filas iniciales: 4502
Filas después de eliminar nulos: 4502
Filas después de eliminar duplicados: 4502

=== DISTRIBUCIÓN DE CLASES ===
amenaza              792
discurso_odio      1276
hecho_delictivo    1177
inofensivo          1257

=== DIVISIÓN TRAIN/TEST ===
Conjunto de entrenamiento: 3601 muestras
Conjunto de prueba: 901 muestras

=== ENTRENAMIENTO DEL MODELO ===
Entrenando modelo con Regresión Logística...
Modelo entrenado exitosamente

=== EVALUACIÓN DEL MODELO ===
Accuracy: 0.9578

Reporte de Clasificación:
              precision    recall  f1-score   support
        amenaza       0.96      0.97      0.97       159
    discurso_odio      0.95      0.96      0.96       255
  hecho_delictivo      0.97      0.97      0.97       247
      inofensivo       0.95      0.93      0.94       240

      macro avg       0.96      0.96      0.96       901
   weighted avg       0.96      0.96      0.96       901

Modelo guardado en: ./models/modelo_clasificador.pkl
============================================================
```

### 2. Hacer predicciones

```bash
python predict.py
```

**Características:**
- ✅ Carga modelo entrenado automáticamente
- ✅ Interfaz interactiva para ingresar textos
- ✅ Muestra categoría predicha y confianza (%)
- ✅ Umbral de confianza del 50% para alertar casos ambiguos
- ✅ Escribe `salir` para terminar

**Ejemplo de uso:**

```
============================================================
CLASIFICADOR DE TEXTO - PREDICCIÓN
============================================================
Cargando modelo...
Modelo cargado correctamente

Categorías disponibles:
  • amenaza: Amenazas o coerción
  • discurso_odio: Contenido discriminatorio
  • hecho_delictivo: Actividades delictivas reportadas
  • inofensivo: Consultas y planteamientos generales

Escribe 'salir' para terminar

Ingresa un texto para clasificar: te voy a hacer daño
============================================================
RESULTADO DE LA PREDICCIÓN
============================================================
Texto: te voy a hacer daño
Categoría: amenaza
Confianza: 96.78%
============================================================

Ingresa un texto para clasificar: necesito información
============================================================
RESULTADO DE LA PREDICCIÓN
============================================================
Texto: necesito información
Categoría: inofensivo
Confianza: 89.23%
============================================================
```

## ⚙️ Configuración del Modelo

### Metodología: KDD (Knowledge Discovery in Databases)

El desarrollo siguió las etapas estándar de KDD:

1. **Selección de datos**: Dataset propio de 4502 textos
2. **Preprocesamiento**: Limpieza y normalización
3. **Transformación**: Vectorización TF-IDF
4. **Minería de datos**: Entrenamiento Regresión Logística
5. **Evaluación**: Validación con métricas estándar

### TfidfVectorizer

```python
TfidfVectorizer(
    ngram_range=(1, 2),      # Unigramas + bigramas
    min_df=2,                # Min 2 documentos
    max_df=0.95,             # Max 95% documentos
    max_features=5000        # 5000 características
)
```

### LogisticRegression

```python
LogisticRegression(
    class_weight='balanced', # Equilibra clases desbalanceadas
    max_iter=2000,           # Iteraciones máximas
    solver='lbfgs',          # Optimizador
    random_state=42          # Reproducibilidad
)
```

## 🧹 Preprocesamiento de Datos

El script aplica automáticamente:

1. ✅ **Eliminación de nulos**: Remueve registros incompletos
2. ✅ **Minúsculas**: Normaliza texto (`"PELIGRO"` → `"peligro"`)
3. ✅ **Limpieza de espacios**: Elimina espacios al inicio/final
4. ✅ **Eliminación de duplicados**: Mantiene primera ocurrencia
5. ✅ **Tokenización**: Divide en palabras/bigramas

## 📈 Tecnologías Utilizadas

- **Python 3.13.3**: Lenguaje de programación
- **pandas 1.3.0+**: Lectura y manipulación de datos CSV
- **scikit-learn 1.0.0+**: ML, NLP, métricas de evaluación
- **joblib 1.1.0+**: Persistencia y carga de modelos
- **Metodología KDD**: Estructura de desarrollo

## 🎯 Rendimiento Computacional

- **Entrenamiento**: < 10 segundos en máquina estándar
- **Predicción por texto**: < 0.1 segundos (instantáneo)
- **Memoria requerida**: ~100 MB

## ⚠️ Manejo de Errores

### "La carpeta 'data' no existe"
```bash
mkdir data
```

### "No se encontró ningún archivo CSV"
Verifica que el archivo esté en `data/datos_entrenamiento.csv`

### "El modelo no existe"
Ejecuta primero: `python train.py`

### "Errores de codificación UTF-8"
Asegúrate que el CSV esté en codificación UTF-8

## 📝 Notas Importantes

- ⚠️ El modelo se sobrescribe cada vez que ejecutas `train.py`
- ⚠️ Las predicciones son determinísticas (mismo resultado para mismo input)
- ⚠️ El umbral de confianza del 50% ayuda a detectar casos ambiguos
- ✅ El modelo maneja bien clases desbalanceadas con `class_weight='balanced'`

## 🔍 Matriz de Confusión

Del conjunto de prueba (901 muestras):

```
Real / Predicho | amenaza | discurso | delictivo | inofensivo
─────────────────────────────────────────────────────────────
amenaza         |   155   |    1     |     0     |     3
discurso_odio   |    1    |   246    |     3     |     5
hecho_delictivo |    1    |    3     |    240    |     3
inofensivo      |    4    |   10     |     4     |    222
```

**Observaciones:**
- Principal confusión: "inofensivo" ↔ "discurso_odio" (10 casos)
- Bajo número de falsos positivos/negativos
- Desempeño excelente en "amenaza" y "hecho_delictivo"

## 🤔 Preguntas Frecuentes

**P: ¿Puedo usar otros formatos de datos?**
R: No, solo CSV con estructura específica (id, texto, categoria).

**P: ¿Cómo mejoro la precisión?**
R: Aumenta datos de entrenamiento o ajusta hiperparámetros en `train.py`.

**P: ¿Puedo integrar esto en otra aplicación?**
R: Sí, el modelo se puede cargar e invocar desde cualquier código Python.

**P: ¿Es reproducible?**
R: Sí, usa `random_state=42` para resultados consistentes.

## 📚 Referencias

Cereijo Mestre, W. (2026). *Modelo de clasificación de textos aplicado a la Fiscalía General de la República de Cuba*. Tesis de Diploma, Universidad de las Ciencias Informáticas, La Habana, Cuba.

## 👨‍💻 Autor Original

**Wilbert Cereijo Mestre** - Desarrollo de tesis y modelo
- Tutores: Ing. Pedro Alejandro Cabrera Enríquez, DrC. Yamil Sánchez Castellanos
- Cotutor: Ing. Lisandra Valdivia Torres

## 📄 Licencia

Libre para uso educativo y en el contexto de la Fiscalía General de la República de Cuba.

---

**Versión del modelo**: 1.0 (Tesis 2026)  
**Última actualización**: Mayo 2026
