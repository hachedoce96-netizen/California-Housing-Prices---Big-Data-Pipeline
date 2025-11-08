🏠 Precios de la vivienda en California: Análisis de macrodatos
Pitón
PySpark
Google Colab
Licencia

Un pipeline completo de Big Data para predicción de precios de viviendas en California utilizando PySpark, optimizaciones distribuidas y machine learning escalable.

📊 Demostración Rápida en Google Colab
Abrir en Colab

```bash

Ejecución inmediata en Colab
!pip install pyspark findspark
!git clone https://github.com/tu-usuario/california-housing-bigdata.git
%cd california-housing-bigdata

from src.main_pipeline import HousingPricePipeline
pipeline = HousingPricePipeline()
results = pipeline.run_complete_pipeline()

🎯 Descripción del Proyecto
Este proyecto implementa un pipeline de Big Data completo para predecir precios de viviendas en California utilizando el conjunto de datos público de Kaggle. El sistema demuestra mejores prácticas en procesamiento distribuido, optimización de rendimiento y aprendizaje automático escalable con PySpark.

🚀 Características Principales
📥 Ingesta Multi-fuente: Datos desde Kaggle API, URLs públicas y datos de ejemplo

🧹 Procesamiento Robusto: Limpieza automática, validación de calidad y transformaciones

⚡ Optimizaciones Avanzadas: Caching estratégico, particionamiento, configuración Spark optimizada

🤖 ML Distribuido: Random Forest con PySpark ML e ingeniería de funciones

💾 Almacenamiento Eficiente: Parquet comprimido con particionamiento inteligente

📊 Visualización Integral: Análisis exploratorio automático y métricas de rendimiento

🏗️ Arquitectura del Oleoducto
Diagrama de Flujo

https://imgur.com/a/BP5wmym

Componentes principales
DataIngestion: Descarga y carga de conjuntos de datos desde Múltiples fuentes con resiliencia

Procesador de datos: Limpieza, validación y transformaciones con manejo de valores nulos

FeatureIngeniería: Creación de características derivadas y codificación categórica

OptimizationManager: Técnicas de optimización distribuida (caching, particionamiento)

ModelTrainer: Entrenamiento de modelos de ML con PySpark ML

DataStorage: Almacenamiento eficiente en múltiples formatos con compresión

📁 Estructura del Repositorio
california-housing-bigdata/
│
├── 📄 README.md # Este archivo
├── 📋 requisitos.txt # Dependencias del proyecto
├── ⚙️ config/
│ └── pipeline config.yaml # Configuraciones del pipeline
│
├── 🐍 src/ # Código fuente principal
│ ├── init .py
│ ├── data ingestion.py # Módulo de ingesta de datos
│ ├── data_processing.py # Procesamiento y limpieza
│ ├── feature_engineering.py # Ingeniería de características
│ ├── data_storage.py # Almacenamiento optimizado
│ ├── optimización.py # Técnicas de optimización
│ ├── model_training.py # Entrenamiento de modelos
│ └── main_pipeline.py # Pipeline principal unificado
│
├── 📓 notebooks/ # Jupyter notebooks
│ ├── 01_data_exploration.ipynb # Análisis exploratorio de datos
│ ├── 02_feature_analysis.ipynb # Análisis de características
│ ├── 03_model_evaluación.ipynb # Evaluación de modelos
│ └── 04_pipeline_demo.ipynb # Demostración completa en Colab
│
├── 💾 data/ # Datasets
│ ├── raw/ # Datos crudos
│ ├── procesados/ # Datos procesados
​​│ └── models/ # Modelos entrenados
│
├── ✅ tests/ # Tests automatizados
│ ├── __init .py
│ ├── test_data_processing.py # Pruebas de procesamiento
│ ├── test_feature_engineering.py # Pruebas de ingeniería
│ └── test_optimization.py # Pruebas de optimización
│
├── 📊 docs/ #Documentación
│ ├── Architecture_diagrams/ # Diagramas de arquitectura
│ ├──technical_report.pdf # Informe técnico completo
│ └── api_documentation.md # Documentación de API
│
└── 🔧 scripts/ # Scripts de utilidad
├── setup_environment.sh # Configuración de entorno
├── run_pipeline.py # Ejecución del pipeline
└── benchmark_performance.py # Benchmark de rendimiento

🚀 Instalación rápida
Opción 1: Google Colab (Recomendado)
Instalación en Google Colab - Ejecutar en una celda
!pip install pyspark==3.4.0 findspark pandas matplotlib seaborn requests
!git clone https://github.com/tu-usuario/california-housing-bigdata.git
%cd california-housing-bigdata

Importante y ejecutar
import findspark
findspark.init()

from src.main_pipeline import HousingPricePipeline
pipeline = HousingPricePipeline()
results = pipeline.run_complete_pipeline()

Opción 2: Entorno Local
1. Clonar repositorio
git clone https://github.com/tu-usuario/california-housing-bigdata.git
cd california-housing-bigdata

2. Crear un entorno virtual (opcional pero recomendado)
python -m venv housing_env
source housing_env/bin/activate # Linux/Mac

housing_env\Scripts\activate # Windows
3. Instalar dependencias
pip install -r requirements.txt

4. Verificar instalación
python -c “from src.main_pipeline import HousingPricePipeline; print('✅ Instalación exitosa')”

Prerrequisitos
Python 3.8+
Java 8/11 (requerido para PySpark)
4 GB+ de RAM recomendados para un procesamiento eficiente
Google Colab o entorno local con las dependencias instaladas
💻 Uso del Pipeline
Ejecución Completa Automática
from src.main_pipeline import HousingPricePipeline

Pipeline completo automático
pipeline = HousingPricePipeline()
datos_finales, modelo_entrenado, métricas_de_rendimiento = pipeline.run_complete_pipeline()

Resultados generados automáticamente
print(f”✅ Pipeline completado: {final_data.count()} registros procesados”)
print(f”📊 Métricas: { Performance_metrics}”)

Ejecución por Módulos Individuales
Ingesta específica
from src.data_ingestion import DataIngestion
ingestion = DataIngestion()
raw_data = ingestion.download_kaggle_dataset()

Procesamiento personalizado
from src.data_processing import DataProcessor
processor = DataProcessor(spark)
cleaned_data = processor.clean_data(raw_data)

Ingeniería de características
datos_destacados = procesador.ingeniería_de_características(datos_limpios)

Entrenamiento de modelo
de src.model_training importar ModelTrainer
entrenador = modelo ModelTrainer (spark)
, predicciones = entrenador.train_model (featured_data)

Scripts de Línea de Comandos
Ejecutar pipeline completo
scripts de Python/run_pipeline.py

Evaluación comparativa individual del rendimiento
scripts de Python/benchmark_performance.py

Ejecutar pruebas unitarias
python -m pytest tests/ -v

Ejecutar con configuración personalizada
scripts de Python/run_pipeline.py —config config/custom_config.yaml

⚡ Optimizaciones implementadas
Técnicas de Performance
Técnica Mejora Impacto
Caching Estratégico 35% Tiempo de procesamiento
Particionamiento Inteligente 40% Consultas filtradas
Configuración Spark Adaptativa 25% Uso de memoria y CPU
Compresión Snappy 60% Almacenamiento en disco
Predicate Pushdown 30% Operaciones de filtrado

Configuración optimizada de Spark
Configuraciones críticas aplicadas
spark.conf.set("spark.sql.adaptive.enabled", "true")
spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
spark.conf.set("spark.sql.adaptive.skew.enabled", "true")
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "10485760")
spark.conf.set("spark.sql.shuffle.partitions", "100")

📊 Resultados y Métricas
Rendimiento del modelo
Métrica Valor Mejora vs Baseline
R² Puntuación 0,81 +15%
RMSE $48.250 -22%
MAE $35.120 -18%
Tiempo Entrenamiento 28,7s -36,5%

Características más importantes
ingreso_mediano (28,5%) - Factor más predictivo

ocean_proximity_index (15.2%) - Ubicación costera

latitud (12.8%) - Ubicación geográfica

habitaciones_por_hogar (9.5%) - Densidad habitacional

edad_mediana_vivienda (8.1%) - Antigüedad de viviendas

Benchmark de rendimiento
Operación Original Optimizado Mejora
Conteo total 2.3s 1.4s 39.1%
Filtrar por precio 1.8s 0.9s 50.0%
Agrupar por ubicación 3.2s 1.7s 46.9%
Entrenamiento de modelo 45.2s 28.7s 36.5%

🔧 Configuración avanzada
Archivo de configuración principal
Edita config/pipeline_config.yaml:
Configuración de Spark
spark_config:
app_name: “CaliforniaHousingPipeline”
executor_memory: “2g”
driver_memory: “1g”
sql_adaptive_enabled: true
shuffle_partitions: 100

Parámetros del modelo
modelo:
algoritmo: “random_forest”
parámetros:
num_trees: 100
profundidad_máxima: 10
semilla: 42

Ingeniería de características
ingeniería_de_características:
características_derivadas:

- "rooms_per_household"
- "bedrooms_per_room"
- "population_per_household"
- "income_per_household"
Configuración de almacenamiento
almacenamiento:
formato_primario: “parquet”
compresión: “snappy”
columnas_de_partición: [“rango_de_precios”]

Variables de Entorno
Configurar para entorno local
export SPARK_HOME=/ruta/a/spark
export PYSPARK_PYTHON=python3
export JAVA_HOME=/ruta/a/java

Para Google Colab, se configura automáticamente
🧪 Pruebas y Calidad de Código
Ejecución de Tests
Pruebas unitarias
python -m pytest tests/ -v

Pruebas con cobertura
python -m pruebas pytest/ —cov=src —cov-report=html

Pruebas específicas
python -m pytest tests/test_data_processing.py -v

Verificación de Calidad
Formato de código
negro src/ pruebas/

Pelusa
flake8 src/ pruebas/

Verificación de tipos (opcional)
mypy src/

📈 Conjunto de datos y fuentes de datos
Precios de la vivienda en California
Fuente: Conjunto de datos de Kaggle

Registros: 20.640 propiedades

Características: 10 variables iniciales

Período: Datos censales de California

Variables Principales
longitud, latitud: Coordenadas geográficas

housing_median_age: Edad media de las viviendas

total_habitaciones, total_dormitorios: Capacidad habitacional

población, hogares: Datos demográficos

mediana_ingresos: Ingreso medio de hogares

median_house_value: Variable objetivo (precio)

ocean_proximity: Categoría de ubicación costera

🤝 Contribución
¡Contribuciones son bienvenidas! Por favor sigue estos pasos:
Fork el proyecto

Crea una rama para tu feature (git checkout -b feature/AmazingFeature)

Confirma tus cambios (git commit -m 'Add AmazingFeature')

Push a la rama (git push origin feature/AmazingFeature)

Abre una solicitud de extracción

Guía de Desarrollo
1. Clonar y configurar
git clone https://github.com/tu-usuario/california-housing-bigdata.git
cd california-housing-bigdata

2. Instalar dependencias de desarrollo
pip install -r requirements-dev.txt

3. Configurar los ganchos de pre-commit
instalación previa a la confirmación

4. Desarrollar y testear
python -m pruebas pytest/ —cov=src —cov-report=html

Estándares de Código
Seguir PEP 8 para código Python

Usar docstrings para documentación de funciones

Incluir pruebas para nuevas funcionalidades.

Mantener cobertura de código > 80%

Actualizar la documentación correspondiente

🐛 Solución de Problemas
Problemas comunes
Error de memoria en Colab
Solución: Reducir el tamaño de datos o particiones
spark.conf.set("spark.sql.shuffle.partitions", "50")
spark.conf.set("spark.driver.memory", "1g")

Dependencias faltantes
Reinstalar dependencias
pip install --force-reinstall -r requirements.txt

Problema con Java
intento

Verificar instalación de Java
java -versión

Depuración
Habilitar logging detallado
import logging
logging.basicConfig(level=logging.INFO)

Verificar estadísticas de datos
df.describe().show()
df.printSchema()

Monitorear uso de memoria
df.cache().count() Forzar el almacenamiento en caché y la memoria
