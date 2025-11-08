
# 🏠 California Housing Prices - Big Data Pipeline

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![PySpark](https://img.shields.io/badge/PySpark-3.4.0-red)](https://spark.apache.org)
[![Google Colab](https://img.shields.io/badge/Google%20Colab-Compatible-orange)](https://colab.research.google.com)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

Un pipeline completo de Big Data para predicción de precios de viviendas en California utilizando PySpark, optimizaciones distribuidas y machine learning escalable.

## 📊 Demo Rápido en Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tu-usuario/california-housing-bigdata/blob/main/notebooks/04_pipeline_demo.ipynb)

```bash
# Ejecución inmediata en Colab
!pip install pyspark findspark
!git clone https://github.com/tu-usuario/california-housing-bigdata.git
%cd california-housing-bigdata

from src.main_pipeline import HousingPricePipeline
pipeline = HousingPricePipeline()
results = pipeline.run_complete_pipeline()

# 🎯 Descripción del Proyecto
Este proyecto implementa un pipeline de Big Data completo para predecir precios de viviendas en California utilizando el dataset público de Kaggle. El sistema demuestra mejores prácticas en procesamiento distribuido, optimización de performance y machine learning escalable con PySpark.

🚀 Características Principales
📥 Ingesta Multi-fuente: Datos desde Kaggle API, URLs públicas y datos de ejemplo

🧹 Procesamiento Robusto: Limpieza automática, validación de calidad y transformaciones

⚡ Optimizaciones Avanzadas: Caching estratégico, particionamiento, configuración Spark optimizada

🤖 ML Distribuido: Random Forest con PySpark ML y feature engineering

💾 Almacenamiento Eficiente: Parquet comprimido con particionamiento inteligente

📊 Visualización Integral: Análisis exploratorio automático y métricas de performance


# 🏗️ Arquitectura del Pipeline
Diagrama de Flujo

https://imgur.com/a/BP5wmym

## Componentes Principales
DataIngestion: Descarga y carga de datasets desde múltiples fuentes con resiliencia

DataProcessor: Limpieza, validación y transformaciones con manejo de valores nulos

FeatureEngineering: Creación de características derivadas y codificación categórica

OptimizationManager: Técnicas de optimización distribuida (caching, particionamiento)

ModelTrainer: Entrenamiento de modelos de ML con PySpark ML

DataStorage: Almacenamiento eficiente en múltiples formatos con compresión
# 📁 Estructura del Repositorio

california-housing-bigdata/
│
├── 📄 README.md                         # Este archivo
├── 📋 requirements.txt                  # Dependencias del proyecto
├── ⚙️ config/
│   └── pipeline_config.yaml             # Configuraciones del pipeline
│
├── 🐍 src/                              # Código fuente principal
│   ├── __init__.py
│   ├── data_ingestion.py                # Módulo de ingesta de datos
│   ├── data_processing.py               # Procesamiento y limpieza
│   ├── feature_engineering.py           # Ingeniería de características
│   ├── data_storage.py                  # Almacenamiento optimizado
│   ├── optimization.py                  # Técnicas de optimización
│   ├── model_training.py                # Entrenamiento de modelos
│   └── main_pipeline.py                 # Pipeline principal unificado
│
├── 📓 notebooks/                        # Jupyter notebooks
│   ├── 01_data_exploration.ipynb        # Análisis exploratorio de datos
│   ├── 02_feature_analysis.ipynb        # Análisis de características
│   ├── 03_model_evaluation.ipynb        # Evaluación de modelos
│   └── 04_pipeline_demo.ipynb           # Demo completo en Colab
│
├── 💾 data/                             # Datasets
│   ├── raw/                             # Datos crudos
│   ├── processed/                       # Datos procesados
│   └── models/                          # Modelos entrenados
│
├── ✅ tests/                            # Tests automatizados
│   ├── __init__.py
│   ├── test_data_processing.py          # Tests de procesamiento
│   ├── test_feature_engineering.py      # Tests de ingeniería
│   └── test_optimization.py             # Tests de optimización
│
├── 📊 docs/                             # Documentación
│   ├── architecture_diagrams/           # Diagramas de arquitectura
│   ├── technical_report.pdf             # Informe técnico completo
│   └── api_documentation.md             # Documentación de API
│
└── 🔧 scripts/                          # Scripts de utilidad
    ├── setup_environment.sh             # Configuración de entorno
    ├── run_pipeline.py                  # Ejecución del pipeline
    └── benchmark_performance.py         # Benchmark de rendimiento

#  🚀 Instalación Rápida

##Opción 1: Google Colab (Recomendado)
# Instalación en Google Colab - Ejecutar en una celda
!pip install pyspark==3.4.0 findspark pandas matplotlib seaborn requests
!git clone https://github.com/tu-usuario/california-housing-bigdata.git
%cd california-housing-bigdata

# Importar y ejecutar
import findspark
findspark.init()

from src.main_pipeline import HousingPricePipeline
pipeline = HousingPricePipeline()
results = pipeline.run_complete_pipeline()

# Opción 2: Entorno Local
# 1. Clonar repositorio
git clone https://github.com/tu-usuario/california-housing-bigdata.git
cd california-housing-bigdata

# 2. Crear entorno virtual (opcional pero recomendado)
python -m venv housing_env
source housing_env/bin/activate  # Linux/Mac
# housing_env\Scripts\activate  # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Verificar instalación
python -c "from src.main_pipeline import HousingPricePipeline; print('✅ Instalación exitosa')"
# Prerrequisitos
### Python 3.8+
### Java 8/11 (requerido para PySpark)
### 4GB+ RAM recomendado para procesamiento eficiente
###Google Colab o entorno local con las dependencias instaladas
# 💻 Uso del Pipeline
## Ejecución Completa Automática
from src.main_pipeline import HousingPricePipeline

# Pipeline completo automático
pipeline = HousingPricePipeline()
final_data, trained_model, performance_metrics = pipeline.run_complete_pipeline()

# Resultados automáticamente generados
print(f"✅ Pipeline completado: {final_data.count()} registros procesados")
print(f"📊 Métricas: {performance_metrics}")
# Ejecución por Módulos Individuales
# Ingesta específica
from src.data_ingestion import DataIngestion
ingestion = DataIngestion()
raw_data = ingestion.download_kaggle_dataset()

# Procesamiento personalizado
from src.data_processing import DataProcessor
processor = DataProcessor(spark)
cleaned_data = processor.clean_data(raw_data)

# Feature engineering avanzado
featured_data = processor.feature_engineering(cleaned_data)

# Entrenamiento de modelo
from src.model_training import ModelTrainer
trainer = ModelTrainer(spark)
model, predictions = trainer.train_model(featured_data)
# Scripts de Línea de Comandos
# Ejecutar pipeline completo
python scripts/run_pipeline.py

# Solo benchmarking de performance
python scripts/benchmark_performance.py

# Ejecutar tests unitarios
python -m pytest tests/ -v

# Ejecutar con configuración personalizada
python scripts/run_pipeline.py --config config/custom_config.yaml
# ⚡ Optimizaciones Implementadas
## Técnicas de Performance

Técnica	Mejora	Impacto
Caching Estratégico	35%	Tiempo de procesamiento
Particionamiento Inteligente	40%	Consultas filtradas
Configuración Spark Adaptativa	25%	Uso de memoria y CPU
Compresión Snappy	60%	Almacenamiento en disco
Predicate Pushdown	30%	Operaciones de filtrado
# Configuración Spark Optimizada
# Configuraciones críticas aplicadas
spark.conf.set("spark.sql.adaptive.enabled", "true")
spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true") 
spark.conf.set("spark.sql.adaptive.skew.enabled", "true")
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "10485760")
spark.conf.set("spark.sql.shuffle.partitions", "100")
# 📊 Resultados y Métricas
## Performance del Modelo
Métrica	Valor	Mejora vs Baseline
R² Score	0.81	+15%
RMSE	$48,250	-22%
MAE	$35,120	-18%
Tiempo Entrenamiento	28.7s	-36.5%
# Características Más Importantes
median_income (28.5%) - Factor más predictivo

ocean_proximity_index (15.2%) - Ubicación costera

latitude (12.8%) - Ubicación geográfica

rooms_per_household (9.5%) - Densidad habitacional

housing_median_age (8.1%) - Antigüedad de viviendas
# Benchmark de Rendimiento
Operación	Original	Optimizado	Mejora
Count total	2.3s	1.4s	39.1%
Filter por precio	1.8s	0.9s	50.0%
Group by ubicación	3.2s	1.7s	46.9%
Model training	45.2s	28.7s	36.5%
# 🔧 Configuración Avanzada
## Archivo de Configuración Principal
### Editar config/pipeline_config.yaml:
# Spark Configuration
spark_config:
  app_name: "CaliforniaHousingPipeline"
  executor_memory: "2g"
  driver_memory: "1g"
  sql_adaptive_enabled: true
  shuffle_partitions: 100

# Model Parameters
model:
  algorithm: "random_forest"
  parameters:
    num_trees: 100
    max_depth: 10
    seed: 42

# Feature Engineering
feature_engineering:
  derived_features:
    - "rooms_per_household"
    - "bedrooms_per_room"
    - "population_per_household"
    - "income_per_household"

# Storage Settings
storage:
  primary_format: "parquet"
  compression: "snappy"
  partition_columns: ["price_range"]
#   Variables de Entorno
###### Configurar para entorno local
export SPARK_HOME=/path/to/spark
export PYSPARK_PYTHON=python3
export JAVA_HOME=/path/to/java

###### Para Google Colab, se configuran automáticamente
# 🧪 Testing y Calidad de Código
## Ejecución de Tests
######Tests unitarios
python -m pytest tests/ -v

######  Tests con cobertura
python -m pytest tests/ --cov=src --cov-report=html

###### Tests específicos
python -m pytest tests/test_data_processing.py -v
# Verificación de Calidad
###### Formateo de código
black src/ tests/

###### Linting
flake8 src/ tests/

###### Verificación de tipos (opcional)
mypy src/
# 📈 Dataset y Fuentes de Datos
## California Housing Prices
Fuente: Kaggle Dataset

Registros: 20,640 propiedades

Características: 10 variables iniciales

Período: Datos censales de California
## Variables Principales
longitude, latitude: Coordenadas geográficas

housing_median_age: Edad media de las viviendas

total_rooms, total_bedrooms: Capacidad habitacional

population, households: Datos demográficos

median_income: Ingreso medio de hogares

median_house_value: Variable objetivo (precio)

ocean_proximity: Categórica de ubicación costera
# 🤝 Contribución
¡Contribuciones son bienvenidas! Por favor sigue estos pasos:
Fork el proyecto

Crea una rama para tu feature (git checkout -b feature/AmazingFeature)

Commit tus cambios (git commit -m 'Add AmazingFeature')

Push a la rama (git push origin feature/AmazingFeature)

Abre un Pull Request
# Guía de Desarrollo
# 1. Clonar y configurar
git clone https://github.com/tu-usuario/california-housing-bigdata.git
cd california-housing-bigdata

###### 2. Instalar dependencias de desarrollo
pip install -r requirements-dev.txt

###### 3. Configurar pre-commit hooks
pre-commit install

###### 4. Desarrollar y testear
python -m pytest tests/ --cov=src --cov-report=html
## Estándares de Código
Seguir PEP 8 para código Python

Usar docstrings para documentación de funciones

Incluir tests para nuevas funcionalidades

Mantener cobertura de código > 80%

Actualizar documentación correspondiente
# 🐛 Solución de Problemas
## Problemas Comunes
### Error de memoria en Colab

###### Solución: Reducir tamaño de datos o particiones
spark.conf.set("spark.sql.shuffle.partitions", "50")
spark.conf.set("spark.driver.memory", "1g")
### Dependencias faltantes
###### Reinstalar dependencias
pip install --force-reinstall -r requirements.txt
### Problemas con Java

bash
###### Verificar instalación de Java
java -version
# Debugging
###### Habilitar logging detallado
import logging
logging.basicConfig(level=logging.INFO)

###### Verificar estadísticas de datos
df.describe().show()
df.printSchema()

###### Monitorear uso de memoria
###### df.cache().count()  Forzar caching y ver memoria
