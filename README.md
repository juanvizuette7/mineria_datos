# Laboratorio de Mineria de Datos con CRISP-DM

Este repositorio contiene el desarrollo de un laboratorio de mineria de datos orientado a aplicar las fases de **CRISP-DM** y a construir modelos de **regresion lineal multiple** para predecir variables cuantitativas en tres contextos:

1. Precio del dolar.
2. Niveles de glucosa en sangre.
3. Consumo de energia electrica.

Ademas del modelado, el proyecto incluye:

- evaluacion del desempeno con metricas de regresion,
- interpretacion de coeficientes,
- exportacion de modelos entrenados con `joblib`,
- una interfaz web en **Streamlit** para generar predicciones.

Este `README.md` tambien puede utilizarse como informe escrito de entrega.

## 1. Objetivo del laboratorio

Aplicar distintas fases de mineria de datos basadas en CRISP-DM para:

- comprender el problema de negocio o analitico,
- explorar y preparar datos,
- entrenar modelos de regresion lineal multiple,
- interpretar resultados,
- evaluar el desempeno,
- exportar modelos,
- desplegar una interfaz web basica para prediccion.

## 2. Metodologia CRISP-DM

### 2.1 Comprension del negocio

Se definieron tres problemas de prediccion numerica:

- **Ejercicio 1 - Dolar:** estimar `Precio_Dolar` a partir de `Dia`, `Inflacion` y `Tasa_interes`.
- **Ejercicio 2 - Glucosa:** estimar `Nivel_Glucosa` a partir de `Edad`, `IMC` y `Actividad_Fisica`.
- **Ejercicio 3 - Energia:** estimar `Consumo_Energia` a partir de `Temperatura`, `Hora` y `Dia_Semana`.

En los tres casos el objetivo es predecir una variable cuantitativa continua, por lo que la tecnica adecuada es regresion lineal multiple.

### 2.2 Comprension de los datos

Se trabajaron tres datasets ubicados en la carpeta `data/`:

- `data/dolar_data.csv` con 500 registros.
- `data/glucosa_data.csv` con 2000 registros.
- `data/energia_data.csv` con 10000 registros.

Hallazgos generales:

- Todos los datasets contienen variables numericas.
- No se detectaron valores nulos en las columnas usadas para modelado.
- Las relaciones entre variables pueden visualizarse con graficas de dispersion, `pairplot` y mapas de calor de correlacion.

### 2.3 Preparacion de los datos

Para cada ejercicio se siguio el mismo flujo base:

1. Carga del dataset con `pandas`.
2. Seleccion de variables independientes y variable objetivo.
3. Verificacion de integridad basica de datos.
4. Separacion entrenamiento/prueba con `train_test_split(test_size=0.2, random_state=42)`.

Nota metodologica:

- Las **metricas** reportadas en este documento corresponden a la particion 80/20.
- Los **modelos exportados** en `models/` se reentrenaron sobre el 100% de cada dataset para despliegue en la app.

### 2.4 Modelado

Se uso `LinearRegression` de `scikit-learn` en los tres ejercicios.

Modelo general:

```text
Y = b0 + b1*X1 + b2*X2 + b3*X3
```

Donde:

- `Y` es la variable dependiente.
- `b0` es la interseccion.
- `b1`, `b2` y `b3` son los coeficientes estimados para cada variable explicativa.

### 2.5 Evaluacion

Se emplearon las siguientes metricas:

- **MSE** para Dolar y Glucosa.
- **RMSE** para Energia.
- **R2** para los tres modelos.

Adicionalmente, para comparar impacto relativo entre variables con diferentes escalas, se revisaron **coeficientes estandarizados**.

### 2.6 Despliegue

Se desarrollo una app en **Streamlit** (`app/app.py`) que permite:

- seleccionar el escenario,
- ingresar los valores por teclado,
- obtener la prediccion en pantalla,
- visualizar contexto estadistico basico del dataset.

## 3. Estructura del proyecto

```text
mineria_datos_lab/
|-- app/
|   `-- app.py
|-- data/
|   |-- dolar_data.csv
|   |-- glucosa_data.csv
|   `-- energia_data.csv
|-- models/
|   |-- modelo_dolar.pkl
|   |-- modelo_glucosa.pkl
|   `-- modelo_energia.pkl
|-- notebooks/
|   `-- 01_analisis.ipynb
|-- train_models.py
`-- README.md
```

## 4. Herramientas utilizadas

- Python 3.11
- pandas
- scikit-learn
- joblib
- matplotlib
- seaborn
- streamlit

## 5. Ejecucion del proyecto

### 5.1 Instalar dependencias

```bash
python -m pip install pandas scikit-learn joblib matplotlib seaborn streamlit
```

### 5.2 Entrenar y exportar los modelos

El script `train_models.py` entrena los tres modelos de regresion lineal multiple, imprime sus metricas y exporta los archivos `.pkl` en la carpeta `models/`.

```bash
python train_models.py
```

### 5.3 Ejecutar la interfaz web

```bash
python -m streamlit run app/app.py
```

Luego abre en el navegador la URL local que Streamlit muestre en consola, normalmente:

```text
http://localhost:8501
```

## 6. Desarrollo por ejercicios

## 6.1 Ejercicio 1 - Prediccion del precio del dolar

### Variables

- `Dia`: numero de dia.
- `Inflacion`: tasa de inflacion diaria.
- `Tasa_interes`: tasa de interes diaria.
- `Precio_Dolar`: variable dependiente.

### Objetivo

Predecir `Precio_Dolar` usando `Dia`, `Inflacion` y `Tasa_interes`.

### Modelo final exportado

Formula aproximada del modelo:

```text
Precio_Dolar = 3978.985 + 4.999*Dia - 338.060*Inflacion - 2.533*Tasa_interes
```

### Interpretacion de coeficientes

- **Dia (`+4.999`)**: manteniendo constantes las otras variables, por cada dia adicional el precio del dolar aumenta en aproximadamente **5 COP**.
- **Inflacion (`-338.060`)**: al aumentar la inflacion en una unidad decimal, el modelo reduce el precio esperado del dolar. Como la variable se mueve en valores pequenos, su efecto practico diario es bajo.
- **Tasa_interes (`-2.533`)**: un incremento de una unidad en la tasa de interes reduce ligeramente el precio esperado del dolar.

### Impacto relativo

Segun los coeficientes estandarizados, la variable con mayor impacto es claramente **Dia**, por lo que el comportamiento del dataset esta dominado por una tendencia temporal.

### Desempeno del modelo

Metricas de validacion 80/20:

| Metrica | Valor |
|---|---:|
| MSE | 2376.9709 |
| RMSE | 48.7542 |
| R2 | 0.9963 |

### Interpretacion del desempeno

- El valor de **R2 = 0.9963** indica un ajuste excelente.
- El error medio es bajo frente al rango observado del dolar.
- El modelo captura muy bien la tendencia principal del dataset.

### Visualizaciones sugeridas

- `Dia` vs `Precio_Dolar`
- `Inflacion` vs `Precio_Dolar`
- `Tasa_interes` vs `Precio_Dolar`
- heatmap de correlacion

## 6.2 Ejercicio 2 - Prediccion de niveles de glucosa

### Variables

- `Edad`: edad del paciente.
- `IMC`: indice de masa corporal.
- `Actividad_Fisica`: horas semanales de ejercicio.
- `Nivel_Glucosa`: variable dependiente en mg/dL.

### Objetivo

Predecir `Nivel_Glucosa` usando `Edad`, `IMC` y `Actividad_Fisica`.

### Modelo final exportado

Formula aproximada del modelo:

```text
Nivel_Glucosa = 66.315 + 1.234*Edad + 0.883*IMC - 2.010*Actividad_Fisica
```

### Interpretacion de coeficientes

- **Edad (`+1.234`)**: por cada ano adicional, el nivel esperado de glucosa aumenta cerca de **1.23 mg/dL**, manteniendo constantes las otras variables.
- **IMC (`+0.883`)**: un aumento de un punto en IMC incrementa la glucosa estimada en aproximadamente **0.88 mg/dL**.
- **Actividad_Fisica (`-2.010`)**: cada hora adicional de actividad fisica semanal reduce el nivel esperado de glucosa en alrededor de **2.01 mg/dL**.

### Variable con mayor impacto

Tomando como referencia los coeficientes estandarizados, la variable mas influyente es **Edad**. En segundo lugar aparece **Actividad_Fisica** con efecto negativo, y despues **IMC**.

### Desempeno del modelo

Metricas de validacion 80/20:

| Metrica | Valor |
|---|---:|
| MSE | 233.6930 |
| RMSE | 15.2870 |
| R2 | 0.6814 |

### Interpretacion del desempeno

- El modelo alcanza un **R2 = 0.6814**, lo que representa un desempeno intermedio.
- Existe capacidad predictiva aceptable, pero tambien margen de mejora.
- La glucosa probablemente responde a relaciones no lineales o a variables no incluidas en el dataset.

### Analisis del impacto

La conclusion principal es que:

- a mayor edad, mayor nivel de glucosa esperado;
- a mayor IMC, mayor glucosa esperada;
- a mayor actividad fisica, menor glucosa esperada.

### Visualizaciones sugeridas

- `Edad` vs `Nivel_Glucosa`
- `IMC` vs `Nivel_Glucosa`
- `Actividad_Fisica` vs `Nivel_Glucosa`
- `pairplot` de todas las variables

## 6.3 Ejercicio 3 - Prediccion del consumo de energia electrica

### Variables

- `Temperatura`: temperatura en grados C.
- `Hora`: hora del dia.
- `Dia_Semana`: dia de la semana (1 a 7).
- `Consumo_Energia`: variable dependiente en kWh.

### Objetivo

Predecir `Consumo_Energia` usando `Temperatura`, `Hora` y `Dia_Semana`.

### Modelo final exportado

Formula aproximada del modelo:

```text
Consumo_Energia = 101.383 + 9.966*Temperatura + 5.012*Hora - 3.089*Dia_Semana
```

### Interpretacion de coeficientes

- **Temperatura (`+9.966`)**: por cada grado adicional, el consumo esperado aumenta en aproximadamente **9.97 kWh**.
- **Hora (`+5.012`)**: por cada hora adicional del dia, el consumo aumenta cerca de **5.01 kWh**.
- **Dia_Semana (`-3.089`)**: dias mas avanzados en la semana se asocian con una ligera disminucion del consumo.

### Variable con mayor impacto

Segun los coeficientes estandarizados, la variable de mayor impacto es **Temperatura**, seguida por **Hora**. `Dia_Semana` tiene una influencia menor.

### Desempeno del modelo

Metricas de validacion 80/20:

| Metrica | Valor |
|---|---:|
| RMSE | 20.7248 |
| R2 | 0.8968 |

Metricas complementarias:

| Metrica | Valor |
|---|---:|
| MSE | 429.5187 |

### Interpretacion del desempeno

- El modelo presenta un **R2 = 0.8968**, por lo que explica gran parte de la variabilidad del consumo.
- El **RMSE = 20.7248** indica un error relativamente bajo frente al nivel promedio del consumo.
- La temperatura y la hora explican la mayor parte del patron observado.

### Visualizaciones sugeridas

- `Temperatura` vs `Consumo_Energia`
- `Hora` vs `Consumo_Energia`
- `Dia_Semana` vs `Consumo_Energia`
- mapa de calor de correlacion

## 7. Resumen comparativo de modelos

| Ejercicio | Variable objetivo | Metrica principal | Valor | R2 | Variable de mayor impacto |
|---|---|---|---:|---:|---|
| Dolar | `Precio_Dolar` | MSE | 2376.9709 | 0.9963 | `Dia` |
| Glucosa | `Nivel_Glucosa` | MSE | 233.6930 | 0.6814 | `Edad` |
| Energia | `Consumo_Energia` | RMSE | 20.7248 | 0.8968 | `Temperatura` |

## 8. Exportacion de modelos

Los modelos finales se almacenan en:

- `models/modelo_dolar.pkl`
- `models/modelo_glucosa.pkl`
- `models/modelo_energia.pkl`

La exportacion se realiza con `joblib` porque:

- es eficiente para objetos de `scikit-learn`,
- facilita guardar y cargar modelos entrenados,
- permite reutilizar el modelo en una aplicacion web.

Ejemplo de carga:

```python
import joblib

modelo = joblib.load("models/modelo_dolar.pkl")
prediccion = modelo.predict([[250, 0.02, 5.0]])
```

## 9. Interfaz web basica

La interfaz se implemento con **Streamlit** en `app/app.py`.

Funcionalidades principales:

- seleccion del ejercicio: Dolar, Glucosa o Energia,
- ingreso manual de las variables independientes,
- ejecucion de la prediccion,
- visualizacion del resultado,
- contexto estadistico del dataset.

La app cumple el requerimiento de permitir predicciones para los tres escenarios desde una sola interfaz.

## 10. Relacion con los requerimientos de entrega

### Requerimiento 1

**Codigo Python de los tres modelos e interpretacion de coeficientes y desempeno**

Se cubre con:

- `train_models.py`
- `notebooks/01_analisis.ipynb`
- secciones 6 y 7 de este README

### Requerimiento 2

**Visualizaciones de relaciones entre variables**

Se cubre con:

- `notebooks/01_analisis.ipynb`
- graficas de dispersion, `regplot`, `pairplot` y mapas de calor

### Requerimiento 3

**Modelos exportados e interfaz web funcional**

Se cubre con:

- `models/*.pkl`
- `app/app.py`

### Requerimiento 4

**Informe escrito explicando pasos, analisis y conclusiones**

Se cubre con:

- `README.md`

## 11. Conclusiones

1. La metodologia CRISP-DM permitio organizar el trabajo desde la comprension del problema hasta el despliegue.
2. La regresion lineal multiple fue adecuada para los tres escenarios como linea base interpretable.
3. El modelo del dolar obtuvo el mejor desempeno, con un `R2` cercano a 1, dominado por la variable temporal `Dia`.
4. El modelo de glucosa mostro un ajuste moderado, lo que sugiere que hay factores clinicos adicionales que no estan presentes en el dataset.
5. El modelo de energia logro un ajuste alto; la temperatura resulto ser la variable mas influyente.
6. La exportacion con `joblib` y el despliegue con Streamlit permiten reutilizar los modelos en una aplicacion sencilla de prediccion.

## 12. Posibles mejoras

- incluir validacion cruzada adicional,
- comparar con modelos no lineales,
- agregar mas variables explicativas,
- guardar visualizaciones en una carpeta `reports/`,
- desplegar la app en la nube.
