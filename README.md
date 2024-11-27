# Análisis de señales de Audio usando Transformada de Fourier
En este repositorio se encuentran el codigo y los datos utilizados para realizar el proyecto final de la materia de Metodos Numericos 2 de la carrera de Licenciatura en Informatica de la Universidad Nacional de Tucuman

El proyecto consite en analizar 3 señales de audio que contienen la pronunciacion de la vocal 'a' en español.

## Herramientas de análisis
* Calculo de envolventes y RMS de las señales
* Calculo de Transformada de Fourier de las señales usando FFT
* Calculo de correlacion cruzada sobre el dominio de las frecuencias
* Calculo de espectrogramas usando Transformada de Fourier de Tiempo Corto (STFT)

## Librerías utilizadas
El análisis se realizó utilizando Python, aprovechando las siguientes librerías:
* Librosa: Proporcionó herramientas para la extracción de características y la generación de espectrogramas.
* SciPy: Implementó funciones avanzadas de FFT y correlación.
* NumPy: Utilizada para cálculos matemáticos y manipulación de datos.
* Matplotlib: Empleada para la visualización de las señales y resultados.

## Conclusiones
Se encontro gran similitud entre las señales aunque las herramientas resultaron insuficientes para garantizart de que se tratan de una misma vocal. Se sugiere investigar las formantes o usar tecnicas de Machine Learning.

## Autores
Este proyecto fue realizado por Franco Muñoz y Germán Kousal.

## Aclaraciones sobre este repositorio
Debido a que se producen una serie de 9 imagenes en este codigo, es probable que la visualizacion de las mismas sea mas comodo usando Jupyter Notebook. El codigo original se hizo en la plataforma Colab de Google. Esta es una version unificada del codigo original.