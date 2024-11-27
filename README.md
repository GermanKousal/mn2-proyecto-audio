# Análisis de Señales de Audio utilizando Transformada de Fourier

Este repositorio contiene el código y los datos utilizados para el proyecto final de la materia **Métodos Numéricos 2** de la carrera de Licenciatura en Informática de la Universidad Nacional de Tucumán.  
El proyecto analiza tres señales de audio con la pronunciación de la vocal **"a"** en español, grabadas a diferentes intensidades (suave, media y fuerte), explorando sus características temporales y espectrales.  

---

## Objetivos del Proyecto

1. Digitalizar y analizar las grabaciones utilizando herramientas de procesamiento de señales.
2. Comparar las frecuencias dominantes empleando la Transformada de Fourier y la correlación cruzada en el dominio frecuencial.
3. Generar espectrogramas para estudiar la evolución temporal de las frecuencias y apoyar las observaciones previas.

---

## Herramientas de Análisis

El análisis incluyó las siguientes técnicas:  
- **Cálculo de envolventes y RMS** para estimar la energía acústica.  
- **Transformada de Fourier** usando el algoritmo FFT para identificar frecuencias dominantes.  
- **Correlación cruzada en el dominio frecuencial** para evaluar similitudes entre señales.  
- **Espectrogramas** basados en la Transformada de Fourier de Tiempo Corto (STFT) para analizar la evolución temporal de las frecuencias.

---

## Librerías Utilizadas

Este proyecto fue desarrollado en Python con las siguientes librerías:  
- **[Librosa](https://librosa.org/doc/latest/index.html):** Extracción de características y espectrogramas.  
- **[SciPy](https://scipy.org/):** Correlación cruzada.  
- **[NumPy](https://numpy.org/):** Manipulación de datos, cálculos numéricos y herramientas para FFT.  
- **[Matplotlib](https://matplotlib.org/):** Visualización de señales y resultados.

---

## Organización del Repositorio

El contenido del repositorio está organizado de la siguiente manera:  
- `main.py`: Script principal que ejecuta todo el análisis.  
- `content/`: Carpeta que contiene los archivos de audio utilizados en el proyecto.  
- `output/`: Carpeta opcional para guardar las imágenes generadas durante la ejecución.  

---

## Conclusiones del Proyecto

Los análisis mostraron una alta similitud entre las señales en las frecuencias bajas (<1500 Hz) y variaciones en frecuencias más altas según la intensidad vocal. Sin embargo, las herramientas empleadas no permitieron garantizar con certeza que las señales correspondan a una misma vocal. Se sugiere explorar el análisis de formantes o implementar técnicas de aprendizaje automático (Machine Learning) para una mayor precisión en futuros estudios.

---

## Instrucciones de Uso

1. Clona este repositorio:  
   ```bash
   git clone https://github.com/GermanKousal/mn2-proyecto-audio.git
   cd mn2-proyecto-audio

2. Asegúrate de instalar las dependencias:
    ```bash
    pip install -r requirements.txt

3. Ejecuta el archivo principal:
    ```bash
    python main.py

4. Revisa las gráficas generadas en tu entorno local o en la carpeta `output/`.

Nota: Aunque el proyecto fue desarrollado inicialmente en Google Colab, esta versión consolidada es funcional en cualquier entorno Python. Para una mejor experiencia interactiva, se recomienda adaptar el script a un Jupyter Notebook.

---

## Autores

* Franco Muñoz - [Perfil de GitHub](https://github.com/Neokai94)
* Germán Kousal - [Perfil de GitHub](https://github.com/GermanKousal)

## Notas Adicionales
* El análisis genera nueve gráficos de salida. Para un mejor manejo de estas imágenes, considera usar un entorno interactivo como Jupyter Notebook.
* Si tienes comentarios o sugerencias, no dudes en abrir un issue o enviar un pull request.