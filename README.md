# Sistema de Asistencia con Reconocimiento Facial

Proyecto desarrollado en **Python** que utiliza técnicas de **visión por computadora** y **reconocimiento facial** para identificar empleados y registrar automáticamente su hora de ingreso en un archivo CSV.

---

## Características

* Detección y reconocimiento facial mediante la librería `face_recognition`
* Captura de imágenes utilizando la cámara web con OpenCV
* Registro automático de asistencia en un archivo CSV
* Comparación de rostros mediante vectores de 128 características faciales
* Gestión de una base de datos de empleados a partir de imágenes

---

## Tecnologías utilizadas

* Python 3
* OpenCV
* face_recognition
* NumPy
* datetime
* os

---

## Estructura del proyecto

```text
proyecto/
│
├── main.py          # Práctica inicial de comparación de rostros
├── asistente.py     # Primera versión del sistema de asistencia
├── jk.py            # Versión mejorada y depurada del sistema
├── registro.csv     # Registro de asistencias
└── Empleados/       # Carpeta con imágenes de empleados (no incluida)
```

---

## Funcionamiento

1. El sistema carga las imágenes almacenadas en la carpeta `Empleados`.
2. Cada rostro es convertido en un vector de **128 características faciales**.
3. Se captura una imagen desde la cámara web.
4. El rostro detectado se compara con los rostros registrados.
5. Si existe coincidencia, se registra el nombre y la hora de ingreso en `registro.csv`.

---

##  Instalación

Instalar las dependencias necesarias:

```bash
pip install opencv-python
pip install face-recognition
pip install numpy
```

---

## Nota sobre privacidad

Las imágenes de la carpeta `Empleados` no se incluyen en este repositorio para proteger la privacidad de las personas utilizadas durante las pruebas.

Para ejecutar el proyecto, crea una carpeta con nombre `Empleados` y agrega tus propias imágenes dentro de la carpeta.

---

##  Mejoras futuras

* [ ] Reconocimiento facial en tiempo real mediante video.
* [ ] Integración con bases de datos.
* [ ] Exportación de registros a Excel.
* [ ] Desarrollo de una interfaz gráfica de usuario (GUI).

---

## Autor

**Evelyn Sosa Rojas**

Proyecto desarrollado como práctica de **visión por computadora** y **reconocimiento facial** utilizando Python.

