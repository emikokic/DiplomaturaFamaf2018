# Laboratorio

Limpiar un set de datos con pandas
* Un set de datos que tengan permisos para compartir con nosotros
    - Más de 100000 registros
    - Más de 20 columnas
    - Con datos con cadenas, números, fechas, y categorías
* Usar los permisos de edificación de San Francisco:
    - https://www.kaggle.com/aparnashastry/building-permit-applications-data/data

El análisis tiene que ser reproducible en las máquinas de los profes: **Docker**

-------------------------------------------------------------------------------

### Descargar Dataset
En un sistema basado en Debian (como Ubuntu), se puede hacer:

    wget https://www.kaggle.com/aparnashastry/building-permit-applications-data/downloads/Building_Permits.csv/1

-------------------------------------------------------------------------------

### Dockerizando el Laboratorio

**IMPORTANTE:** Se debe estar posicionado en el directorio **Laboratorio**.

##### 1. Crear imagen Docker del Laboratorio
    docker build -t lab-ayc .

##### 2. Correr la imagen creda
    docker run --rm -p 8888:8888 --net=host lab-ayc
