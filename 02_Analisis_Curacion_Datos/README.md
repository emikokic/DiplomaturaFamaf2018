Diplomatura en Ciencia de Datos, Aprendizaje Automático y sus Aplicaciones
==========================================================================

Pagina
------

http://diplodatos.famaf.unc.edu.ar/


Repositorio
-----------

https://github.com/DiploDatos

-------------------------------------------------------------------------------

Análisis Exploratorio y Curación de Datos
=========================================

Pagina
------

http://diplodatos.famaf.unc.edu.ar/analisis-y-curacion-de-datos/

Para acceder a la pagina de la primera parte de materia, seguir los siguientes pasos:
1. Ingresar a [Google Classroom](https://edu.google.com/intl/es-419/products/productivity-tools/classroom/).
2. Iniciar sesion desde su cuenta de Google.
3. Seleccionar el signo **+** y luego **Apuntarse a una clase**.
4. Igresar el siguiente código: _q2f3enu_.


Repositorios (Parte 2 - Python)
-------------------------------

* https://github.com/DiploDatos/AnalisisyCuracion
* https://github.com/bitlogic/hello-docker
* https://github.com/gmiretti/DataScienceExamples


Instalación
-----------

1.  Se necesita el siguiente software:

    -   Git
    -   Pip
    -   Python 3.6
    -   TkInter
    -   Virtualenv
    -   R
    -   RStudio

    Para la instalacion de R, leer el siguiente Blog: [Install R on Ubuntu 18.04 Bionic Beaver Linux](https://linuxconfig.org/install-r-on-ubuntu-18-04-bionic-beaver-linux).

    En un sistema basado en Debian (como Ubuntu), se puede hacer:

        sudo apt-get install git python-pip python3 python3-tk virtualenv r-base

2.  Crear y activar un nuevo [virtualenv]. Recomiendo usar [virtualenvwrapper]. Se puede instalar así:

        sudo pip install virtualenvwrapper

    Y luego agregando la siguiente línea al final del archivo `.bashrc`:

        export WORKON_HOME=$HOME/.virtualenvs
        export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python
        export VIRTUALENVWRAPPER_VIRTUALENV=/usr/local/bin/virtualenv
        [[ -s "/usr/local/bin/virtualenvwrapper.sh" ]] && source "/usr/local/bin/virtualenvwrapper.sh"

3.  Instalar **RStudio**:

    Para instalar RStudio ir a su [pagina principal](https://www.rstudio.com/products/rstudio/download/) y descargar la version **Free** segun su version de Ubuntu.

4.  Bajar el código:

        git clone https://github.com/DiploDatos2018/Analisis_Exploratorio_Curacion_Datos.git


Entorno Virtual
---------------

1.  Para crear y activar nuestro virtualenv:

        mkvirtualenv --python=/usr/bin/python3.6 diplodatos-ayc

4.  Instalar dependencias:

        cd Analisis_Exploratorio_Curacion_Datos
        pip install -r Parte02_Python/requirements.txt


Ejecución
---------

1.  Activar el entorno virtual con:

        workon diplodatos-ayc



<!---------------------- Links ---------------------->
[virtualenv]: http://virtualenv.readthedocs.org/en/latest/virtualenv.html
[virtualenvwrapper]: http://virtualenvwrapper.readthedocs.org/en/latest/install.html#basic-installation
