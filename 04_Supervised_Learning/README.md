Diplomatura en Ciencia de Datos, Aprendizaje Supervisado
========================================================

Pagina
------

http://diplodatos.famaf.unc.edu.ar/


Repositorio
-----------

https://github.com/DiploDatos

-------------------------------------------------------------------------------

Aprendizaje Supervisado
=======================

Pagina
------

Para acceder a la pagina de la materia, seguir los siguientes pasos:
1. Ingresar a [Google Classroom](https://edu.google.com/intl/es-419/products/productivity-tools/classroom/).
2. Iniciar sesion desde su cuenta de Google.
3. Seleccionar el signo **+** y luego **Apuntarse a una clase**.
4. Igresar el siguiente código: _mx1v0ay_.

Repositorio
-----------

https://github.com/DiploDatos/AprendizajeSupervisado


Instalación
-----------

1.  Se necesita el siguiente software:

    -   Git
    -   Pip
    -   Python 3.6
    -   TkInter
    -   Virtualenv

    En un sistema basado en Debian (como Ubuntu), se puede hacer:

        sudo apt-get install git python-pip python3 python3-tk virtualenv

2.  Crear y activar un nuevo [virtualenv]. Recomiendo usar [virtualenvwrapper]. Se puede instalar así:

        sudo pip install virtualenvwrapper

    Y luego agregando la siguiente línea al final del archivo `.bashrc`:

        export WORKON_HOME=$HOME/.virtualenvs
        export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python
        export VIRTUALENVWRAPPER_VIRTUALENV=/usr/local/bin/virtualenv
        [[ -s "/usr/local/bin/virtualenvwrapper.sh" ]] && source "/usr/local/bin/virtualenvwrapper.sh"

3.  Bajar el código:

        git clone https://github.com/DiploDatos2018/Aprendizaje_Supervisado.git


Entorno Virtual
---------------

1.  Para crear y activar nuestro virtualenv:

        mkvirtualenv --python=/usr/bin/python3.6 diplodatos-as

4.  Instalar dependencias:

        cd Aprendizaje_Supervisado
        pip install -r requirements.txt


Ejecución
---------

1.  Activar el entorno virtual con:

        workon diplodatos-as



<!---------------------- Links ---------------------->
[virtualenv]: http://virtualenv.readthedocs.org/en/latest/virtualenv.html
[virtualenvwrapper]: http://virtualenvwrapper.readthedocs.org/en/latest/install.html#basic-installation
