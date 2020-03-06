Diplomatura en Ciencia de Datos, Aprendizaje por Refuerzo
=========================================================

Pagina
------

http://diplodatos.famaf.unc.edu.ar/


Repositorio
-----------

https://github.com/DiploDatos

-------------------------------------------------------------------------------

Aprendizaje por Refuerzo
========================

Repositorio
-----------

https://github.com/DiploDatos/AprendizajePorRefuerzos


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

        git clone https://github.com/DiploDatos2018/Aprendizaje_Refuerzo.git

4. TensorFlow para Python3.6 (**CPU**):

        URL: https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.10.0-cp36-cp36m-linux_x86_64.whl

        pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.10.0-cp36-cp36m-linux_x86_64.whl

5. TensorFlow para Python3.6 (**GPU**):

        URL: https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.10.0-cp36-cp36m-linux_x86_64.whl

        pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.10.0-cp36-cp36m-linux_x86_64.whl

6. Seguir las instrucciones de la siguiente Notebook:

        https://github.com/DiploDatos/AprendizajeProfundo/blob/master/0_set_up.ipynb

7. (Opcional) Instalar PyCharm Community Edition:

    En un sistema basado en Debian (como Ubuntu), se puede hacer:

        sudo snap install pycharm-community --classic

Entorno Virtual
---------------

1.  Para crear y activar nuestro virtualenv:

        mkvirtualenv --python=/usr/bin/python3.6 diplodatos-rl

4.  Instalar dependencias:

        cd Aprendizaje_Refuerzo
        pip install -r requirements.txt


Ejecución
---------

1.  Activar el entorno virtual con:

        workon diplodatos-rl



<!---------------------- Links ---------------------->
[virtualenv]: http://virtualenv.readthedocs.org/en/latest/virtualenv.html
[virtualenvwrapper]: http://virtualenvwrapper.readthedocs.org/en/latest/install.html#basic-installation
