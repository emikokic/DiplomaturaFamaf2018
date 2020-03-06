FROM jupyter/minimal-notebook

COPY 1.Laboratorio_Importacion.ipynb .
COPY 2.Laboratorio_Pasos.ipynb .
COPY Building_Permits.csv .
COPY building_permits-for-pandas.csv .
COPY README.md .
COPY requirements.txt .

RUN pip install -r requirements.txt

EXPOSE 8888
