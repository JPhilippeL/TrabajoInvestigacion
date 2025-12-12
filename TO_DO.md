# COSAS Q HACER
Normalizar todos los valores de E antes de meterlo al lime usando minmax
Programar lo de fidelity
Hacer muestras de lo de fidelity

# No tan importante
Que el calculo de ver la importancia se pueda hacer con alfa y beta, alfa o beta?

# Miscelaneo
Ordenar el código de menu bar dividiendolo

# COSAS Q PREGUNTAR
Preguntar si lo de grid search es de cara al tfm o lo hago ahora
Hago Beta^t * E * alfa^t pq los vectores se inicializan como columnas en pytorch
Para las distancias, sigma es la desviacion estandar = 2*mean²
Las distancias se normalizan haciendo que la suma de todas de 1, para que el loss sea independiente del numero de samples
Luego se multiplican por numero de samples porque al hacer el loss se hace .mean(N)
entonces te quedaria 1/N mientras que si lo multiplicas te queda alrededor de N/N

El nuestro ve como muy importante el atomo mientras que el GNNexplainer no pq:
En GNNExplainer se "oculta" el simbolo del atomo, pero se sigue teniendo la informacion
de las conexiones y demás, por lo que puede "deducir" que atomo es, por ejemplo:
Si un nodo tiene solo una conexion, aunque no sepa el simbolo, puede deducir que es hidrogeno

Mientras que el nuestro, ve que si cambias el simbolo, influye mucho en la pendiente
del modelo local, por lo que le da mucha importancia.

El nuestro responde a "Si cambio este atomo, cambia la prediccion?" Si, entonces es importante
El otro responde "Aunque no tenga esta informacion, puedo predecir lo mismo? Si, pq sigo sabiendo que
atomo es aunque no tenga esa info, por lo que no es importante.

El nuestro mide Sensibilidad
El otro mide Suficiencia/Informacion