# Deep Learning con Python y Keras

## Reconocimiento de dígitos

### Cargar MNIST

El conjunto de datos se descarga automáticamente la primera vez que se llama a esta función y se almacena en su directorio de inicio en `~/.keras/datasets/mnist.pkl.gz` como un archivo de 15 megabytes. 

Primero escribiremos un pequeño script para descargar y visualizar las primeras 4 imágenes mediante la función `mnist.load data()`.

![image](https://github.com/PJBigBoss115/reconocimientoDigitos/assets/65696918/0c6a0200-a9a0-433b-864f-c25844f035d7)

### MLP de línea base

Vamos a usar un MLP clásico como base para la comparación con modelos de redes neuronales convolucionales. 
Importamos las clases, funciones y el dataset MNIST.

Para un MLP clásico debemos reducir las imágenes a un vector de píxeles. En este caso, las imágenes de tamaño $28 × 28$ serán vectores de entrada de 784 píxeles. 
Realizamos esta transformación meidante la función `reshape()`. 

Los valores de los píxeles son números enteros, por lo que los convertimos a punto flotante para poder normalizarlos.

![image](https://github.com/PJBigBoss115/reconocimientoDigitos/assets/65696918/0812652c-08ce-4942-9421-2b77ca14acbc)

![image](https://github.com/PJBigBoss115/reconocimientoDigitos/assets/65696918/a365e02c-3c18-4dd1-ac95-83751b065ffa)

### CNN para MNIST

Ahora que hemos visto cómo cargar el conjunto de datos MNIST y entrenar un modelo simple de perceptrón multicapa en él, es hora de desarrollar una red neuronal convolucional más sofisticada o un modelo CNN. 

Crearemos una CNN simple para MNIST que demuestra cómo utilizar todos los aspectos de una implementación de CNN moderna, incluidas las capas convolucionales, las capas de agrupación y las capas de dropout. 

![image](https://github.com/PJBigBoss115/reconocimientoDigitos/assets/65696918/9b7b00ba-2cfb-4c06-b7c7-501959608a82)

![image](https://github.com/PJBigBoss115/reconocimientoDigitos/assets/65696918/f862029f-e882-4c7a-af80-fd9d0b6a7ee1)

### CNN más profunda para MNIST

Esta vez definimos una arquitectura con más capas de convolucionales, Max-pooling y capas completamente conectadas.

1. Capa convolucional con 30 mapas de tamaño $5 × 5$.
2. Capa de Pooling con patch de $2 × 2$.
3. Capa convolucional con 15 mapas de tamaño $3 × 3$.
4. Capa de Pooling con patch de $2 × 2$.
5. Capa de Dropout del 20%.
6. Capa Flatten.
7. Capa completamente conectada con 128 neuronas y ReLu.
8. Capa completamente conectada con 50 neuronas y ReLu
9. Capa de salida cpm activación Softmax.
10. La compilación con ADAM, pérdida logarítmica como función de coste y Accuracy como métrica.

![image](https://github.com/PJBigBoss115/reconocimientoDigitos/assets/65696918/104feece-fc9c-4450-ac10-793a62ad47c4)

![image](https://github.com/PJBigBoss115/reconocimientoDigitos/assets/65696918/1de6ec5a-f03c-46dd-a782-ea217411e1fb)
