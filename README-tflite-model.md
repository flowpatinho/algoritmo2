# Segundo algoritmo para asignatura de finalización con el tema "prototipo de aplicación móvil para el aporte de lectura de clientes con tarifa básica para una empresa de distribución eléctrica"

En el repositorio se entrega el codigo fuente para la confeccion del mododelo de deteccion, OCR y aplicacion final

Para el modelo de detección con Tensorflow Lite se utilizó Anaconda para la creacion de ambientes virtuales y python 3.7 para la ejecución

# 1. Instalar librerias:

    pip install tflite-model-maker
    pip install pycocotools
    pip install ipykernel
    pip install opencv-python==4.5.5.64
    
# 2. Se importan los paquetes requeridos para la creacion del modelo de entrenamiento:

    import numpy as np
    import os
    from tflite_model_maker.config import ExportFormat
    from tflite_model_maker import model_spec
    from tflite_model_maker import object_detector
    import tensorflow as tf
    assert tf.__version__.startswith('2')
    
Tip: Para agilizar el proceso de entrenamiento, se recomienda hacer uso de trajeta de video dedicada (CUDA Toolkit 11.2 y CuDNN 8.1.0)

    print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")

# 3. Cargar base de datos previamente descargados de Roboflow

Se declaran los directorios correspondientes al entrenamiento, anotaciones, validacion y evaluacion del modelo, tambien se enumeran los tipos de clases en los cuales el etiquetado del dataset se llevó a cabo.

    tr_image_dir= 'modelo-tflite-fotos/train'
tr_image_annotations= 'modelo-tflite-fotos/train'
val_image_dir='modelo-tflite-fotos/valid'
test_image_dir= 'modelo-tflite-fotos/test'

label_map={1:'0', 2:'1', 3:'2', 4:'3', 5: '4', 6:'5', 7: '6', 8:'7', 9:'8', 10:'9'}
print(label_map)

    # Muestras de entrenamiento: la informacion es caragda como tfrecord y almacenada en la carpeta de cache para un uso futuro
    train_ds = object_detector.DataLoader.from_pascal_voc(images_dir=tr_image_dir,
                                                          annotations_dir= tr_image_dir,
                                                          label_map=label_map
                                                         )
    # Muestras de validacion.
    val_ds = object_detector.DataLoader.from_pascal_voc(images_dir=val_image_dir,
                                                          annotations_dir= val_image_dir,
                                                          label_map=label_map,
                                                         )
    # Muestras de evaluacion.
    test_ds = object_detector.DataLoader.from_pascal_voc(images_dir=test_image_dir,
                                                          annotations_dir= test_image_dir,
                                                          label_map=label_map,
                                                         )

    print("Train dataset contains {} images".format(train_ds.__len__()))
