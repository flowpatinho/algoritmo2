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
Reemplazar directorio con ubicación de archivos para entrenamiento "modelo-tflite-fotos.rar"

    tr_image_dir= 'modelo-tflite-fotos/train'
    tr_image_annotations= 'modelo-tflite-fotos/train'
    val_image_dir='modelo-tflite-fotos/valid'
    test_image_dir= 'modelo-tflite-fotos/test'

    label_map={1:'0', 2:'1', 3:'2', 4:'3', 5: '4', 6:'5', 7: '6', 8:'7', 9:'8', 10:'9'}
    print(label_map)

    #Muestras de entrenamiento: la informacion es caragda como tfrecord y almacenada en la carpeta de cache para un uso futuro
    train_ds = object_detector.DataLoader.from_pascal_voc(images_dir=tr_image_dir,
                                                          annotations_dir= tr_image_dir,
                                                          label_map=label_map
                                                         )
    #Muestras de validacion.
    val_ds = object_detector.DataLoader.from_pascal_voc(images_dir=val_image_dir,
                                                          annotations_dir= val_image_dir,
                                                          label_map=label_map,
                                                         )
    #Muestras de evaluacion.
    test_ds = object_detector.DataLoader.from_pascal_voc(images_dir=test_image_dir,
                                                          annotations_dir= test_image_dir,
                                                          label_map=label_map,
                                                         )

    print("Train dataset contains {} images".format(train_ds.__len__()))
    
Especificar el tipo de arquitectura que utilizará el modelo, para este caso, "EfficientNet-Lite2"

    spec = model_spec.get('efficientdet_lite2')
    
# 4. Entrenar el modelo de TensorFlow con los datos de entrenamiento

Se define el modelo en base al tipo, EfficientDet-Lite0, es un tipo de modelo el cual es bastante liviano en comparacion al resto, ademas, es bastante liviano y permite el uso offline de esta, debido a que el modelo es almacenado en el dispositivo, esto favorece el propósito de la aplicacion.

'EfficientDet-Lite2': utiliza de forma predetermianda 50 epocas, lo que significa que repasará las muestras 50 veces, la precisión en la etapa de validacion puede ser revisada en tiempo real mientras se ejecuta el entrenamiento. Para este caso se emplearon 200 epocas.
'batch_size': es el tamaño de la tanda de muestras que analiza el modelo por cada etapa, para este caso, es 10, ya que 10 veces 110 imagenes resultan 1101 muestras totales.
'train_whole_model': para afinar la precision del modelo completo se debe fijar en 'true', este entrena el modelo completo y no solo la priemra capa, esto puede mejorar sustancialemnte la precision, pero puede tardar mas tiempo en entrenarse.

    model = object_detector.create(train_ds, 
                                    model_spec=spec, 
                                    batch_size=10, 
                                    train_whole_model=True, 
                                    validation_data=val_ds,epochs = 200)
    
# 5. Medir precisión y perdidas despues del entrenamiento

    final_loss, final_accuracy = model.evaluate(valid_generator, steps = val_steps_per_epoch)
    print("Final loss: {:.2f}".format(final_loss))
    print("Final accuracy: {:.2f}%".format(final_accuracy * 100))
    
# 6. Exportar el modelo

Se exporta el modelo entrenado en formato de TensorFlow Lite indicando la carpeta de destino. La cuantizacion de post-entrenado predeterminada es una tecnica de cuantizacion integrada.
    
    model.export(export_dir='modelo-tflite-fotos/modelo-tflite-exportado/')
    
# 5. Modelo exportado

los factores que pueden afectar la exactitud del modelo pueden ser:

*   La cuantizacion ayuda a disminuir alrededor de 4 veces el tamaño del modelo anterior respecto de la precision
*   el modelo TFlite utiliza menos espacio pero disminuye la precision, esto debido a que otros modelo utilizan una supresion sin maximo de clases (NMS) para el post-procesado, mientras que TFLite utiliza una supresion de clases global, el cuale s mucho mas rapida, pero menos precisa. El resultado máxzimo de Keras es hasta 100 detecciones, mientras que TFLite solo admite 25.

La etapa de entrenamiento se finalizó, ahora se procede a la confeccion e implementacion del modelo entrenado en la plataforma movil.
