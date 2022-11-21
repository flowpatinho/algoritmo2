# model.tflite

Código fuente para la creación de modelo de detección personalizado, encargado de detectar el consumo para medidores eléctricos no inteligentes a partir de una fotografía.

# 1. Se importan los paquetes requeridos para la creacion del modelo de entrenamiento

    import numpy as np
    import os
    from tflite_model_maker.config import ExportFormat
    from tflite_model_maker import model_spec
    from tflite_model_maker import object_detector
    import tensorflow as tf
    #assert tf.__version__.startswith('2'
    
Para agilizar el proceso de entrenamiento, se recomienda hacer uso de trajeta de video dedicada (CUDA Toolkit 11.2 y CuDNN 8.1.0)

# 2. Cargar base de datos previamente descargados de Roboflow

    tr_image_dir= 'dataset/train'
    tr_image_annotations= 'dataset/train'
    val_image_dir='dataset/valid'
    test_image_dir= 'dataset/test'

    label_map={1:'0', 2:'1', 3:'2', 4:'3', 5: '4', 6:'5', 7: '6', 8:'7', 9:'8', 10:'9', 11:'consumption'}
    print(label_map)

Muestras de entrenamiento: la informacion es caragda como tfrecord y almacenada en la carpeta de cache para un uso futuro

    train_ds = object_detector.DataLoader.from_pascal_voc(images_dir=tr_image_dir,
                                                          annotations_dir= tr_image_dir,
                                                          label_map=label_map
                                                         )
Muestras de validacion.

    val_ds = object_detector.DataLoader.from_pascal_voc(images_dir=val_image_dir,
                                                          annotations_dir= val_image_dir,
                                                          label_map=label_map,
                                                         )
Muestras de evaluacion.

    test_ds = object_detector.DataLoader.from_pascal_voc(images_dir=test_image_dir,
                                                          annotations_dir= test_image_dir,
                                                          label_map=label_map,
                                                         )

    print("Train dataset contains {} images".format(train_ds.__len__()))

Se especifica el tipo de arquitectura que se desea implementar en el entrenamiento del modelo, para este caso, se utilizará EfficientNet-Lite0

    spec = model_spec.get('efficientdet_lite0')
    
# 3. Entrenar el modelo de TensorFlow con los datos de entrenamiento

Se define el modelo en base al tipo, EfficientDet-Lite0, es un tipo de modelo el cual es bastante liviano en comparacion al resto, ademas, es bastante liviano y permite el uso offline de esta, debido a que el modelo es almacenado en el dispositivo, esto favorece el propósito de la aplicacion.


1.   'EfficientDet-Lite0': utiliza de forma predetermianda 50 epocas, lo que significa que repasará las muestras 50 veces, la precision en la etapa de validacion puede ser revisada en tiempo real mientras se ejecuta el entrenamiento, asi como peude ser abortada para evitar el solapamiento.
2.   'batch_size': es el tamaño de la tanda de muestras que analiza el modelo por cada etapa, para este caso, es 6, ya que 6 veces 100 imagenes resultan 600 muestras totales.
3.   'train_whole_model': para afinar la precision del modelo completo se debe fijar en 'true', este entrena el modelo completo y no solo la priemra capa, esto puede mejorar sustancialemnte la precision, pero puede tardar mas tiempo en entrenarse.
    
    model = object_detector.create(train_ds, model_spec=spec, batch_size=6, train_whole_model=True, validation_data=val_ds,epochs = 300)
    
# 4. Exportar el modelo

Se exporta el modelo entrenado en formato de TensorFlow Lite indicando la carpeta de destino. La cuantizacion de post-entrenado predeterminada es una tecnica de cuantizacion integrada

    model.export(export_dir='Insertar directorio de destino/dataset')

# 5. Modelo exportado

los factores que pueden afectar la exactitud del modelo pueden ser:

*   La cuantizacion ayuda a disminuir alrededor de 4 veces el tamaño del modelo anterior respecto de la precision
*   el modelo TFlite utiliza menos espacio pero disminuye la precision, esto debido a que otros modelo utilizan una supresion sin maximo de clases (NMS) para el post-procesado, mientras que TFLite utiliza una supresion de clases global, el cuale s mucho mas rapida, pero menos precisa. El resultado máxzimo de Keras es hasta 100 detecciones, mientras que TFLite solo admite 25.

La etapa de entrenamiento se finalizó, ahora se procede a la confeccion e implementacion del modelo entrenado en la plataforma movil.
