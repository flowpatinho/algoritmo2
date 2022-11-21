# algoritmo2
Código fuente para la creación de modelo de detección personalizado, encargado de detectar el consumo para medidores eléctricos no inteligentes con certificacion nueva (2015) a partir de una fotografía 

Se importan los paquetes requeridos para la creacion del modelo de entrenamiento

    import numpy as np
    import os
    from tflite_model_maker.config import ExportFormat
    from tflite_model_maker import model_spec
    from tflite_model_maker import object_detector
    import tensorflow as tf
    #assert tf.__version__.startswith('2'
    
