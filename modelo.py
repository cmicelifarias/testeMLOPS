import tensorflow as tf
import numpy as np
from tensorflow import keras

#rede neural com 1 camada e 1 neuronio
minha_camada = keras.layers.Dense(units=1,input_shape=[1])
modelo = tf.keras.Sequential([minha_camada])
modelo.compile(optimizer="sgd", loss="mean_squared_error")

# y = 2x+1
# meu pseudo BD - nao da para ficar trocando manualmente o tempo todo
xs = np.array([0,  1, 2, 3, 4], dtype=float)
ys = np.array([1, 3, 5, 7, 9], dtype=float)

modelo.fit(xs,ys, epochs=1000)

print(modelo.predict([10.0])) # deveria ser 21

print(minha_camada.get_weights())

