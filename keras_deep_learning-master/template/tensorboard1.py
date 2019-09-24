import os,sys
import keras
from keras import layers
from keras.datasets import imdb
from keras.preprocessing import sequence

os.chdir('D:\\workspace\\machine-learning-study\\keras_deep_learning-master')

## GPU config
import tensorflow as tf
from keras.backend import tensorflow_backend as K
gpu_fraction = 0.3
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = gpu_fraction
K.set_session(tf.compat.v1.Session(config=config))

max_features = 2000
max_len = 500

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
x_train = sequence.pad_sequences(x_train, maxlen=max_len)
x_test = sequence.pad_sequences(x_test, maxlen=max_len)

model = keras.models.Sequential()
model.add(layers.Embedding(max_features, 128,
                        input_length=max_len,
                        name='embed'))
model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.MaxPooling1D(5))
model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.GlobalMaxPool1D())
model.add(layers.Dense(1))
model.summary()
model.compile(optimizer='rmsprop',
            loss='binary_crossentropy',
            metrics=['acc'])

# $ mkdir my_log_dir
callbacks = [
    keras.callbacks.TensorBoard(
        log_dir='my_log_dir',
        histogram_freq=1,
        #embeddings_data=x_test, ## oom occur
        #embeddings_freq=1
    )
]
history = model.fit(x_train, y_train,
                    epochs=20,
                    batch_size=64,
                    validation_split=0.2,
                    callbacks=callbacks)




