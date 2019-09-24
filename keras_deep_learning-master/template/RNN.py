from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.LSTM(32, return_sequences=True,
                    input_shape=(num_timesteps, num_features)))
model.add(layers.LSTM(32, return_sequences=True))
model.add(layers.LSTM(32))
model.add(layers.Dense(num_classes, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy')