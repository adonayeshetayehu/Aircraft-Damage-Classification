from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam
from .config import input_shape

def build_model():
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    x = Flatten()(base_model.output)
    base_model = Model(inputs=base_model.input, outputs=x)
    for layer in base_model.layers:
        layer.trainable = False

    model = Sequential([
        base_model,
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])
    return model