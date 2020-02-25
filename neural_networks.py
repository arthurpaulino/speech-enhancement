import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import *
from keras.models import Model

from sklearn.preprocessing import MinMaxScaler

from parameters import *
from utils import *


early_stop = EarlyStopping(monitor="val_loss", patience=PATIENCE)


def build_tensor(layers, tensor_input):
    tensor = layers[0](tensor_input)
    for layer in layers[1:]:
        tensor = layer(tensor)
    return tensor


def build_nn(input_dim, output_dim):
    model_input = Input((input_dim,))

    dense_layers = [
        Dense(input_dim, activation="relu"),
        Dropout(0.25)
    ]

    dense_tensor = build_tensor(dense_layers, model_input)

    conv_layers_1 = [
        Reshape((LOOK_BACK + 1 + LOOK_AFTER, output_dim, 1)),
        Conv2D(32, (3, 3), activation="relu", padding="same"),
        MaxPooling2D((2, 2))
    ]

    conv_layers_2 = [
        Conv2D(64, (3, 3), activation="relu", padding="same"),
        MaxPooling2D((2, 2))
    ]

    conv_tensor_1 = build_tensor(conv_layers_1, model_input)
    conv_tensor_2 = build_tensor(conv_layers_2, conv_tensor_1)

    conv_tensor_1_flat = Flatten()(conv_tensor_1)
    conv_tensor_2_flat = Flatten()(conv_tensor_2)

    concat = Concatenate()([dense_tensor,
                            conv_tensor_1_flat,
                            conv_tensor_2_flat])

    output = Dense(output_dim, activation="sigmoid")(concat)

    nn = Model(model_input, output)

    nn.compile(optimizer="adam", loss="mae")

    return nn


def train_and_predict(X_train_t, Y_train_t,
                      X_train_v, Y_train_v,
                      X_valid, Y_valid,
                      seed, model_id):
    input_len, input_dim = X_train_t.shape
    output_dim = Y_train_t.shape[1]

    X_scaler, Y_scaler = MinMaxScaler(), MinMaxScaler()

    X_train_t_scaled = X_scaler.fit_transform(X_train_t)
    Y_train_t_scaled = Y_scaler.fit_transform(Y_train_t)

    X_train_v_scaled = X_scaler.transform(X_train_v)
    Y_train_v_scaled = Y_scaler.transform(Y_train_v)

    X_valid_scaled = X_scaler.transform(X_valid)
    Y_valid_scaled = Y_scaler.transform(Y_valid)

    nn = build_nn(input_dim, output_dim)

    callbacks = [
        early_stop,
        ModelCheckpoint(
            filepath=EXPERIMENT_FOLDER + "models/" + model_id + ".h5",
            monitor="val_loss",
            save_best_only=True
        )
    ]

    nn.fit(X_train_t_scaled,
           Y_train_t_scaled,
           batch_size=round(BATCH_SIZE_RATIO * input_len),
           epochs=1_000_000, # going to use early stop instead
           verbose=VERBOSE,
           callbacks=callbacks,
           validation_data=(X_train_v_scaled, Y_train_v_scaled))

    Y_model_scaled = nn.predict(X_valid_scaled)
    return Y_scaler.inverse_transform(Y_model_scaled)
