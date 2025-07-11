import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization
from tensorflow.keras.layers import MultiHeadAttention, Add
from tensorflow.keras.callbacks import EarlyStopping
from kerastuner.tuners import RandomSearch


class PaymentPredictor:
    def __init__(self, input_shape, max_trials=10, executions_per_trial=1):
        self.input_shape = input_shape
        self.best_model = None
        self.tuner = RandomSearch(
            self._build_model,
            objective='val_loss',
            max_trials=max_trials,
            executions_per_trial=executions_per_trial,
            directory='tuning_logs',
            project_name='payment_delay_transformer'
        )

    def _build_model(self, hp):
        input_layer = Input(shape=self.input_shape)

        # Transformer Encoder Block
        embed_dim = self.input_shape[-1]
        num_heads = hp.Int("num_heads", min_value=2, max_value=8, step=2)
        ff_dim = hp.Int("ff_dim", min_value=64, max_value=256, step=64)
        dropout_rate = hp.Float('dropout', min_value=0.1, max_value=0.5, step=0.1)

        # Multi-Head Attention + Residual + LayerNorm
        attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(input_layer, input_layer)
        attention_output = Dropout(dropout_rate)(attention_output)
        out1 = LayerNormalization(epsilon=1e-6)(Add()([input_layer, attention_output]))
        ff_output = Dense(ff_dim, activation='relu')(out1)
        ff_output = Dense(embed_dim)(ff_output)
        ff_output = Dropout(dropout_rate)(ff_output)
        out2 = LayerNormalization(epsilon=1e-6)(Add()([out1, ff_output]))
        pooled = tf.reduce_mean(out2, axis=1)
        output = Dense(1, activation='linear')(pooled)

        model = Model(inputs=input_layer, outputs=output)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
            ),
            loss='mse',
            metrics=['mae']
        )

        return model

    def train(self, x_train, y_train, x_val, y_val, epochs=50, batch_size=32):
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        self.tuner.search(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop],
            verbose=0
        )

        self.best_model = self.tuner.get_best_models(num_models=1)[0]

    def predict_payment_delay(self, invoice_data):
        if self.best_model is None:
            raise ValueError("Model has not been trained yet.")
        prediction = self.best_model.predict(np.array([invoice_data]))
        return prediction[0][0]
