import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
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
            project_name='payment_delay_lstm'
        )

    def _build_model(self, hp):
        model = Sequential()
        model.add(LSTM(
            units=hp.Int('units', min_value=32, max_value=128, step=32),
            return_sequences=False,
            input_shape=self.input_shape
        ))
        model.add(Dropout(hp.Float('dropout', min_value=0.1, max_value=0.5, step=0.1)))
        model.add(Dense(1, activation='linear'))

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
            verbose=0  # Set to 1 if you want to see tuning progress
        )

        self.best_model = self.tuner.get_best_models(num_models=1)[0]

    def predict_payment_delay(self, invoice_data):
        if self.best_model is None:
            raise ValueError("Model has not been trained yet.")
        prediction = self.best_model.predict(np.array([invoice_data]))
        return prediction[0][0]  # Return scalar
