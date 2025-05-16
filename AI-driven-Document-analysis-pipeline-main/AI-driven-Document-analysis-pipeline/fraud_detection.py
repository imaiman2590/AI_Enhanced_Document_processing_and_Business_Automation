import shap
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Attention, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import StandardScaler
import numpy as np
import os

class FraudDetection:
    def __init__(self, input_shape, model_dir='fraud_model', max_trials=10, executions_per_trial=1):
        self.input_shape = input_shape
        self.model_dir = model_dir
        self.best_model = None
        self.scaler = StandardScaler()

        self.tuner = RandomSearch(
            self._build_model,
            objective='val_accuracy',
            max_trials=max_trials,
            executions_per_trial=executions_per_trial,
            directory='fraud_tuning_logs',
            project_name='fraud_detection_lstm'
        )

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

    def _build_model(self, hp):
        model = Sequential()
        model.add(Bidirectional(LSTM(
            units=hp.Int('units', min_value=32, max_value=128, step=32),
            return_sequences=True,
            input_shape=self.input_shape
        )))
        model.add(Attention())
        model.add(Flatten())
        model.add(Dropout(hp.Float('dropout', min_value=0.1, max_value=0.5, step=0.1)))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
            ),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model

    def _normalize(self, x_train, x_val):
        b, t, f = x_train.shape
        x_train_flat = x_train.reshape(-1, f)
        x_val_flat = x_val.reshape(-1, f)

        x_train_scaled = self.scaler.fit_transform(x_train_flat).reshape(b, t, f)
        x_val_scaled = self.scaler.transform(x_val_flat).reshape(x_val.shape[0], t, f)

        return x_train_scaled, x_val_scaled

    def train(self, x_train, y_train, x_val, y_val, epochs=50, batch_size=32):
        x_train, x_val = self._normalize(x_train, x_val)

        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        checkpoint_path = os.path.join(self.model_dir, 'best_model.h5')
        checkpoint = ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_loss')

        self.tuner.search(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, checkpoint],
            verbose=0
        )
        self.best_model = self.tuner.get_best_models(num_models=1)[0]
        self.best_model.save(checkpoint_path)

    def explain_prediction(self, x_sample):
        explainer = shap.KernelExplainer(self.best_model.predict, x_sample)
        shap_values = explainer.shap_values(x_sample)
        shap.summary_plot(shap_values, x_sample)

    def predict(self, features):
        if self.best_model is None:
            raise ValueError("Model has not been trained or loaded.")
        features_flat = features.reshape(-1, features.shape[-1])
        features_scaled = self.scaler.transform(features_flat).reshape(features.shape)
        prediction = self.best_model.predict(features_scaled)
        return int(prediction[0][0] > 0.5)

    def evaluate(self, x_test, y_test, metric='f1'):
        if self.best_model is None:
            raise ValueError("Model has not been trained or loaded.")

        x_flat = x_test.reshape(-1, x_test.shape[-1])
        x_test_scaled = self.scaler.transform(x_flat).reshape(x_test.shape)

        y_pred_probs = self.best_model.predict(x_test_scaled).flatten()
        y_pred = (y_pred_probs > 0.5).astype(int)

        if metric == 'f1':
            return f1_score(y_test, y_pred)
        elif metric == 'auc':
            return roc_auc_score(y_test, y_pred_probs)
        elif metric == 'precision':
            return precision_score(y_test, y_pred)
        elif metric == 'recall':
            return recall_score(y_test, y_pred)
        elif metric == 'confusion_matrix':
            return confusion_matrix(y_test, y_pred)
        else:
            return np.mean(y_pred == y_test)
