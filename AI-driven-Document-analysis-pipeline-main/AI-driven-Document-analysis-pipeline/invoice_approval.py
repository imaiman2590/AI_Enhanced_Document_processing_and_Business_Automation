import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
import os

class InvoiceApproval:
    def __init__(self, model_type='rf'):
        """
        Initialize the model(s) to be used for invoice approval.
        model_type options: 'dt' (Decision Tree), 'rf' (Random Forest), 'dl' (Deep Learning)
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()

        # Load or train models based on choice
        if model_type == 'dt':
            self.model = self._train_decision_tree()
        elif model_type == 'rf':
            self.model = self._train_random_forest()
        elif model_type == 'dl':
            self.model = self._train_deep_learning_model()

    def _train_decision_tree(self):
        """
        Train a Decision Tree model.
        """
        model = DecisionTreeClassifier(max_depth=5)
        return model

    def _train_random_forest(self):
        """
        Train a Random Forest model.
        """
        model = RandomForestClassifier(n_estimators=100, max_depth=5)
        return model

    def _build_deep_learning_model(self):
        """
        Build and compile the deep learning model.
        """
        model = Sequential()
        model.add(Dense(64, activation='relu', input_dim=5))
        model.add(Dropout(0.3))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def _train_deep_learning_model(self):
        """
        Train the deep learning model (Neural Network).
        """
        # Example random data for training
        X_train = np.random.rand(1000, 5)  # 1000 samples, 5 features
        y_train = np.random.randint(0, 2, size=(1000,))

        model = self._build_deep_learning_model()

        early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, callbacks=[early_stop])

        return model

    def train(self, x_train, y_train):
        """
        Train the model on the provided data.
        """
        if self.model_type == 'dl':
            self._train_deep_learning_model()
        else:
            self.model.fit(x_train, y_train)

    def predict(self, invoice_features):
        """
        Predict whether an invoice is approved or rejected.
        """
        try:
            features_scaled = self.scaler.fit_transform(invoice_features)  # Fit and scale the features
            prediction = self.model.predict(features_scaled)[0]
            return "Approved" if prediction == 1 else "Rejected"
        except Exception as e:
            return f"Error in invoice approval: {str(e)}"

    def evaluate(self, x_test, y_test):
        """
        Evaluate the performance of the model on test data.
        """
        x_test_scaled = self.scaler.transform(x_test)  # Scale the test data

        if self.model_type == 'dl':
            y_pred = self.model.predict(x_test_scaled)
            y_pred = (y_pred > 0.5).astype(int)
        else:
            y_pred = self.model.predict(x_test_scaled)

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        return {"accuracy": accuracy, "f1_score": f1, "roc_auc": auc, "confusion_matrix": cm}

