from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding, Input
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import preprocessing
import evaluation
import joblib
from tensorflow.keras.layers import Bidirectional

file_path = 'Symptom2Disease.csv'
data = pd.read_csv(file_path)

# Step 1: Shuffle the Data
data = shuffle(data, random_state=42)  # Shuffle the data with a fixed random seed for reproducibility

# Step 2: Text Cleaning

data['cleaned_text'] = data['text'].apply(preprocessing.clean_text)

# Step 3: Vectorization
label_encoder = LabelEncoder()
data['label_encoded'] = label_encoder.fit_transform(data['label'])  # Encode target labels
joblib.dump(data['label_encoded'], 'label_encoder.pkl')
vectorizer = TfidfVectorizer(max_features=5000)  # Limit to 5000 features
X = vectorizer.fit_transform(data['cleaned_text']).toarray()  # Vectorize cleaned text
y = data['label_encoded']  # Target variable

# Split into training and validation sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape X for compatibility with RNN
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

# Step 4: Build the RNN Model
model = Sequential([
    Input(shape=(1, X_train.shape[2])),  # Input layer for vectorized data
    LSTM(128, return_sequences=False),  # LSTM layer
    Dropout(0.3),  # Dropout to prevent overfitting
    Dense(64, activation='relu'),  # Fully connected layer
    Dropout(0.3),  # Dropout
    Dense(len(label_encoder.classes_), activation='softmax')  # Output layer
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Step 5: Train the Model
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_test, y_test)
)

#dnn
model_dnn = Sequential([
    Input(shape=(X_train.shape[2],)),  # Flatten input for dense layers
    Dense(128, activation='relu'),  # Fully connected layer
    Dropout(0.3),
    Dense(64, activation='relu'),  # Additional dense layer
    Dropout(0.3),
    Dense(len(label_encoder.classes_), activation='softmax')  # Output layer
])

model_dnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history_dnn = model_dnn.fit(
    X_train.reshape(X_train.shape[0], X_train.shape[2]), y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_test.reshape(X_test.shape[0], X_test.shape[2]), y_test)
)

#bidirectional-lstm

model_bilstm = Sequential([
    Input(shape=(1, X_train.shape[2])),
    Bidirectional(LSTM(128, return_sequences=False)),  # Bidirectional LSTM layer
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(len(label_encoder.classes_), activation='softmax')
])

model_bilstm.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history_bilstm = model_bilstm.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_test, y_test)
)

model_bilstm.save('disease_model.h5')

# Adjust the training sections to compute confusion matrices
# For RNN:
y_pred_rnn = model.predict(X_test)
evaluation.evaluation_graphs(history, model, X_test, y_test, label_encoder)

# For DNN:
X_test_dnn = X_test.reshape(X_test.shape[0], X_test.shape[2])
y_pred_dnn = model_dnn.predict(X_test_dnn)
evaluation.evaluation_graphs(history_dnn, model_dnn, X_test_dnn, y_test, label_encoder)

# For BiLSTM:
y_pred_bilstm = model_bilstm.predict(X_test)
evaluation.evaluation_graphs(history_bilstm, model_bilstm, X_test, y_test, label_encoder)

evaluation.compare_accuracies(history, history_dnn, history_bilstm)

# Convert predicted probabilities to class labels for each model
y_pred_rnn_classes = np.argmax(y_pred_rnn, axis=1)
y_pred_dnn_classes = np.argmax(y_pred_dnn, axis=1)
y_pred_bilstm_classes = np.argmax(y_pred_bilstm, axis=1)

classes = label_encoder.classes_
# Evaluate RNN
print("Evaluation for RNN Model:")
evaluation.print_evaluation_scores(y_test, y_pred_rnn_classes,classes)

# Evaluate DNN
print("Evaluation for DNN Model:")
evaluation.print_evaluation_scores(y_test, y_pred_dnn_classes,classes)

# Evaluate BiLSTM
print("Evaluation for BiLSTM Model:")
evaluation.print_evaluation_scores(y_test, y_pred_bilstm_classes,classes)