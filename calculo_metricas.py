from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Verifica a versão do TensorFlow
tf.__version__

# Diretório para salvar os logs do TensorBoard
logdir = 'log'

# Carrega o dataset MNIST
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

# Redimensiona as imagens para incluir o canal de cor
train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

# Normaliza as imagens para o intervalo [0, 1]
train_images, test_images = train_images / 255.0, test_images / 255.0

# Define as classes do dataset MNIST
classes = [0,1,2,3,4,5,6,7,8,9]

# Cria o modelo de rede neural convolucional
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# Adiciona camadas densas ao modelo
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# Configura o callback do TensorBoard
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

# Compila o modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Treina o modelo
model.fit(x=train_images,
          y=train_labels,
          epochs=5,
          validation_data=(test_images, test_labels))

# Define os rótulos verdadeiros
y_true = test_labels

# Faz previsões no conjunto de teste
y_pred = np.argmax(model.predict(test_images), axis=-1)

# Define novamente as classes (redundante, pois já foi definido anteriormente)
classes = [0,1,2,3,4,5,6,7,8,9]

# Calcula a matriz de confusão
con_mat = tf.math.confusion_matrix(labels=y_true, predictions=y_pred).numpy()
con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)

# Cria um DataFrame para a matriz de confusão normalizada
con_mat_df = pd.DataFrame(con_mat_norm,
                          index=classes,
                          columns=classes)

# Plota a matriz de confusão normalizada
figure = plt.figure(figsize=(8, 8))
sns.heatmap(con_mat_df, annot=True, cmap=plt.cm.Blues)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# Calcula as métricas
accuracy = accuracy_score(y_true, y_pred)  # Acurácia
precision = precision_score(y_true, y_pred, average='weighted')  # Precisão
recall = recall_score(y_true, y_pred, average='weighted')  # Sensibilidade (Recall)
f1 = f1_score(y_true, y_pred, average='weighted')  # F-score

# Sensibilidade (Recall) e Especificidade por classe
sensitivity = recall_score(y_true, y_pred, average=None)  # Sensibilidade por classe
specificity = []
for i in range(len(classes)):
    tn = con_mat.sum() - (con_mat[i, :].sum() + con_mat[:, i].sum() - con_mat[i, i])  # Verdadeiros negativos
    fp = con_mat[:, i].sum() - con_mat[i, i]  # Falsos positivos
    specificity.append(tn / (tn + fp))  # Especificidade

# Imprime as métricas
print(f'Acurácia: {accuracy}')
print(f'Precisão: {precision}')
print(f'Sensibilidade: {sensitivity}')
print(f'Especificidade: {specificity}')
print(f'F-score: {f1}')