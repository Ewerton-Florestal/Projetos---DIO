from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, SeparableConv2D

# Modelo aprimorado
model = Sequential([
    # Primeira camada convolucional
    Input(shape=(224, 224, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    # Segunda camada convolucional
    SeparableConv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    # Terceira camada convolucional
    SeparableConv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    # Quarta camada convolucional
    SeparableConv2D(256, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    # Flatten e fully connected
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),  # Regularização para evitar overfitting
    Dense(1, activation='sigmoid')  # Saída binária
])

# Compilar o modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

train_datagen= ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    'data/Imagens',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

# Treinar o modelo
model.fit(train_generator, epochs=15)

# Salvar o modelo
model.save('modelo_classificacao.keras')

# Carregar o modelo Haar Cascade para detecção de faces
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Carregar o modelo de classificação treinado
model = load_model('modelo_classificacao.keras')  # Substitua pelo caminho do seu modelo

# Função para classificar a face
def classify_face(face_image):
    # Redimensionar a imagem para o tamanho esperado pelo modelo (224x224)
    face_image = cv2.resize(face_image, (224, 224))
    face_image = np.expand_dims(face_image, axis=0)  # Adicionar dimensão do batch
    face_image = face_image / 255.0  # Normalizar a imagem

    # Fazer a predição
    prediction = model.predict(face_image)
    if prediction[0][0] > 0.5:
        return "Amy"
    else:
        return "Raj"

# Carregar a imagem
image = cv2.imread('data/testea.jpg')  # Substitua pelo caminho da sua imagem
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Converter para escala de cinza

# Detectar faces
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Processar cada face detectada
for (x, y, w, h) in faces:
    # Recortar a região da face
    face = image[y:y+h, x:x+w]

    # Classificar a face
    label = classify_face(face)

    # Desenhar um retângulo ao redor da face
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Escrever o nome da pessoa acima do retângulo
    cv2.putText(image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

cv2.imshow('Faces Detectadas e Classificadas', image)
cv2.waitKey(0)
cv2.destroyAllWindows()