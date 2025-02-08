import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt
import numpy as np

# Configurações iniciais
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 3

# Caminho para o diretório das categorias
base_dir = os.getcwd()
categories_dir = os.path.join(base_dir, 'Projeto de Transfer Learning em Python', 'data', 'Fashion_data', 'categories')

# Verificar se o diretório existe
if not os.path.exists(categories_dir):
    raise FileNotFoundError(f"Diretório {categories_dir} não encontrado.")

# Geradores de dados
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = datagen.flow_from_directory(
    categories_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

valid_generator = datagen.flow_from_directory(
    categories_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

print("Imagens de treino encontradas:", train_generator.samples)
print("Imagens de validação encontradas:", valid_generator.samples)

# Construção do modelo
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(train_generator.num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Treinamento do modelo
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_data=valid_generator,
    validation_steps=valid_generator.samples // BATCH_SIZE,
    epochs=EPOCHS
)

# Visualização dos resultados
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()

# Salvar modelo

model.save('Projeto de Transfer Learning em Python/modelos/fashion_model_classification.keras')

# Teste do modelo com uma imagem
def load_and_prepare_image(img_path):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=IMAGE_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Cria um batch de tamanho 1
    return img_array / 255.0

# Escolha uma imagem de teste (substitua pelo caminho da sua imagem)
test_image_path = os.path.join(base_dir, 'Projeto de Transfer Learning em Python', 'data', 'testes','teste_p.jpg')  # Exemplo
if not os.path.exists(test_image_path):
    raise FileNotFoundError(f"Imagem de teste {test_image_path} não encontrada.")

test_image = load_and_prepare_image(test_image_path)
predictions = model.predict(test_image)
predicted_class = np.argmax(predictions, axis=1)

print(f"Classe prevista: {predicted_class}")
print(f"Nomes das classes: {train_generator.class_indices}")