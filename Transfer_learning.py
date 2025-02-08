
%matplotlib inline  # Necessário para exibir gráficos no Jupyter Notebook

import os  # Biblioteca para manipulação de diretórios e arquivos

#if using Theano with GPU
#os.environ["KERAS_BACKEND"] = "tensorflow"

import random  # Biblioteca para operações aleatórias
import numpy as np  # Biblioteca para operações com arrays
import keras  # Biblioteca para construção de redes neurais

import matplotlib.pyplot as plt  # Biblioteca para plotagem de gráficos
from matplotlib.pyplot import imshow  # Função para exibir imagens

from keras.preprocessing import image  # Função para pré-processamento de imagens
from keras.applications.imagenet_utils import preprocess_input  # Função para pré-processamento de imagens
from keras.models import Sequential  # Classe para criar um modelo sequencial
from keras.layers import Dense, Dropout, Flatten, Activation  # Camadas para a rede neural
from keras.layers import Conv2D, MaxPooling2D  # Camadas de convolução e pooling
from keras.models import Model  # Classe para criar um modelo

# Diretório raiz contendo as categorias de imagens
root = 'data\gatos_cachorros'

# Proporção de dados para treino e validação
train_split, val_split = 0.7, 0.15

# Lista de categorias (subdiretórios) no diretório raiz, excluindo as categorias especificadas
categories = [x[0] for x in os.walk(root) if x[0]][1:]
categories = [c for c in categories if c not in [os.path.join(root, e) for e in exclude]]

print(categories)  # Exibe as categorias selecionadas

# Função auxiliar para carregar uma imagem e retornar a imagem e o vetor de entrada
def get_image(path):
    img = image.load_img(path, target_size=(224, 224))  # Carrega a imagem e redimensiona
    x = image.img_to_array(img)  # Converte a imagem para um array
    x = np.expand_dims(x, axis=0)  # Expande a dimensão do array
    x = preprocess_input(x)  # Pré-processa a imagem
    return img, x

data = []  # Lista para armazenar os dados
for c, category in enumerate(categories):
    images = [os.path.join(dp, f) for dp, dn, filenames
              in os.walk(category) for f in filenames
              if os.path.splitext(f)[1].lower() in ['.jpg','.png','.jpeg']]  # Lista de caminhos das imagens
    for img_path in images:
        img, x = get_image(img_path)  # Carrega a imagem
        data.append({'x':np.array(x[0]), 'y':c})  # Adiciona a imagem e a categoria aos dados

# Conta o número de classes
num_classes = len(categories)

random.shuffle(data)  # Embaralha os dados

# Índices para divisão dos dados em treino, validação e teste
idx_val = int(train_split * len(data))
idx_test = int((train_split + val_split) * len(data))
train = data[:idx_val]
val = data[idx_val:idx_test]
test = data[idx_test:]

# Separa os dados e rótulos para treino, validação e teste
x_train, y_train = np.array([t["x"] for t in train]), [t["y"] for t in train]
x_val, y_val = np.array([t["x"] for t in val]), [t["y"] for t in val]
x_test, y_test = np.array([t["x"] for t in test]), [t["y"] for t in test]
print(y_test)

# Normaliza os dados
x_train = x_train.astype('float32') / 255.
x_val = x_val.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

# Converte os rótulos para vetores one-hot
y_train = keras.utils.to_categorical(y_train, num_classes)
y_val = keras.utils.to_categorical(y_val, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print(y_test.shape)

# Resumo dos dados carregados
print("finished loading %d images from %d categories"%(len(data), num_classes))
print("train / validation / test split: %d, %d, %d"%(len(x_train), len(x_val), len(x_test)))
print("training data shape: ", x_train.shape)
print("training labels shape: ", y_train.shape)

# Exibe algumas imagens de exemplo
images = [os.path.join(dp, f) for dp, dn, filenames in os.walk(root) for f in filenames if os.path.splitext(f)[1].lower() in ['.jpg','.png','.jpeg']]
idx = [int(len(images) * random.random()) for i in range(8)]
imgs = [image.load_img(images[i], target_size=(224, 224)) for i in idx]
concat_image = np.concatenate([np.asarray(img) for img in imgs], axis=1)
plt.figure(figsize=(16,4))
plt.imshow(concat_image)
plt.show()

# Constrói a rede neural
model = Sequential()
print("Input dimensions: ",x_train.shape[1:])

model.add(Conv2D(32, (3, 3), input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.summary()

# Compila o modelo para usar a função de perda de entropia cruzada categórica e o otimizador Adam
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Treina o modelo
history = model.fit(x_train, y_train,
                    batch_size=128,
                    epochs=25,
                    validation_data=(x_val, y_val))

# Plota a perda e a precisão da validação
fig = plt.figure(figsize=(16,4))
ax = fig.add_subplot(121)
ax.plot(history.history["val_loss"])
ax.set_title("validation loss")
ax.set_xlabel("epochs")

ax2 = fig.add_subplot(122)
ax2.plot(history.history["val_accuracy"])
ax2.set_title("validation accuracy")
ax2.set_xlabel("epochs")
ax2.set_ylim(0, 1)

plt.show()

# Avalia o modelo nos dados de teste
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', loss)
print('Test accuracy:', accuracy)

# Carrega o modelo VGG16 pré-treinado com os pesos do ImageNet
vgg = keras.applications.VGG16(weights='imagenet', include_top=True)
vgg.summary()

# Faz uma referência à camada de entrada do VGG
inp = vgg.input

# Cria uma nova camada softmax com num_classes neurônios
new_classification_layer = Dense(num_classes, activation='softmax')

# Conecta nossa nova camada à penúltima camada do VGG e faz uma referência a ela
out = new_classification_layer(vgg.layers[-2].output)

# Cria uma nova rede entre inp e out
model_new = Model(inp, out)

# Torna todas as camadas não treináveis, exceto a última camada
for l, layer in enumerate(model_new.layers[:-1]):
    layer.trainable = False

# Garante que a última camada seja treinável/não congelada
for l, layer in enumerate(model_new.layers[-1:]):
    layer.trainable = True

# Compila o novo modelo
model_new.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model_new.summary()

# Treina o novo modelo
history2 = model_new.fit(x_train, y_train,
                         batch_size=128,
                         epochs=10,
                         validation_data=(x_val, y_val))

# Plota a perda e a precisão da validação para o novo modelo
fig = plt.figure(figsize=(16,4))
ax = fig.add_subplot(121)
ax.plot(history.history["val_loss"])
ax.plot(history2.history["val_loss"])
ax.set_title("validation loss")
ax.set_xlabel("epochs")

ax2 = fig.add_subplot(122)
ax2.plot(history.history["val_acc"])
ax2.plot(history2.history["val_acc"])
ax2.set_title("validation accuracy")
ax2.set_xlabel("epochs")
ax2.set_ylim(0, 1)

plt.show()

# Avalia o novo modelo nos dados de teste
loss, accuracy = model_new.evaluate(x_test, y_test, verbose=0)

print('Test loss:', loss)
print('Test accuracy:', accuracy)

# Carrega uma imagem de teste e faz a predição
img, x = get_image('data\Cachorro.jpg')
probabilities = model_new.predict([x])

# Exibe as probabilidades
print(probabilities)

# Mapeia as probabilidades para as categorias
predicted_class = np.argmax(probabilities, axis=1)
print(f"Predicted class: {categories[predicted_class[0]]}")