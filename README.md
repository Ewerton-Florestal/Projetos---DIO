# 🖼️📊 Repositório de Processamento de Imagens e Assistente Virtual 🤖🎤

Bem-vindo ao repositório de **Processamento de Imagens e Assistente Virtual**! Aqui você encontrará uma coleção de scripts Python que cobrem desde processamento de imagens até reconhecimento facial e um assistente virtual básico. Cada script foi desenvolvido para demonstrar técnicas de machine learning, visão computacional e processamento de áudio. Vamos explorar!

---

## 🚀 **Scripts Disponíveis**

Aqui estão os scripts disponíveis, cada um com sua funcionalidade única:

### 1. **Assistente Virtual** 🤖🎤
**Arquivo:** `Assistente_virtual.py`

Um assistente virtual simples que converte texto em fala e fala em texto. Ele pode executar comandos básicos, como abrir a Wikipedia ou o YouTube.

- **Funcionalidades:**
  - 🎤 **Texto para fala** (Text-to-Speech) usando `pyttsx3`.
  - 🎙️ **Fala para texto** (Speech-to-Text) usando `speech_recognition`.
  - 🚀 **Execução de comandos de voz** (abrir sites como Wikipedia e YouTube).

- **Como usar:**
  1. Execute o script.
  2. Fale um comando como "abrir Wikipedia" ou "abrir YouTube".
  3. O assistente responderá e executará o comando.

---

### 2. **Cálculo de Métricas de Modelo** 📊🧠
**Arquivo:** `calculo_metricas.py`

Este script treina uma rede neural convolucional (CNN) no dataset MNIST e calcula métricas de avaliação como acurácia, precisão, sensibilidade, especificidade e F-score.

- **Funcionalidades:**
  - 🧠 **Treinamento de uma CNN** no dataset MNIST.
  - 📈 **Cálculo de métricas** (acurácia, precisão, sensibilidade, especificidade, F-score).
  - 📊 **Visualização da matriz de confusão**.

- **Como usar:**
  1. Execute o script.
  2. O modelo será treinado e as métricas serão exibidas no console.
  3. A matriz de confusão será exibida em um gráfico.

---

### 3. **Detecção de Objetos com YOLOv5** 🕵️‍♂️📦
**Arquivo:** `detecção_yolo.py`

Este script utiliza o modelo YOLOv5 para detectar objetos em imagens do dataset COCO. Ele baixa imagens e anotações do COCO e aplica o modelo YOLOv5 para detecção.

- **Funcionalidades:**
  - 📥 **Download de imagens e anotações do COCO**.
  - 🕵️‍♂️ **Detecção de objetos** usando YOLOv5.
  - 🖼️ **Visualização dos resultados**.

- **Como usar:**
  1. Execute o script.
  2. As imagens serão baixadas e processadas pelo YOLOv5.
  3. Os resultados serão exibidos e você poderá avançar para a próxima imagem pressionando Enter.

---

### 4. **Sistema de Recomendação de Imagens** 🖼️🔍
**Arquivo:** `Sistema_recomendação_imagens.py`

Este script implementa um sistema de recomendação de imagens usando transfer learning. Ele treina um modelo de classificação de imagens em um dataset de moda e prevê a classe de uma imagem de teste.

- **Funcionalidades:**
  - 🧠 **Treinamento de um modelo de classificação de imagens**.
  - 🔍 **Previsão da classe de uma imagem de teste**.
  - 📈 **Visualização da acurácia durante o treinamento**.

- **Como usar:**
  1. Coloque as imagens de treino no diretório correto.
  2. Execute o script.
  3. O modelo será treinado e a acurácia será exibida em um gráfico.
  4. Uma imagem de teste será classificada e a classe prevista será exibida.

---

### 5. **Sistema de Reconhecimento Facial** 👤🔍
**Arquivo:** `Sistema_reconhecimento_face.py`

Este script implementa um sistema de reconhecimento facial usando uma CNN. Ele detecta faces em uma imagem e as classifica como pertencentes a uma de duas pessoas.

- **Funcionalidades:**
  - 👤 **Detecção de faces** usando Haar Cascade.
  - 🧠 **Classificação de faces** usando uma CNN treinada.
  - 🖼️ **Visualização das faces detectadas e classificadas**.

- **Como usar:**
  1. Treine o modelo com imagens de treino.
  2. Execute o script com uma imagem de teste.
  3. As faces detectadas serão classificadas e exibidas com um retângulo e o nome da pessoa.

---

### 6. **Suavização e Binarização de Imagens** 🖼️🔳
**Arquivo:** `Suavização_imagem.py`

Este script aplica suavização (blur) e binarização em uma imagem.

- **Funcionalidades:**
  - 🌫️ **Aplicação de suavização Gaussiana**.
  - 🔳 **Binarização da imagem**.
  - 🖼️ **Visualização dos resultados**.

- **Como usar:**
  1. Coloque uma imagem no diretório correto.
  2. Execute o script.
  3. A imagem original, suavizada e binarizada será exibida.

---

## 🛠️ **Requisitos**

Para executar os scripts, você precisará das seguintes bibliotecas Python:

- Python 3.x
  
---

## 🚀 **Como Executar**

Clone o repositório:
   ```bash
   git clone https://github.com/seu-usuario/seu-repositorio.git
   ```
---

## 🤝 **Contribuição**

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues ou pull requests. Vamos construir algo incrível juntos! 🚀

---

## 📜 **Licença**

Este projeto está licenciado sob a licença MIT. Veja o arquivo LICENSE para mais detalhes.
