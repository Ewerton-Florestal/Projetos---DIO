import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from pycocotools.coco import COCO
import requests
import zipfile
import os
import cv2
import torch

# Define o diretório para salvar as imagens e anotações dentro da pasta 'data'
base_dir = 'data'
save_dir = os.path.join(base_dir, 'coco_images')
annotations_dir = os.path.join(base_dir, 'coco_annotations')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
if not os.path.exists(annotations_dir):
    os.makedirs(annotations_dir)

# URLs dos datasets COCO
dataDir = 'http://images.cocodataset.org/zips/train2017.zip'
annFile = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'

# Função para baixar e extrair arquivos ZIP
def download_and_extract(url, extract_to):
    local_zip = os.path.join(extract_to, url.split('/')[-1])
    with requests.get(url, stream=True) as r:
        with open(local_zip, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    with zipfile.ZipFile(local_zip, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    os.remove(local_zip)

# Baixa e extrai as anotações
download_and_extract(annFile, annotations_dir)

# Carrega as anotações
annFile = os.path.join(annotations_dir, 'annotations', 'instances_train2017.json')
coco = COCO(annFile)

# Define as categorias de interesse (por exemplo, 'person' e 'dog')
catIds = coco.getCatIds(catNms=['person', 'dog'])
imgIds = coco.getImgIds(catIds=catIds)
images = coco.loadImgs(imgIds)

# Baixa as imagens (limitado a 100 imagens)
max_images = 100
for i, im in enumerate(images):
    if i >= max_images:
        break
    img_data = requests.get(im['coco_url']).content
    with open(os.path.join(save_dir, im['file_name']), 'wb') as handler:
        handler.write(img_data)

# Carrega o modelo YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)

# Lista todas as imagens no diretório
images = [os.path.join(save_dir, img) for img in os.listdir(save_dir) if img.endswith('.jpg')]

for img_path in images:
    # Carrega a imagem
    img = cv2.imread(img_path)

    # Realiza a detecção
    results = model(img)

    # Exibe os resultados
    results.show()

    # Espera pela entrada do usuário antes de passar para a próxima imagem
    input("Pressione Enter para ver a próxima imagem...")