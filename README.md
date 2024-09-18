# Classificação de Imagens de Japoneses x Coreanos sem Aumento de Imagem

Este projeto demonstra como construir um classificador de imagens para distinguir fotos de japoneses e coreanos usando TensorFlow e Keras. O modelo utiliza uma rede neural convolucional (CNN) e é treinado sem técnicas de aumento de dados.

## Descrição do Código

O código realiza as seguintes etapas:

1. **Importação de bibliotecas:** Inclui bibliotecas essenciais como TensorFlow, Keras, NumPy, Matplotlib e OS.

   ```python
   import tensorflow as tf
   from tensorflow.keras.preprocessing.image import ImageDataGenerator
   import os
   import matplotlib.pyplot as plt
   import numpy as np
   import zipfile
   import logging
   logger = tf.get_logger()
   logger.setLevel(logging.ERROR)
   ```

      * import tensorflow as tf: Importa a biblioteca TensorFlow e a renomeia para tf para facilitar o uso no código. O TensorFlow é uma plataforma de código aberto para machine learning que oferece um ecossistema abrangente de ferramentas, bibliotecas e recursos da comunidade para construir e implantar aplicativos de IA.

      * from tensorflow.keras.preprocessing.image import ImageDataGenerator: Importa a classe ImageDataGenerator do módulo tensorflow.keras.preprocessing.image. Essa classe é usada para realizar aumento de dados em imagens, criando variações das imagens originais para aumentar o conjunto de dados de treinamento e melhorar a capacidade de generalização do modelo.

      * import os: Importa a biblioteca os, que fornece funções para interagir com o sistema operacional, como manipulação de arquivos e diretórios.

      * import matplotlib.pyplot as plt: Importa o módulo pyplot da biblioteca Matplotlib e o renomeia para plt. O Matplotlib é uma biblioteca para criar visualizações estáticas, animadas e interativas em Python. Neste caso, será usado para plotar gráficos e exibir imagens.

      * import numpy as np: Importa a biblioteca NumPy e a renomeia para np. O NumPy é uma biblioteca fundamental para computação científica em Python, fornecendo suporte para arrays multidimensionais, matrizes e uma coleção de funções matemáticas de alto nível para operar nesses arrays.

      * import zipfile: Importa a biblioteca zipfile, que fornece ferramentas para trabalhar com arquivos ZIP, permitindo extrair seu conteúdo.

      * import logging: Importa a biblioteca logging, que permite registrar mensagens durante a execução do código.

      * logger = tf.get_logger(): Obtém o logger do TensorFlow.

      * logger.setLevel(logging.ERROR): Define o nível de registro para ERROR, o que significa que apenas mensagens de erro serão registradas, ignorando mensagens de aviso e informações.


3. **Carregamento de dados:** Carrega um conjunto de dados de imagens de japoneses e coreanos a partir de um arquivo ZIP.

```python
from google.colab import drive
drive.mount('/content/drive')

!cp -r /content/drive/MyDrive/koreans_and_japanese_filtered.zip /content

#_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
_URL = './koreans_and_japanese_filtered.zip'
#zip_dir = tf.keras.utils.get_file('koreans_and_japanese_filtered.zip', origin=_URL, extract=True)

with zipfile.ZipFile(_URL, 'r') as zip_ref:
    zip_ref.extractall('/content')
```

    * O conjunto de dados é dividido em conjuntos de treinamento e validação.
    * A estrutura do diretório é importante para o carregamento adequado dos dados.

<pre style="font-size: 10.0pt; font-family: Arial; line-height: 2; letter-spacing: 1.0pt;" >
<b>koreans_and_japanese_filtered</b>
|__ <b>train</b>
    |______ <b>koreans</b>: [Koreans_201_1726343003.jpg, Koreans_202_1726343003.jpg, Koreans_203_1726343003.jpg ...]
    |______ <b>japanese</b>: [Japanese_201_1726343071.jpg, Japanese_202_1726343071.jpg, Japanese_203_1726343071.jpg ...]
|__ <b>validation</b>
    |______ <b>koreans</b>: [Koreans_201_1726343003.jpg, Koreans_202_1726343003.jpg, Koreans_203_1726343003.jpg ...]
    |______ <b>japanese</b>: [Japanese_201_1726343071.jpg, Japanese_202_1726343071.jpg, Japanese_203_1726343071.jpg ...]
</pre>

4. **Pré-processamento de dados:**
    * Usa `ImageDataGenerator` para redimensionar as imagens e realizar aumento de dados no conjunto de treinamento (embora o aumento seja desativado neste código).
    * As imagens são carregadas em lotes e redimensionadas para um tamanho uniforme.

```python
base_dir = os.path.join(os.path.dirname(_URL), 'koreans_and_japanese_filtered')
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

train_koreans_dir = os.path.join(train_dir, 'koreans')  # directory with our training cat pictures
train_japanese_dir = os.path.join(train_dir, 'japanese')  # directory with our training dog pictures
validation_koreans_dir = os.path.join(validation_dir, 'koreans')  # directory with our validation cat pictures
validation_japanese_dir = os.path.join(validation_dir, 'japanese')  # directory with our validation dog pictures
num_koreans_tr = len(os.listdir(train_koreans_dir))
num_japanese_tr = len(os.listdir(train_japanese_dir))

num_koreans_val = len(os.listdir(validation_koreans_dir))
num_japanese_val = len(os.listdir(validation_japanese_dir))

total_train = num_koreans_tr + num_japanese_tr
total_val = num_koreans_val + num_japanese_val
print('total training koreans images:', num_koreans_tr)
print('total training japanese images:', num_japanese_tr)

print('total validation koreans images:', num_koreans_val)
print('total validation japanese images:', num_japanese_val)
print("--")
print("Total training images:", total_train)
print("Total validation images:", total_val)

BATCH_SIZE = 100  # Number of training examples to process before updating our models variables
IMG_SHAPE  = 150  # Our training data consists of images with width of 150 pixels and height of 150 pixels

train_image_generator = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

#train_image_generator      = ImageDataGenerator(rescale=1./255)  # Generator for our training data
validation_image_generator = ImageDataGenerator(rescale=1./255)  # Generator for our validation data

train_data_gen = train_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_SHAPE,IMG_SHAPE), #(150,150)
                                                           class_mode='binary')

val_data_gen = validation_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                              directory=validation_dir,
                                                              shuffle=False,
                                                              target_size=(IMG_SHAPE,IMG_SHAPE), #(150,150)
                                                              class_mode='binary')

sample_training_images, _ = next(train_data_gen)

# This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
    plt.tight_layout()
    plt.show()

plotImages(sample_training_images[:5])  # Plot images 0-4
```

5. **Criação do modelo:** Define um modelo `Sequential` com várias camadas convolucionais, camadas de pooling máximo e camadas densas.
    * A função de ativação ReLU é usada para as camadas convolucionais e densas.
    * A camada de saída usa uma função softmax para classificação binária.

```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(2)
])

```

6. **Compilação do modelo:**
    * Usa o otimizador `adam`.
    * Usa `SparseCategoricalCrossentropy` como função de perda.
    * Monitora a precisão como métrica.

```python
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()
```

7. **Treinamento do modelo:**
    * Treina o modelo usando os dados de treinamento.
    * Valida o modelo durante o treinamento usando os dados de validação.
```python
EPOCHS = 100
history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=int(np.ceil(total_train / float(BATCH_SIZE))),
    epochs=EPOCHS,
    validation_data=val_data_gen,
    #validation_steps=None
    validation_steps=int(np.ceil(total_val / float(BATCH_SIZE)))
)
```
8. **Visualização dos resultados:**
    * Plota gráficos de precisão e perda do treinamento e validação.
    * Analisa os gráficos para identificar overfitting.

```python
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.savefig('./foo.png')
plt.show()
```

![image](https://github.com/user-attachments/assets/923b446b-cc3d-42c2-9f1a-0c1a3176337f)

## Resultados

O modelo atinge uma precisão de aproximadamente 70% no conjunto de validação. Os gráficos de treinamento e validação mostram sinais de overfitting, o que indica que o modelo está memorizando os dados de treinamento e não generalizando bem para novos dados.

## Próximos Passos

Para melhorar o modelo, as seguintes etapas podem ser consideradas:

* Implementar técnicas de aumento de dados para aumentar a diversidade dos dados de treinamento.
* Ajustar a arquitetura do modelo, como adicionar mais camadas ou usar diferentes funções de ativação.
* Experimentar diferentes otimizadores e taxas de aprendizado.
* Regularizar o modelo usando técnicas como dropout ou L2 regularization.
* Coletar mais dados para treinamento.
