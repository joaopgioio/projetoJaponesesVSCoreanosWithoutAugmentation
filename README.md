# Classificação de Imagens de Japoneses x Coreanos sem Aumento de Imagem

Este projeto demonstra como construir um classificador de imagens para distinguir fotos de japoneses e coreanos usando TensorFlow e Keras. O modelo utiliza uma rede neural convolucional (CNN) e é treinado sem técnicas de aumento de dados.

## Descrição do Código

O código realiza as seguintes etapas:

1. **Importação de bibliotecas:** Inclui bibliotecas essenciais como TensorFlow, Keras, NumPy, Matplotlib e OS.
2. **Carregamento de dados:** Carrega um conjunto de dados de imagens de japoneses e coreanos a partir de um arquivo ZIP.
    * O conjunto de dados é dividido em conjuntos de treinamento e validação.
    * A estrutura do diretório é importante para o carregamento adequado dos dados.
3. **Pré-processamento de dados:**
    * Usa `ImageDataGenerator` para redimensionar as imagens e realizar aumento de dados no conjunto de treinamento (embora o aumento seja desativado neste código).
    * As imagens são carregadas em lotes e redimensionadas para um tamanho uniforme.
4. **Criação do modelo:** Define um modelo `Sequential` com várias camadas convolucionais, camadas de pooling máximo e camadas densas.
    * A função de ativação ReLU é usada para as camadas convolucionais e densas.
    * A camada de saída usa uma função softmax para classificação binária.
5. **Compilação do modelo:**
    * Usa o otimizador `adam`.
    * Usa `SparseCategoricalCrossentropy` como função de perda.
    * Monitora a precisão como métrica.
6. **Treinamento do modelo:**
    * Treina o modelo usando os dados de treinamento.
    * Valida o modelo durante o treinamento usando os dados de validação.
7. **Visualização dos resultados:**
    * Plota gráficos de precisão e perda do treinamento e validação.
    * Analisa os gráficos para identificar overfitting.

## Resultados

O modelo atinge uma precisão de aproximadamente 70% no conjunto de validação. Os gráficos de treinamento e validação mostram sinais de overfitting, o que indica que o modelo está memorizando os dados de treinamento e não generalizando bem para novos dados.

## Próximos Passos

Para melhorar o modelo, as seguintes etapas podem ser consideradas:

* Implementar técnicas de aumento de dados para aumentar a diversidade dos dados de treinamento.
* Ajustar a arquitetura do modelo, como adicionar mais camadas ou usar diferentes funções de ativação.
* Experimentar diferentes otimizadores e taxas de aprendizado.
* Regularizar o modelo usando técnicas como dropout ou L2 regularization.
* Coletar mais dados para treinamento.
