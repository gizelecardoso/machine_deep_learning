{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "classificacao_roupas.ipynb",
      "provenance": [],
      "collapsed_sections": [],
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/gizelecardoso/machine_deep_learning/blob/master/classificacao_roupas.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "DL6kdCli6tKS",
        "colab_type": "text"
      },
      "source": [
        "Imports"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "o3JR3z4-6gDo",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "import tensorflow\n",
        "from tensorflow import keras\n",
        "#usando uma biblioteca do python para visualizar um imagem\n",
        "import matplotlib.pyplot as plt\n",
        "import numpy as np\n",
        "from tensorflow.keras.models import load_model"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "pVtd9ttb6yFV",
        "colab_type": "text"
      },
      "source": [
        "Carregar o dataset\n"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "FZ0FErv76j45",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "dataset = keras.datasets.fashion_mnist\n",
        "((imagens_treino, identificacoes_treino), (imagens_teste, identificacoes_teste)) = dataset.load_data() #devolve duas listas padroes que nao podemos modificar - treino e teste\n"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "trqCMi4Z62tN",
        "colab_type": "text"
      },
      "source": [
        "Explorar os dados"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "X2t7EE0J6mAs",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "len(imagens_treino)\n",
        "imagens_treino.shape #retorna a quantidade de imagens e o tamanho da imagem\n",
        "imagens_teste.shape \n",
        "len(identificacoes_treino)\n",
        "identificacoes_treino.min()\n",
        "identificacoes_treino.max()\n"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "Q3grYWhp67WE",
        "colab_type": "text"
      },
      "source": [
        "Exibir os dados\n"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "3NXq-xctzp6R",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "total_de_classificaoes = 10\n",
        "nomes_de_classificacoes = [\"Camiseta\", \"Calça\", \"Pullover\", \"Vestido\", \"Casaco\", \"Sandália\", \"Camisa\", \"Tênis\", \"Bolsa\", \"Bota\"]\n",
        "\n",
        "plt.imshow(imagens_treino[0])\n",
        "plt.colorbar()"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "wKcMKa10RGpd",
        "colab_type": "text"
      },
      "source": [
        "Normalizando as imagens"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "W5aH3UJ48RIE",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "#dimuindo o tamanho do processamento - o loss foi de 3 a 0.48 \n",
        "#normalização\n",
        "imagens_treino = imagens_treino/float(255)\n",
        "\n"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "0oe-ZiUaRbyD",
        "colab_type": "text"
      },
      "source": [
        "Criando, compilando, treinando e normalizando o modelo"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "tqwz4nGORRAZ",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "#multicamadas\n",
        "modelo = keras.Sequential([\n",
        "  #entrada - camada 0 achatada(flatten) - transforma 28x28 em 1x56 ficando com uma dimensão só\n",
        "  keras.layers.Flatten(input_shape = (28, 28)),\n",
        "  #processamento - precisamos criar uma camada dense, totalmente conectada onde todo mundo fala com todo mundo\n",
        "  #camadas ocultas - não sabemos exatamento o que acontece por dentro delas\n",
        "  keras.layers.Dense(256, activation = tensorflow.nn.relu),#colocar um numero multiplo de dois - que sao os neuronios ou unidades. / relu = funcao muito usada no deep - dentro do tensorflow usa nn - redes neurais\n",
        "  #normalização - dropout - faz algumas unidades adormecer para nao ficar tao viciado.\n",
        "  keras.layers.Dropout(0.2),\n",
        "    #keras.layers.Dense(128, activation = tensorflow.nn.relu),\n",
        "  #keras.layers.Dense(64, activation = tensorflow.nn.relu),   \n",
        "    #relu todos os valores negativos viram zero, os positivos se mantem positivos.# introduz as funções não linear. para classicar dados.\n",
        "  #imagem - porcentagem para cada categoria, para identificar o que essa imagem é (probabilidade) - soma das categorias = 1 - e a categoria com maior porcentagem é que tem maior probabilidade de categorizar aquela imagem\n",
        "  #saida\n",
        "  keras.layers.Dense(10, activation = tensorflow.nn.softmax)\n",
        "])\n",
        "\n",
        "#compilando o modelo\n",
        "modelo.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics = ['accuracy'])\n",
        "\n",
        "#treinando o modelo\n",
        "historico = modelo.fit(imagens_treino, identificacoes_treino, epochs = 5, validation_split = 0.2)"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "YOlYq4kQR8ww",
        "colab_type": "text"
      },
      "source": [
        "Salvando e carregando o modelo treinado"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "P2uB9VORP9PE",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "modelo.save('modelo.h5')\n",
        "modelo_salvo = load_model('modelo.h5')"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "T_1cbVrTSCRx",
        "colab_type": "text"
      },
      "source": [
        "Visualizando as acuracias de treino e validacao por epoca"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "O1Oji4dwJZ0e",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "plt.plot(historico.history['acc'])\n",
        "plt.plot(historico.history['val_acc'])\n",
        "plt.title('Acurácia por epocas')\n",
        "plt.xlabel('epocas')\n",
        "plt.ylabel('acuracia')\n",
        "plt.legend(['treino', 'validação'])\n",
        "\n",
        "#grafico de acuracia se perceber que a linha de treino esta muito bem e a linha de teste(validacao) esta baixando - overfiting - sua rede esta entendendo o treino muito melhor que a validação\n",
        "#overfiting - encaixando muito os dados de treino, teste ruim\n",
        "\n",
        "#grafico de perda se perceber que a linha de teste esta muito menor que o de validação\n",
        "\n",
        "\n",
        "#underfitting - encaixando pouco os dados de treino , teste bom\n",
        "  #testar umas 30 vezes.\n"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "Bg9PbGxfSN1T",
        "colab_type": "text"
      },
      "source": [
        "Visuzalizando as perdas de treino e validacao por epoca"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "uocj3iXUKsGH",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "plt.plot(historico.history['loss'])\n",
        "plt.plot(historico.history['val_loss'])\n",
        "plt.title('Perda por epocas')\n",
        "plt.xlabel('epocas')\n",
        "plt.ylabel('perda')\n",
        "plt.legend(['treino', 'validação'])\n",
        "\n"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "b_P6L1ykSWXj",
        "colab_type": "text"
      },
      "source": [
        "Testando do modelo e o modelo salvo"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "S8YaeY6aHZEM",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "testes = modelo.predict(imagens_teste)\n",
        "print('resultado teste: ', np.argmax(testes[1]))\n",
        "print('número da imagem teste: ', identificacoes_teste[1])\n",
        "\n",
        "testes_modelo_salvo = modelo_salvo.predict(imagens_teste)\n",
        "print('resultado teste modelo salvo: ', np.argmax(testes_modelo_salvo[1]))\n",
        "print('número da imagem teste: ', identificacoes_teste[1])\n"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "XaKUtxohSbHl",
        "colab_type": "text"
      },
      "source": [
        "Avaliando o modelo"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "nYuOOlKLIWJ7",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "perda_teste, acuracia_teste = modelo.evaluate(imagens_teste, identificacoes_teste)\n",
        "\n",
        "print('Perda do teste',  perda_teste)\n",
        "print('Acurácia do teste', acuracia_teste)"
      ],
      "execution_count": 0,
      "outputs": []
    }
  ]
}