GPU Computing
================

# Indice

1. [Informazioni](#1-informazioni)
2. [Modalità di lavoro](#2-modalità-di-lavoro)
  1. [TODO](#21-todo)
  2. [Processo di lavoro](#22-processo-di-lavoro)
  3. [Divisione dei compiti](#23-divisione-dei-compiti)
  4. [Link Utili](#24-link-utili)
3. [Comunicazioni](#3-comunicazioni)


-----------------

# 1. Informazioni

- Linguaggio di programmazione: Cuda
- Versione Toolkit: Cuda Toolkit 8.0
- Cross-compiling da Linux per Windows con NVCC: non fattibile
- Possibili Librerie
    - NVIDIA cuDNN

# 2. Modalità di lavoro

## 2.1. TODO

- Valutare la velocità delle rete neurale sequenziale a 3 livelli creata dallo sviluppatore
- Riprodurre la stessa rete in CUDA e valutare se:
    - Diminuisce il tempo di esecuzione, quindi la rete ha un notevole Speed-up
    - Diminuisce l'errore della rete (cosa altamente improbabile)
- Produrre la documentazione necessaria (report) da presentare

## 2.2. Processo di lavoro

- Valutare come parallelizzare il più possibile:
    - Il livello di input
    - Il livello hidden
    - Il livello di output
- Analizzare i risultati ottenuti e poi fare il tuning dei parametri per vedere se si ottengono risultati
migliori (es. dimensione del blocco, uso di stream ecc...)
- Produrre il report
 

## 2.3. Divisione dei compiti

NICHOLAS:


MICHELE:

## 2.4. Link Utili

- Creazione di reti neurali con NVidia: https://developer.nvidia.com/cudnn
- Rete convoluzionale: https://en.wikipedia.org/wiki/Convolutional_neural_network#GPU_implementations
- Spiegazione approfondita reti convoluzionali: https://adeshpande3.github.io/adeshpande3.github.io/A-Beginner%27s-Guide-To-Understanding-Convolutional-Neural-Networks/
- Famoso database contenente training set e test set di caratteri digitalizzati: http://yann.lecun.com/exdb/mnist/
- Spiegazione reti neurali: https://mmlind.github.io/
- Esempio di una rete neurale a 3 livelli (input - hidden - output): https://mmlind.github.io/Simple_3-Layer_Neural_Network_for_MNIST_Handwriting_Recognition/
- Esempio di una rete neurale convoluzionale: https://mmlind.github.io/Deep_Neural_Network_for_MNIST_Handwriting_Recognition/
- Spiegazione della rete LeCun Net (meglio conosciuta come LeNet) per il riconoscimento dei caratteri: https://www.pyimagesearch.com/2016/08/01/lenet-convolutional-neural-network-in-python/
- Rete neurale convoluzionale LeCun Net per il riconoscimento di caratteri sequenziale: https://github.com/tatsy/educnn
- Seconda rete neurale convoluzionale per il riconoscimento dei caratteri con video YouTube: https://github.com/can1357/simple_cnn


# 3. Comunicazioni

DA NICHOLAS:



A NICHOLAS:



DA MICHELE:


A MICHELE:
