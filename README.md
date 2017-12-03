GPU Computing
================

# Indice

1. [Informazioni](#1-informazioni)
2. [Modalità di lavoro](#2-modalità-di-lavoro)
  1. [TODO](#21-todo)
  2. [Processo di lavoro](#22-processo-di-lavoro)
  3. [Divisione dei compiti](#23-divisione-dei-compiti)
  4. [Link Utili](#24-link-utili)
3. [Considerazioni](#3-considerazioni)


-----------------

# 1. Informazioni

- Linguaggio di programmazione: Cuda
- Versione Toolkit: Cuda Toolkit 8.0
- Cross-compiling da Linux per Windows con NVCC: non fattibile
- Possibili Librerie
    - NVIDIA cuDNN

# 2. Modalità di lavoro

## 2.1. TODO

- Valutare la velocità delle rete neurale sequenziale contenuta nella directory sequential su diversi terminali
- Riprodurre la stessa rete in CUDA e valutare se:
    - Diminuisce il tempo di esecuzione, quindi si ottiene un notevole Speed-up
    - Diminuisce l'errore della rete (molto improbabile in quanto dipende dal learning rate e da parametri sulla quale bisogna fare il tuning)
- Produrre la documentazione necessaria da presentare (report, slide)

## 2.2. Processo di lavoro

- Parallelizzare il più possibile la rete, partendo dalla costruzione dei livelli e poi dalla loro composizione 
- Analizzare i risultati ottenuti e successivamente fare il tuning dei parametri per vedere se si riescono ad ottenere risultati
migliori (es. dimensione del blocco, uso di stream ecc...)
- Produrre il report, slide ecc...
 

## 2.3. Divisione dei compiti

NICHOLAS:

- Dubbio correlazione e convoluzione
- Controllare con sm_50 se i risultati sono corretti
- Test per vedere quanto la rete impara
- Inserimento della depth nella backpropagation
- Risolvere i problemi su tutti i memcpy per velocizzare la rete
- Inserire gli stream
- Possibile implementazione di stride (???)


MICHELE:

- Aggiungere la funzione calcError come ho scritto in Network
- Togliere l'output inutile da Relu
- Fixare alignedNodes e THREADS
- Gestire la velocità della forward

## 2.4. Link Utili

- Creazione di reti neurali con NVidia: https://developer.nvidia.com/cudnn
- Rete convoluzionale: https://en.wikipedia.org/wiki/Convolutional_neural_network#GPU_implementations
- Spiegazione approfondita reti convoluzionali: https://adeshpande3.github.io/adeshpande3.github.io/A-Beginner%27s-Guide-To-Understanding-Convolutional-Neural-Networks/
- Famoso database contenente training set e test set di caratteri digitalizzati: http://yann.lecun.com/exdb/mnist/
- Spiegazione reti neurali: https://mmlind.github.io/
- Partendo da diverse configurazioni di blocchi e thread iniziali, ottenere i rispettivi indici degli array: http://www.martinpeniak.com/index.php?option=com_content&view=article&id=288:cuda-thread-indexing-explained&catid=17:updates
- Informazioni utili sull'architettura Fermi e relative capacità computazionali: http://blog.cuvilib.com/2010/06/09/nvidia-cuda-difference-between-fermi-and-previous-architectures/


- Esempio di una rete neurale a 3 livelli (input - hidden - output): https://mmlind.github.io/Simple_3-Layer_Neural_Network_for_MNIST_Handwriting_Recognition/
- Esempio di una rete neurale convoluzionale: https://mmlind.github.io/Deep_Neural_Network_for_MNIST_Handwriting_Recognition/
- Spiegazione della rete LeCun Net (meglio conosciuta come LeNet) per il riconoscimento dei caratteri: https://www.pyimagesearch.com/2016/08/01/lenet-convolutional-neural-network-in-python/
- Rete neurale convoluzionale LeCun Net per il riconoscimento di caratteri sequenziale: https://github.com/tatsy/educnn
- Seconda rete neurale convoluzionale per il riconoscimento dei caratteri con video YouTube: https://github.com/can1357/simple_cnn


- link utili convoluzione
    - http://cs231n.github.io/optimization-2/
    - http://www.simon-hohberg.de/2014/10/10/conv-net.html
    - http://andrew.gibiansky.com/blog/machine-learning/convolutional-neural-networks/
    - https://grzegorzgwardys.wordpress.com/2016/04/22/8/
    - http://www.jefkine.com/general/2016/09/05/backpropagation-in-convolutional-neural-networks/

# 3. Considerazioni

- La dnn **originale** risulta più veloce perché utilizza uno stride pari a 2 nei livelli convoluzionali e non usa le epoche
- La educnn oltre ai livelli convoluzionali ha i livelli di max_pooling e di average_pooling
