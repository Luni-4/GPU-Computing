GPU Computing
================

# Indice

1. [Informazioni](#1-informazioni)
2. [Modalità di lavoro](#2-modalità-di-lavoro)
  1. [TODO](#21-todo)
  2. [Divisione dei compiti](#22-divisione-dei-compiti)
  3. [Link Utili](#23-link-utili)
3. [Considerazioni](#3-considerazioni)


-----------------

# 1. Informazioni

- Linguaggio di programmazione: Cuda
- Versione Toolkit: Cuda Toolkit 8.0
- Cross-compiling da Linux per Windows con NVCC: non fattibile
- Possibili Librerie
    - NVIDIA cuDNN

# 2. Modalità di lavoro

## 2.1 TODO
### 2.1.1 TODO MIGLIORAMENTI

- (FATTO) Usare gli stream sulle fasi della rete e non su singole funzioni (caso convoluzionale, lanciare le convoluzioni di forward propagation di un livello in diversi stream) => non porta ad uno speedup in quanto i dati sono tutti in memoria gpu e i kernel così suddivisi impiegano troppo poco tempo per portare ad un vantaggio.
- (FATTO) guardare l'occupancy e se migliorandola può portare a dei miglioramenti.
- (FATTO) implementare i livelli batch veri e propri => sono un metodo diverso di fare la convoluzione ma non aumenta la parallelizzazione e quindi non ci porta un vantaggio nell'implementarla.
- provare ad utilizzare i metodi batched di cublas nella convoluzione
- provare a ripensare l'algoritmo della convoluzione
- provare ad utilizzare la libreria di cuda per le reti neurali, CUDNN, in modo integrale o su solo una parte del nostro progetto.
- implementare le epoche.

### 2.1.2 TODO TEST
- Provare ad utilizzare lo stesso tipo di random su entrambe le reti, nei test successivi comunque utilizzare dei pesi fissi.
- Effettuare più test sulla Educnn usando il livello di batch per confrontare le accuratezze a partire dagli stessi pesi iniziali.
- Effettuare dei test usando Cifar per valutare l'accuratezza ed i tempi computazionali ottenuti dalla nostra rete.
- Creare una rete grossa, composta da più livelli e più profondità, e con le epoche e utilizzarla col MNIST per fare un confronto con la educnn. NB: se i pesi iniziali sono gli stessi e usiamo il livello chiamato "batch", il miglior learning rate dovrebbe essere lo stesso per entrambe le reti.
- Usare la rete precedente su CIFAR e raccogliere dati per il prof senza poter fare un confronto.

### 2.1.3 TODO ALTRO

- Usare un approccio mini-batch per ridurre il tempo computazionale
- Confrontare la nostra rete con un'altra rete implementata in CUDA e vedere se speedup ed accuratezza sono simili tra loro
- Utilizzare algoritmi di convoluzioni per immagini trovati da Nicholas (bassa priorità, solo nel caso in cui tutti gli altri metodi fallissero)
- Implementare gli stride per i livelli convoluzionali(facoltativo)


## 2.2. Divisione dei compiti

NICHOLAS:

- Testare la EduCnn e la nostra rete con il livello di batch per vedere come cambiano i tempi di esecuzioni e l'accuratezza
- Aggiungere gli Stream alle varie fasi del livello convoluzionale e non ai singoli kernel che compongono le fasi

MICHELE:

- Verificare perché con un unico livello fully-connected e pesi iniziali fissati la EduCNN e la nostra rete producono risultati diversi
- Aggiungere alle varie fasi del livello fully-connected e non ai singoli kernel che compongono le fasi
- Testare con Cifar la nostra rete

## 2.3. Link Utili

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
- Batch Normalization: http://yeephycho.github.io/2016/08/03/Normalizations-in-neural-networks/


- link utili convoluzione
    - http://cs231n.github.io/optimization-2/
    - http://www.simon-hohberg.de/2014/10/10/conv-net.html
    - http://andrew.gibiansky.com/blog/machine-learning/convolutional-neural-networks/
    - https://grzegorzgwardys.wordpress.com/2016/04/22/8/
    - http://www.jefkine.com/general/2016/09/05/backpropagation-in-convolutional-neural-networks/

# 3. Considerazioni

- La dnn **originale** risulta più veloce perché utilizza uno stride pari a 2 nei livelli convoluzionali e non usa le epoche
- La educnn oltre ai livelli convoluzionali ha i livelli di max_pooling e di average_pooling
- Definire un intervallo per il learning rate, ad esempio [0.001, 0.8], e campionarlo con diverse frequenze di campionamento (già fatto)
- Testare i learning rate in maniera decrescente per vedere se l'accuratezza migliora al diminuire dei valori (aumento del tempo di esecuzione)
- Partire con dei pesi iniziali bassi fissi, non generati casualmente, per poter confrontare le due reti senza dover dipendere dai diversi algoritmi
per la generazione di numeri pseudo-casuali
- Costruire una rete con tanti livelli convoluzionali e fully-connected
