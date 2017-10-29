# Definizione dei Livelli

Sia *i* un generico livello che può essere dei seguenti tipi


## Input Layer

- Id: 0
- Numero di pesi: 0

## FullyConnected

- Numero di pesi: Nodi[i-1] * Nodi[i]

## Convoluzionale

- Id: 1
- Numero di pesi: filtro[i] * filtro[i] * profondità[i] * profondità[i-1]


## Output

- Id: 2
- Numero di pesi: Nodi[i-1] * Nodi[i]


# Connessioni SINGOLO Nodo

es. Dato nodo *n* di un livello *i*, sapendo che il livello *i+1* è FullyConnected, quante connessioni avrà? Guardo "Numero di connessioni in avanti".
Se il livello *i-1* è Convoluzionale, guardo "Numero di connessioni all'indietro".

## FullyConnected

- Numero di connessioni all'indietro (connessioni in entrata): Nodi[i-1]
- Numero di connessioni in avanti (connessioni in uscita): Nodi[i+1]

## Convoluzionale

- Numero di connessioni all'indietro (connessioni in entrata): filtro[i] * filtro[i] * profondità[i-1]
- Numero di connessioni in avanti (connessioni in uscita), variano con nodi di tipo convoluzionale, ma hanno un valore massimo che è: filtro[i+1] * filtro[i+1] * profondità[i+1]

## OutPut

- Numero di connessioni all'indietro (connessioni in entrata): Nodi[i-1]
- Numero di connessioni in avanti (connessioni in uscita): Nodi[i+1]


# Colonne

- Numero di nodi in colonna: profondità[i]
- Numero di connessioni, sia in entrata che in uscita, dei nodi che fanno parte della colonna: (Numero di connessioni all'indietro + Numero di connessioni in avanti)   


# Connessioni

Non si hanno connessioni sul nodo di input, solo per altri tipi di livello.

Per ogni nodo di una colonna vengono riazzerate le connessioni e viene ristabilito il numero di connessioni in entrata e in uscita del nodo stesso a seconda
del livello di cui fa parte

Per prima cosa vengono settate le connessioni all'indietro, poi quelle in avanti

Le connessioni all'indietro sono usate per il feed_forward, mentre quelle in avanti per la backpropagation.


# Note

Ogni identificativo, dei livelli, delle colonne, dei nodi e dei pesi, corrisponde all'indice di un vettore!

Numero di pesi totale della rete: somma del numero dei pesi dei singoli livelli + numero di nodi della rete (bias)

Identificativo del livello usato per individuare il puntatore di quel livello


# GPU

## Memoria
- Immagini di input
- Pesi generali
- Ouput dei nodi di ciascun livello
- Errori dei nodi di ciascun livello

## Immagini di input

Matrice che ha per righe le immagini e per colonne i pixel di ciascuna immagine

## Pesi

Matrice dei pesi, le righe corrispondono ai livelli (eccetto quello di input) e le colonne ai pesi del singolo livello. In fondo all'ultima colonna aggiungere
il bias associato ai nodi del livello


## Nodi dei livelli

Matrice che ha per righe i singoli livelli e per colonne i valori di output dei nodi associati a quel livello
Matrice che ha per righe i singoli livelli e per colonne i valori di errore dei nodi associati a quel livello
