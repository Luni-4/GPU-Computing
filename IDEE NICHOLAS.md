NB: per coerenza nello pseudo codice uso tutte matrici e array ma ovviamente sono sostituibili con vettori.

COSTRUZIONE MATRICE IMMAGINI:

immagini [][][];

for(j in {ogni immagine}){

	image = loadImage(j);

	for(k in {ogni pixel})
		for(i in {ogni RGB})
			immagini[j][k][i] = image.getFromImage();
}

Quindi alla fine il primo posto della matrice rappresenta l'immagine, il secondo posto rappresenta il pixel e il terzo posto rappresenta RGB.
Nel MNIST il terzo posto avrà sempre grandezza 1.

COSTRUZIONE LIVELLI:

livelli[];

livelli[0] = inputLayer();
livelli[1] = fullyConnected();
livelli[2] = convolutionalLayer();
livelli[3] = outputLayer;

COSTRUZIONE MATRICE PESI:

pesi[][][];

for(j in {ogni livello}){
	switch livelli[j].type:
	
	case INPUT:
		//non ci sono pesi
		break;
	case FULLYCONNECTED:
		for(k in {ogni nodo livello j-1}){
			for(i in {profondità ogni nodo livello j-1}){
				pesi[j][k][i] = randomPeso();
 			}
		}
		break;
	case CONVOLUTIONAL:
		for(k in {filter * filter}){
			for(i in {numero di filtri}){
				pesi[j][k][i] = randomPeso();
			}
		}
		break;
	case OUTPUT:
		//domanda: output è un tipo a parte o è uno degli altri tipi?
		break;
	default:
		error();
		break;
	
}

CREAZIONE MATRICE OUTPUT:

ogni livello ha una matrice
	output[nodo][RGB/profondità]

ESECUZIONE TRAIN:

for(q in {ogni immagine}){
	for(j in {ogni livello}){
		livelli[j].train(); //diverso in base al tipo di livello

		- livello di input prepara solo i dati dell'immagine nei suoi nodi, cioè nella sua matrice di output
		for(k in {ogni pixel}){
			for(i in {ogni RGB}{
				output[k][i] = immagini[q][k][i];
			}
		} 

		- livello fullyConnected
		for(l in {ogni nodo livello j}){
			for(k in {ogni nodo livello j-1}){
				for(i in {profondità ogni nodo j-1}){
					peso = pesi[j][k][i];
					input = outputLayerPrecedente[k][i];
					output[l][0] += calcolo(peso, input); //l'output del livello fullyConnected sarà sempre di profondità 1.
				}
			}
		}
		
		-livello convoluzionale
		for(l in {ogni nodo livello j})
			for(k in {filter * filter}){
				for(i in {numero di filtri}){
					peso = pesi[j][k][i];
					input = outputLayerPrecedenteCalcolato(l, k, i); //calcolato in base a movimento del filtro, cioè in base al nodo in cui siamo
					output[l][i] += calcolo(peso, input);// l'output del livello convoluzionale avrà profondita uguale al numero di filtri (?)
				}
			}
		}
	}
}


































