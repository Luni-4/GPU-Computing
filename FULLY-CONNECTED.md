# C onsiderazioni

## Setting iniziale

- Pesi 0.01
- Learning Rate 1
- Epoche 1

## Accuratezza

- Per valori di learning rate bassi (es. 0.0018) le due accuratezze sono uguali
- Un unico livello vinciamo con più di 0.4% noi (89.56 contro 89.13)
- 2 livelli e pochi nodi (es. 100) vinciamo noi di quasi l'1% sulla educnn (26.81 contro 27.74)
- 2 livelli e nodi intermedi (es.200) vince la educnn di 0.2% (29.52 contro 29.26)
- 2 livelli con tanti nodi (es. 300) vince la educnn di circa 0.6% rispetto alla nostra. (10.32 contro 9.74)
- Presumo che i problemi delle ultime due configurazioni siano dovuti alla grande quantità di memoria utilizzata dalle variabili private delle classi
e questo comporta problemi nel poter trovare un'accuratezza migliore
- Con 3 livelli (100, 50, 10) stessa accuratezza

## Tempo

- In tempo vinciamo sempre noi, tranne quando si ha solo un fully da 10. Questo è dovuto all'overhead delle nostre funzioni, ma non
è importante in quanto questa configurazione non ha senso e non sarà usata (2.89 s la edu e 5 s noi)

# Importante

- ESEGUIRE I TEST SENZA AVERE ALTRE APPLICAZIONI APERTE (TIPO MOZILLA O CHROME) PERCHÈ IL PROCESSO DI COMPUTAZIONE RALLENTA MOLTO E ANCHE QUESTI
SOFTWARE OCCUPANO SPAZIO IN GRAM E LE ACCURATEZZE OTTENUTE POSSONO ESSERE DIVERSE DA UN ESECUZIONE AD UN ALTRA A CAUSA DEL FATTO CHE LA GRAM È
VICINA AL SUO LIMITE (NEL MIO CASO 1GB)



TEST NICHOLAS
ho fatto tutti test con learning rate = 1, pesi = 0.01 e epoch = 1.
non ho potuto spegnere gli altri programmi perchè dovevo almeno far finta di lavorare XD

1)		 	    10 = 89.56
2)		 100 -> 10 = 29.23%
3)	 	 200 -> 10 = 29.26%
4)		 300 -> 10 = 9.74%
5)100 ->  50 -> 10 = 32.93%
6)300 -> 100 -> 10 = 32.01%
6)500 -> 300 -> 10 = 9.74%
