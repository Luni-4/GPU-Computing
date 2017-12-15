# Numeri Casuali

- I numeri casuali generati da Cuda usano di default l'algoritmo xorshift di tipo xorwow e io ho deciso di usarlo perché è rapido e passa tutti i test creati per
analizzare gli algoritmi per la generazione di numeri pseudo-casuali

- La educnn usa lo xorshift chiamato xorshift128, che risulta meno stabile rispetto allo xorwow e presenta un implmentazione non
compatibile con lo xorwow 

- La dnn usa il rand del C

- In cuda non esiste lo xorshift128, per cui dovrei implementarlo e mi sembra inutile avendo già lo xorwow. Per migliorare l'accuratezza della
rete preferirei concentrarmi sul learning rate o sui valori (tipo lo 0.4 nel fullyconnected) che moltiplicano i pesi casuali

- Link che spiega gli algoritmi xorshift128 e xorwow con relativi codici scritti in C (https://www.google.it/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&ved=0ahUKEwiDsZrvmIrYAhXK2RoKHQp2DOMQFggxMAA&url=https%3A%2F%2Fwww.jstatsoft.org%2Farticle%2Fview%2Fv008i14%2Fxorshift.pdf&usg=AOvVaw0BPuy1unBMwZ0igdHgDD8H)


# Risultati Ottenuti

- Con learningRate pari a 1 l'accuratezza media è del 47%

- Variando il learning rate si arriva ad un'accuratezza del 81%

- La rete viene eseguita in 55 secondi contro i 110 della educnn e i 212 della dnn (senza usare gli stream)
