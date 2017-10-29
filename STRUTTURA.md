Struttura Programma

typedef enum LayerType {INPUT, CONVOLUTIONAL, FULLY_CONNECTED, OUTPUT} LayerType;
typedef enum ActFctType {SIGMOID, TANH, RELU, NONE} ActFctType;

class LayerDefinition
{

protected:
    LayerType l;
    int width;
    int height;
    std::vector<Column*> columns;

public:

    LayerDefinition();
    ~LayerDefinition();
    
    virtual void initColumn(LayerDefinition& l1, LayerDefinition& l2) = 0;   
    
    virtual int getWeightCount() = 0; // Numero di pesi di un singolo livello
    virtual int getLayerNodeCount() = 0; // Numero di nodi di un singolo livello (ci serve per il bias)
    virtual int getColumnCount() = 0; // Numero di colonne di un singolo livello
    
    virtual void forward_propagation(LayerDefinition& l); // Solo il livello precedente mi interessa
    
    virtual void back_propagation(); // Aggiorno i pesi di questo livello  
    
    
    virtual int getNodeBackwardConnectionCount(LayerDefinition& l); // Numero di connessioni backward del nodo n con il livello l precedente
    //virtual int getNodeForwardConnectionCount(LayerDefinition& l); // Numero di connessioni forward del nodo n con il livello l successivo
    
    
    
};

class FullyConnected : public LayerDefinition
{

public:
    FullyConnected(width = 1, height = 1, ActFctType = RELU); // costruttore definisce layertype = FULLY_CONNECTED
    ~FullyConnected();
    
    void initColumn(LayerDefinition& l1, LayerDefinition& l2)
    {
      // ciclo for su colonne
      // definisco numero di nodi in colonne (=depth)
      // definisco connessioni massime per nodi in colonne (=backforward + forward)  
    
    }
    
    
private:
    ActFctType f;    
        
};


class Column
{

public:
    Column(int maxConnCountPerNode, int nodeCount, int backforward); // Ciclo for per i nodi dentro il costruttore
    ~Column(); // Cancellare i nodi nel distruttore
    
    int getMaxConnCountPerNode();
    int getNodeCount();
    int getBackforward();
    
private:
    int maxConnCountPerNode;
    int nodeCount;
    int backforward;
    std::vector<Node*> nodes;

};


class Node
{

public:
    Node(bias = 0, output = 0, errorsum = 0, backwardConnCount = 0, forwardConnCount = 0);
    ~Node(); // Cancellare le connessioni nel distruttore 


private:
    double bias;                // value of the bias weight of this node
    double output;              // result of activation function applied to this node
    double errorSum;            // result of error back propagation applied to this node
    int backwardConnCount;      // number of connections to the previous layer
    //int forwardConnCount;       // number of connections to the following layer
    std::vector<Connection*> connections; 
    



};    
