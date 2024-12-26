#include "neural_network.h"

double activate_relu(double x) {
    return x > 0 ? x : 0;
}

double activate_sigmoid(double x) {
    return 1.0/(1.0 + exp(-x));
}

double derivative_relu(double x) {
    return x > 0 ? 1 : 0;
}

double derivative_sigmoid(double x) {
    double sigmoid = activate_sigmoid(x);
    return sigmoid * (1 - sigmoid);
}


// Fungsi untuk membuat Neural Network
NeuralNetwork* create_neural_network(int *layer_sizes, int num_layers, double learning_rate) {
    NeuralNetwork *nn = (NeuralNetwork *)malloc(sizeof(NeuralNetwork));
    if (!nn) {
        fprintf(stderr, "Error: Tidak bisa mengalokasikan memori untuk Neural Network.\n");
        exit(EXIT_FAILURE);
    }

    nn->learning_rate = learning_rate;
    nn->num_layers = num_layers;

    // Alokasi layer pertama (input layer)
    nn->input_layer = (Layer *)malloc(sizeof(Layer));
    nn->input_layer->num_neurons = layer_sizes[0];
    nn->input_layer->neurons = (Neuron *)malloc(layer_sizes[0] * sizeof(Neuron));
    nn->input_layer->prev = NULL;

    // Alokasi weights untuk setiap neuron di input layer
    for (int i = 0; i < layer_sizes[0]; i++) {
        nn->input_layer->neurons[i].weights = NULL;  // Tidak ada weights untuk input layer
        nn->input_layer->neurons[i].bias = 0.0;
        nn->input_layer->neurons[i].output = 0.0;
        nn->input_layer->neurons[i].delta = 0.0;
    }

    // Membuat layer lainnya
    Layer *prev_layer = nn->input_layer;
    for (int l = 1; l < num_layers; l++) {
        Layer *current_layer = (Layer *)malloc(sizeof(Layer));
        current_layer->num_neurons = layer_sizes[l];
        current_layer->neurons = (Neuron *)malloc(layer_sizes[l] * sizeof(Neuron));
        current_layer->prev = prev_layer;
        current_layer->next = NULL;

        // Hubungkan layer sebelumnya dengan layer saat ini
        prev_layer->next = current_layer;

        // Alokasi weights dan bias untuk neuron di layer ini
        for (int i = 0; i < layer_sizes[l]; i++) {
            current_layer->neurons[i].num_weights = prev_layer->num_neurons;
            current_layer->neurons[i].weights = (double *)malloc(prev_layer->num_neurons * sizeof(double));
            current_layer->neurons[i].bias = ((double)rand() / RAND_MAX) * 2 - 1;  // Bias random antara -1 dan 1
            current_layer->neurons[i].output = 0.0;
            current_layer->neurons[i].delta = 0.0;

            // Inisialisasi weights dengan nilai random antara -1 dan 1
            for (int j = 0; j < prev_layer->num_neurons; j++) {
                current_layer->neurons[i].weights[j] = ((double)rand() / RAND_MAX) * 2 - 1;
            }
        }

        prev_layer = current_layer;
    }

    nn->output_layer = prev_layer;
    return nn;
}

// Fungsi untuk membebaskan memori Neural Network
void free_neural_network(NeuralNetwork *nn) {
    Layer *current = nn->input_layer;

    while (current != NULL) {
        Layer *next = current->next;

        // Bebaskan memori weights dan neurons di layer saat ini
        for (int i = 0; i < current->num_neurons; i++) {
            if (current->neurons[i].weights != NULL) {
                free(current->neurons[i].weights);
            }
        }
        free(current->neurons);
        free(current);

        current = next;
    }

    free(nn);
}

void forward_propagation(NeuralNetwork *nn, double *inputs) {
    Layer *current = nn->input_layer;
    
    // Set input values
    for(int i = 0; i < current->num_neurons; i++) {
        current->neurons[i].output = inputs[i];
    }
    
    // Propagate through hidden and output layers
    current = current->next;
    while(current != NULL) {
        for (int i = 0; i < current->num_neurons;i++) {
            double sum = current->neurons[i].bias;
            
            // Calculate weighted sum
            for (int j = 0; j < current->prev->num_neurons; j++) {
                sum += current->neurons[i].weights[j] * current->prev->neurons[j].output;
            }
            
            // Apply activation function
            current->neurons[i].output = activate_relu(sum);
        }
        current = current->next;
    }
}

void backward_propagation(NeuralNetwork *nn, double *targets) {
    // Calculate output layer deltas
    Layer *current = nn->output_layer;
    for (int i = 0; i < current->num_neurons; i++) {
        double output = current->neurons[i].output;
        current->neurons[i].delta = (targets[i] - output) * derivative_relu(output);
    }
    
    // Calculate hidden layer deltas
    current = current->prev;
    while (current != nn->input_layer) {
        for (int i = 0; i < current->num_neurons; i++) {
            double error = 0.0;
            Layer *next_layer = current->next;
            
            // Calculate error contribution from next  layer
            for (int j = 0; j < next_layer->num_neurons; j++) {
                error += next_layer->neurons[j].weights[i] * next_layer->neurons[j].delta;
            }
            
            current->neurons[i].delta = error * derivative_relu(current->neurons[i].output);
        }
        current =  current->prev;
    }
    
    update_weights_and_biases(nn);
}

void update_weights_and_biases(NeuralNetwork *nn) {
    Layer *current = nn->input_layer->next;
    
    while(current != NULL) {
        for (int i = 0; i < current->num_neurons; i++) {
            // Update bias
            current->neurons[i].bias += nn->learning_rate * current->neurons[i].delta;
            
            // Update weights
            for(int j = 0; j < current->prev->num_neurons; j++) {
                current->neurons[i].weights[j] += nn->learning_rate * current->neurons[i].delta * current->prev->neurons[j].output;
            }
        }
        current = current->next;
    }
}

double calculate_loss(double *predictions, double *targets, int output_size) {
    double loss = 0.0;
    for (int i = 0; i < output_size; i++) {
        double error = predictions[i] - targets[i];
        loss += error * error;
    }
    return loss / output_size;
}