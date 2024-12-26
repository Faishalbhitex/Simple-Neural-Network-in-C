#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// Struktur untuk menyimpan neuron
typedef struct Neuron {
    double *weights;
    double bias;
    double output;
    double delta;
    int num_weights;
} Neuron;

// Struktur untuk layer
typedef struct Layer {
    Neuron *neurons;
    int num_neurons;
    struct Layer *next;  // pointer ke layer selanjutnya
    struct Layer *prev; // pointer ke layer sebelumnya
} Layer;

// Strucktur utama Neural Network
typedef struct NeuralNetwork {
    Layer *input_layer;
    Layer *output_layer;
    int num_layers;
    double learning_rate;
} NeuralNetwork;

// Function prototypes untuk neural_network.h
NeuralNetwork* create_neural_network(int *layer_sizes, int num_layers, double learning_rate);
void free_neural_network(NeuralNetwork *nn);
void forward_propagation(NeuralNetwork *nn, double *inputs);
void backward_propagation(NeuralNetwork *nn, double *targets);
double calculate_loss(double *predictions,double *targets, int output_size);

// Function prototypes untuk algorithm_nn.c
double activate_relu(double x);
double activate_sigmoid(double x);
double derivative_relu(double x);
double derivative_sigmoid(double x);
void update_weights_and_biases(NeuralNetwork *nn);

#endif