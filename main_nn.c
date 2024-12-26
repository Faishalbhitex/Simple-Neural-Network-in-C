#include "neural_network.h"

int main() {
    // Contoh arsitektur: input(2) -> Hidden(3) -> Output(1)
    int layer_sizes[] = {2, 3, 1};
    int num_layers = 3;
    double learning_rate = 0.01;

    // Inisialisasi Neural Network
    NeuralNetwork *nn = create_neural_network(layer_sizes, num_layers, learning_rate);

    // Data training (XOR problem)
    double training_inputs[][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    double training_outputs[] = {0, 1, 1, 0};

    // Training loop
    int epochs = 1000;
    for (int epoch = 0; epoch < epochs; epoch++) {
        double total_loss = 0;

        for (int i = 0; i < 4; i++) {
            // Forward propagation
            forward_propagation(nn, training_inputs[i]);

            // Backward propagation
            backward_propagation(nn, &training_outputs[i]);

            // Calculate loss
            total_loss += calculate_loss(&nn->output_layer->neurons[0].output, &training_outputs[i], 1);
        }

        // Print loss setiap 100 epoch
        if (epoch % 100 == 0) {
            printf("Epoch %d, Loss: %f\n", epoch, total_loss / 4);
        }
    }

    // Cleanup
    free_neural_network(nn);

    return 0;
}