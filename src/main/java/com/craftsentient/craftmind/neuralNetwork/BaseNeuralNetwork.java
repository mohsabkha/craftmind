package com.craftsentient.craftmind.neuralNetwork;

import com.craftsentient.craftmind.activation.DEFAULT_ACTIVATIONS;
import com.craftsentient.craftmind.derivitives.activationDerivatives.ActivationDerivatives;
import com.craftsentient.craftmind.derivitives.errorLossDerivatives.ErrorLossDerivatives;
import com.craftsentient.craftmind.errorLoss.DEFAULT_LOSSES;
import com.craftsentient.craftmind.errorLoss.ErrorLossFunctions;
import com.craftsentient.craftmind.layer.DenseLayer;
import com.craftsentient.craftmind.learningRate.DEFAULT_LEARNING_RATE;
import com.craftsentient.craftmind.utils.FileUtils;
import com.craftsentient.craftmind.utils.craftmath.MathUtils;
import com.craftsentient.craftmind.neuron.Neuron;
import lombok.Getter;
import lombok.NoArgsConstructor;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;
import java.util.stream.IntStream;

import static com.craftsentient.craftmind.learningRate.LearningRate.updateLearningRate;
import static com.craftsentient.craftmind.utils.craftmath.MathUtils.getHotOneVecIndexValue;
import static com.craftsentient.craftmind.utils.PrintUtils.*;


@Getter
public class BaseNeuralNetwork {
    private final ArrayList<DenseLayer> layerList;
    private final double[][] initialInput;
    private int dataCounter = 0;
    private int miniBatchSize = 1;
    private int epoch = 1;
    private int step = miniBatchSize;
    private boolean epochDecay = false;
    private boolean stepDecay = false;
    private final Map<Integer, Double> decisions;
    private final int[] decisionsIndex;
    private int[] trueValueIndices;
    private double accuracy;
    private double loss;
    private double sum;
    private double learningRate = 1.0;
    private double learningRateDecay = 0.0;
    private double momentum;
    private DEFAULT_LEARNING_RATE learningRateFunction;
    private int[][] hotOneVec;
    public static double ALPHA = 1.0;
    public static double GAMMA = 1.0;
    public static double DELTA = 1.0;
    public static double MARGIN = 1.0;
    public static double EPSILON = 1e-15;
    private DEFAULT_LOSSES lossFunction;
    public static final Random random = new Random(0);

    private BaseNeuralNetwork(int layers, DEFAULT_ACTIVATIONS activationFunction, Map<Integer, DEFAULT_ACTIVATIONS> activationFunctionsMap) {
        // create layer list
        this.layerList = new ArrayList<>();
        // create an initial input array of random numbers of size layers
        this.initialInput = randn(1,layers);
        // instantiate the decisions index
        this.decisionsIndex = new int[initialInput.length];
        // instantiate the hash map for the values of the decision
        this.decisions = new HashMap<>();

        // loop over layer count and generate network
        IntStream.range(0, layers).forEach(i -> {
            // set activation function
            DEFAULT_ACTIVATIONS activationFunctionToUse = activationFunctionsMap.getOrDefault(i, activationFunction);
            // randomly initialize weights matrix
            double[][] weights = randn(layers, layers);
            try {
                // if not starting layer, feed the previous layers outputs to the current layer
                if (i != 0) {
                    layerList.add(new DenseLayer(weights, layerList.get(i - 1).getLayerOutputs(), activationFunctionToUse));
                } else {
                    layerList.add(new DenseLayer(weights, activationFunctionToUse));
                }
            } catch (Exception e){
                throw new RuntimeException("[DenseLayers Constructor] Error During Layer Creation! " + e.getMessage());
            }
        });
    }
    private BaseNeuralNetwork(int layers, double[][] initialInput, DEFAULT_ACTIVATIONS activationFunction, Map<Integer, DEFAULT_ACTIVATIONS> activationFunctionsMap) {
        this.initialInput = initialInput;
        this.layerList = new ArrayList<>();
        this.decisionsIndex = new int[initialInput.length];
        this.decisions = new HashMap<>();
        IntStream.range(0, layers).forEach(i -> {
            DEFAULT_ACTIVATIONS activationFunctionToUse = activationFunctionsMap.getOrDefault(i, activationFunction);
            try {
                if (i != 0) {
                    double[][] weights = randn(initialInput.length, layerList.get(i - 1).getLayerOutputs().length);
                    layerList.add(new DenseLayer(weights, layerList.get(i - 1).getLayerOutputs(), activationFunctionToUse));
                } else {
                    double[][] weights = randn(initialInput.length, initialInput[dataCounter].length);
                    layerList.add(new DenseLayer(weights, initialInput[dataCounter], activationFunctionToUse));
                }
            } catch (Exception e){
                throw new RuntimeException("[DenseLayers Constructor] Error During Layer Creation! " + e.getMessage());
            }
        });
    }
    private BaseNeuralNetwork(int layers, double[][] initialWeights, double[][] initialInput, DEFAULT_ACTIVATIONS activationFunction, Map<Integer, DEFAULT_ACTIVATIONS> activationFunctionsMap) {
        this.initialInput = initialInput;
        this.layerList = new ArrayList<>();
        this.decisionsIndex = new int[initialInput.length];
        this.decisions = new HashMap<>();

        IntStream.range(0, layers).forEach(i -> {
            DEFAULT_ACTIVATIONS activationFunctionToUse = activationFunctionsMap.getOrDefault(i, activationFunction);
            try {
                if (i != 0) {
                    double[][] weights = randn(initialWeights.length, layerList.get(i - 1).getLayerOutputs().length);
                    layerList.add(new DenseLayer(weights, layerList.get(i - 1).getLayerOutputs(), activationFunctionToUse));
                } else {
                    layerList.add(new DenseLayer(initialWeights, initialInput[0], activationFunctionToUse));
                }
            } catch(Exception e){
                throw new RuntimeException("[DenseLayers Constructor] Error During Layer Creation! " + e.getMessage());
            }
        });
    }
    private BaseNeuralNetwork(int layers, double[][] initialWeights, double[] biases, double[][] initialInput, DEFAULT_ACTIVATIONS activationFunction, Map<Integer, DEFAULT_ACTIVATIONS> activationFunctionsMap) {
        this.layerList = new ArrayList<>();
        this.initialInput = initialInput;
        this.decisionsIndex = new int[initialInput.length];
        this.decisions = new HashMap<>();

        IntStream.range(0, layers).forEach(i -> {
            DEFAULT_ACTIVATIONS activationFunctionToUse = activationFunctionsMap.getOrDefault(i, activationFunction);
            try {
                if (i != 0) {
                    double[][] weights = randn(initialWeights.length, layerList.get(i - 1).getLayerOutputs().length);
                    layerList.add(new DenseLayer(weights, layerList.get(i - 1).getLayerOutputs(), activationFunctionToUse));
                } else {
                    layerList.add(new DenseLayer(initialWeights, biases, initialInput[0], activationFunctionToUse));
                }
            } catch (Exception e){
                throw new RuntimeException("[DenseLayers Constructor] Error During Layer Creation! " + e.getMessage());
            }
        });
    }
    private BaseNeuralNetwork(int layers, int numberOfNeurons, DEFAULT_ACTIVATIONS activationFunction, Map<Integer, DEFAULT_ACTIVATIONS> activationFunctionsMap) {
        this.layerList = new ArrayList<>();
        this.initialInput = randn(1,numberOfNeurons);
        this.decisionsIndex = new int[initialInput.length];
        this.decisions = new HashMap<>();

        IntStream.range(0, layers).forEach(i -> {
            DEFAULT_ACTIVATIONS activationFunctionToUse = activationFunctionsMap.getOrDefault(i, activationFunction);
            double[][] weights = randn(numberOfNeurons, numberOfNeurons);
            try {
                if (i != 0) {
                    layerList.add(new DenseLayer(weights, layerList.get(i - 1).getLayerOutputs(), activationFunctionToUse));
                }
                else {
                    layerList.add(new DenseLayer(weights, initialInput[0], activationFunctionToUse));
                }
            } catch (Exception e){
                throw new RuntimeException("[DenseLayers Constructor] Error During Layer Creation! " + e.getMessage());
            }
        });
    }
    private BaseNeuralNetwork(int layers, int numberOfNeurons, double[][] initialInput, DEFAULT_ACTIVATIONS activationFunction, Map<Integer, DEFAULT_ACTIVATIONS> activationFunctionsMap) {
        if(numberOfNeurons != initialInput.length) { throw new IllegalArgumentException("neuronsPerLayer of " + numberOfNeurons + " and initialInput size of " + initialInput.length + " do not match!");}
        this.initialInput = initialInput;
        this.layerList = new ArrayList<>();
        this.decisionsIndex = new int[initialInput.length];
        this.decisions = new HashMap<>();

        IntStream.range(0, layers).forEach(i -> {
            DEFAULT_ACTIVATIONS activationFunctionToUse = activationFunctionsMap.getOrDefault(i, activationFunction);
            try {
                if (i != 0) {
                    double[][] weights = randn(numberOfNeurons, layerList.get(i - 1).getLayerOutputs().length);
                    layerList.add(new DenseLayer(weights, layerList.get(i - 1).getLayerOutputs(), activationFunctionToUse));
                }
                else {
                    double[][] weights = randn(numberOfNeurons, initialInput[0].length);
                    layerList.add(new DenseLayer(weights, initialInput[0], activationFunctionToUse));
                }
            } catch (Exception e) {
                throw new RuntimeException("[DenseLayers Constructor] Error During Layer Creation! " + e.getMessage());
            }
        });
    }
    private BaseNeuralNetwork(int layers, int numberOfNeurons, double[][] initialWeights, double[][] initialInput, DEFAULT_ACTIVATIONS activationFunction, Map<Integer, DEFAULT_ACTIVATIONS> activationFunctionsMap) {
        this.initialInput = initialInput;
        this.layerList = new ArrayList<>();
        this.decisionsIndex = new int[initialInput.length];
        this.decisions = new HashMap<>();

        IntStream.range(0, layers).forEach(i -> {
            DEFAULT_ACTIVATIONS activationFunctionToUse = activationFunctionsMap.getOrDefault(i, activationFunction);
            try {
                if (i != 0) {
                    double[][] weights = randn(numberOfNeurons, numberOfNeurons);
                    layerList.add(new DenseLayer(weights, layerList.get(i - 1).getLayerOutputs(), activationFunctionToUse));
                } else {
                    layerList.add(new DenseLayer(initialWeights, initialInput[0], activationFunctionToUse));
                }
            } catch (Exception e) {
                throw new RuntimeException("[DenseLayers Constructor] Error During Layer Creation! " + e.getMessage());
            }
        });
    }
    private BaseNeuralNetwork(int layers, int numberOfNeurons, double[][] initialWeights, double[] biases, double[][] initialInput, DEFAULT_ACTIVATIONS activationFunction, Map<Integer, DEFAULT_ACTIVATIONS> activationFunctionsMap) {
        this.layerList = new ArrayList<>();
        this.initialInput = initialInput;
        this.decisionsIndex = new int[initialInput.length];
        this.decisions = new HashMap<>();

        IntStream.range(0, layers).forEach(i -> {
            DEFAULT_ACTIVATIONS activationFunctionToUse = activationFunctionsMap.getOrDefault(i, activationFunction);
            try {
                if (i != 0) {
                    double[][] weights = randn(numberOfNeurons, numberOfNeurons);
                    layerList.add(new DenseLayer(weights, layerList.get(i - 1).getLayerOutputs(), activationFunctionToUse));
                } else {
                    layerList.add(new DenseLayer(initialWeights, biases, initialInput[0], activationFunctionToUse));
                }
            } catch (Exception e) {
                throw new RuntimeException("[DenseLayers Constructor] Error During Layer Creation! " + e.getMessage());
            }
        });
    }
    private BaseNeuralNetwork(int layers, int[] numberOfNeuronsPerLayer, DEFAULT_ACTIVATIONS activationFunction, Map<Integer, DEFAULT_ACTIVATIONS> activationFunctionsMap) {
        if(layers != numberOfNeuronsPerLayer.length) {
            throw new IllegalArgumentException(layers + " Layers given but only " + numberOfNeuronsPerLayer.length
                    + " layers described!\nAdjust neuronsPerLayer to be of same length as number of layers!");
        }
        this.layerList = new ArrayList<>();
        this.initialInput = randn(1, numberOfNeuronsPerLayer[0]);
        this.decisionsIndex = new int[initialInput.length];
        this.decisions = new HashMap<>();

        IntStream.range(0, layers).forEach(i -> {
            DEFAULT_ACTIVATIONS activationFunctionToUse = activationFunctionsMap.getOrDefault(i, activationFunction);
            double[][] weights;
            try {
                if (i != 0) {
                    weights = randn(numberOfNeuronsPerLayer[i], numberOfNeuronsPerLayer[i - 1]);
                    layerList.add(new DenseLayer(weights, layerList.get(i - 1).getLayerOutputs(), activationFunctionToUse));
                } else {
                    weights = randn(numberOfNeuronsPerLayer[i], numberOfNeuronsPerLayer[i]);
                    layerList.add(new DenseLayer(weights, initialInput[0], activationFunctionToUse));
                }
            } catch (Exception e) {
                throw new RuntimeException("[DenseLayers Constructor] Error During Layer Creation! " + e.getMessage());
            }
        });
    }
    private BaseNeuralNetwork(int layers, int[] numberOfNeuronsPerLayer, double[][] initialInput, DEFAULT_ACTIVATIONS activationFunction, Map<Integer, DEFAULT_ACTIVATIONS> activationFunctionsMap) {
        if(layers != numberOfNeuronsPerLayer.length) {
            throw new IllegalArgumentException(layers + " Layers given but only " + numberOfNeuronsPerLayer.length
                    + " layers described!\nAdjust neuronsPerLayer to be of same length as number of layers!");
        }
        this.initialInput = initialInput;
        this.layerList = new ArrayList<>();
        this.decisionsIndex = new int[initialInput.length];
        this.decisions = new HashMap<>();

        IntStream.range(0, layers).forEach(i -> {
            DEFAULT_ACTIVATIONS activationFunctionToUse = activationFunctionsMap.getOrDefault(i, activationFunction);
            double[][] weights;
            try {
                if (i != 0) {
                    weights = randn(numberOfNeuronsPerLayer[i], numberOfNeuronsPerLayer[i - 1]);
                    layerList.add(new DenseLayer(weights, layerList.get(i - 1).getLayerOutputs(), activationFunctionToUse));
                } else {
                    weights = randn(numberOfNeuronsPerLayer[i], initialInput[0].length);
                    layerList.add(new DenseLayer(weights, initialInput[0], activationFunctionToUse));
                }
            } catch (Exception e) {
                throw new RuntimeException("[DenseLayers Constructor] Error During Layer Creation! " + e.getMessage());
            }
        });
    }
    private BaseNeuralNetwork(int layers, int[] numberOfNeuronsPerLayer, double[][] initialWeights, double[][] initialInput, DEFAULT_ACTIVATIONS activationFunction, Map<Integer, DEFAULT_ACTIVATIONS> activationFunctionsMap) {
        if(layers != numberOfNeuronsPerLayer.length) {
            throw new IllegalArgumentException(layers + " Layers given but only " + numberOfNeuronsPerLayer.length
                    + " layers described!\nAdjust neuronsPerLayer to be of same length as number of layers!");
        }
        this.layerList = new ArrayList<>();
        this.initialInput = initialInput;
        this.decisionsIndex = new int[initialInput.length];
        this.decisions = new HashMap<>();

        IntStream.range(0, layers).forEach(i -> {
            DEFAULT_ACTIVATIONS activationFunctionToUse = activationFunctionsMap.getOrDefault(i, activationFunction);
            try {
                if (i != 0) {
                    double[][] weights = randn(numberOfNeuronsPerLayer[i], numberOfNeuronsPerLayer[i - 1]);
                    layerList.add(new DenseLayer(weights, layerList.get(i - 1).getLayerOutputs(), activationFunctionToUse));
                } else {
                    layerList.add(new DenseLayer(initialWeights, initialInput[0], activationFunctionToUse));
                }
            } catch (Exception e){
                throw new RuntimeException("[DenseLayers Constructor] Error During Layer Creation! " + e.getMessage());
            }
        });
    }
    private BaseNeuralNetwork(int layers, int[] numberOfNeuronsPerLayer, double[][] initialWeights, double[] initialBiases, double[][] initialInput, DEFAULT_ACTIVATIONS activationFunction, Map<Integer, DEFAULT_ACTIVATIONS> activationFunctionsMap) {
        if(layers != numberOfNeuronsPerLayer.length) {
            throw new IllegalArgumentException(layers + " Layers given but only " + numberOfNeuronsPerLayer.length
                    + " layers described!\nAdjust neuronsPerLayer to be of same length as number of layers!");
        }
        this.layerList = new ArrayList<>();
        this.initialInput = initialInput;
        this.decisionsIndex = new int[initialInput.length];
        this.decisions = new HashMap<>();

        // create first layer by multiplying the initial input by the  initial weights and adding the initial biases
        IntStream.range(0, layers).forEach(i -> {
            DEFAULT_ACTIVATIONS activationFunctionToUse = activationFunctionsMap.getOrDefault(i, activationFunction);
            try {
                if (i != 0) {
                    double[][] weights = randn(numberOfNeuronsPerLayer[i], numberOfNeuronsPerLayer[i - 1]);
                    double[] biases = randn(numberOfNeuronsPerLayer[i]);
                    DenseLayer layer = new DenseLayer(weights, biases, layerList.get(i - 1).getLayerOutputs(), activationFunctionToUse);
                    layerList.add(layer);
                } else {
                    DenseLayer layer = new DenseLayer(initialWeights, initialBiases, initialInput[dataCounter], activationFunctionToUse);
                    layerList.add(layer);
                }
            } catch (Exception e) {
                throw new RuntimeException("[DenseLayers Constructor] Error During Layer Creation! " + e.getMessage());
            }
        });
        printPositive("Construct built!");
    }

    public void train() {
        int epochCounter = 0;
        while(epochCounter < this.epoch) {
            for(int y = 0; y < (this.initialInput.length-1)/this.miniBatchSize; y++) {
                this.backPropagate();
                double tempLoss = 0;
                for(int x = 0; x < this.miniBatchSize; x++) {
                    if(dataCounter >= this.initialInput.length - 1) {
                        break;
                    }
                    forward();
                    tempLoss += generateLoss();
                    this.dataCounter++;
                }
                this.loss = tempLoss / this.miniBatchSize;
                print("Current Epoch ", epochCounter);
                //printLayers("Neural Network", this);
                print("Accuracy:", String.format("%.2f", (this.getAccuracy() * 100)) + "%");
                print("Loss:", String.format("%.2f", this.getLoss()));
                printSubTitle("Epoch Complete: " + epochCounter);
                print("");
            }

            // if not all inputs processed, process remaining
            if((((this.initialInput.length-1) % this.miniBatchSize)) != 0) {
                this.backPropagate();
                double tempLoss = 0;
                for(int x = 0; x < (((this.initialInput.length-1) % this.miniBatchSize)); x++) {
                    if(this.dataCounter >= this.trueValueIndices.length - 1) {
                        break;
                    }
                    tempLoss += this.generateLoss();
                    this.forward();
                    this.dataCounter++;
                }
                this.loss = tempLoss / this.miniBatchSize;
            }
            if(this.learningRateFunction == DEFAULT_LEARNING_RATE.EPOCH_DECAY) {
                this.learningRate = updateLearningRate(this.learningRateFunction, this.learningRate, this.learningRateDecay, epochCounter);
            }

            this.sum = 0;
            this.accuracy = 0;
            this.dataCounter = 0;
            epochCounter++;
        }
        printTitle("Training Complete!");
    }

    private void forward() {
        for(int i = 0; i < this.getLayerList().size(); i ++) {
            if (i != 0) {
                this.getLayerAt(i).setInputs(this.getLayerAt(i-1).getLayerOutputs());
            } else {
                this.getLayerAt(i).setInputs(this.initialInput[dataCounter]);
            }
            this.getLayerAt(i).regenerateLayerOutput();
        }
    }

    private void backPropagate() {
        double[][] gradients = new double[this.layerList.size()][];
        int outputIndex = this.getLayerList().size() - 1;
        DenseLayer outputLayer = this.getLayerAt(outputIndex);
        // loss function derivative
        if(this.trueValueIndices != null) {
            gradients[outputIndex] = ErrorLossDerivatives.derivative(lossFunction,
                    this.getTrueValueIndices()[dataCounter],
                    this.getDecisionsIndex()[dataCounter],
                    this.getLayerAt(outputIndex).getLayerOutputs());
        } else {
            gradients[outputIndex] = ErrorLossDerivatives.derivative(lossFunction,
                    this.getHotOneVec()[dataCounter],
                    this.getDecisionsIndex()[dataCounter],
                    this.getLayerAt(outputIndex).getLayerOutputs());
        }

        // output layer weights and biases
        for (int layerNeuronIterator = 0; layerNeuronIterator < outputLayer.getNeuronList().size(); layerNeuronIterator++) {
            Neuron neuron = this.getNeuronFromLayerAt(outputIndex, layerNeuronIterator);
            // bias update
            double biasUpdate;
            if(this.momentum != 0) {
                biasUpdate = (this.momentum * outputLayer.getBiasMomentums()[layerNeuronIterator]) + (this.learningRate * gradients[outputIndex][layerNeuronIterator]);
                outputLayer.updateBiasMomentum(layerNeuronIterator, biasUpdate);
            } else {
                biasUpdate = (this.learningRate * gradients[outputIndex][layerNeuronIterator]);
            }
            neuron.setBias(neuron.getBias() - biasUpdate);
            // weights update
            double[] inputs = this.getLayerAt(outputIndex - 1).getLayerOutputs();
            for (int neuronWeightIterator = 0; neuronWeightIterator < neuron.getWeights().length; neuronWeightIterator++) {
                double weightUpdate = 0;
                if(this.learningRateFunction == DEFAULT_LEARNING_RATE.MOMENTUM) {
                    weightUpdate = (this.momentum * outputLayer.getWeightMomentums()[layerNeuronIterator][neuronWeightIterator]) + (this.learningRate * gradients[outputIndex][layerNeuronIterator]);
                    outputLayer.updateWeightMomentum(layerNeuronIterator, neuronWeightIterator, weightUpdate);
                } else if(this.learningRateFunction == DEFAULT_LEARNING_RATE.ADAGRAD) {
                    // increment weight cache by (weight cache squared)
                   outputLayer.getWeightsCache()[neuronWeightIterator] =
                           MathUtils.addArrays(
                                   MathUtils.addArrays(
                                           outputLayer.getWeightsCache()[neuronWeightIterator],
                                           gradients[outputIndex]),
                                   outputLayer.getWeightsCache()[neuronWeightIterator]);
                } else if(this.learningRateFunction == DEFAULT_LEARNING_RATE.RMSPROP) {

                } else if(this.learningRateFunction == DEFAULT_LEARNING_RATE.ADAM) {

                }
                else {
                    weightUpdate = (this.learningRate * gradients[outputIndex][layerNeuronIterator]);
                }

                double deltaWeight = weightUpdate * inputs[neuronWeightIterator];
                neuron.setWeight(neuronWeightIterator, neuron.getWeights()[neuronWeightIterator] + deltaWeight);
            }
        }

        // hidden layers
        for (int index = outputIndex - 1; index >= 0; index--) {
            DenseLayer currentLayer = this.getLayerAt(index+1);
            // create gradient for current layer
            gradients[index] = new double[this.getLayerAt(index).getNeuronList().size()];
            double[] derivatives = ActivationDerivatives.derivative(
                    this.getLayerAt(index).getActivationFunction(),
                    this.getLayerAt(index).getLayerOutputs()
            );
            for (int j = 0; j < gradients[index].length; j++) {
                double gradientSum = 0;
                for (int k = 0; k < currentLayer.getNeuronList().size(); k++) {
                    gradientSum += gradients[index + 1][k] * currentLayer.getNeuronList().get(k).getWeights()[j];
                }
                gradients[index][j] = gradientSum * derivatives[j];
            }

            // update each neuron
            for (int layerNeuronIterator= 0; layerNeuronIterator< this.getLayerAt(index).getNeuronList().size(); layerNeuronIterator++) {
                Neuron neuron = this.getNeuronFromLayerAt(index, layerNeuronIterator);
                double biasUpdate;
                if(this.momentum != 0) {
                    biasUpdate = (this.momentum * this.getLayerAt(index).getBiasMomentums()[layerNeuronIterator])  + (this.learningRate * gradients[index][layerNeuronIterator]);
                    this.getLayerAt(index).updateBiasMomentum(layerNeuronIterator, biasUpdate);
                } else {
                    biasUpdate = (this.learningRate * gradients[index][layerNeuronIterator]);
                }
                neuron.setBias(neuron.getBias() - biasUpdate );

                double[] inputs = (index == 0) ? this.getInitialInput()[dataCounter] : this.getLayerAt(index - 1).getLayerOutputs();
                for (int neuronWeightIterator = 0; neuronWeightIterator < neuron.getWeights().length; neuronWeightIterator++) {
                    double weightUpdate;
                    if(this.momentum != 0) {
                        weightUpdate = (this.momentum * this.getLayerAt(index).getWeightMomentums()[layerNeuronIterator][neuronWeightIterator]) + (this.learningRate * gradients[index][layerNeuronIterator]);
                        this.getLayerAt(index).updateWeightMomentum(layerNeuronIterator, neuronWeightIterator, weightUpdate);
                    } else {
                        weightUpdate = (this.learningRate * gradients[index][layerNeuronIterator]);
                    }
                    double deltaWeight = this.learningRate * gradients[index][layerNeuronIterator] * inputs[neuronWeightIterator];
                    neuron.setWeight(neuronWeightIterator, neuron.getWeights()[neuronWeightIterator] - deltaWeight);
                }
            }
        }
    }
    private double generateLoss(){
        double loss = 0;
        if(this.hotOneVec != null && this.trueValueIndices != null) { throw new RuntimeException("Cannot initialize both Hot-One-Vector and a True-Value! You must select one method of error/loss checking!"); }
        // check if both the one hot vector mappings true values mappings are empty
        if(this.hotOneVec == null && this.trueValueIndices == null) { throw new RuntimeException("Must initialize either Hot-One-Vector or a True-Value!"); }

        if(this.hotOneVec != null) {
            int hotValueIndex = 0;
            if(this.hotOneVec.length <= this.dataCounter) { throw new RuntimeException("Too few true values for the number of inputs given!"); }
            for(int i = 0; i < this.hotOneVec.length; i++) {
                if(this.hotOneVec[this.dataCounter][i] != 0) {
                    hotValueIndex = i;
                    break;
                }
            }
            this.generateDecisionsMap(hotValueIndex);
            loss = ErrorLossFunctions.lossFunction(lossFunction, hotValueIndex, this.getDecisionsIndex()[this.dataCounter], this.getLastLayer().getLayerOutputs());
        } else {
            if(trueValueIndices.length <= this.dataCounter) { throw new RuntimeException("Too few true values for the number of inputs given!"); }
            this.generateDecisionsMap(trueValueIndices[this.dataCounter]);
            loss = ErrorLossFunctions.lossFunction(lossFunction, this.trueValueIndices[this.dataCounter],  this.getDecisionsIndex()[this.dataCounter], this.getLastLayer().getLayerOutputs());
        }
        return loss;
    }
    private double accuracy(int trueIndex, int predictedIndex) {
        if(trueIndex == predictedIndex) { this.sum+=1; }
        this.accuracy = sum / (this.dataCounter + EPSILON);
        return this.accuracy;
    }
    private void generateDecisionsMap(int trueValueIndex) {
        // go through and get the decision made at the output layer
        double[] indexAndMax = decision(this.getLastLayer().getLayerOutputs());
        // store the index of that decision for each cycle of training
        this.decisionsIndex[dataCounter] = (int)indexAndMax[0];
        // store the actual value of the decision as well
        this.decisions.put(dataCounter, indexAndMax[1]);
        // determine the accuracy of the decision
        this.accuracy = accuracy(trueValueIndex, this.decisionsIndex[dataCounter]);
    }
    private void generateDecisionsMap(int[] hotOneVec) {
        double[] indexAndMax = decision(this.getLastLayer().getLayerOutputs());
        this.decisionsIndex[dataCounter] = (int)indexAndMax[0];
        this.decisions.put(dataCounter, indexAndMax[1]);
        this.accuracy = accuracy(getHotOneVecIndexValue(this.hotOneVec[dataCounter]), this.decisionsIndex[dataCounter]);
    }
    private double[] batchDecisions() {
        double[] decisions = new double[this.getLastLayer().getLayerOutputs().length];
        for(int i = 0; i < decisions.length; i++) { decisions[i] = decision(this.getLastLayer().getLayerOutputs())[1]; }
        return decisions;
    }
    private double[] decision(double[] values) {
        return MathUtils.indexAndMax(values);
    }

    // helper functions
    private static double[][] randn(int rows, int cols) {
        return getRandomMatrix(rows, cols);
    }
    private static double[] randn(int elementCount) {
        return getRandomArray(elementCount);
    }
    private static double[][] getRandomMatrix(int rows, int cols) {
        double[][] output = new double[rows][cols];
        IntStream.range(0, output.length).parallel().forEachOrdered(i -> {
            for (int j = 0; j < output[0].length; j++) {
                output[i][j] = (0.1 * random.nextGaussian());
            }
        });
        return output;
    }
    private static double[] getRandomArray(int elementCount) {
        double[] output = new double[elementCount];
        for(int i = 0; i < output.length; i++) {
            output[i] = (0.1 * random.nextGaussian());
        }
        return output;
    }
    public ArrayList<Neuron> getNeuronsFromLayerAt(int index) {
        return this.getLayerList().get(index).getNeuronList();
    }
    public Neuron getNeuronFromLayerAt(int layerIndex, int neuronIndex) {
        return this.getLayerList().get(layerIndex).getNeuronList().get(neuronIndex);
    }
    public double[] getOutputs() {
        return getLayerAt(getLayerList().size()-1).getLayerOutputs();
    }
    public DEFAULT_ACTIVATIONS getActivationFunctionFrom(int index) {
        return this.getLayerAt(index).getActivationFunction();
    }
    public DenseLayer getLayerAt(int index) {
        return this.getLayerList().get(index);
    }
    public DenseLayer getFirstLayer() {
        return this.getLayerAt(0);
    }
    public DenseLayer getLastLayer() {
        return this.getLayerAt(this.getLayerList().size()-1);
    }
    public static void appendNumberToFile(String filePath, String number) {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(filePath, true))) {
            writer.write(String.valueOf(number));
            writer.newLine();  // This will add a new line after writing the number
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void printLogs(int y) {
        printLayers("Middle Layer", this);
        print("Accuracy:", String.format("%.2f", (this.getAccuracy() * 100)) + "%");
        print("Loss:", String.format("%.2f", this.getLoss()));
        printSubTitle("Iteration Complete: " + y);
        print("");
    }

    // builder class for the neural network
    @NoArgsConstructor
    public static class DenseLayersBuilder {
        private ArrayList<DenseLayer> layerList;
        private boolean isUsingListOfLayers = true;

        private DEFAULT_ACTIVATIONS activationFunction = DEFAULT_ACTIVATIONS.SOFTMAX_ACTIVATION_FUNCTION;
        private final Map<Integer, DEFAULT_ACTIVATIONS> activationFunctionsMap = new HashMap<>();
        private boolean hasSetSpecificLayerActivationFunctions = false;

        private DEFAULT_LOSSES lossFunction = DEFAULT_LOSSES.NLL_LOSS_FUNCTION;
        private int[][] hotOneVec;
        private int[] trueValueIndices;
        private double[] trueValues;
        private double[][][] trueValuesMatrix;
        private final Map<Integer, DEFAULT_LOSSES> lossFunctionsMap = new HashMap<>();

        private boolean hasSetSpecificLayerLossFunctions = false;
        private boolean isUsingTrueValueIndex = false;
        private boolean isUsingHotEncodedVec = false;
        private boolean isUsingTrueValues = false;

        private int numberOfLayers = 1;
        private boolean isUsingNumberOfLayers = false;

        private int numberOfInputs;
        private double[][] initialInput;
        private double[][][] initialMatrixInput;
        private boolean isUsingStochasticGradientDescent = true;
        private boolean isUsingBatchInputs = false;
        private int miniBatchSize = 1;
        private int epoch = 1;

        private double[][] initialWeights;
        private boolean isUsingSpecificWeights = false;

        private double[] initialBiases;
        private boolean isUsingSpecificBiases = false;
        private boolean isSettingSpecificBiases = false;

        private int[] numberOfNeuronsPerLayer;
        private boolean isUsingSpecificNeurons = false;
        private int numberOfNeurons;
        private boolean isUsingNumberOfNeurons = false;
        private boolean isUsingFileAsInput = false;

        double learningRate = 0.01;
        double learningRateDecay = 0;
        DEFAULT_LEARNING_RATE learningRateFunction = DEFAULT_LEARNING_RATE.EPOCH_DECAY;
        double momentum = 0.0;

        double alpha = 1.0;
        double gamma = 1.0;
        double delta = 1.0;
        double margin = 1.0;

        public DenseLayersBuilder withTextFileAsInput(String filePath, String delimiter) {
            this.isUsingFileAsInput = true;
            this.isUsingBatchInputs = false;
            char char1 = filePath.charAt(filePath.length()-1);
            char char2 = filePath.charAt(filePath.length()-2);
            char char3 = filePath.charAt(filePath.length()-3);
            char char4 = filePath.charAt(filePath.length()-4);
            if (char1 == 't' && char2 == 'x' && char3 == 't' && char4 == '.') {
                this.initialInput = (double[][])FileUtils.readTextFile(filePath, delimiter);
            } else if (char1 == 'v' && char2 == 's' && char3 == 'c' && char4 == '.') {
                this.initialInput = FileUtils.readCsvFile(filePath);
            } else {
                throw new IllegalArgumentException("Error: file must be of type txt or csv");
            }
            return this;
        }

        public DenseLayersBuilder withLayerList(ArrayList<DenseLayer> layerList) {
            this.isUsingListOfLayers = true;
            this.layerList = layerList;
            return this;
        }

        public DenseLayersBuilder withNumberOfLayers(int numberOfLayers) {
            this.isUsingNumberOfLayers = true;
            this.numberOfLayers = numberOfLayers;
            return this;
        }

        public DenseLayersBuilder withNumberOfInputs(int numberOfInputs) {
            this.numberOfInputs = numberOfInputs;
            return this;
        }

        public DenseLayersBuilder withInitialInput(double[][] initialInput) {
            this.isUsingFileAsInput = false;
            this.isUsingBatchInputs = true;
            this.initialInput = initialInput;
            return this;
        }

        public DenseLayersBuilder withInitialWeights(double[][] initialWeights) {
            this.isUsingSpecificWeights = true;
            this.initialWeights = initialWeights;
            return this;
        }

        public DenseLayersBuilder withSpecificBiasForNeuronInLayer(int layer, int neuron, double bias) {
            this.isSettingSpecificBiases = true;
            this.layerList.get(layer).getNeuronList().get(neuron).setBias(bias);
            return this;
        }

        public DenseLayersBuilder withInitialBiases(double[] initialBiases) {
            this.isUsingSpecificBiases = true;
            this.initialBiases = initialBiases;
            return this;
        }

        public DenseLayersBuilder withNumberOfNeuronsPerLayer(int[] numberOfNeuronsPerLayer) {
            this.isUsingSpecificNeurons = true;
            this.isUsingNumberOfNeurons = false;
            this.numberOfNeuronsPerLayer = numberOfNeuronsPerLayer;
            return this;
        }

        public DenseLayersBuilder withNumberOfNeurons(int numberOfNeurons) {
            this.isUsingNumberOfNeurons = true;
            this.isUsingSpecificNeurons = false;
            this.numberOfNeurons = numberOfNeurons;
            return this;
        }

        public DenseLayersBuilder withActivationFunction(DEFAULT_ACTIVATIONS activationFunction) {
            this.activationFunction = activationFunction;
            return this;
        }

        public DenseLayersBuilder withActivationFunctionForSingleLayer(int layer, DEFAULT_ACTIVATIONS activationFunction) {
            activationFunctionsMap.put(layer, activationFunction);
            return this;
        }

        public DenseLayersBuilder withActivationFunctionForMultipleLayers(int startingLayer, int endingLayer, DEFAULT_ACTIVATIONS activationFunction) {
            IntStream.range(startingLayer, endingLayer+1).forEachOrdered( i -> activationFunctionsMap.put(i, activationFunction));
            return this;
        }

        public DenseLayersBuilder withActivationFunctionForOutput(DEFAULT_ACTIVATIONS activationFunction) {
            if(numberOfLayers > 1){
                activationFunctionsMap.put(numberOfLayers-1, activationFunction);
            }
            return this;
        }

        public DenseLayersBuilder withLearningRate(double learningRate) {
            this.learningRate = learningRate;
            return this;
        }

        public DenseLayersBuilder withLearningRateDecay(double learningRateDecay) {
            this.learningRateDecay = learningRateDecay;
            return this;
        }

        public DenseLayersBuilder withLearningRateFunction(DEFAULT_LEARNING_RATE learningRateFunction) {
            this.learningRateFunction = learningRateFunction;
            return this;
        }

        public DenseLayersBuilder withLearningRateFunction(DEFAULT_LEARNING_RATE learningRateFunction, double learningRateDecay) {
            this.learningRateFunction = learningRateFunction;
            if(learningRateFunction != DEFAULT_LEARNING_RATE.EPOCH_DECAY ||
                    learningRateFunction != DEFAULT_LEARNING_RATE.STEP_DECAY ||
                    learningRateFunction != DEFAULT_LEARNING_RATE.EXPONENTIAL_DECAY ){
                throw new RuntimeException("Learning rate decay is not applicable with + " + learningRateFunction.name());
            }
            this.learningRateDecay = learningRateDecay;
            return this;
        }

        public DenseLayersBuilder withMomentum(double momentum) {
            if(momentum < 0 || momentum > 1){
                throw new RuntimeException("Momentum must be between 0 and 1!");
            }
            this.momentum = momentum;
            return this;
        }

        public DenseLayersBuilder withTrueValueIndices(int[] trueValueIndices) {
            this.isUsingTrueValueIndex = true;
            this.isUsingHotEncodedVec = false;
            this.trueValueIndices = trueValueIndices;
            return this;
        }

        public DenseLayersBuilder withTextFileAsTrueValueIndices(String filePath, String delimiter) {
            this.isUsingFileAsInput = true;
            char char1 = filePath.charAt(filePath.length()-1);
            char char2 = filePath.charAt(filePath.length()-2);
            char char3 = filePath.charAt(filePath.length()-3);
            char char4 = filePath.charAt(filePath.length()-4);
            if (char1 == 't' && char2 == 'x' && char3 == 't' && char4 == '.'){
                this.trueValueIndices = (int[])FileUtils.readTextFile(filePath, delimiter);

            } else if (char1 == 'v' && char2 == 's' && char3 == 'c' && char4 == '.'){
                this.initialInput = FileUtils.readCsvFile(filePath);
            } else {
                throw new IllegalArgumentException("Error: file must be of type txt or csv");
            }
            return this;
        }

        public DenseLayersBuilder withTrueValues(double[] trueValues) {
            this.isUsingTrueValues = true;
            this.trueValues = trueValues;
            return this;
        }

        public DenseLayersBuilder withTrueValues(double[][][] trueValues) {
            this.isUsingTrueValues = true;
            this.trueValuesMatrix = trueValues;
            return this;
        }

        public DenseLayersBuilder withHotOneVector(int[][] hotOneVec) {
            this.isUsingTrueValueIndex = false;
            this.isUsingHotEncodedVec = true;
            this.hotOneVec = hotOneVec;
            return this;
        }

        public DenseLayersBuilder withLossFunction(DEFAULT_LOSSES lossFunction) {
            this.lossFunction = lossFunction;
            return this;
        }

        public DenseLayersBuilder withAlpha(double alpha) {
            this.alpha = alpha;
            return this;
        }

        public DenseLayersBuilder withGamma(double gamma) {
            this.gamma = gamma;
            return this;
        }

        public DenseLayersBuilder withDelta(double delta) {
            this.delta = delta;
            return this;
        }

        public DenseLayersBuilder withMargin(double margin) {
            this.margin = margin;
            return this;
        }

        public DenseLayersBuilder withLossFunctionAndHotEncodedVectors(DEFAULT_LOSSES lossFunction, int[][] hotOneOutput) {
            this.lossFunction = lossFunction;
            this.hotOneVec = hotOneOutput;
            return this;
        }

        public DenseLayersBuilder withLossFunctionAndTrueValues(DEFAULT_LOSSES lossFunction, int[] trueValueIndices) {
            this.lossFunction = lossFunction;
            this.trueValueIndices = trueValueIndices;
            return this;
        }

        public DenseLayersBuilder withEpoch(int epoch) {
            this.epoch = epoch;
            return this;
        }

        public DenseLayersBuilder withMiniBatchProcessing(int miniBatchSize){
            if(miniBatchSize < 1) {
                throw new RuntimeException("Mini-batch size must be greater than 0!");
            }
            this.miniBatchSize = miniBatchSize;
            return this;
        }

        public DenseLayersBuilder withFullBatchProcessing(){
            this.isUsingStochasticGradientDescent = false;
            this.isUsingBatchInputs = true;
            return this;
        }

        public BaseNeuralNetwork build() {
            BaseNeuralNetwork built = null;
            if(!this.isUsingNumberOfLayers && !this.isUsingListOfLayers) { throw new IllegalArgumentException("Please use the numberOfLayers() builder method to initialize, or provide an ArrayList<DenseLayer> using the withLayerList() builder method!"); }

            if(!this.isUsingSpecificNeurons && !this.isUsingNumberOfNeurons && this.isUsingNumberOfLayers && !this.isUsingBatchInputs) {
                if(this.isUsingFileAsInput && this.isUsingSpecificWeights && this.isUsingSpecificBiases) {
                    printTitle("Using construct 1.1");
                    built = new BaseNeuralNetwork(this.numberOfLayers, this.initialWeights, this.initialBiases, this.initialInput, this.activationFunction, this.activationFunctionsMap);
                }
                else if(this.isUsingFileAsInput && this.isUsingSpecificWeights) {
                    printTitle("Using construct 1.2");
                    built = new BaseNeuralNetwork(this.numberOfLayers, this.initialWeights, this.initialInput, this.activationFunction, this.activationFunctionsMap);
                }
                else if(this.isUsingFileAsInput) {
                    printTitle("Using construct 1.3");
                    built = new BaseNeuralNetwork(this.numberOfLayers, this.initialInput, this.activationFunction, this.activationFunctionsMap);
                }
                else if(!this.isUsingSpecificWeights && !this.isUsingSpecificBiases) {
                    printTitle("Using construct 1.4");
                    built = new BaseNeuralNetwork(this.numberOfLayers, this.activationFunction, this.activationFunctionsMap);
                }
            }
            else if(this.isUsingNumberOfNeurons && this.isUsingNumberOfLayers && !this.isUsingBatchInputs) {
                if(this.isUsingSpecificWeights && this.isUsingSpecificBiases) {
                    printTitle("Using construct 2.1");
                    built = new BaseNeuralNetwork(this.numberOfLayers, this.numberOfNeurons, this.initialWeights, this.initialBiases, this.initialInput, this.activationFunction, this.activationFunctionsMap);
                }
                else if(this.isUsingFileAsInput  && this.isUsingSpecificWeights) {
                    printTitle("Using construct 2.2");
                    built = new BaseNeuralNetwork(this.numberOfLayers, this.numberOfNeurons, this.initialWeights, this.initialInput, this.activationFunction, this.activationFunctionsMap);
                }
                else if(this.isUsingFileAsInput) {
                    printTitle("Using construct 2.3");
                    built = new BaseNeuralNetwork(this.numberOfLayers, this.numberOfNeurons, this.initialInput, this.activationFunction, this.activationFunctionsMap);
                }
                else if(!this.isUsingFileAsInput && !this.isUsingSpecificWeights && !this.isUsingSpecificBiases) {
                    printTitle("Using construct 2.4");
                    built = new BaseNeuralNetwork(this.numberOfLayers, this.numberOfNeurons, this.activationFunction, this.activationFunctionsMap);
                }
            }
            else if(this.isUsingSpecificNeurons && this.isUsingNumberOfLayers && !this.isUsingBatchInputs) {
                if(this.isUsingFileAsInput && this.isUsingSpecificWeights && this.isUsingSpecificBiases) {
                    printTitle("Using construct 3.1");
                    built = new BaseNeuralNetwork(this.numberOfLayers, this.numberOfNeuronsPerLayer, this.initialWeights, this.initialBiases, this.initialInput, this.activationFunction, this.activationFunctionsMap);
                }
                else if(this.isUsingFileAsInput && this.isUsingSpecificWeights) {
                    printTitle("Using construct 3.2");
                    built = new BaseNeuralNetwork(this.numberOfLayers, this.numberOfNeuronsPerLayer, this.initialWeights, this.initialInput, this.activationFunction, this.activationFunctionsMap);
                }
                else if(this.isUsingFileAsInput) {
                    printTitle("Using construct 3.3");
                    built = new BaseNeuralNetwork(this.numberOfLayers, this.numberOfNeuronsPerLayer, this.initialInput, this.activationFunction, this.activationFunctionsMap);
                }
                else if(!this.isUsingFileAsInput && !this.isUsingSpecificWeights && !this.isUsingSpecificBiases) {
                    printTitle("Using construct 3.4");
                    built = new BaseNeuralNetwork(this.numberOfLayers, this.numberOfNeuronsPerLayer, this.activationFunction, this.activationFunctionsMap);
                }
            }
            // neurons are not being specified, file input is not being used
            else if(!this.isUsingSpecificNeurons && !this.isUsingNumberOfNeurons && this.isUsingNumberOfLayers && !this.isUsingFileAsInput) {
                if(this.isUsingBatchInputs && this.isUsingSpecificWeights && this.isUsingSpecificBiases) {
                    printTitle("Using construct 4.1");
                    built = new BaseNeuralNetwork(this.numberOfLayers, this.initialWeights, this.initialBiases, this.initialInput, this.activationFunction, this.activationFunctionsMap);
                }
                else if(this.isUsingBatchInputs  && this.isUsingSpecificWeights) {
                    printTitle("Using construct 4.2");
                    built = new BaseNeuralNetwork(this.numberOfLayers, this.initialWeights, this.initialInput, this.activationFunction, this.activationFunctionsMap);
                }
                else if(this.isUsingBatchInputs) {
                    printTitle("Using construct 4.3");
                    built = new BaseNeuralNetwork(this.numberOfLayers, this.initialInput, this.activationFunction, this.activationFunctionsMap);
                }
                else if(!this.isUsingBatchInputs && !this.isUsingSpecificWeights && !this.isUsingSpecificBiases) {
                    printTitle("Using construct 4.4");
                    built = new BaseNeuralNetwork(this.numberOfLayers, this.activationFunction, this.activationFunctionsMap);
                }
            }
            else if(this.isUsingNumberOfNeurons && this.isUsingNumberOfLayers && !this.isUsingFileAsInput) {
                if(this.isUsingBatchInputs && this.isUsingSpecificWeights && this.isUsingSpecificBiases) {
                    printTitle("Using construct 5.1");
                    built = new BaseNeuralNetwork(this.numberOfLayers, this.numberOfNeurons, this.initialWeights, this.initialBiases, this.initialInput, this.activationFunction, this.activationFunctionsMap);
                }
                else if(this.isUsingBatchInputs  && this.isUsingSpecificWeights) {
                    printTitle("Using construct 5.2");
                    built = new BaseNeuralNetwork(this.numberOfLayers, this.numberOfNeurons, this.initialWeights, this.initialInput, this.activationFunction, this.activationFunctionsMap);
                }
                else if(this.isUsingBatchInputs) {
                    printTitle("Using construct 5.3");
                    built = new BaseNeuralNetwork(this.numberOfLayers, this.numberOfNeurons, this.initialInput, this.activationFunction, this.activationFunctionsMap);
                }
                else if(!this.isUsingBatchInputs && !this.isUsingSpecificWeights && !this.isUsingSpecificBiases) {
                    printTitle("Using construct 5.4");
                    built = new BaseNeuralNetwork(this.numberOfLayers, this.numberOfNeurons, this.activationFunction, this.activationFunctionsMap);
                }
            }
            else if(this.isUsingSpecificNeurons && this.isUsingNumberOfLayers && !this.isUsingFileAsInput) {
                if(this.isUsingBatchInputs && this.isUsingSpecificWeights && this.isUsingSpecificBiases) {
                    printTitle("Using construct 6.1");
                    built = new BaseNeuralNetwork(this.numberOfLayers, this.numberOfNeuronsPerLayer, this.initialWeights, this.initialBiases, this.initialInput, this.activationFunction, this.activationFunctionsMap);
                }
                else if(this.isUsingBatchInputs  && this.isUsingSpecificWeights) {
                    printTitle("Using construct 6.2");
                    built = new BaseNeuralNetwork(this.numberOfLayers, this.numberOfNeuronsPerLayer, this.initialWeights, this.initialInput, this.activationFunction, this.activationFunctionsMap);
                }
                else if(this.isUsingBatchInputs) {
                    printTitle("Using construct 6.3");
                    built = new BaseNeuralNetwork(this.numberOfLayers, this.numberOfNeuronsPerLayer, this.initialInput, this.activationFunction, this.activationFunctionsMap);
                }
                else if(!this.isUsingBatchInputs && !this.isUsingSpecificWeights && !this.isUsingSpecificBiases) {
                    printTitle("Using construct 6.4");
                    built = new BaseNeuralNetwork(this.numberOfLayers, this.numberOfNeuronsPerLayer, this.activationFunction, this.activationFunctionsMap);
                }
            }
            else {
                printTitle("Using construct 7");
                built = new BaseNeuralNetwork(2, DEFAULT_ACTIVATIONS.LINEAR_ACTIVATION_FUNCTION, this.activationFunctionsMap);
                if(this.isUsingBatchInputs && this.isUsingFileAsInput) {
                    throw new RuntimeException(bold(red("Builder Not Configured Properly! Do Not Use File As Input and Batch Input Together!")));
                }
                throw new RuntimeException(bold(red("Builder Not Configured Properly!")));
            }
            assert built != null;

            built.learningRate = this.learningRate;
            built.learningRateDecay = this.learningRateDecay;
            built.learningRateFunction = this.learningRateFunction;
            built.momentum = momentum;
            built.loss = 0;
            built. accuracy = 0;
            // create momentum arrays for each layer
            if(learningRateFunction == DEFAULT_LEARNING_RATE.MOMENTUM) {
                for (int i = 0; i < built.getLayerList().size(); i++) {
                    built.getLayerAt(i).setWeightMomentums(new double[built.getLayerAt(i).getNeuronList().size()][built.getLayerAt(i).getNeuronList().get(0).getWeights().length]);
                    built.getLayerAt(i).setBiasMomentums(new double[built.getLayerAt(i).getNeuronList().size()]);
                }
            } else if (learningRateFunction == DEFAULT_LEARNING_RATE.ADAGRAD) {
                for (int i = 0; i < built.getLayerList().size(); i++) {
                    built.getLayerAt(i).setWeightsCache(new double[built.getLayerAt(i).getNeuronList().size()][built.getLayerAt(i).getNeuronList().get(0).getWeights().length]);
                    built.getLayerAt(i).setBiasCache(new double[built.getLayerAt(i).getNeuronList().size()]);
                }
            }
            built.lossFunction = this.lossFunction;
            built.miniBatchSize = this.miniBatchSize;
            built.epoch = this.epoch;
            if(isUsingTrueValueIndex) {
                built.trueValueIndices = this.trueValueIndices;
                printPositive("True Values Set!");
            } else {
                built.hotOneVec = this.hotOneVec;
                printPositive("Hot Vector Set!");
            }
            if(built.miniBatchSize > built.initialInput.length){
                throw new RuntimeException("The miniBatchSize exceeds the provided data!");
            }
            ALPHA = this.alpha;
            GAMMA = this.gamma;
            DELTA = this.delta;
            MARGIN = this.margin;
            if(built.hotOneVec != null && built.trueValueIndices != null) { throw new RuntimeException("Cannot initialize both Hot-One-Vector and a True-Value! You must select one method of error/loss checking!"); }
            // check if both the one hot vector mappings true values mappings are empty
            if(built.hotOneVec == null && trueValueIndices == null) { throw new RuntimeException("Must initialize either Hot-One-Vector or a True-Value!"); }
            printPositive("Neural network successfully built!");
            printPositive("Created dense layer neural network: " + built);
            printPositive("Number of layers set to " + this.numberOfLayers);
            printPositive("Initial activation function set to " + this.activationFunction.name());
            printPositive("Loss function set to " + this.lossFunction.name());
            printPositive("Learning rate set to " + built.learningRate);
            printPositive("Learning rate decay set to " + built.learningRateDecay);
            printPositive("Batch size set to " + built.miniBatchSize);
            printPositive("Epochs set to " + built.epoch);

            return built;
        }
    }
}
