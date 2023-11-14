package com.craftsentient.craftmind.layers;

import com.craftsentient.craftmind.activation.DEFAULT_ACTIVATION_FUNCTIONS;
import com.craftsentient.craftmind.derivitives.activationDerivatives.ActivationDerivatives;
import com.craftsentient.craftmind.derivitives.errorLossDerivatives.ErrorLossDerivatives;
import com.craftsentient.craftmind.errorLoss.DEFAULT_LOSS_FUNCTIONS;
import com.craftsentient.craftmind.errorLoss.ErrorLossFunctions;
import com.craftsentient.craftmind.layer.DenseLayer;
import com.craftsentient.craftmind.layer.Layer;
import com.craftsentient.craftmind.utils.FileUtils;
import com.craftsentient.craftmind.utils.MathUtils;
import com.craftsentient.craftmind.neuron.Neuron;
import lombok.Getter;
import lombok.NoArgsConstructor;

import java.util.*;
import java.util.stream.IntStream;

import static com.craftsentient.craftmind.utils.MathUtils.getHotOneVecIndexValue;
import static com.craftsentient.craftmind.utils.PrintUtils.*;
import static java.lang.Thread.sleep;

@Getter
public class DenseLayers {
    private final ArrayList<DenseLayer> layerList;
    private final double[][] initialInput;
    private int dataCounter = 0;
    private int batchCounter = 0;
    private int miniBatchSize = 1;
    private final Map<Integer, Double> decisions;
    private final int[] decisionsIndex;
    private int[] trueValueIndices;
    private double accuracy;
    private double loss;
    private double sum;
    private double learningRate = 1.0;
    private double learningRateDecay = 1.0;
    private int[][] hotOneVec;
    public static double ALPHA = 1.0;
    public static double GAMMA = 1.0;
    public static double DELTA = 1.0;
    public static double MARGIN = 1.0;
    private DEFAULT_LOSS_FUNCTIONS lossFunction;
    public static final Random random = new Random(0);

    private DenseLayers(int layers, DEFAULT_ACTIVATION_FUNCTIONS activationFunction, Map<Integer, DEFAULT_ACTIVATION_FUNCTIONS> activationFunctionsMap) {
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
            DEFAULT_ACTIVATION_FUNCTIONS activationFunctionToUse = activationFunctionsMap.getOrDefault(i, activationFunction);
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
    private DenseLayers(int layers, double[][] initialInput, DEFAULT_ACTIVATION_FUNCTIONS activationFunction, Map<Integer, DEFAULT_ACTIVATION_FUNCTIONS> activationFunctionsMap) {
        this.initialInput = initialInput;
        this.layerList = new ArrayList<>();
        this.decisionsIndex = new int[initialInput.length];
        this.decisions = new HashMap<>();
        IntStream.range(0, layers).forEach(i -> {
            DEFAULT_ACTIVATION_FUNCTIONS activationFunctionToUse = activationFunctionsMap.getOrDefault(i, activationFunction);
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
    private DenseLayers(int layers, double[][] initialWeights, double[][] initialInput, DEFAULT_ACTIVATION_FUNCTIONS activationFunction, Map<Integer, DEFAULT_ACTIVATION_FUNCTIONS> activationFunctionsMap) {
        this.initialInput = initialInput;
        this.layerList = new ArrayList<>();
        // instantiate the decisions index
        this.decisionsIndex = new int[initialInput.length];
        // instantiate the hash map for the values of the decision
        this.decisions = new HashMap<>();

        IntStream.range(0, layers).forEach(i -> {
            DEFAULT_ACTIVATION_FUNCTIONS activationFunctionToUse = activationFunctionsMap.getOrDefault(i, activationFunction);
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
    private DenseLayers(int layers, double[][] initialWeights, double[] biases, double[][] initialInput, DEFAULT_ACTIVATION_FUNCTIONS activationFunction, Map<Integer, DEFAULT_ACTIVATION_FUNCTIONS> activationFunctionsMap) {
        this.layerList = new ArrayList<>();
        this.initialInput = initialInput;
        // instantiate the decisions index
        this.decisionsIndex = new int[initialInput.length];
        // instantiate the hash map for the values of the decision
        this.decisions = new HashMap<>();

        IntStream.range(0, layers).forEach(i -> {
            DEFAULT_ACTIVATION_FUNCTIONS activationFunctionToUse = activationFunctionsMap.getOrDefault(i, activationFunction);
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
    private DenseLayers(int layers, int numberOfNeurons, DEFAULT_ACTIVATION_FUNCTIONS activationFunction, Map<Integer, DEFAULT_ACTIVATION_FUNCTIONS> activationFunctionsMap) {
        this.layerList = new ArrayList<>();
        this.initialInput = randn(1,numberOfNeurons);
        // instantiate the decisions index
        this.decisionsIndex = new int[initialInput.length];
        // instantiate the hash map for the values of the decision
        this.decisions = new HashMap<>();

        IntStream.range(0, layers).forEach(i -> {
            DEFAULT_ACTIVATION_FUNCTIONS activationFunctionToUse = activationFunctionsMap.getOrDefault(i, activationFunction);
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
    private DenseLayers(int layers, int numberOfNeurons, double[][] initialInput, DEFAULT_ACTIVATION_FUNCTIONS activationFunction, Map<Integer, DEFAULT_ACTIVATION_FUNCTIONS> activationFunctionsMap) {
        if(numberOfNeurons != initialInput.length) { throw new IllegalArgumentException("neuronsPerLayer of " + numberOfNeurons + " and initialInput size of " + initialInput.length + " do not match!");}
        this.initialInput = initialInput;
        this.layerList = new ArrayList<>();
        // instantiate the decisions index
        this.decisionsIndex = new int[initialInput.length];
        // instantiate the hash map for the values of the decision
        this.decisions = new HashMap<>();

        IntStream.range(0, layers).forEach(i -> {
            DEFAULT_ACTIVATION_FUNCTIONS activationFunctionToUse = activationFunctionsMap.getOrDefault(i, activationFunction);
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
    private DenseLayers(int layers, int numberOfNeurons, double[][] initialWeights, double[][] initialInput, DEFAULT_ACTIVATION_FUNCTIONS activationFunction, Map<Integer, DEFAULT_ACTIVATION_FUNCTIONS> activationFunctionsMap) {
        this.initialInput = initialInput;
        this.layerList = new ArrayList<>();
        // instantiate the decisions index
        this.decisionsIndex = new int[initialInput.length];
        // instantiate the hash map for the values of the decision
        this.decisions = new HashMap<>();

        IntStream.range(0, layers).forEach(i -> {
            DEFAULT_ACTIVATION_FUNCTIONS activationFunctionToUse = activationFunctionsMap.getOrDefault(i, activationFunction);
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
    private DenseLayers(int layers, int numberOfNeurons, double[][] initialWeights, double[] biases, double[][] initialInput, DEFAULT_ACTIVATION_FUNCTIONS activationFunction, Map<Integer, DEFAULT_ACTIVATION_FUNCTIONS> activationFunctionsMap) {
        this.layerList = new ArrayList<>();
        this.initialInput = initialInput;
        // instantiate the decisions index
        this.decisionsIndex = new int[initialInput.length];
        // instantiate the hash map for the values of the decision
        this.decisions = new HashMap<>();

        IntStream.range(0, layers).forEach(i -> {
            DEFAULT_ACTIVATION_FUNCTIONS activationFunctionToUse = activationFunctionsMap.getOrDefault(i, activationFunction);
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
    private DenseLayers(int layers, int[] numberOfNeuronsPerLayer, DEFAULT_ACTIVATION_FUNCTIONS activationFunction, Map<Integer, DEFAULT_ACTIVATION_FUNCTIONS> activationFunctionsMap) {
        if(layers != numberOfNeuronsPerLayer.length) {
            throw new IllegalArgumentException(layers + " Layers given but only " + numberOfNeuronsPerLayer.length
                    + " layers described!\nAdjust neuronsPerLayer to be of same length as number of layers!");
        }
        this.layerList = new ArrayList<>();
        this.initialInput = randn(1, numberOfNeuronsPerLayer[0]);
        // instantiate the decisions index
        this.decisionsIndex = new int[initialInput.length];
        // instantiate the hash map for the values of the decision
        this.decisions = new HashMap<>();

        IntStream.range(0, layers).forEach(i -> {
            DEFAULT_ACTIVATION_FUNCTIONS activationFunctionToUse = activationFunctionsMap.getOrDefault(i, activationFunction);
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
    private DenseLayers(int layers, int[] numberOfNeuronsPerLayer, double[][] initialInput, DEFAULT_ACTIVATION_FUNCTIONS activationFunction, Map<Integer, DEFAULT_ACTIVATION_FUNCTIONS> activationFunctionsMap) {
        if(layers != numberOfNeuronsPerLayer.length) {
            throw new IllegalArgumentException(layers + " Layers given but only " + numberOfNeuronsPerLayer.length
                    + " layers described!\nAdjust neuronsPerLayer to be of same length as number of layers!");
        }
        this.initialInput = initialInput;
        this.layerList = new ArrayList<>();
        // instantiate the decisions index
        this.decisionsIndex = new int[initialInput.length];
        // instantiate the hash map for the values of the decision
        this.decisions = new HashMap<>();

        IntStream.range(0, layers).forEach(i -> {
            DEFAULT_ACTIVATION_FUNCTIONS activationFunctionToUse = activationFunctionsMap.getOrDefault(i, activationFunction);
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
    private DenseLayers(int layers, int[] numberOfNeuronsPerLayer, double[][] initialWeights, double[][] initialInput, DEFAULT_ACTIVATION_FUNCTIONS activationFunction, Map<Integer, DEFAULT_ACTIVATION_FUNCTIONS> activationFunctionsMap) {
        if(layers != numberOfNeuronsPerLayer.length) {
            throw new IllegalArgumentException(layers + " Layers given but only " + numberOfNeuronsPerLayer.length
                    + " layers described!\nAdjust neuronsPerLayer to be of same length as number of layers!");
        }
        this.layerList = new ArrayList<>();
        this.initialInput = initialInput;
        // instantiate the decisions index
        this.decisionsIndex = new int[initialInput.length];
        // instantiate the hash map for the values of the decision
        this.decisions = new HashMap<>();

        IntStream.range(0, layers).forEach(i -> {
            DEFAULT_ACTIVATION_FUNCTIONS activationFunctionToUse = activationFunctionsMap.getOrDefault(i, activationFunction);
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
    private DenseLayers(int layers, int[] numberOfNeuronsPerLayer, double[][] initialWeights, double[] initialBiases, double[][] initialInput, DEFAULT_ACTIVATION_FUNCTIONS activationFunction, Map<Integer, DEFAULT_ACTIVATION_FUNCTIONS> activationFunctionsMap) {
        if(layers != numberOfNeuronsPerLayer.length) {
            throw new IllegalArgumentException(layers + " Layers given but only " + numberOfNeuronsPerLayer.length
                    + " layers described!\nAdjust neuronsPerLayer to be of same length as number of layers!");
        }
        this.layerList = new ArrayList<>();
        this.initialInput = initialInput;
        // instantiate the decisions index
        this.decisionsIndex = new int[initialInput.length];
        // instantiate the hash map for the values of the decision
        this.decisions = new HashMap<>();

        // create first layer by multiplying the initial input by the  initial weights and adding the initial biases
        IntStream.range(0, layers).forEach(i -> {
            DEFAULT_ACTIVATION_FUNCTIONS activationFunctionToUse = activationFunctionsMap.getOrDefault(i, activationFunction);
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
        if(dataCounter == this.initialInput.length - 1){ return; }
        IntStream.range(0, ((this.initialInput.length-1)/miniBatchSize)).parallel().forEachOrdered(y -> {
            double tempLoss = 0;
            // call back-propagate
            this.backPropagate();
            // mini batch processing
            for(int x = 0; x < miniBatchSize; x++){
                if(dataCounter >= this.initialInput.length - 1){
                    break;
                }
                this.dataCounter++;
                // set buffer to hold inputs
                double[][] inputs = new double[this.getLayerList().size()][];
                forward(inputs);
                tempLoss += generateLoss();
            }
            this.loss = tempLoss / miniBatchSize;
            this.batchCounter++;
            print("Loss and Accuracy Data For Batch: " + this.batchCounter);
            print("Accuracy:", this.getAccuracy());
            print("Loss:", this.getLoss());
            print("Layer Outputs:", this.getLastLayer().getLayerOutputs());
            print("");
        });
        printTitle("Training Complete!");
    }

    private double generateLoss(){
        double loss = 0;
        if(hotOneVec != null && trueValueIndices != null) { throw new RuntimeException("Cannot initialize both Hot-One-Vector and a True-Value! You must select one method of error/loss checking!"); }
        // check if both the one hot vector mappings true values mappings are empty
        if(hotOneVec == null && trueValueIndices == null) { throw new RuntimeException("Must initialize either Hot-One-Vector or a True-Value!"); }
        assert trueValueIndices != null;
        this.generateDecisionsMap(trueValueIndices[this.dataCounter]);
        if(hotOneVec != null) {
            int hotValueIndex = 0;
            for(int i = 0; i < hotOneVec.length; i++){
                if(hotOneVec[this.dataCounter][i] != 0){
                    hotValueIndex = i;
                    break;
                }
            }
            loss = ErrorLossFunctions.lossFunction(lossFunction, hotValueIndex, this.getDecisionsIndex()[this.dataCounter], this.getLastLayer().getLayerOutputs());
        } else if (trueValueIndices != null){
            loss =  ErrorLossFunctions.lossFunction(lossFunction, trueValueIndices[this.dataCounter],  this.getDecisionsIndex()[this.dataCounter], this.getLastLayer().getLayerOutputs());
        }
        return loss;
    }

    private void forward(double[][] inputs) {
        // do forward pass with next batch of input
        IntStream.range(0, inputs.length).parallel().forEachOrdered(i -> {
            if (i != 0) {
                this.getLayerAt(i).setInputs(inputs[i-1]); // use previous layers input
            } else {
                this.getLayerAt(i).setInputs(this.initialInput[dataCounter]); // use user provided input
            }
            inputs[i] = this.getLayerAt(i).regenerateLayerOutput();
        });
    }

    private void initialForward(double[][] inputs) {
        // do forward pass with next batch of input
        IntStream.range(0, inputs.length).parallel().forEachOrdered(i -> {
            if (i != 0) {
                this.getLayerAt(i).setInputs(inputs[i-1]); // use previous layers input
            } else {
                this.getLayerAt(i).setInputs(this.initialInput[dataCounter]); // use user provided input
            }
            inputs[i] = this.getLayerAt(i).regenerateLayerOutput();
        });
    }

    private void backPropagate() {
        print("Back-Propagating...");
        // Output layer error gradient
        double[][] gradients = new double[this.layerList.size()][];
        // index of the output layer
        int outputIndex = this.getLayerList().size() - 1;
        // loss function derivative
        gradients[outputIndex] = ErrorLossDerivatives.derivative(
                lossFunction,
                this.getTrueValueIndices()[dataCounter],
                this.getDecisionsIndex()[dataCounter],
                this.getLayerAt(outputIndex).getLayerOutputs()
        );
        // Update output layer weights and biases
        for (int j = 0; j < this.getLayerAt(outputIndex).getNeuronList().size(); j++) {
            // update biases
            Neuron neuron = this.getNeuronFromLayerAt(outputIndex, j);
            neuron.setBias(neuron.getBias() - (this.learningRate * gradients[outputIndex][j]));

            double[] inputs = this.getLayerAt(outputIndex - 1).getLayerOutputs();
            for (int k = 0; k < neuron.getWeights().length; k++) {
                // update weights with fractional change (learning rate should be between 0.01 and 0.0001
                double deltaWeight = this.learningRate * gradients[outputIndex][j] * inputs[k]; // previously multiplied by inputs[j]
                neuron.setWeight(k, neuron.getWeights()[k] - deltaWeight);
            }
        }

        // Backpropagation through the hidden layers
        for (int index = outputIndex - 1; index >= 0; index--) {
            // create gradient for current layer
            gradients[index] = new double[this.getLayerAt(index).getNeuronList().size()];
            // activation function derivative
            double[] derivatives = ActivationDerivatives.derivative(
                    this.getLayerAt(index).getActivationFunction(),
                    this.getLayerAt(index).getLayerOutputs()
            );
            for (int j = 0; j < gradients[index].length; j++) {
                double gradientSum = 0;
                // sum the weights related to the given input by the gradients related to the given neuron (derivative related to input)
                for (int k = 0; k < this.getLayerAt(index + 1).getNeuronList().size(); k++) {
                    gradientSum += gradients[index + 1][k] * this.getLayerAt(index + 1).getNeuronList().get(k).getWeights()[j];
                }
                // multiply the sum of the gradients and the derivative of that activation function to get the current error signal
                gradients[index][j] = gradientSum * derivatives[j];
            }
            for (int j = 0; j < this.getLayerAt(index).getNeuronList().size(); j++) {
                Neuron neuron = this.getNeuronFromLayerAt(index, j);
                // update biases
                neuron.setBias(neuron.getBias() - (this.learningRate * gradients[index][j]));
                double[] inputs = (index == 0) ? this.getInitialInput()[dataCounter] :
                        this.getLayerAt(index - 1).getLayerOutputs();
                for (int k = 0; k < neuron.getWeights().length; k++) {
                    double deltaWeight = this.learningRate * gradients[index][j] * inputs[k];
                    neuron.setWeight(k, neuron.getWeights()[k] - deltaWeight);
                }
            }
        }
        printPositive("Finished back-propagation!");
    }

    private static double[][] randn(int rows, int cols) {
        return getRandomMatrix(rows, cols);
    }
    private static double[] randn(int elementCount) {
        return getRandomArray(elementCount);
    }
    private static double[][] getRandomMatrix(int rows, int cols) {
        double[][] output = new double[rows][cols];
        for (int i = 0; i < output.length; i++)
            for (int j = 0; j < output[0].length; j++)
                output[i][j] = (0.1 * random.nextGaussian());
        return output;
    }
    private static double[] getRandomArray(int elementCount) {
        double[] output = new double[elementCount];
        for(int i = 0; i < output.length; i++){
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
    private double[] batchDecisions() {
        double[] decisions = new double[this.getLastLayer().getLayerOutputs().length];
        IntStream.range(0, decisions.length).parallel().forEachOrdered( i -> decisions[i] = decision(this.getLastLayer().getLayerOutputs())[1]);
        return decisions;
    }
    private double[] decision(double[] values) {
        return MathUtils.indexAndMax(values);
    }
    public double[] getOutputs() {
        return getLayerAt(getLayerList().size()-1).getLayerOutputs();
    }
    private double accuracy(int trueIndex, int predictedIndex) {
        if(trueIndex == predictedIndex) {
            sum+=1;
        }
        this.accuracy = sum / (dataCounter + 1);
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

    public DEFAULT_ACTIVATION_FUNCTIONS getActivationFunctionFrom(int index){
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

    // builder class for the neural network
    @NoArgsConstructor
    public static class DenseLayersBuilder {
        private ArrayList<DenseLayer> layerList;
        private boolean isUsingListOfLayers = true;

        private DEFAULT_ACTIVATION_FUNCTIONS activationFunction = DEFAULT_ACTIVATION_FUNCTIONS.SOFTMAX_ACTIVATION_FUNCTION;
        private final Map<Integer, DEFAULT_ACTIVATION_FUNCTIONS> activationFunctionsMap = new HashMap<>();
        private boolean hasSetSpecificLayerActivationFunctions = false;

        //
        private DEFAULT_LOSS_FUNCTIONS lossFunction = DEFAULT_LOSS_FUNCTIONS.NLL_LOSS_FUNCTION;
        private int[][] hotOneVec;
        private int[] trueValueIndices;
        private double[] trueValues;
        private double[][][] trueValuesMatrix;
        private final Map<Integer, DEFAULT_LOSS_FUNCTIONS> lossFunctionsMap = new HashMap<>();

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
        private boolean isUsingBatchInputs = true;
        private int miniBatchSize = 1;

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
            if (char1 == 't' && char2 == 'x' && char3 == 't' && char4 == '.'){
                this.initialInput = (double[][])FileUtils.readTextFile(filePath, delimiter);

            } else if (char1 == 'v' && char2 == 's' && char3 == 'c' && char4 == '.'){
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

        public DenseLayersBuilder withActivationFunction(DEFAULT_ACTIVATION_FUNCTIONS activationFunction) {
            this.activationFunction = activationFunction;
            return this;
        }

        public DenseLayersBuilder withActivationFunctionForSingleLayer(int layer, DEFAULT_ACTIVATION_FUNCTIONS activationFunction) {
            activationFunctionsMap.put(layer, activationFunction);
            return this;
        }

        public DenseLayersBuilder withActivationFunctionForMultipleLayers(int startingLayer, int endingLayer, DEFAULT_ACTIVATION_FUNCTIONS activationFunction) {
            IntStream.range(startingLayer, endingLayer+1).forEachOrdered( i -> activationFunctionsMap.put(i, activationFunction));
            return this;
        }

        public DenseLayersBuilder withActivationFunctionForOutput(DEFAULT_ACTIVATION_FUNCTIONS activationFunction) {
            if(numberOfLayers > 1){
                activationFunctionsMap.put(numberOfLayers-1, activationFunction);
            }
            return this;
        }

        public DenseLayersBuilder withLearningRate(double learningRate) {
            this.learningRate = learningRate;
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

        public DenseLayersBuilder withLossFunction(DEFAULT_LOSS_FUNCTIONS lossFunction) {
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

        public DenseLayersBuilder withLossFunctionAndHotEncodedVectors(DEFAULT_LOSS_FUNCTIONS lossFunction, int[][] hotOneOutput) {
            this.lossFunction = lossFunction;
            this.hotOneVec = hotOneOutput;
            return this;
        }

        public DenseLayersBuilder withLossFunctionAndTrueValues(DEFAULT_LOSS_FUNCTIONS lossFunction, int[] trueValueIndices) {
            this.lossFunction = lossFunction;
            this.trueValueIndices = trueValueIndices;
            return this;
        }

        public DenseLayersBuilder withMiniBatchProcessing(int miniBatchSize){
            this.miniBatchSize = miniBatchSize;
            return this;
        }

        public DenseLayersBuilder withFullBatchProcessing(){
            this.isUsingStochasticGradientDescent = false;
            this.isUsingBatchInputs = true;
            return this;
        }

        public DenseLayers build() {
            DenseLayers built = null;
            if(!this.isUsingNumberOfLayers && !this.isUsingListOfLayers) { throw new IllegalArgumentException("Please use the numberOfLayers() builder method to initialize, or provide an ArrayList<DenseLayer> using the withLayerList() builder method!"); }

            if(!this.isUsingSpecificNeurons && !this.isUsingNumberOfNeurons && this.isUsingNumberOfLayers && !this.isUsingBatchInputs) {
                if(this.isUsingFileAsInput && this.isUsingSpecificWeights && this.isUsingSpecificBiases) {
                    printTitle("Using construct 1.1");
                    built = new DenseLayers(this.numberOfLayers, this.initialWeights, this.initialBiases, this.initialInput, this.activationFunction, this.activationFunctionsMap);
                }
                else if(this.isUsingFileAsInput && this.isUsingSpecificWeights) {
                    printTitle("Using construct 1.2");
                    built = new DenseLayers(this.numberOfLayers, this.initialWeights, this.initialInput, this.activationFunction, this.activationFunctionsMap);
                }
                else if(this.isUsingFileAsInput) {
                    printTitle("Using construct 1.3");
                    built = new DenseLayers(this.numberOfLayers, this.initialInput, this.activationFunction, this.activationFunctionsMap);
                }
                else if(!this.isUsingSpecificWeights && !this.isUsingSpecificBiases) {
                    printTitle("Using construct 1.4");
                    built = new DenseLayers(this.numberOfLayers, this.activationFunction, this.activationFunctionsMap);
                }
            }
            else if(this.isUsingNumberOfNeurons && this.isUsingNumberOfLayers && !this.isUsingBatchInputs) {
                if(this.isUsingSpecificWeights && this.isUsingSpecificBiases) {
                    printTitle("Using construct 2.1");
                    built = new DenseLayers(this.numberOfLayers, this.numberOfNeurons, this.initialWeights, this.initialBiases, this.initialInput, this.activationFunction, this.activationFunctionsMap);
                }
                else if(this.isUsingFileAsInput  && this.isUsingSpecificWeights) {
                    printTitle("Using construct 2.2");
                    built = new DenseLayers(this.numberOfLayers, this.numberOfNeurons, this.initialWeights, this.initialInput, this.activationFunction, this.activationFunctionsMap);
                }
                else if(this.isUsingFileAsInput) {
                    printTitle("Using construct 2.3");
                    built = new DenseLayers(this.numberOfLayers, this.numberOfNeurons, this.initialInput, this.activationFunction, this.activationFunctionsMap);
                }
                else if(!this.isUsingFileAsInput && !this.isUsingSpecificWeights && !this.isUsingSpecificBiases) {
                    printTitle("Using construct 2.4");
                    built = new DenseLayers(this.numberOfLayers, this.numberOfNeurons, this.activationFunction, this.activationFunctionsMap);
                }
            }
            else if(this.isUsingSpecificNeurons && this.isUsingNumberOfLayers && !this.isUsingBatchInputs) {
                if(this.isUsingFileAsInput && this.isUsingSpecificWeights && this.isUsingSpecificBiases) {
                    printTitle("Using construct 3.1");
                    built = new DenseLayers(this.numberOfLayers, this.numberOfNeuronsPerLayer, this.initialWeights, this.initialBiases, this.initialInput, this.activationFunction, this.activationFunctionsMap);
                }
                else if(this.isUsingFileAsInput && this.isUsingSpecificWeights) {
                    printTitle("Using construct 3.2");
                    built = new DenseLayers(this.numberOfLayers, this.numberOfNeuronsPerLayer, this.initialWeights, this.initialInput, this.activationFunction, this.activationFunctionsMap);
                }
                else if(this.isUsingFileAsInput) {
                    printTitle("Using construct 3.3");
                    built = new DenseLayers(this.numberOfLayers, this.numberOfNeuronsPerLayer, this.initialInput, this.activationFunction, this.activationFunctionsMap);
                }
                else if(!this.isUsingFileAsInput && !this.isUsingSpecificWeights && !this.isUsingSpecificBiases) {
                    printTitle("Using construct 3.4");
                    built = new DenseLayers(this.numberOfLayers, this.numberOfNeuronsPerLayer, this.activationFunction, this.activationFunctionsMap);
                }
            }
            else if(!this.isUsingSpecificNeurons && !this.isUsingNumberOfNeurons && this.isUsingNumberOfLayers && !this.isUsingFileAsInput) {
                if(this.isUsingBatchInputs && this.isUsingSpecificWeights && this.isUsingSpecificBiases) {
                    printTitle("Using construct 4.1");
                    built = new DenseLayers(this.numberOfLayers, this.initialWeights, this.initialBiases, this.initialInput, this.activationFunction, this.activationFunctionsMap);
                }
                else if(this.isUsingBatchInputs  && this.isUsingSpecificWeights) {
                    printTitle("Using construct 4.2");
                    built = new DenseLayers(this.numberOfLayers, this.initialWeights, this.initialInput, this.activationFunction, this.activationFunctionsMap);
                }
                else if(this.isUsingBatchInputs) {
                    printTitle("Using construct 4.3");
                    built = new DenseLayers(this.numberOfLayers, this.initialInput, this.activationFunction, this.activationFunctionsMap);
                }
                else if(!this.isUsingBatchInputs && !this.isUsingSpecificWeights && !this.isUsingSpecificBiases) {
                    printTitle("Using construct 4.4");
                    built = new DenseLayers(this.numberOfLayers, this.activationFunction, this.activationFunctionsMap);
                }
            }
            else if(this.isUsingNumberOfNeurons && this.isUsingNumberOfLayers && !this.isUsingFileAsInput) {
                if(this.isUsingBatchInputs && this.isUsingSpecificWeights && this.isUsingSpecificBiases) {
                    printTitle("Using construct 5.1");
                    built = new DenseLayers(this.numberOfLayers, this.numberOfNeurons, this.initialWeights, this.initialBiases, this.initialInput, this.activationFunction, this.activationFunctionsMap);
                }
                else if(this.isUsingBatchInputs  && this.isUsingSpecificWeights) {
                    printTitle("Using construct 5.2");
                    built = new DenseLayers(this.numberOfLayers, this.numberOfNeurons, this.initialWeights, this.initialInput, this.activationFunction, this.activationFunctionsMap);
                }
                else if(this.isUsingBatchInputs) {
                    printTitle("Using construct 5.3");
                    built = new DenseLayers(this.numberOfLayers, this.numberOfNeurons, this.initialInput, this.activationFunction, this.activationFunctionsMap);
                }
                else if(!this.isUsingBatchInputs && !this.isUsingSpecificWeights && !this.isUsingSpecificBiases) {
                    printTitle("Using construct 5.4");
                    built = new DenseLayers(this.numberOfLayers, this.numberOfNeurons, this.activationFunction, this.activationFunctionsMap);
                }
            }
            else if(this.isUsingSpecificNeurons && this.isUsingNumberOfLayers && !this.isUsingFileAsInput) {
                if(this.isUsingBatchInputs && this.isUsingSpecificWeights && this.isUsingSpecificBiases) {
                    printTitle("Using construct 6.1");
                    built = new DenseLayers(this.numberOfLayers, this.numberOfNeuronsPerLayer, this.initialWeights, this.initialBiases, this.initialInput, this.activationFunction, this.activationFunctionsMap);
                }
                else if(this.isUsingBatchInputs  && this.isUsingSpecificWeights) {
                    printTitle("Using construct 6.2");
                    built = new DenseLayers(this.numberOfLayers, this.numberOfNeuronsPerLayer, this.initialWeights, this.initialInput, this.activationFunction, this.activationFunctionsMap);
                }
                else if(this.isUsingBatchInputs) {
                    printTitle("Using construct 6.3");
                    built = new DenseLayers(this.numberOfLayers, this.numberOfNeuronsPerLayer, this.initialInput, this.activationFunction, this.activationFunctionsMap);
                }
                else if(!this.isUsingBatchInputs && !this.isUsingSpecificWeights && !this.isUsingSpecificBiases) {
                    printTitle("Using construct 6.4");
                    built = new DenseLayers(this.numberOfLayers, this.numberOfNeuronsPerLayer, this.activationFunction, this.activationFunctionsMap);
                }
            }
            else {
                printTitle("Using construct 7");
                built = new DenseLayers(2, DEFAULT_ACTIVATION_FUNCTIONS.LINEAR_ACTIVATION_FUNCTION, this.activationFunctionsMap);
                if(this.isUsingBatchInputs && this.isUsingFileAsInput) {
                    throw new RuntimeException(bold(red("Builder Not Configured Properly! Do Not Use File As Input and Batch Input Together!")));
                }
                throw new RuntimeException(bold(red("Builder Not Configured Properly!")));
            }
            assert built != null;

            printPositive("Neural network successfully built!");
            printPositive("Created dense layer neural network: " + built);
            printPositive("Number of layers set to " + this.numberOfLayers);
            printPositive("Initial activation function set to " + this.activationFunction.name());
            printPositive("Loss function set to " + this.lossFunction.name());

            built.learningRate = this.learningRate;
            built.lossFunction = this.lossFunction;
            built.miniBatchSize = this.miniBatchSize;
            if(isUsingTrueValueIndex) {
                built.trueValueIndices = this.trueValueIndices;
                printPositive("True Values Set!");
            } else {
                built.hotOneVec = this.hotOneVec;
                printPositive("Hot Vector Set!");
            }
            ALPHA = this.alpha;
            GAMMA = this.gamma;
            DELTA = this.delta;
            MARGIN = this.margin;
            if(built.hotOneVec != null && built.trueValueIndices != null) { throw new RuntimeException("Cannot initialize both Hot-One-Vector and a True-Value! You must select one method of error/loss checking!"); }
            // check if both the one hot vector mappings true values mappings are empty
            if(built.hotOneVec == null && trueValueIndices == null) { throw new RuntimeException("Must initialize either Hot-One-Vector or a True-Value!"); }
            if(built.hotOneVec != null) {
                built.generateDecisionsMap(built.hotOneVec[built.miniBatchSize]);
                built.loss =  ErrorLossFunctions.lossFunction(lossFunction, built.hotOneVec[built.miniBatchSize], built.getDecisionsIndex()[built.miniBatchSize], built.getLastLayer().getLayerOutputs());

            } else if (built.trueValueIndices != null){
                built.generateDecisionsMap(built.trueValueIndices[built.miniBatchSize]);
                built.loss = ErrorLossFunctions.lossFunction(lossFunction, trueValueIndices[built.miniBatchSize],  built.getDecisionsIndex()[built.miniBatchSize], built.getLastLayer().getLayerOutputs());
            }
            double tl = built.loss;
            for(int y = 0; y < built.miniBatchSize-1; y++) {
                if(built.dataCounter >= built.initialInput.length - 1){
                    break;
                }
                built.dataCounter++;
                // set buffer to hold inputs
                double[][] inputs = new double[built.getLayerList().size()][];
                built.forward(inputs);
                tl += built.generateLoss();
            }
            built.loss = tl / built.miniBatchSize;
            printSubTitle("Starting back propagation!");
            built.backPropagate();
            printSubTitle("Initialization Stats:");
            print("Loss and Accuracy Data For Batch: " + built.batchCounter);
            print("Accuracy:", built.getAccuracy());
            print("Loss:", built.getLoss());
            print("Layer Outputs:", built.getLastLayer().getLayerOutputs());
            printTitle("Finished Initialization!");
            print("");
            return built;
        }
    }
}
