package com.craftsentient.craftmind.layers;

import com.craftsentient.craftmind.activation.DEFAULT_ACTIVATION_FUNCTIONS;
import com.craftsentient.craftmind.errorLoss.DEFAULT_LOSS_FUNCTIONS;
import com.craftsentient.craftmind.errorLoss.ErrorLossFunctions;
import com.craftsentient.craftmind.layer.DenseLayer;
import com.craftsentient.craftmind.utils.FileUtils;
import com.craftsentient.craftmind.utils.MathUtils;
import com.craftsentient.craftmind.neuron.Neuron;
import lombok.Getter;
import lombok.NoArgsConstructor;

import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;

import static com.craftsentient.craftmind.utils.MathUtils.accuracy;
import static com.craftsentient.craftmind.utils.PrintUtils.*;

@Getter
public class DenseLayers {
    private final ArrayList<DenseLayer> layerList;
    private final double[][] initialInput;
    private double[][] trueValues;
    private double[][] hotOneVecs;
    private int batchCounter = 0;
    private Map<Integer, Double> decisions;
    private int[] decisionsIndex;
    private double accuracy;
    private double meanLoss;
    private double loss;
    public static final Random random = new Random(0);

    private DenseLayers(int layers, DEFAULT_ACTIVATION_FUNCTIONS activationFunction, Map<Integer, DEFAULT_ACTIVATION_FUNCTIONS> activationFunctionsMap, DEFAULT_LOSS_FUNCTIONS lossFunction, int[][] hotOneVec, int[] trueValues) throws Exception {
        if(hotOneVec.length != 0 && trueValues.length != 0) { throw new RuntimeException("Cannot initialize both Hot-One-Vector and a True-Value! You must select one method of error/loss checking!"); }
        // check if both the one hot vector mappings true values mappings are empty
        if(hotOneVec.length == 0 && trueValues.length == 0) { throw new RuntimeException("Must initialize either Hot-One-Vector or a True-Value!"); }
        // create layer list
        this.layerList = new ArrayList<>();
        // create an initial input array of random numbers of size layers
        this.initialInput = randn(1,layers);

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
                    layerList.add(new DenseLayer(weights, initialInput[i], activationFunctionToUse));
                }
            } catch (Exception e){
                throw new RuntimeException("[DenseLayers Constructor] Error During Layer Creation!");
            }
        });

        // if the true value mapping is empty, set up the one hot vector
        if(lossFunction == DEFAULT_LOSS_FUNCTIONS.NLL_LOSS_FUNCTION) {
            this.loss =  MathUtils.mean(ErrorLossFunctions.lossFunction(lossFunction, hotOneVec[0], this.getLastLayer().getLayerOutputs()));
        } else {
            this.loss = ErrorLossFunctions.lossFunction(lossFunction, trueValues[0], this.getLastLayer().getLayerOutputs());
        }
    }

    private DenseLayers(int layers, double[][] initialInput, DEFAULT_ACTIVATION_FUNCTIONS activationFunction, Map<Integer, DEFAULT_ACTIVATION_FUNCTIONS> activationFunctionsMap, DEFAULT_LOSS_FUNCTIONS lossFunction, int[][] hotOneVec, int[] trueValues) throws Exception {
        if(hotOneVec.length != 0 && trueValues.length != 0) { throw new RuntimeException("Cannot initialize both Hot-One-Vector and a True-Value! You must select one method of error/loss checking!"); }
        // check if both the one hot vector mappings true values mappings are empty
        if(hotOneVec.length == 0 && trueValues.length == 0) { throw new RuntimeException("Must initialize either Hot-One-Vector or a True-Value!"); }
        this.initialInput = initialInput;
        this.layerList = new ArrayList<>();

        IntStream.range(0, layers).forEach(i -> {
            DEFAULT_ACTIVATION_FUNCTIONS activationFunctionToUse = activationFunctionsMap.getOrDefault(i, activationFunction);
            double[][] weights = randn(initialInput.length, initialInput[0].length);
            try {
                if (i != 0) {
                    layerList.add(new DenseLayer(weights, layerList.get(i - 1).getLayerOutputs(), activationFunctionToUse));
                } else {
                    layerList.add(new DenseLayer(weights, initialInput[i], activationFunctionToUse));
                }
            } catch (Exception e){
                throw new RuntimeException("[DenseLayers Constructor] Error During Layer Creation!");
            }
        });

        if(lossFunction == DEFAULT_LOSS_FUNCTIONS.NLL_LOSS_FUNCTION) {
            this.loss =  MathUtils.mean(ErrorLossFunctions.lossFunction(lossFunction, hotOneVec[0], this.getLastLayer().getLayerOutputs()));
        } else {
            this.loss = ErrorLossFunctions.lossFunction(lossFunction, trueValues[0], this.getLastLayer().getLayerOutputs());
        }
    }

    private DenseLayers(int layers, double[][] initialWeights, double[][] initialInput, DEFAULT_ACTIVATION_FUNCTIONS activationFunction, Map<Integer, DEFAULT_ACTIVATION_FUNCTIONS> activationFunctionsMap, DEFAULT_LOSS_FUNCTIONS lossFunction, int[][] hotOneVec, int[] trueValues) throws Exception {
        if(hotOneVec.length != 0 && trueValues.length != 0) { throw new RuntimeException("Cannot initialize both Hot-One-Vector and a True-Value! You must select one method of error/loss checking!"); }
        // check if both the one hot vector mappings true values mappings are empty
        if(hotOneVec.length == 0 && trueValues.length == 0) { throw new RuntimeException("Must initialize either Hot-One-Vector or a True-Value!"); }
        this.initialInput = initialInput;
        this.layerList = new ArrayList<>();

        IntStream.range(0, layers).forEach(i -> {
            DEFAULT_ACTIVATION_FUNCTIONS activationFunctionToUse = activationFunctionsMap.getOrDefault(i, activationFunction);
            try {
                if (i != 0) {
                    double[][] weights = randn(initialWeights.length, initialInput[0].length);
                    layerList.add(new DenseLayer(weights, layerList.get(i - 1).getLayerOutputs(), activationFunctionToUse));
                } else {
                    layerList.add(new DenseLayer(initialWeights, initialInput[i], activationFunctionToUse));
                }
            } catch(Exception e){
                throw new RuntimeException("[DenseLayers Constructor] Error During Layer Creation!");
            }
        });

        if(lossFunction == DEFAULT_LOSS_FUNCTIONS.NLL_LOSS_FUNCTION) {
            this.loss =  MathUtils.mean(ErrorLossFunctions.lossFunction(lossFunction, hotOneVec[0], this.getLastLayer().getLayerOutputs()));
        } else {
            this.loss = ErrorLossFunctions.lossFunction(lossFunction, trueValues[0], this.getLastLayer().getLayerOutputs());
        }
    }

    private DenseLayers(int layers, double[][] initialWeights, double[]biases, double[][] initialInput, DEFAULT_ACTIVATION_FUNCTIONS activationFunction, Map<Integer, DEFAULT_ACTIVATION_FUNCTIONS> activationFunctionsMap, DEFAULT_LOSS_FUNCTIONS lossFunction, int[][] hotOneVec, int[] trueValues) throws Exception {
        if(hotOneVec.length != 0 && trueValues.length != 0) { throw new RuntimeException("Cannot initialize both Hot-One-Vector and a True-Value! You must select one method of error/loss checking!"); }
        // check if both the one hot vector mappings true values mappings are empty
        if(hotOneVec.length == 0 && trueValues.length == 0) { throw new RuntimeException("Must initialize either Hot-One-Vector or a True-Value!"); }
        this.layerList = new ArrayList<>();
        this.initialInput = initialInput;
        IntStream.range(0, layers).forEach(i -> {
            DEFAULT_ACTIVATION_FUNCTIONS activationFunctionToUse = activationFunctionsMap.getOrDefault(i, activationFunction);
            try {
                if (i != 0) {
                    double[][] weights = randn(initialWeights.length, initialInput[0].length);
                    layerList.add(new DenseLayer(weights, layerList.get(i - 1).getLayerOutputs(), activationFunctionToUse));
                } else {
                    layerList.add(new DenseLayer(initialWeights, biases, initialInput[i], activationFunctionToUse));
                }
            } catch (Exception e){
                throw new RuntimeException("[DenseLayers Constructor] Error During Layer Creation!");
            }
        });

        if(lossFunction == DEFAULT_LOSS_FUNCTIONS.NLL_LOSS_FUNCTION) {
            this.loss =  MathUtils.mean(ErrorLossFunctions.lossFunction(lossFunction, hotOneVec[0], this.getLastLayer().getLayerOutputs()));
        } else {
            this.loss = ErrorLossFunctions.lossFunction(lossFunction, trueValues[0], this.getLastLayer().getLayerOutputs());
        }
    }

    private DenseLayers(int layers, int numberOfNeurons, DEFAULT_ACTIVATION_FUNCTIONS activationFunction, Map<Integer, DEFAULT_ACTIVATION_FUNCTIONS> activationFunctionsMap, DEFAULT_LOSS_FUNCTIONS lossFunction, int[][] hotOneVec, int[] trueValues) throws Exception {
        if(hotOneVec.length != 0 && trueValues.length != 0) { throw new RuntimeException("Cannot initialize both Hot-One-Vector and a True-Value! You must select one method of error/loss checking!"); }
        // check if both the one hot vector mappings true values mappings are empty
        if(hotOneVec.length == 0 && trueValues.length == 0) { throw new RuntimeException("Must initialize either Hot-One-Vector or a True-Value!"); }
        this.layerList = new ArrayList<>();
        this.initialInput = randn(1,numberOfNeurons);

        IntStream.range(0, layers).forEach(i -> {
            DEFAULT_ACTIVATION_FUNCTIONS activationFunctionToUse = activationFunctionsMap.getOrDefault(i, activationFunction);
            double[][] weights = randn(numberOfNeurons, numberOfNeurons);
            try {
                if (i != 0)
                    layerList.add(new DenseLayer(weights, layerList.get(i - 1).getLayerOutputs(), activationFunctionToUse));
                else
                    layerList.add(new DenseLayer(weights, initialInput[i], activationFunctionToUse));
            } catch (Exception e){
                throw new RuntimeException("[DenseLayers Constructor] Error During Layer Creation!");
            }
        });

        if(lossFunction == DEFAULT_LOSS_FUNCTIONS.NLL_LOSS_FUNCTION) {
            this.loss =  MathUtils.mean(ErrorLossFunctions.lossFunction(lossFunction, hotOneVec[0], this.getLastLayer().getLayerOutputs()));
        } else {
            this.loss = ErrorLossFunctions.lossFunction(lossFunction, trueValues[0], this.getLastLayer().getLayerOutputs());
        }
    }

    private DenseLayers(int layers, int numberOfNeurons, double[][] initialInput, DEFAULT_ACTIVATION_FUNCTIONS activationFunction, Map<Integer, DEFAULT_ACTIVATION_FUNCTIONS> activationFunctionsMap, DEFAULT_LOSS_FUNCTIONS lossFunction, int[][] hotOneVec, int[] trueValues) throws Exception {
        if(hotOneVec.length != 0 && trueValues.length != 0) { throw new RuntimeException("Cannot initialize both Hot-One-Vector and a True-Value! You must select one method of error/loss checking!"); }
        // check if both the one hot vector mappings true values mappings are empty
        if(hotOneVec.length == 0 && trueValues.length == 0) { throw new RuntimeException("Must initialize either Hot-One-Vector or a True-Value!"); }
        if(numberOfNeurons != initialInput.length) { throw new IllegalArgumentException("neuronsPerLayer of " + numberOfNeurons + " and initialInput size of " + initialInput.length + " do not match!");}
        this.initialInput = initialInput;
        this.layerList = new ArrayList<>();

        IntStream.range(0, layers).forEach(i -> {
            DEFAULT_ACTIVATION_FUNCTIONS activationFunctionToUse = activationFunctionsMap.getOrDefault(i, activationFunction);
            double[][] weights = randn(numberOfNeurons, initialInput[0].length);
            try {
                if (i != 0)
                    layerList.add(new DenseLayer(weights, layerList.get(i - 1).getLayerOutputs(), activationFunctionToUse));
                else
                    layerList.add(new DenseLayer(weights, initialInput[i], activationFunctionToUse));
            } catch (Exception e) {
                throw new RuntimeException("[DenseLayers Constructor] Error During Layer Creation!");
            }
        });

        if(lossFunction == DEFAULT_LOSS_FUNCTIONS.NLL_LOSS_FUNCTION) {
            this.loss =  MathUtils.mean(ErrorLossFunctions.lossFunction(lossFunction, hotOneVec[0], this.getLastLayer().getLayerOutputs()));
        } else {
            this.loss = ErrorLossFunctions.lossFunction(lossFunction, trueValues[0], this.getLastLayer().getLayerOutputs());
        }
    }

    private DenseLayers(int layers, int numberOfNeurons, double[][] initialWeights, double[][] initialInput, DEFAULT_ACTIVATION_FUNCTIONS activationFunction, Map<Integer, DEFAULT_ACTIVATION_FUNCTIONS> activationFunctionsMap, DEFAULT_LOSS_FUNCTIONS lossFunction, int[][] hotOneVec, int[] trueValues) throws Exception {
        if(hotOneVec.length != 0 && trueValues.length != 0) { throw new RuntimeException("Cannot initialize both Hot-One-Vector and a True-Value! You must select one method of error/loss checking!"); }
        // check if both the one hot vector mappings true values mappings are empty
        if(hotOneVec.length == 0 && trueValues.length == 0) { throw new RuntimeException("Must initialize either Hot-One-Vector or a True-Value!"); }
        this.initialInput = initialInput;
        this.layerList = new ArrayList<>();

        IntStream.range(0, layers).forEach(i -> {
            DEFAULT_ACTIVATION_FUNCTIONS activationFunctionToUse = activationFunctionsMap.getOrDefault(i, activationFunction);
            try {
                if (i != 0) {
                    double[][] weights = randn(numberOfNeurons, numberOfNeurons);
                    layerList.add(new DenseLayer(weights, layerList.get(i - 1).getLayerOutputs(), activationFunctionToUse));
                } else {
                    layerList.add(new DenseLayer(initialWeights, initialInput[i], activationFunctionToUse));
                }
            } catch (Exception e) {
                throw new RuntimeException("[DenseLayers Constructor] Error During Layer Creation!");
            }
        });

        if(lossFunction == DEFAULT_LOSS_FUNCTIONS.NLL_LOSS_FUNCTION) {
            this.loss =  MathUtils.mean(ErrorLossFunctions.lossFunction(lossFunction, hotOneVec[0], this.getLastLayer().getLayerOutputs()));
        } else {
            this.loss = ErrorLossFunctions.lossFunction(lossFunction, trueValues[0], this.getLastLayer().getLayerOutputs());
        }
    }

    private DenseLayers(int layers, int numberOfNeurons, double[][] initialWeights, double[]biases, double[][] initialInput, DEFAULT_ACTIVATION_FUNCTIONS activationFunction, Map<Integer, DEFAULT_ACTIVATION_FUNCTIONS> activationFunctionsMap, DEFAULT_LOSS_FUNCTIONS lossFunction, int[][] hotOneVec, int[] trueValues) throws Exception {
        if(hotOneVec.length != 0 && trueValues.length != 0) { throw new RuntimeException("Cannot initialize both Hot-One-Vector and a True-Value! You must select one method of error/loss checking!"); }
        // check if both the one hot vector mappings true values mappings are empty
        if(hotOneVec.length == 0 && trueValues.length == 0) { throw new RuntimeException("Must initialize either Hot-One-Vector or a True-Value!"); }
        this.layerList = new ArrayList<>();
        this.initialInput = initialInput;

        IntStream.range(0, layers).forEach(i -> {
            DEFAULT_ACTIVATION_FUNCTIONS activationFunctionToUse = activationFunctionsMap.getOrDefault(i, activationFunction);
            try {
                if (i != 0) {
                    double[][] weights = randn(numberOfNeurons, numberOfNeurons);
                    layerList.add(new DenseLayer(weights, layerList.get(i - 1).getLayerOutputs(), activationFunctionToUse));
                } else {
                    layerList.add(new DenseLayer(initialWeights, biases, initialInput[i], activationFunctionToUse));
                }
            } catch (Exception e) {
                throw new RuntimeException("[DenseLayers Constructor] Error During Layer Creation!");
            }
        });

        if(lossFunction == DEFAULT_LOSS_FUNCTIONS.NLL_LOSS_FUNCTION) {
            this.loss =  MathUtils.mean(ErrorLossFunctions.lossFunction(lossFunction, hotOneVec[0], this.getLastLayer().getLayerOutputs()));
        } else {
            this.loss = ErrorLossFunctions.lossFunction(lossFunction, trueValues[0], this.getLastLayer().getLayerOutputs());
        }
    }

    private DenseLayers(int layers, int[] numberOfNeuronsPerLayer, DEFAULT_ACTIVATION_FUNCTIONS activationFunction, Map<Integer, DEFAULT_ACTIVATION_FUNCTIONS> activationFunctionsMap, DEFAULT_LOSS_FUNCTIONS lossFunction, int[][] hotOneVec, int[] trueValues) throws Exception {
        if(hotOneVec.length != 0 && trueValues.length != 0) { throw new RuntimeException("Cannot initialize both Hot-One-Vector and a True-Value! You must select one method of error/loss checking!"); }
        // check if both the one hot vector mappings true values mappings are empty
        if(hotOneVec.length == 0 && trueValues.length == 0) { throw new RuntimeException("Must initialize either Hot-One-Vector or a True-Value!"); }
        if(layers != numberOfNeuronsPerLayer.length) {
            throw new IllegalArgumentException(layers + " Layers given but only " + numberOfNeuronsPerLayer.length
                    + " layers described!\nAdjust neuronsPerLayer to be of same length as number of layers!");
        }
        this.layerList = new ArrayList<>();
        this.initialInput = randn(1, numberOfNeuronsPerLayer[0]);

        IntStream.range(0, layers).forEach(i -> {
            DEFAULT_ACTIVATION_FUNCTIONS activationFunctionToUse = activationFunctionsMap.getOrDefault(i, activationFunction);
            double[][] weights;
            try {
                if (i != 0) {
                    weights = randn(numberOfNeuronsPerLayer[i], numberOfNeuronsPerLayer[i - 1]);
                    layerList.add(new DenseLayer(weights, layerList.get(i - 1).getLayerOutputs(), activationFunctionToUse));
                } else {
                    weights = randn(numberOfNeuronsPerLayer[i], numberOfNeuronsPerLayer[i]);
                    layerList.add(new DenseLayer(weights, initialInput[i], activationFunctionToUse));
                }
            } catch (Exception e) {
                throw new RuntimeException("[DenseLayers Constructor] Error During Layer Creation!");
            }
        });

        if(lossFunction == DEFAULT_LOSS_FUNCTIONS.NLL_LOSS_FUNCTION) {
            this.loss =  MathUtils.mean(ErrorLossFunctions.lossFunction(lossFunction, hotOneVec[0], this.getLastLayer().getLayerOutputs()));
        } else {
            this.loss = ErrorLossFunctions.lossFunction(lossFunction, trueValues[0], this.getLastLayer().getLayerOutputs());
        }
    }

    private DenseLayers(int layers, int[] numberOfNeuronsPerLayer, double[][] initialInput, DEFAULT_ACTIVATION_FUNCTIONS activationFunction, Map<Integer, DEFAULT_ACTIVATION_FUNCTIONS> activationFunctionsMap, DEFAULT_LOSS_FUNCTIONS lossFunction, int[][] hotOneVec, int[] trueValues) throws Exception {
        if(hotOneVec.length != 0 && trueValues.length != 0) { throw new RuntimeException("Cannot initialize both Hot-One-Vector and a True-Value! You must select one method of error/loss checking!"); }
        // check if both the one hot vector mappings true values mappings are empty
        if(hotOneVec.length == 0 && trueValues.length == 0) { throw new RuntimeException("Must initialize either Hot-One-Vector or a True-Value!"); }
        this.initialInput = initialInput;
        this.layerList = new ArrayList<>();

        IntStream.range(0, layers).forEach(i -> {
            DEFAULT_ACTIVATION_FUNCTIONS activationFunctionToUse = activationFunctionsMap.getOrDefault(i, activationFunction);
            double[][] weights;
            try {
                if (i != 0) {
                    weights = randn(numberOfNeuronsPerLayer[i], numberOfNeuronsPerLayer[i - 1]);
                    layerList.add(new DenseLayer(weights, layerList.get(i - 1).getLayerOutputs(), activationFunctionToUse));
                } else {
                    weights = randn(numberOfNeuronsPerLayer[i], initialInput[0].length);
                    layerList.add(new DenseLayer(weights, initialInput[i], activationFunctionToUse));
                }
            } catch (Exception e) {
                throw new RuntimeException("[DenseLayers Constructor] Error During Layer Creation!");
            }
        });

        if(lossFunction == DEFAULT_LOSS_FUNCTIONS.NLL_LOSS_FUNCTION) {
            this.loss =  MathUtils.mean(ErrorLossFunctions.lossFunction(lossFunction, hotOneVec[0], this.getLastLayer().getLayerOutputs()));
        } else {
            this.loss = ErrorLossFunctions.lossFunction(lossFunction, trueValues[0], this.getLastLayer().getLayerOutputs());
        }
    }

    private DenseLayers(int layers, int[] numberOfNeuronsPerLayer, double[][] initialWeights, double[][] initialInput, DEFAULT_ACTIVATION_FUNCTIONS activationFunction, Map<Integer, DEFAULT_ACTIVATION_FUNCTIONS> activationFunctionsMap, DEFAULT_LOSS_FUNCTIONS lossFunction, int[][] hotOneVec, int[] trueValues) throws Exception {
        if(hotOneVec.length != 0 && trueValues.length != 0) { throw new RuntimeException("Cannot initialize both Hot-One-Vector and a True-Value! You must select one method of error/loss checking!"); }
        // check if both the one hot vector mappings true values mappings are empty
        if(hotOneVec.length == 0 && trueValues.length == 0) { throw new RuntimeException("Must initialize either Hot-One-Vector or a True-Value!"); }
        this.layerList = new ArrayList<>();
        this.initialInput = initialInput;
        IntStream.range(0, layers).forEach(i -> {
            DEFAULT_ACTIVATION_FUNCTIONS activationFunctionToUse = activationFunctionsMap.getOrDefault(i, activationFunction);
            try {
                if (i != 0) {
                    double[][] weights = randn(numberOfNeuronsPerLayer[i], numberOfNeuronsPerLayer[i - 1]);
                    layerList.add(new DenseLayer(weights, layerList.get(i - 1).getLayerOutputs(), activationFunctionToUse));
                } else {
                    layerList.add(new DenseLayer(initialWeights, initialInput[i], activationFunctionToUse));
                }
            } catch (Exception e){
                throw new RuntimeException("[DenseLayers Constructor] Error During Layer Creation!");
            }
        });

        if(lossFunction == DEFAULT_LOSS_FUNCTIONS.NLL_LOSS_FUNCTION) {
            this.loss =  MathUtils.mean(ErrorLossFunctions.lossFunction(lossFunction, hotOneVec[0], this.getLastLayer().getLayerOutputs()));
        } else {
            this.loss = ErrorLossFunctions.lossFunction(lossFunction, trueValues[0], this.getLastLayer().getLayerOutputs());
        }
    }

    private DenseLayers(int layers, int[] numberOfNeuronsPerLayer, double[][] initialWeights, double[]biases, double[][] initialInput, DEFAULT_ACTIVATION_FUNCTIONS activationFunction, Map<Integer, DEFAULT_ACTIVATION_FUNCTIONS> activationFunctionsMap, DEFAULT_LOSS_FUNCTIONS lossFunction, int[][] hotOneVec, int[] trueValues) throws Exception {
        if(hotOneVec.length != 0 && trueValues.length != 0) { throw new RuntimeException("Cannot initialize both Hot-One-Vector and a True-Value! You must select one method of error/loss checking!"); }
        // check if both the one hot vector mappings true values mappings are empty
        if(hotOneVec.length == 0 && trueValues.length == 0) { throw new RuntimeException("Must initialize either Hot-One-Vector or a True-Value!"); }
        this.layerList = new ArrayList<>();
        this.initialInput = initialInput;

        IntStream.range(0, layers).forEach(i -> {
            DEFAULT_ACTIVATION_FUNCTIONS activationFunctionToUse = activationFunctionsMap.getOrDefault(i, activationFunction);
            try {
                if (i != 0) {
                    double[][] weights = randn(numberOfNeuronsPerLayer[i], numberOfNeuronsPerLayer[i - 1]);
                    layerList.add(new DenseLayer(weights, layerList.get(i - 1).getLayerOutputs(), activationFunctionToUse));
                } else {
                    layerList.add(new DenseLayer(initialWeights, biases, initialInput[i], activationFunctionToUse));
                }
            } catch (Exception e){
                throw new RuntimeException("[DenseLayers Constructor] Error During Layer Creation!");
            }
        });

        if(lossFunction == DEFAULT_LOSS_FUNCTIONS.NLL_LOSS_FUNCTION) {
            this.loss =  MathUtils.mean(ErrorLossFunctions.lossFunction(lossFunction, hotOneVec[0], this.getLastLayer().getLayerOutputs()));
        } else {
            this.loss = ErrorLossFunctions.lossFunction(lossFunction, trueValues[0], this.getLastLayer().getLayerOutputs());
        }
    }

    // method to keep going depending on the accuracy and loss
    public void train() {
        // this.generateBatchDecisionsMap();
        // LOG
        // keep calling backPropagate until desired accuracy or loss is reached
    }

//    // method to enable backpropagation and training
//    public void backPropagate(Map<Integer, Double> decisions){
//        // LOG
//        printInfo("Back-Propagating...");
//        AtomicInteger index = new AtomicInteger(this.getLayerList().size() - 1);
//        double[] gradient;
//        // This will parallelize the traversal of the array from back to front.
//        IntStream.range(0, this.getLayerList().size()).parallel().map(i -> index.getAndDecrement()).forEachOrdered(i -> {
//            // generate decisions
//            this.generateBatchDecisionsMap();
//            if(index.get() == this.getLayerList().size() - 1){ // if output layer
//                IntStream.range(0, this.getLayerAt(index.get()).getNeuronList().size()).parallel().forEachOrdered( j -> {
//                    try {
//                        Neuron neuron = this.getNeuronFromLayerAt(i, j);
//                        // derivative of activation function, pass in raw output from selected neuron
//                        double A_prime = Activation.derivative(this.getActivationFunctionFrom(index.get()), neuron.getOutput());
//                        // derivative of loss function, pass in outputs of output layer and true values
//                        double[] EL_prime = ErrorLoss.derivative(this.getLayerAt(index.get()).getLossFunction(), this.getLayerAt(index.get()).getBatchTrueValues()[0], this.getLayerAt(index.get()).getLayerOutputs());
//                        double errorSignal =
//                    } catch (Exception e) {
//                        throw new RuntimeException(e);
//                    }
//                });
//
//            } else {
//            }
//        });
//        // iterate over the array from back to front
//            // if output layer
//                // call the derivative of the activation function (pass raw input in), the derivative of the loss function for each neuron to create error signal (pass activation function output)
//                // update bias for each neuron
//                // update the weight for each neuron by multiplying output by error signal
//            // if not output layer
//                // sum the derivate of the acttivation function multiplied by each weight, sum the gradient from the next layer, multiply both sums to get the error signal for the hidden layer
//                // update bias for each neuron
//                // update the weigh for each neuron by muliplying output by associated input by error signal
//            // LOG
//    }
     // calls the getDoubles random matrix generator
    public static double[][] randn(int rows, int cols) {
        return getRandomMatrix(rows, cols);
    }

    // creates a metrix of randomly generated numbers
    static double[][] getRandomMatrix(int rows, int cols) {
        double[][] output = new double[rows][cols];
        for (int i = 0; i < output.length; i++)
            for (int j = 0; j < output[0].length; j++)
                output[i][j] = (0.1 * random.nextGaussian());
        return output;
    }

    public ArrayList<Neuron> getNeuronsFromLayerAt(int index){
        return this.getLayerList().get(index).getNeuronList();
    }

    public Neuron getNeuronFromLayerAt(int layerIndex, int nueronIndex){
        return this.getLayerList().get(layerIndex).getNeuronList().get(nueronIndex);
    }

    // gets the outputs (or decisions) of each layer in the network
    public double[] batchDecisions(){
        double[] decisions = new double[this.getLastLayer().getLayerOutputs().length];
        IntStream.range(0, decisions.length).parallel().forEachOrdered( i -> {
            decisions[i] = decision(this.getLastLayer().getLayerOutputs())[1];
        });
        return decisions;
    }

    public double[] decision(double[] values){
        return MathUtils.indexAndMax(values);
    }

    public double[] getOutputs(){
        return getLayerAt(getLayerList().size()-1).getLayerOutputs();
    }

    /**
     * creates a map of decisions made in each layer, and notes the index of the node that was chosen in each layer
     */
    public void generateDecisionsMap(){
        // create the decision map
        this.decisions = new HashMap<>();
        this.decisionsIndex = new int[this.getLastLayer().getLayerOutputs().length];
        IntStream.range(0, this.decisionsIndex.length).parallel().forEachOrdered( i -> {
            double[] indexAndMax = decision(this.getLastLayer().getLayerOutputs());
            this.decisionsIndex[i] = (int)indexAndMax[0];
            this.decisions.put(i, indexAndMax[1]);
        });
        if(this.getTrueValues()[this.batchCounter].length > 0){
            this.accuracy = accuracy(this.getTrueValues()[this.batchCounter], this.decisionsIndex);
            printInfo("Using Batch True Values");
            printInfo("Decisions:", this.decisions);
            printInfo("Decision Indices:", this.decisionsIndex);
            printInfo("True Values:", this.getTrueValues()[this.batchCounter]);
            printInfo("Accuracy: ", this.accuracy);
            this.batchCounter++;
        }
        else if(this.getHotOneVecs()[this.batchCounter].length > 0){
            this.accuracy = accuracy(this.getHotOneVecs()[this.batchCounter], this.decisionsIndex);
            printInfo("Using Batch Hot One Vectors");
            printInfo("Decisions:", this.decisions);
            printInfo("Decision Indices:", this.decisionsIndex);
            printInfo("Accuracy: ", this.accuracy);
            this.batchCounter++;
        }
    }

    public DEFAULT_ACTIVATION_FUNCTIONS getActivationFunctionFrom(int index){
        return this.getLayerAt(index).getActivationFunction();
    }

    public DenseLayer getLayerAt(int index){
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
    public static class DenseLayersBuilder{
        private ArrayList<DenseLayer> layerList;
        private boolean isUsingListOfLayers = true;

        private DEFAULT_ACTIVATION_FUNCTIONS activationFunction = DEFAULT_ACTIVATION_FUNCTIONS.SOFTMAX_ACTIVATION_FUNCTION;
        private final Map<Integer, DEFAULT_ACTIVATION_FUNCTIONS> activationFunctionsMap = new HashMap<>();
        private boolean hasSetSpecificLayerActivationFunctions = false;

        private DEFAULT_LOSS_FUNCTIONS lossFunction = DEFAULT_LOSS_FUNCTIONS.NLL_LOSS_FUNCTION;
        private int[][] hotOneVec;
        private int[] trueValueIndex;
        private final Map<Integer, DEFAULT_LOSS_FUNCTIONS> lossFunctionsMap = new HashMap<>();
        private final Map<Integer, int[][]> hotOneMap = new HashMap<>();
        private final Map<Integer, int[]> trueValueIndexMap = new HashMap<>();
        private boolean hasSetSpecificLayerLossFunctions = false;
        private boolean isUsingTrueValueIndex = false;
        private boolean isUsingHotEncodedVec = true;

        private int numberOfLayers = 1;
        private boolean isUsingNumberOfLayers = false;

        private int numberOfInputs;
        private double[][] initialInput;
        private boolean isUsingBatchInputs = false;

        private double[][] initialWeights;
        private boolean isUsingSpecificWeights = false;

        private double[] initialBiases;
        private boolean isUsingSpecificBiases = false;
        private boolean isSettingSpecificBiases = false;

        private int[] numberOfNeuronsPerLayer;
        private boolean isUsingSpecificNeurons = false;
        private int numberOfNeurons;
        private boolean isUsingNumberOfNeurons = false;
        private String filePath = "";
        private boolean isUsingFileAsInput= false;


        public DenseLayersBuilder withTextFileAsInput(String filePath, String delimiter){
            this.filePath = filePath;
            this.isUsingFileAsInput = true;
            if (filePath.charAt(filePath.length()-1) == 't' && filePath.charAt(filePath.length()-2) == 'x' && filePath.charAt(filePath.length()-3) == 't' && filePath.charAt(filePath.length()-4) == '.'){
                this.initialInput = FileUtils.readTextFile(filePath, delimiter);

            } else if (filePath.charAt(filePath.length()-1) == 'v' && filePath.charAt(filePath.length()-2) == 's' && filePath.charAt(filePath.length()-3) == 'c' && filePath.charAt(filePath.length()-4) == '.'){
                this.initialInput = FileUtils.readCsvFile(filePath);
            } else {
                throw new IllegalArgumentException("Error: file must be of type txt or csv");
            }
            return this;
        }

        public DenseLayersBuilder withLayerList(ArrayList<DenseLayer> layerList){
            this.isUsingListOfLayers = true;
            this.layerList = layerList;
            return this;
        }

        public DenseLayersBuilder withNumberOfLayers(int numberOfLayers){
            this.isUsingNumberOfLayers = true;
            this.numberOfLayers = numberOfLayers;
            return this;
        }

        public DenseLayersBuilder withNumberOfInputs(int numberOfInputs){
            this.isUsingBatchInputs = false;
            this.numberOfInputs = numberOfInputs;
            return this;
        }

        public DenseLayersBuilder withInitialInput(double[][] initialInput){
            this.isUsingBatchInputs = true;
            this.isUsingFileAsInput = false;
            this.initialInput = initialInput;
            return this;
        }

        public DenseLayersBuilder withInitialWeights(double[][] initialWeights){
            this.isUsingSpecificWeights = true;
            this.initialWeights = initialWeights;
            return this;
        }

        public DenseLayersBuilder withSpecificBiasForNeuronInLayer(int layer, int neuron, double bias){
            this.isSettingSpecificBiases = true;
            this.layerList.get(layer).getNeuronList().get(neuron).setBias(bias);
            return this;
        }

        public DenseLayersBuilder withInitialBiases(double[] initialBiases){
            this.isUsingSpecificBiases = true;
            this.initialBiases = initialBiases;
            return this;
        }

        public DenseLayersBuilder withNumberOfNeuronsPerLayer(int[] numberOfNeuronsPerLayer){
            this.isUsingSpecificNeurons = true;
            this.isUsingNumberOfNeurons = false;
            this.numberOfNeuronsPerLayer = numberOfNeuronsPerLayer;
            return this;
        }

        public DenseLayersBuilder withNumberOfNeurons(int numberOfNeurons){
            this.isUsingNumberOfNeurons = true;
            this.isUsingSpecificNeurons = false;
            this.numberOfNeurons = numberOfNeurons;
            return this;
        }

        public DenseLayersBuilder withActivationFunction(DEFAULT_ACTIVATION_FUNCTIONS activationFunction){
            this.activationFunction = activationFunction;
            return this;
        }

        public DenseLayersBuilder withSingleActivationFunctionForSingleLayer(int layer, DEFAULT_ACTIVATION_FUNCTIONS activationFunction) {
            activationFunctionsMap.put(layer, activationFunction);
            return this;
        }

        public DenseLayersBuilder withSingleActivationFunctionForMultipleLayers(int startingLayer, int endingLayer, DEFAULT_ACTIVATION_FUNCTIONS activationFunction) {
            IntStream.range(startingLayer, endingLayer+1).forEachOrdered( i -> activationFunctionsMap.put(i, activationFunction));
            return this;
        }
        public DenseLayersBuilder withLossFunction(DEFAULT_LOSS_FUNCTIONS lossFunction){
             this.lossFunction = lossFunction;
             return this;
        }
        public DenseLayersBuilder withTrueValue(int[] trueValueIndex){
            this.isUsingTrueValueIndex = true;
            this.isUsingHotEncodedVec = false;
            this.trueValueIndex = trueValueIndex;
            return this;
        }
        public DenseLayersBuilder withHotOneVector(int[][] hotOneVec){
            this.isUsingTrueValueIndex = false;
            this.isUsingHotEncodedVec = true;
            this.hotOneVec = hotOneVec;
            return this;
        }

        public DenseLayersBuilder withLossFunction(DEFAULT_LOSS_FUNCTIONS lossFunction, int[][] hotOneOutput){
            this.lossFunction = lossFunction;
            this.hotOneVec = hotOneOutput;
            return this;
        }
        public DenseLayersBuilder withLossFunction(DEFAULT_LOSS_FUNCTIONS lossFunction, int[] trueValueIndex){
            this.lossFunction = lossFunction;
            this.trueValueIndex = trueValueIndex;
            return this;
        }
        public DenseLayersBuilder withSingleLossFunctionForSingleLayer(int layer, DEFAULT_LOSS_FUNCTIONS lossFunction, int[][] hotOneOutput){
            lossFunctionsMap.put(layer, lossFunction);
            hotOneMap.put(layer, hotOneOutput);
            return this;
        }
        public DenseLayersBuilder withSingleLossFunctionForSingleLayer(int layer, DEFAULT_LOSS_FUNCTIONS lossFunction, int[] trueValueIndex){
            lossFunctionsMap.put(layer, lossFunction);
            trueValueIndexMap.put(layer, trueValueIndex);
            return this;
        }
        public DenseLayersBuilder withSingleLossFunctionForMultipleLayers(int startingLayer, int endingLayer, DEFAULT_LOSS_FUNCTIONS lossFunction, int[][][] hotOneOutputs){
            IntStream.range(startingLayer, endingLayer+1).forEachOrdered(i -> {
                lossFunctionsMap.put(i, lossFunction);
                hotOneMap.put(i, hotOneOutputs[i]);
            });
            this.hasSetSpecificLayerLossFunctions = true;
            return this;
        }
        public DenseLayersBuilder withSingleLossFunctionForMultipleLayers(int startingLayer, int endingLayer, DEFAULT_LOSS_FUNCTIONS lossFunction, int[][] trueIndices){
            IntStream.range(startingLayer, endingLayer+1).forEachOrdered(i -> {
                lossFunctionsMap.put(i, lossFunction);
                trueValueIndexMap.put(i, trueIndices[i]);
            });
            this.hasSetSpecificLayerLossFunctions = true;
            return this;
        }

        public DenseLayers build(){
            DenseLayers built = null;
            if(!this.isUsingNumberOfLayers && !this.isUsingListOfLayers) throw new IllegalArgumentException("Please use the numberOfLayers() builder method to initialize, or provide an ArrayList<DenseLayer> using the withLayerList() builder method!");
            if(!this.isUsingSpecificNeurons && !this.isUsingNumberOfNeurons && this.isUsingNumberOfLayers && !this.isUsingBatchInputs){
                if(this.isUsingFileAsInput && this.isUsingSpecificWeights && this.isUsingSpecificBiases) {
                    printInfo("Using construct 1.1");
                    built = new DenseLayers(this.numberOfLayers, this.initialWeights, this.initialBiases, this.initialInput, this.activationFunction, this.activationFunctionsMap,
                            this.lossFunction, this.lossFunctionsMap, this.hotOneVec, this.hotOneMap, this.trueValueIndex, this.trueValueIndexMap, this.isUsingHotEncodedVec);
                    printPositive("Neural network successfully build!");
                    printInfo("Created dense layer neural network: " + built);
                    printInfo("Number of layers set to " + this.numberOfLayers);
                    printInfo("Initial weights set to", this.initialWeights);
                    printInfo("Initial biases set to", this.initialBiases);
                    printInfo("Initial inputs set to", this.initialInput);
                    printInfo("Initial activation function set to " + this.activationFunction.name());
                    printInfo("Initial activation function set to " + this.activationFunction.name());
                    printInfo("Initial activation function map set to", this.activationFunctionsMap);
                    printInfo("Initial loss function map set to", this.lossFunctionsMap);
                    if (isUsingTrueValueIndex) {printInfo("Initial true value index set to", this.trueValueIndex);}
                    else {printInfo("Initial hot one vector set to", this.hotOneVec);}
                    printInfo("Initial hot one map set to", this.hotOneMap);
                    printInfo("Initial true value index map set to", this.trueValueIndexMap);
                }
                else if(this.isUsingFileAsInput && this.isUsingSpecificWeights) {
                    printInfo("Using construct 1.2");
                    built = new DenseLayers(this.numberOfLayers, this.initialWeights, this.initialInput, this.activationFunction, this.activationFunctionsMap,
                            this.lossFunction, this.lossFunctionsMap, this.hotOneVec, this.hotOneMap, this.trueValueIndex, this.trueValueIndexMap, this.isUsingHotEncodedVec);
                    printPositive("Neural network successfully build!");
                    printInfo("Created dense layer neural network: " + built);
                    printInfo("Number of layers set to " + this.numberOfLayers);
                    printInfo("Initial weights set to", this.initialWeights);
                    printInfo("Initial biases not set!");
                    printInfo("Initial weights set to", this.initialInput);
                    printInfo("Initial weights set to " + this.activationFunction.name());
                    printInfo("Initial activation function set to " + this.activationFunction.name());
                    printInfo("Initial activation function map set to", this.activationFunctionsMap);
                    printInfo("Initial loss function map set to", this.lossFunctionsMap);
                    if (isUsingTrueValueIndex) {printInfo("Initial true value index set to", this.trueValueIndex);}
                    else {printInfo("Initial hot one vector set to", this.hotOneVec);}
                    printInfo("Initial hot one map set to", this.hotOneMap);
                    printInfo("Initial true value index map set to", this.trueValueIndexMap);
                }
                else if(this.isUsingFileAsInput) {
                    printInfo("Using construct 1.3");
                    built = new DenseLayers(this.numberOfLayers, this.initialInput, this.activationFunction, this.activationFunctionsMap,
                            this.lossFunction, this.lossFunctionsMap, this.hotOneVec, this.hotOneMap, this.trueValueIndex, this.trueValueIndexMap, this.isUsingHotEncodedVec);
                    printPositive("Neural network successfully build!");
                    printInfo("Created dense layer neural network: " + built);
                    printInfo("Number of layers set to " + this.numberOfLayers);
                    printInfo("Initial weights not set!");
                    printInfo("Initial biases not set!");
                    printInfo("Initial weights set to", this.initialInput);
                    printInfo("Initial activation function set to " + this.activationFunction.name());
                    printInfo("Initial activation function set to " + this.activationFunction.name());
                    printInfo("Initial activation function map set to", this.activationFunctionsMap);
                    printInfo("Initial loss function map set to", this.lossFunctionsMap);
                    if (isUsingTrueValueIndex) {printInfo("Initial true value index set to", this.trueValueIndex);}
                    else {printInfo("Initial hot one vector set to", this.hotOneVec);}
                    printInfo("Initial hot one map set to", this.hotOneMap);
                    printInfo("Initial true value index map set to", this.trueValueIndexMap);
                }
                else if(!this.isUsingFileAsInput && !this.isUsingSpecificWeights && !this.isUsingSpecificBiases) {
                    printInfo("Using construct 1.4");
                    built = new DenseLayers(this.numberOfLayers, this.activationFunction, this.activationFunctionsMap,
                            this.lossFunction, this.lossFunctionsMap, this.hotOneVec, this.hotOneMap, this.trueValueIndex, this.trueValueIndexMap, this.isUsingHotEncodedVec);
                    printPositive("Neural network successfully build!");
                    printInfo("Created dense layer neural network: " + built);
                    printInfo("Number of layers set to " + this.numberOfLayers);
                    printInfo("Initial weights not set!");
                    printInfo("Initial biases not set!");
                    printInfo("Initial inputs not set!");
                    printInfo("Initial activation function set to " + this.activationFunction.name());
                    printInfo("Initial activation function map set to", this.activationFunctionsMap);
                    printInfo("Initial loss function map set to", this.lossFunctionsMap);
                    if (isUsingTrueValueIndex) {printInfo("Initial true value index set to", this.trueValueIndex);}
                    else {printInfo("Initial hot one vector set to", this.hotOneVec);}
                    printInfo("Initial hot one map set to", this.hotOneMap);
                    printInfo("Initial true value index map set to", this.trueValueIndexMap);
                }
            }
            else if(this.isUsingNumberOfNeurons && this.isUsingNumberOfLayers && !this.isUsingBatchInputs) {
                if(this.isUsingSpecificWeights && this.isUsingSpecificBiases) {
                    printInfo("Using construct 2.1");
                    built = new DenseLayers(this.numberOfLayers, this.numberOfNeurons, this.initialWeights, this.initialBiases, this.initialInput, this.activationFunction, this.activationFunctionsMap,
                            this.lossFunction, this.lossFunctionsMap, this.hotOneVec, this.hotOneMap, this.trueValueIndex, this.trueValueIndexMap, this.isUsingHotEncodedVec);
                    printPositive("Neural network successfully build!");
                    printInfo("Created dense layer neural network: " + built);
                    printInfo("Number of layers set to " + this.numberOfLayers);
                    printInfo("Initial weights set to", this.initialWeights);
                    printInfo("Initial biases set to", this.initialBiases);
                    printInfo("Initial input set to", this.initialInput);
                    printInfo("Initial activation function set to " + this.activationFunction.name());
                    printInfo("Initial activation function map set to", this.activationFunctionsMap);
                    printInfo("Initial loss function map set to", this.lossFunctionsMap);
                    if (isUsingTrueValueIndex) {printInfo("Initial true value index set to", this.trueValueIndex);}
                    else {printInfo("Initial hot one vector set to", this.hotOneVec);}
                    printInfo("Initial hot one map set to", this.hotOneMap);
                    printInfo("Initial true value index map set to", this.trueValueIndexMap);
                }
                else if(this.isUsingFileAsInput  && this.isUsingSpecificWeights) {
                    printInfo("Using construct 2.2");
                    built = new DenseLayers(this.numberOfLayers, this.numberOfNeurons, this.initialWeights, this.initialInput, this.activationFunction, this.activationFunctionsMap,
                            this.lossFunction, this.lossFunctionsMap, this.hotOneVec, this.hotOneMap, this.trueValueIndex, this.trueValueIndexMap, this.isUsingHotEncodedVec);
                    printPositive("Neural network successfully build!");
                    printInfo("Created dense layer neural network: " + built);
                    printInfo("Number of layers set to " + this.numberOfLayers);
                    printInfo("Initial weights set to", this.initialWeights);
                    printInfo("Initial biases not set!");
                    printInfo("Initial inputs set to", this.initialInput);
                    printInfo("Initial activation function set to " + this.activationFunction.name());
                    printInfo("Initial activation function map set to", this.activationFunctionsMap);
                    printInfo("Initial loss function map set to", this.lossFunctionsMap);
                    if (isUsingTrueValueIndex) {printInfo("Initial true value index set to", this.trueValueIndex);}
                    else {printInfo("Initial hot one vector set to", this.hotOneVec);}
                    printInfo("Initial hot one map set to", this.hotOneMap);
                    printInfo("Initial true value index map set to", this.trueValueIndexMap);
                }
                else if(this.isUsingFileAsInput) {
                    printInfo("Using construct 2.3");
                    built = new DenseLayers(this.numberOfLayers, this.numberOfNeurons, this.initialInput, this.activationFunction, this.activationFunctionsMap,
                            this.lossFunction, this.lossFunctionsMap, this.hotOneVec, this.hotOneMap, this.trueValueIndex, this.trueValueIndexMap, this.isUsingHotEncodedVec);
                    printPositive("Neural network successfully build!");
                    printInfo("Created dense layer neural network: " + built);
                    printInfo("Number of layers set to " + this.numberOfLayers);
                    printInfo("Initial weights not set!");
                    printInfo("Initial biases not set!");
                    printInfo("Initial weights set to", this.initialInput);
                    printInfo("Initial activation function set to " + this.activationFunction.name());
                    printInfo("Initial activation function map set to", this.activationFunctionsMap);
                    printInfo("Initial loss function map set to", this.lossFunctionsMap);
                    if (isUsingTrueValueIndex) {printInfo("Initial true value index set to", this.trueValueIndex);}
                    else {printInfo("Initial hot one vector set to", this.hotOneVec);}
                    printInfo("Initial hot one map set to", this.hotOneMap);
                    printInfo("Initial true value index map set to", this.trueValueIndexMap);
                }
                else if(!this.isUsingFileAsInput && !this.isUsingSpecificWeights && !this.isUsingSpecificBiases) {
                    printInfo("Using construct 2.4");
                    built = new DenseLayers(this.numberOfLayers, this.numberOfNeurons, this.activationFunction, this.activationFunctionsMap,
                            this.lossFunction, this.lossFunctionsMap, this.hotOneVec, this.hotOneMap, this.trueValueIndex, this.trueValueIndexMap, this.isUsingHotEncodedVec);
                    printPositive("Neural network successfully build!");
                    printInfo("Created dense layer neural network: " + built);
                    printInfo("Number of layers set to " + this.numberOfLayers);
                    printInfo("Initial weights not set!");
                    printInfo("Initial biases not set!");
                    printInfo("Initial inputs not set!");
                    printInfo("Initial activation function set to " + this.activationFunction.name());
                    printInfo("Initial activation function map set to", this.activationFunctionsMap);
                    printInfo("Initial loss function map set to", this.lossFunctionsMap);
                    if (isUsingTrueValueIndex) {printInfo("Initial true value index set to", this.trueValueIndex);}
                    else {printInfo("Initial hot one vector set to", this.hotOneVec);}
                    printInfo("Initial hot one map set to", this.hotOneMap);
                    printInfo("Initial true value index map set to", this.trueValueIndexMap);
                }
            }
            else if(this.isUsingSpecificNeurons && this.isUsingNumberOfLayers && !this.isUsingBatchInputs) {
                if(this.isUsingFileAsInput && this.isUsingSpecificWeights && this.isUsingSpecificBiases) {
                    printInfo("Using construct 3.1");
                    built = new DenseLayers(this.numberOfLayers, this.numberOfNeuronsPerLayer, this.initialWeights, this.initialBiases, this.initialInput, this.activationFunction, this.activationFunctionsMap,
                            this.lossFunction, this.lossFunctionsMap, this.hotOneVec, this.hotOneMap, this.trueValueIndex, this.trueValueIndexMap, this.isUsingHotEncodedVec);
                    printPositive("Neural network successfully build!");
                    printInfo("Created dense layer neural network: " + built);
                    printInfo("Number of layers set to " + this.numberOfLayers);
                    printInfo("Initial weights set to", this.initialWeights);
                    printInfo("Initial biases set to", this.initialBiases);
                    printInfo("Initial inputs set to", this.initialInput);
                    printInfo("Initial activation function set to " + this.activationFunction.name());
                    printInfo("Initial activation function map set to", this.activationFunctionsMap);
                    printInfo("Initial loss function map set to", this.lossFunctionsMap);
                    if (isUsingTrueValueIndex) {printInfo("Initial true value index set to", this.trueValueIndex);}
                    else {printInfo("Initial hot one vector set to", this.hotOneVec);}
                    printInfo("Initial hot one map set to", this.hotOneMap);
                    printInfo("Initial true value index map set to", this.trueValueIndexMap);
                }
                else if(this.isUsingFileAsInput && this.isUsingSpecificWeights) {
                    printInfo("Using construct 3.2");
                    built = new DenseLayers(this.numberOfLayers, this.numberOfNeuronsPerLayer, this.initialWeights, this.initialInput, this.activationFunction, this.activationFunctionsMap,
                            this.lossFunction, this.lossFunctionsMap, this.hotOneVec, this.hotOneMap, this.trueValueIndex, this.trueValueIndexMap, this.isUsingHotEncodedVec);
                    printPositive("Neural network successfully build!");
                    printInfo("Created dense layer neural network: " + built);
                    printInfo("Number of layers set to " + this.numberOfLayers);
                    printInfo("Initial weights set to", this.initialWeights);
                    printInfo("Initial biases not set!");
                    printInfo("Initial inputs set to", this.initialInput);
                    printInfo("Initial activation function set to " + this.activationFunction.name());
                    printInfo("Initial activation function map set to", this.activationFunctionsMap);
                    printInfo("Initial loss function map set to", this.lossFunctionsMap);
                    if (isUsingTrueValueIndex) {printInfo("Initial true value index set to", this.trueValueIndex);}
                    else {printInfo("Initial hot one vector set to", this.hotOneVec);}
                    printInfo("Initial hot one map set to", this.hotOneMap);
                    printInfo("Initial true value index map set to", this.trueValueIndexMap);
                }
                else if(this.isUsingFileAsInput) {
                    printInfo("Using construct 3.3");
                    built = new DenseLayers(this.numberOfLayers, this.numberOfNeuronsPerLayer, this.initialInput, this.activationFunction, this.activationFunctionsMap,
                            this.lossFunction, this.lossFunctionsMap, this.hotOneVec, this.hotOneMap, this.trueValueIndex, this.trueValueIndexMap, this.isUsingHotEncodedVec);
                    printPositive("Neural network successfully build!");
                    printInfo("Created dense layer neural network: " + built);
                    printInfo("Number of layers set to " + this.numberOfLayers);
                    printInfo("Initial weights not set!");
                    printInfo("Initial biases not set!");
                    printInfo("Initial inputs set to", this.initialInput);
                    printInfo("Initial activation function set to " + this.activationFunction.name());
                    printInfo("Initial activation function map set to", this.activationFunctionsMap);
                    printInfo("Initial loss function map set to", this.lossFunctionsMap);
                    if (isUsingTrueValueIndex) {printInfo("Initial true value index set to", this.trueValueIndex);}
                    else {printInfo("Initial hot one vector set to", this.hotOneVec);}
                    printInfo("Initial hot one map set to", this.hotOneMap);
                    printInfo("Initial true value index map set to", this.trueValueIndexMap);
                }
                else if(!this.isUsingFileAsInput && !this.isUsingSpecificWeights && !this.isUsingSpecificBiases) {
                    printInfo("Using construct 3.4");
                    built = new DenseLayers(this.numberOfLayers, this.numberOfNeuronsPerLayer, this.activationFunction, this.activationFunctionsMap,
                            this.lossFunction, this.lossFunctionsMap, this.hotOneVec, this.hotOneMap, this.trueValueIndex, this.trueValueIndexMap, this.isUsingHotEncodedVec);
                    printPositive("Neural network successfully build!");
                    printInfo("Created dense layer neural network: " + built);
                    printInfo("Number of layers set to " + this.numberOfLayers);
                    printInfo("Initial weights not set!");
                    printInfo("Initial biases not set!");
                    printInfo("Initial inputs not set!");
                    printInfo("Initial activation function set to " + this.activationFunction.name());
                    printInfo("Initial activation function map set to", this.activationFunctionsMap);
                    printInfo("Initial loss function map set to", this.lossFunctionsMap);
                    if (isUsingTrueValueIndex) {printInfo("Initial true value index set to", this.trueValueIndex);}
                    else {printInfo("Initial hot one vector set to", this.hotOneVec);}
                    printInfo("Initial hot one map set to", this.hotOneMap);
                    printInfo("Initial true value index map set to", this.trueValueIndexMap);
                }
            }
            else if(!this.isUsingSpecificNeurons && !this.isUsingNumberOfNeurons && this.isUsingNumberOfLayers && !this.isUsingFileAsInput){
                if(this.isUsingBatchInputs && this.isUsingSpecificWeights && this.isUsingSpecificBiases) {
                    printInfo("Using construct 4.1");
                    built = new DenseLayers(this.numberOfLayers, this.initialWeights, this.initialBiases, this.initialInput, this.activationFunction, this.activationFunctionsMap,
                            this.lossFunction, this.lossFunctionsMap, this.hotOneVec, this.hotOneMap, this.trueValueIndex, this.trueValueIndexMap, this.isUsingHotEncodedVec);
                    printPositive("Neural network successfully build!");
                    printInfo("Created dense layer neural network: " + built);
                    printInfo("Number of layers set to " + this.numberOfLayers);
                    printInfo("Initial weights set to", this.initialWeights);
                    printInfo("Initial biases set to", this.initialBiases);
                    printInfo("Initial inputs set to", this.initialInput);
                    printInfo("Initial activation function set to " + this.activationFunction.name());
                    printInfo("Initial activation function map set to", this.activationFunctionsMap);
                    printInfo("Initial loss function map set to", this.lossFunctionsMap);
                    if (isUsingTrueValueIndex) {printInfo("Initial true value index set to", this.trueValueIndex);}
                    else {printInfo("Initial hot one vector set to", this.hotOneVec);}
                    printInfo("Initial hot one map set to", this.hotOneMap);
                    printInfo("Initial true value index map set to", this.trueValueIndexMap);
                }
                else if(this.isUsingBatchInputs  && this.isUsingSpecificWeights) {
                    printInfo("Using construct 4.2");
                    built = new DenseLayers(this.numberOfLayers, this.initialWeights, this.initialInput, this.activationFunction, this.activationFunctionsMap,
                            this.lossFunction, this.lossFunctionsMap, this.hotOneVec, this.hotOneMap, this.trueValueIndex, this.trueValueIndexMap, this.isUsingHotEncodedVec);
                    printPositive("Neural network successfully build!");
                    printInfo("Created dense layer neural network: " + built);
                    printInfo("Number of layers set to " + this.numberOfLayers);
                    printInfo("Initial weights set to", this.initialWeights);
                    printInfo("Initial biases not set!");
                    printInfo("Initial inputs set to", this.initialInput);
                    printInfo("Initial activation function set to " + this.activationFunction.name());
                    printInfo("Initial activation function map set to", this.activationFunctionsMap);
                    printInfo("Initial loss function map set to", this.lossFunctionsMap);
                    if (isUsingTrueValueIndex) {printInfo("Initial true value index set to", this.trueValueIndex);}
                    else {printInfo("Initial hot one vector set to", this.hotOneVec);}
                    printInfo("Initial hot one map set to", this.hotOneMap);
                    printInfo("Initial true value index map set to", this.trueValueIndexMap);
                }
                else if(this.isUsingBatchInputs) {
                    printInfo("Using construct 4.3");
                    built = new DenseLayers(this.numberOfLayers, this.initialInput, this.activationFunction, this.activationFunctionsMap,
                            this.lossFunction, this.lossFunctionsMap, this.hotOneVec, this.hotOneMap, this.trueValueIndex, this.trueValueIndexMap, this.isUsingHotEncodedVec);
                    printPositive("Neural network successfully build!");
                    printInfo("Created dense layer neural network: " + built);
                    printInfo("Number of layers set to " + this.numberOfLayers);
                    printInfo("Initial weights not set!");
                    printInfo("Initial biases not set!");
                    printInfo("Initial inputs set to", this.initialInput);
                    printInfo("Initial activation function set to " + this.activationFunction.name());
                    printInfo("Initial activation function map set to", this.activationFunctionsMap);
                    printInfo("Initial loss function map set to", this.lossFunctionsMap);
                    if (isUsingTrueValueIndex) {printInfo("Initial true value index set to", this.trueValueIndex);}
                    else {printInfo("Initial hot one vector set to", this.hotOneVec);}
                    printInfo("Initial hot one map set to", this.hotOneMap);
                    printInfo("Initial true value index map set to", this.trueValueIndexMap);
                }
                else if(!this.isUsingBatchInputs && !this.isUsingSpecificWeights && !this.isUsingSpecificBiases) {
                    printInfo("Using construct 4.4");
                    built = new DenseLayers(this.numberOfLayers, this.activationFunction, this.activationFunctionsMap,
                            this.lossFunction, this.lossFunctionsMap, this.hotOneVec, this.hotOneMap, this.trueValueIndex, this.trueValueIndexMap, this.isUsingHotEncodedVec);
                    printPositive("Neural network successfully build!");
                    printInfo("Created dense layer neural network: " + built);
                    printInfo("Number of layers set to " + this.numberOfLayers);
                    printInfo("Initial weights not set!");
                    printInfo("Initial biases not set!");
                    printInfo("Initial inputs not set!");
                    printInfo("Initial activation function set to " + this.activationFunction.name());
                    printInfo("Initial activation function map set to", this.activationFunctionsMap);
                    printInfo("Initial loss function map set to", this.lossFunctionsMap);
                    if (isUsingTrueValueIndex) {printInfo("Initial true value index set to", this.trueValueIndex);}
                    else {printInfo("Initial hot one vector set to", this.hotOneVec);}
                    printInfo("Initial hot one map set to", this.hotOneMap);
                    printInfo("Initial true value index map set to", this.trueValueIndexMap);
                }
            }
            else if(this.isUsingNumberOfNeurons && this.isUsingNumberOfLayers && !this.isUsingFileAsInput) {
                if(this.isUsingBatchInputs && this.isUsingSpecificWeights && this.isUsingSpecificBiases) {
                    printInfo("Using construct 5.1");
                    built = new DenseLayers(this.numberOfLayers, this.numberOfNeurons, this.initialWeights, this.initialBiases, this.initialInput, this.activationFunction, this.activationFunctionsMap,
                            this.lossFunction, this.lossFunctionsMap, this.hotOneVec, this.hotOneMap, this.trueValueIndex, this.trueValueIndexMap, this.isUsingHotEncodedVec);
                    printPositive("Neural network successfully build!");
                    printInfo("Created dense layer neural network: " + built);
                    printInfo("Number of layers set to " + this.numberOfLayers);
                    printInfo("Initial weights set to", this.initialWeights);
                    printInfo("Initial biases set to", this.initialBiases);
                    printInfo("Initial inputs set to", this.initialInput);
                    printInfo("Initial activation function set to " + this.activationFunction.name());
                    printInfo("Initial activation function map set to", this.activationFunctionsMap);
                    printInfo("Initial loss function map set to", this.lossFunctionsMap);
                    if (isUsingTrueValueIndex) {printInfo("Initial true value index set to", this.trueValueIndex);}
                    else {printInfo("Initial hot one vector set to", this.hotOneVec);}
                    printInfo("Initial hot one map set to", this.hotOneMap);
                    printInfo("Initial true value index map set to", this.trueValueIndexMap);
                }
                else if(this.isUsingBatchInputs  && this.isUsingSpecificWeights) {
                    printInfo("Using construct 5.2");
                    built = new DenseLayers(this.numberOfLayers, this.numberOfNeurons, this.initialWeights, this.initialInput, this.activationFunction, this.activationFunctionsMap,
                            this.lossFunction, this.lossFunctionsMap, this.hotOneVec, this.hotOneMap, this.trueValueIndex, this.trueValueIndexMap, this.isUsingHotEncodedVec);
                    printPositive("Neural network successfully build!");
                    printInfo("Created dense layer neural network: " + built);
                    printInfo("Number of layers set to " + this.numberOfLayers);
                    printInfo("Initial weights set to", this.initialWeights);
                    printInfo("Initial biases not set!");
                    printInfo("Initial inputs set to", this.initialInput);
                    printInfo("Initial activation function set to " + this.activationFunction.name());
                    printInfo("Initial activation function map set to", this.activationFunctionsMap);
                    printInfo("Initial loss function map set to", this.lossFunctionsMap);
                    if (isUsingTrueValueIndex) {printInfo("Initial true value index set to", this.trueValueIndex);}
                    else {printInfo("Initial hot one vector set to", this.hotOneVec);}
                    printInfo("Initial hot one map set to", this.hotOneMap);
                    printInfo("Initial true value index map set to", this.trueValueIndexMap);
                }
                else if(this.isUsingBatchInputs) {
                    printInfo("Using construct 5.3");
                    built = new DenseLayers(this.numberOfLayers, this.numberOfNeurons, this.initialInput, this.activationFunction, this.activationFunctionsMap,
                            this.lossFunction, this.lossFunctionsMap, this.hotOneVec, this.hotOneMap, this.trueValueIndex, this.trueValueIndexMap, this.isUsingHotEncodedVec);
                    printPositive("Neural network successfully build!");
                    printInfo("Created dense layer neural network: " + built);
                    printInfo("Number of layers set to " + this.numberOfLayers);
                    printInfo("Initial weights not set!");
                    printInfo("Initial biases not set!");
                    printInfo("Initial inputs set to", this.initialInput);
                    printInfo("Initial activation function set to " + this.activationFunction.name());
                    printInfo("Initial activation function map set to", this.activationFunctionsMap);
                    printInfo("Initial loss function map set to", this.lossFunctionsMap);
                    if (isUsingTrueValueIndex) {printInfo("Initial true value index set to", this.trueValueIndex);}
                    else {printInfo("Initial hot one vector set to", this.hotOneVec);}
                    printInfo("Initial hot one map set to", this.hotOneMap);
                    printInfo("Initial true value index map set to", this.trueValueIndexMap);
                }
                else if(!this.isUsingBatchInputs && !this.isUsingSpecificWeights && !this.isUsingSpecificBiases) {
                    printInfo("Using construct 5.4");
                    built = new DenseLayers(this.numberOfLayers, this.numberOfNeurons, this.activationFunction, this.activationFunctionsMap,
                            this.lossFunction, this.lossFunctionsMap, this.hotOneVec, this.hotOneMap, this.trueValueIndex, this.trueValueIndexMap, this.isUsingHotEncodedVec);
                    printPositive("Neural network successfully build!");
                    printInfo("Created dense layer neural network: " + built);
                    printInfo("Number of layers set to " + this.numberOfLayers);
                    printInfo("Initial weights not set!");
                    printInfo("Initial biases not set!");
                    printInfo("Initial inputs not set!");
                    printInfo("Initial activation function set to " + this.activationFunction.name());
                    printInfo("Initial activation function map set to", this.activationFunctionsMap);
                    printInfo("Initial loss function map set to", this.lossFunctionsMap);
                    if (isUsingTrueValueIndex) {printInfo("Initial true value index set to", this.trueValueIndex);}
                    else {printInfo("Initial hot one vector set to", this.hotOneVec);}
                    printInfo("Initial hot one map set to", this.hotOneMap);
                    printInfo("Initial true value index map set to", this.trueValueIndexMap);
                }
            }
            else if(this.isUsingSpecificNeurons && this.isUsingNumberOfLayers && !this.isUsingFileAsInput) {
                if(this.isUsingBatchInputs && this.isUsingSpecificWeights && this.isUsingSpecificBiases) {
                    printInfo("Using construct 6.1");
                    built = new DenseLayers(this.numberOfLayers, this.numberOfNeuronsPerLayer, this.initialWeights, this.initialBiases, this.initialInput, this.activationFunction, this.activationFunctionsMap,
                            this.lossFunction, this.lossFunctionsMap, this.hotOneVec, this.hotOneMap, this.trueValueIndex, this.trueValueIndexMap, this.isUsingHotEncodedVec);
                    printPositive("Neural network successfully build!");
                    printInfo("Created dense layer neural network: " + built);
                    printInfo("Number of layers set to " + this.numberOfLayers);
                    printInfo("Initial weights set to", this.initialWeights);
                    printInfo("Initial biases set to", this.initialBiases);
                    printInfo("Initial inputs set to", this.initialInput);
                    printInfo("Initial activation function set to " + this.activationFunction.name());
                    printInfo("Initial activation function map set to", this.activationFunctionsMap);
                    printInfo("Initial loss function map set to", this.lossFunctionsMap);
                    if (isUsingTrueValueIndex) {printInfo("Initial true value index set to", this.trueValueIndex);}
                    else {printInfo("Initial hot one vector set to", this.hotOneVec);}
                    printInfo("Initial hot one map set to", this.hotOneMap);
                    printInfo("Initial true value index map set to", this.trueValueIndexMap);
                }
                else if(this.isUsingBatchInputs  && this.isUsingSpecificWeights) {
                    printInfo("Using construct 6.2");
                    built = new DenseLayers(this.numberOfLayers, this.numberOfNeuronsPerLayer, this.initialWeights, this.initialInput, this.activationFunction, this.activationFunctionsMap,
                            this.lossFunction, this.lossFunctionsMap, this.hotOneVec, this.hotOneMap, this.trueValueIndex, this.trueValueIndexMap, this.isUsingHotEncodedVec);
                    printPositive("Neural network successfully build!");
                    printInfo("Created dense layer neural network: " + built);
                    printInfo("Number of layers set to " + this.numberOfLayers);
                    printInfo("Initial weights set to", this.initialWeights);
                    printInfo("Initial biases not set!");
                    printInfo("Initial inputs set to", this.initialInput);
                    printInfo("Initial activation function set to " + this.activationFunction.name());
                    printInfo("Initial activation function map set to", this.activationFunctionsMap);
                    printInfo("Initial loss function map set to", this.lossFunctionsMap);
                    if (isUsingTrueValueIndex) {printInfo("Initial true value index set to", this.trueValueIndex);}
                    else {printInfo("Initial hot one vector set to", this.hotOneVec);}
                    printInfo("Initial hot one map set to", this.hotOneMap);
                    printInfo("Initial true value index map set to", this.trueValueIndexMap);
                }
                else if(this.isUsingBatchInputs) {
                    printInfo("Using construct 6.3");
                    built = new DenseLayers(this.numberOfLayers, this.numberOfNeuronsPerLayer, this.initialInput, this.activationFunction, this.activationFunctionsMap,
                            this.lossFunction, this.lossFunctionsMap, this.hotOneVec, this.hotOneMap, this.trueValueIndex, this.trueValueIndexMap, this.isUsingHotEncodedVec);
                    printPositive("Neural network successfully build!");
                    printInfo("Created dense layer neural network: " + built);
                    printInfo("Number of layers set to " + this.numberOfLayers);
                    printInfo("Initial weights not set!");
                    printInfo("Initial biases not set!");
                    printInfo("Initial inputs set to", this.initialInput);
                    printInfo("Initial activation function set to " + this.activationFunction.name());
                    printInfo("Initial activation function map set to", this.activationFunctionsMap);
                    printInfo("Initial loss function map set to", this.lossFunctionsMap);
                    if (isUsingTrueValueIndex) {printInfo("Initial true value index set to", this.trueValueIndex);}
                    else {printInfo("Initial hot one vector set to", this.hotOneVec);}
                    printInfo("Initial hot one map set to", this.hotOneMap);
                    printInfo("Initial true value index map set to", this.trueValueIndexMap);
                }
                else if(!this.isUsingBatchInputs && !this.isUsingSpecificWeights && !this.isUsingSpecificBiases) {
                    printInfo("Using construct 6.4");
                    built = new DenseLayers(this.numberOfLayers, this.numberOfNeuronsPerLayer, this.activationFunction, this.activationFunctionsMap,
                            this.lossFunction, this.lossFunctionsMap, this.hotOneVec, this.hotOneMap, this.trueValueIndex, this.trueValueIndexMap, this.isUsingHotEncodedVec);
                    printPositive("Neural network successfully build!");
                    printInfo("Created dense layer neural network: " + built);
                    printInfo("Number of layers set to " + this.numberOfLayers);
                    printInfo("Initial weights not set!");
                    printInfo("Initial biases not set!");
                    printInfo("Initial inputs not set!");
                    printInfo("Initial activation function set to " + this.activationFunction.name());
                    printInfo("Initial activation function map set to", this.activationFunctionsMap);
                    printInfo("Initial loss function map set to", this.lossFunctionsMap);
                    if (isUsingTrueValueIndex) {printInfo("Initial true value index set to", this.trueValueIndex);}
                    else {printInfo("Initial hot one vector set to", this.hotOneVec);}
                    printInfo("Initial hot one map set to", this.hotOneMap);
                    printInfo("Initial true value index map set to", this.trueValueIndexMap);
                }
            }
            else {
                built = new DenseLayers(2, DEFAULT_ACTIVATION_FUNCTIONS.LINEAR_ACTIVATION_FUNCTION, this.activationFunctionsMap,
                        this.lossFunction, this.lossFunctionsMap, this.hotOneVec, this.hotOneMap, this.trueValueIndex, this.trueValueIndexMap, this.isUsingHotEncodedVec);
                if(this.isUsingBatchInputs && this.isUsingFileAsInput){
                    throw new RuntimeException(bold(red("Builder Not Configured Properly! Do Not Use File As Input and Batch Input Together!")));
                }
                throw new RuntimeException(bold(red("Builder Not Configured Properly!")));
            }
            built.generateMeanLoss();
            built.generateBatchDecisionsMap();
            printInfo("Predicited values", built.getDecisionsIndex());
            printInfo("Loss is", built.getMeanLoss());
            printInfo("Accuracy is", built.getAccuracy());
            printInfo("Neuron indices per batch are", built.getDecisionsIndex());
            printInfo("Actual output values of those neurons are", built.getDecisionsIndex());
            return built;
        }
    }
}
