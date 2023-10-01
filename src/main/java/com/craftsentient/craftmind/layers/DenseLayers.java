package com.craftsentient.craftmind.layers;

import com.craftsentient.craftmind.activation.DEFAULT_ACTIVATION_FUNCTIONS;
import com.craftsentient.craftmind.errorLoss.DEFAULT_LOSS_FUNCTIONS;
import com.craftsentient.craftmind.layer.DenseLayer;
import com.craftsentient.craftmind.utils.FileUtils;
import com.craftsentient.craftmind.utils.MathUtils;
import com.craftsentient.craftmind.neuron.Neuron;
import lombok.Getter;
import lombok.NoArgsConstructor;

import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;

import static com.craftsentient.craftmind.utils.PrintUtils.*;

@Getter
public class DenseLayers {
    private final ArrayList<DenseLayer> layerList;
    private final double[][] initialInput;
    public static final Random random = new Random(0);

    private DenseLayers(int layers, DEFAULT_ACTIVATION_FUNCTIONS activationFunction, Map<Integer, DEFAULT_ACTIVATION_FUNCTIONS> activationFunctionsMap,
                        DEFAULT_LOSS_FUNCTIONS lossFunction, Map<Integer, DEFAULT_LOSS_FUNCTIONS> lossFunctionsMap, int[][] hotOneVec, Map<Integer, int[][]> hotOneVecMap, int[] trueValues, Map<Integer, int[]> trueValuesMap) {
        if(!hotOneVecMap.isEmpty() && !trueValuesMap.isEmpty()) throw new RuntimeException("Cannot initialize both Hot-One-Vector and a True-Value! You must select one method of error/loss checking!");
        this.layerList = new ArrayList<>();
        this.initialInput = randn(layers,layers);
        IntStream.range(0, layers).forEach(i -> {
                DEFAULT_ACTIVATION_FUNCTIONS activationFunctionToUse = activationFunctionsMap.getOrDefault(i, activationFunction);
                DEFAULT_LOSS_FUNCTIONS lossFunctionToUse = lossFunctionsMap.getOrDefault(i, lossFunction);

            int[] trueValueToUse = trueValuesMap.getOrDefault(i, trueValues);
            int[][] hotOneVecToUse = new int[0][0];
            if (trueValueToUse.length == 0) hotOneVecMap.getOrDefault(i, hotOneVec);

            double[][] weights = randn(layers, layers);
            if(i != 0) layerList.add(new DenseLayer(weights, layerList.get(i-1).getBatchLayerOutputs(), activationFunctionToUse, lossFunctionToUse, hotOneVecToUse, trueValueToUse));
            else layerList.add(new DenseLayer(weights, initialInput, activationFunctionToUse, lossFunction, hotOneVecToUse, trueValueToUse));
        });
    }

    private DenseLayers(int layers, double[][] initialInput, DEFAULT_ACTIVATION_FUNCTIONS activationFunction, Map<Integer, DEFAULT_ACTIVATION_FUNCTIONS> activationFunctionsMap,
                        DEFAULT_LOSS_FUNCTIONS lossFunction, Map<Integer, DEFAULT_LOSS_FUNCTIONS> lossFunctionsMap, int[][] hotOneVec, Map<Integer, int[][]> hotOneVecMap, int[] trueValues, Map<Integer, int[]> trueValuesMap) {
        if(!hotOneVecMap.isEmpty() && !trueValuesMap.isEmpty()) throw new RuntimeException("Cannot initialize both Hot-One-Vector and a True-Value! You must select one method of error/loss checking!");
        this.initialInput = initialInput;
        this.layerList = new ArrayList<>();
        IntStream.range(0, layers).forEach(i -> {
            DEFAULT_ACTIVATION_FUNCTIONS activationFunctionToUse = activationFunctionsMap.getOrDefault(i, activationFunction);
            DEFAULT_LOSS_FUNCTIONS lossFunctionToUse = lossFunctionsMap.getOrDefault(i, lossFunction);

            int[] trueValueToUse = trueValuesMap.getOrDefault(i, trueValues);
            int[][] hotOneVecToUse = new int[0][0];
            if (trueValueToUse.length == 0) hotOneVecMap.getOrDefault(i, hotOneVec);

            double[][] weights = randn(initialInput.length, initialInput[0].length);
            if(i != 0) layerList.add(new DenseLayer(weights, layerList.get(i-1).getBatchLayerOutputs(), activationFunctionToUse, lossFunctionToUse, hotOneVecToUse, trueValueToUse));
            else layerList.add(new DenseLayer(weights, initialInput, activationFunctionToUse, lossFunctionToUse, hotOneVecToUse, trueValueToUse));
        });
    }

    private DenseLayers(int layers, double[][] initialWeights, double[][] initialInput, DEFAULT_ACTIVATION_FUNCTIONS activationFunction, Map<Integer, DEFAULT_ACTIVATION_FUNCTIONS> activationFunctionsMap,
                        DEFAULT_LOSS_FUNCTIONS lossFunction, Map<Integer, DEFAULT_LOSS_FUNCTIONS> lossFunctionsMap, int[][] hotOneVec, Map<Integer, int[][]> hotOneVecMap, int[] trueValues, Map<Integer, int[]> trueValuesMap) {
        if(!hotOneVecMap.isEmpty() && !trueValuesMap.isEmpty()) throw new RuntimeException("Cannot initialize both Hot-One-Vector and a True-Value! You must select one method of error/loss checking!");
        this.initialInput = initialInput;
        this.layerList = new ArrayList<>();
        IntStream.range(0, layers).forEach(i -> {
            DEFAULT_ACTIVATION_FUNCTIONS activationFunctionToUse = activationFunctionsMap.getOrDefault(i, activationFunction);
            DEFAULT_LOSS_FUNCTIONS lossFunctionToUse = lossFunctionsMap.getOrDefault(i, lossFunction);

            int[] trueValueToUse = trueValuesMap.getOrDefault(i, trueValues);
            int[][] hotOneVecToUse = new int[0][0];
            if (trueValueToUse.length == 0) hotOneVecMap.getOrDefault(i, hotOneVec);

            if(i != 0) {
                double[][] weights = randn(initialWeights.length, initialInput[0].length);
                layerList.add(new DenseLayer(weights, layerList.get(i - 1).getBatchLayerOutputs(), activationFunctionToUse, lossFunctionToUse, hotOneVecToUse, trueValueToUse));
            } else {
                layerList.add(new DenseLayer(initialWeights, initialInput, activationFunctionToUse, lossFunctionToUse, hotOneVecToUse, trueValueToUse));
            }
        });
    }

    private DenseLayers(int layers, double[][] initialWeights, double[]biases, double[][] initialInput, DEFAULT_ACTIVATION_FUNCTIONS activationFunction, Map<Integer, DEFAULT_ACTIVATION_FUNCTIONS> activationFunctionsMap,
                        DEFAULT_LOSS_FUNCTIONS lossFunction, Map<Integer, DEFAULT_LOSS_FUNCTIONS> lossFunctionsMap, int[][] hotOneVec, Map<Integer, int[][]> hotOneVecMap, int[] trueValues, Map<Integer, int[]> trueValuesMap) {
        if(!hotOneVecMap.isEmpty() && !trueValuesMap.isEmpty()) throw new RuntimeException("Cannot initialize both Hot-One-Vector and a True-Value! You must select one method of error/loss checking!");
        this.layerList = new ArrayList<>();
        this.initialInput = initialInput;
        IntStream.range(0, layers).forEach(i -> {
            DEFAULT_ACTIVATION_FUNCTIONS activationFunctionToUse = activationFunctionsMap.getOrDefault(i, activationFunction);
            DEFAULT_LOSS_FUNCTIONS lossFunctionToUse = lossFunctionsMap.getOrDefault(i, lossFunction);

            int[] trueValueToUse = trueValuesMap.getOrDefault(i, trueValues);
            int[][] hotOneVecToUse = new int[0][0];
            if (trueValueToUse.length == 0) hotOneVecMap.getOrDefault(i, hotOneVec);

            if(i != 0) {
                double[][] weights = randn(initialWeights.length, initialInput[0].length);
                layerList.add(new DenseLayer(weights, layerList.get(i - 1).getBatchLayerOutputs(), activationFunctionToUse, lossFunctionToUse, hotOneVecToUse, trueValueToUse));
            } else {
                layerList.add(new DenseLayer(initialWeights, biases, initialInput, activationFunctionToUse, lossFunctionToUse, hotOneVecToUse, trueValueToUse));
            }
        });
    }

    private DenseLayers(int layers, int numberOfNeurons, DEFAULT_ACTIVATION_FUNCTIONS activationFunction, Map<Integer, DEFAULT_ACTIVATION_FUNCTIONS> activationFunctionsMap,
                        DEFAULT_LOSS_FUNCTIONS lossFunction, Map<Integer, DEFAULT_LOSS_FUNCTIONS> lossFunctionsMap, int[][] hotOneVec, Map<Integer, int[][]> hotOneVecMap, int[] trueValues, Map<Integer, int[]> trueValuesMap) {
        if(!hotOneVecMap.isEmpty() && !trueValuesMap.isEmpty()) throw new RuntimeException("Cannot initialize both Hot-One-Vector and a True-Value! You must select one method of error/loss checking!");
        this.layerList = new ArrayList<>();
        this.initialInput = randn(1,numberOfNeurons);

        IntStream.range(0, layers).forEach(i -> {
            DEFAULT_ACTIVATION_FUNCTIONS activationFunctionToUse = activationFunctionsMap.getOrDefault(i, activationFunction);
            DEFAULT_LOSS_FUNCTIONS lossFunctionToUse = lossFunctionsMap.getOrDefault(i, lossFunction);

            int[] trueValueToUse = trueValuesMap.getOrDefault(i, trueValues);
            int[][] hotOneVecToUse = new int[0][0];
            if (trueValueToUse.length == 0) hotOneVecMap.getOrDefault(i, hotOneVec);

            double[][] weights = randn(numberOfNeurons, numberOfNeurons);
            if(i != 0) layerList.add(new DenseLayer(weights, layerList.get(i-1).getBatchLayerOutputs(), activationFunctionToUse, lossFunctionToUse, hotOneVecToUse, trueValueToUse));
            else layerList.add(new DenseLayer(weights, initialInput, activationFunctionToUse, lossFunctionToUse, hotOneVecToUse, trueValueToUse));
        });
    }

    private DenseLayers(int layers, int numberOfNeurons, double[][] initialInput, DEFAULT_ACTIVATION_FUNCTIONS activationFunction, Map<Integer, DEFAULT_ACTIVATION_FUNCTIONS> activationFunctionsMap,
                        DEFAULT_LOSS_FUNCTIONS lossFunction, Map<Integer, DEFAULT_LOSS_FUNCTIONS> lossFunctionsMap, int[][] hotOneVec, Map<Integer, int[][]> hotOneVecMap, int[] trueValues, Map<Integer, int[]> trueValuesMap) {
        if(!hotOneVecMap.isEmpty() && !trueValuesMap.isEmpty()) throw new RuntimeException("Cannot initialize both Hot-One-Vector and a True-Value! You must select one method of error/loss checking!");
        if(numberOfNeurons != initialInput.length) {
            throw new IllegalArgumentException("neuronsPerLayer of " + numberOfNeurons + " and initialInput size of " + initialInput.length + " do not match!");
        }
        this.initialInput = initialInput;
        this.layerList = new ArrayList<>();
        IntStream.range(0, layers).forEach(i -> {
            DEFAULT_ACTIVATION_FUNCTIONS activationFunctionToUse = activationFunctionsMap.getOrDefault(i, activationFunction);
            DEFAULT_LOSS_FUNCTIONS lossFunctionToUse = lossFunctionsMap.getOrDefault(i, lossFunction);

            int[] trueValueToUse = trueValuesMap.getOrDefault(i, trueValues);
            int[][] hotOneVecToUse = new int[0][0];
            if (trueValueToUse.length == 0) hotOneVecMap.getOrDefault(i, hotOneVec);

            double[][] weights = randn(numberOfNeurons, initialInput[0].length);
            if(i != 0) layerList.add(new DenseLayer(weights, layerList.get(i-1).getBatchLayerOutputs(), activationFunctionToUse, lossFunctionToUse, hotOneVecToUse, trueValueToUse));
            else layerList.add(new DenseLayer(weights, initialInput, activationFunctionToUse, lossFunctionToUse, hotOneVecToUse, trueValueToUse));
        });
    }

    private DenseLayers(int layers, int numberOfNeurons, double[][] initialWeights, double[][] initialInput, DEFAULT_ACTIVATION_FUNCTIONS activationFunction, Map<Integer, DEFAULT_ACTIVATION_FUNCTIONS> activationFunctionsMap,
                        DEFAULT_LOSS_FUNCTIONS lossFunction, Map<Integer, DEFAULT_LOSS_FUNCTIONS> lossFunctionsMap, int[][] hotOneVec, Map<Integer, int[][]> hotOneVecMap, int[] trueValues, Map<Integer, int[]> trueValuesMap) {
        if(!hotOneVecMap.isEmpty() && !trueValuesMap.isEmpty()) throw new RuntimeException("Cannot initialize both Hot-One-Vector and a True-Value! You must select one method of error/loss checking!");
        this.initialInput = initialInput;
        this.layerList = new ArrayList<>();
        IntStream.range(0, layers).forEach(i -> {
            DEFAULT_ACTIVATION_FUNCTIONS activationFunctionToUse = activationFunctionsMap.getOrDefault(i, activationFunction);
            DEFAULT_LOSS_FUNCTIONS lossFunctionToUse = lossFunctionsMap.getOrDefault(i, lossFunction);

            int[] trueValueToUse = trueValuesMap.getOrDefault(i, trueValues);
            int[][] hotOneVecToUse = new int[0][0];
            if (trueValueToUse.length == 0) hotOneVecMap.getOrDefault(i, hotOneVec);

            if(i != 0) {
                double[][] weights = randn(numberOfNeurons, numberOfNeurons);
                layerList.add(new DenseLayer(weights, layerList.get(i - 1).getBatchLayerOutputs(), activationFunctionToUse, lossFunctionToUse, hotOneVecToUse, trueValueToUse));
            } else {
                layerList.add(new DenseLayer(initialWeights, initialInput, activationFunctionToUse, lossFunctionToUse, hotOneVecToUse, trueValueToUse));
            }
        });
    }

    private DenseLayers(int layers, int numberOfNeurons, double[][] initialWeights, double[]biases, double[][] initialInput, DEFAULT_ACTIVATION_FUNCTIONS activationFunction, Map<Integer, DEFAULT_ACTIVATION_FUNCTIONS> activationFunctionsMap,
                        DEFAULT_LOSS_FUNCTIONS lossFunction, Map<Integer, DEFAULT_LOSS_FUNCTIONS> lossFunctionsMap, int[][] hotOneVec, Map<Integer, int[][]> hotOneVecMap, int[] trueValues, Map<Integer, int[]> trueValuesMap) {
        if(!hotOneVecMap.isEmpty() && !trueValuesMap.isEmpty()) throw new RuntimeException("Cannot initialize both Hot-One-Vector and a True-Value! You must select one method of error/loss checking!");
        this.layerList = new ArrayList<>();
        this.initialInput = initialInput;
        IntStream.range(0, layers).forEach(i -> {
            DEFAULT_ACTIVATION_FUNCTIONS activationFunctionToUse = activationFunctionsMap.getOrDefault(i, activationFunction);
            DEFAULT_LOSS_FUNCTIONS lossFunctionToUse = lossFunctionsMap.getOrDefault(i, lossFunction);

            int[] trueValueToUse = trueValuesMap.getOrDefault(i, trueValues);
            int[][] hotOneVecToUse = new int[0][0];
            if (trueValueToUse.length == 0) hotOneVecMap.getOrDefault(i, hotOneVec);

            if(i != 0) {
                double[][] weights = randn(numberOfNeurons, numberOfNeurons);
                layerList.add(new DenseLayer(weights, layerList.get(i - 1).getBatchLayerOutputs(), activationFunctionToUse, lossFunctionToUse, hotOneVecToUse, trueValueToUse));
            } else {
                layerList.add(new DenseLayer(initialWeights, biases, initialInput, activationFunctionToUse, lossFunctionToUse, hotOneVecToUse, trueValueToUse));
            }
        });
    }

    private DenseLayers(int layers, int[] numberOfNeuronsPerLayer, DEFAULT_ACTIVATION_FUNCTIONS activationFunction, Map<Integer, DEFAULT_ACTIVATION_FUNCTIONS> activationFunctionsMap,
                        DEFAULT_LOSS_FUNCTIONS lossFunction, Map<Integer, DEFAULT_LOSS_FUNCTIONS> lossFunctionsMap, int[][] hotOneVec, Map<Integer, int[][]> hotOneVecMap, int[] trueValues, Map<Integer, int[]> trueValuesMap) {
        if(!hotOneVecMap.isEmpty() && !trueValuesMap.isEmpty()) throw new RuntimeException("Cannot initialize both Hot-One-Vector and a True-Value! You must select one method of error/loss checking!");
        if(layers != numberOfNeuronsPerLayer.length) {
            throw new IllegalArgumentException(layers + " Layers given but only " + numberOfNeuronsPerLayer.length
                    + " layers described!\nAdjust neuronsPerLayer to be of same length as number of layers!");
        }
        this.layerList = new ArrayList<>();
        this.initialInput = randn(numberOfNeuronsPerLayer[0], numberOfNeuronsPerLayer[0]);
        IntStream.range(0, layers).forEach(i -> {
            DEFAULT_ACTIVATION_FUNCTIONS activationFunctionToUse = activationFunctionsMap.getOrDefault(i, activationFunction);
            DEFAULT_LOSS_FUNCTIONS lossFunctionToUse = lossFunctionsMap.getOrDefault(i, lossFunction);

            int[] trueValueToUse = trueValuesMap.getOrDefault(i, trueValues);
            int[][] hotOneVecToUse = new int[0][0];
            if (trueValueToUse.length == 0) hotOneVecMap.getOrDefault(i, hotOneVec);

            double[][] weights;
            if(i != 0) {
                weights = randn(numberOfNeuronsPerLayer[i], numberOfNeuronsPerLayer[i-1]);
                layerList.add(new DenseLayer(weights, layerList.get(i-1).getBatchLayerOutputs(), activationFunctionToUse, lossFunctionToUse, hotOneVecToUse, trueValueToUse));
            }
            else {
                weights = randn(numberOfNeuronsPerLayer[i], numberOfNeuronsPerLayer[i]);
                layerList.add(new DenseLayer(weights, initialInput, activationFunctionToUse, lossFunctionToUse, hotOneVecToUse, trueValueToUse));
            }
        });
    }

    private DenseLayers(int layers, int[] numberOfNeuronsPerLayer, double[][] initialInput, DEFAULT_ACTIVATION_FUNCTIONS activationFunction, Map<Integer, DEFAULT_ACTIVATION_FUNCTIONS> activationFunctionsMap,
                        DEFAULT_LOSS_FUNCTIONS lossFunction, Map<Integer, DEFAULT_LOSS_FUNCTIONS> lossFunctionsMap, int[][] hotOneVec, Map<Integer, int[][]> hotOneVecMap, int[] trueValues, Map<Integer, int[]> trueValuesMap) {
        if(!hotOneVecMap.isEmpty() && !trueValuesMap.isEmpty()) throw new RuntimeException("Cannot initialize both Hot-One-Vector and a True-Value! You must select one method of error/loss checking!");
        this.initialInput = initialInput;
        this.layerList = new ArrayList<>();
        IntStream.range(0, layers).forEach(i -> {
            DEFAULT_ACTIVATION_FUNCTIONS activationFunctionToUse = activationFunctionsMap.getOrDefault(i, activationFunction);
            DEFAULT_LOSS_FUNCTIONS lossFunctionToUse = lossFunctionsMap.getOrDefault(i, lossFunction);

            int[] trueValueToUse = trueValuesMap.getOrDefault(i, trueValues);
            int[][] hotOneVecToUse = new int[0][0];
            if (trueValueToUse.length == 0) hotOneVecMap.getOrDefault(i, hotOneVec);

            double[][] weights;
            if(i != 0) {
                weights = randn(numberOfNeuronsPerLayer[i], numberOfNeuronsPerLayer[i-1]);
                layerList.add(new DenseLayer(weights, layerList.get(i-1).getBatchLayerOutputs(), activationFunctionToUse, lossFunctionToUse, hotOneVecToUse, trueValueToUse));
            }
            else {
                weights = randn(numberOfNeuronsPerLayer[i], initialInput[0].length);
                layerList.add(new DenseLayer(weights, initialInput, activationFunctionToUse, lossFunctionToUse, hotOneVecToUse, trueValueToUse));
            }
        });
    }

    private DenseLayers(int layers, int[] numberOfNeuronsPerLayer, double[][] initialWeights, double[][] initialInput, DEFAULT_ACTIVATION_FUNCTIONS activationFunction, Map<Integer, DEFAULT_ACTIVATION_FUNCTIONS> activationFunctionsMap,
                        DEFAULT_LOSS_FUNCTIONS lossFunction, Map<Integer, DEFAULT_LOSS_FUNCTIONS> lossFunctionsMap, int[][] hotOneVec, Map<Integer, int[][]> hotOneVecMap, int[] trueValues, Map<Integer, int[]> trueValuesMap) {
        if(!hotOneVecMap.isEmpty() && !trueValuesMap.isEmpty()) throw new RuntimeException("Cannot initialize both Hot-One-Vector and a True-Value! You must select one method of error/loss checking!");
        this.layerList = new ArrayList<>();
        this.initialInput = initialInput;
        IntStream.range(0, layers).forEach(i -> {
            DEFAULT_ACTIVATION_FUNCTIONS activationFunctionToUse = activationFunctionsMap.getOrDefault(i, activationFunction);
            DEFAULT_LOSS_FUNCTIONS lossFunctionToUse = lossFunctionsMap.getOrDefault(i, lossFunction);

            int[] trueValueToUse = trueValuesMap.getOrDefault(i, trueValues);
            int[][] hotOneVecToUse = new int[0][0];
            if (trueValueToUse.length == 0) hotOneVecMap.getOrDefault(i, hotOneVec);

            if(i != 0) {
                double[][] weights = randn(numberOfNeuronsPerLayer[i], numberOfNeuronsPerLayer[i-1]);
                layerList.add(new DenseLayer(weights, layerList.get(i - 1).getBatchLayerOutputs(), activationFunctionToUse, lossFunctionToUse, hotOneVecToUse, trueValueToUse));
            } else {
                layerList.add(new DenseLayer(initialWeights, initialInput, activationFunctionToUse, lossFunctionToUse, hotOneVecToUse, trueValueToUse));
            }
        });
    }

    private DenseLayers(int layers, int[] numberOfNeuronsPerLayer, double[][] initialWeights, double[]biases, double[][] initialInput, DEFAULT_ACTIVATION_FUNCTIONS activationFunction, Map<Integer, DEFAULT_ACTIVATION_FUNCTIONS> activationFunctionsMap,
                        DEFAULT_LOSS_FUNCTIONS lossFunction, Map<Integer, DEFAULT_LOSS_FUNCTIONS> lossFunctionsMap, int[][] hotOneVec, Map<Integer, int[][]> hotOneVecMap, int[] trueValues, Map<Integer, int[]> trueValuesMap) {
        if(!hotOneVecMap.isEmpty() && !trueValuesMap.isEmpty()) throw new RuntimeException("Cannot initialize both Hot-One-Vector and a True-Value! You must select one method of error/loss checking!");
        this.layerList = new ArrayList<>();
        this.initialInput = initialInput;
        IntStream.range(0, layers).forEach(i -> {
            DEFAULT_ACTIVATION_FUNCTIONS activationFunctionToUse = activationFunctionsMap.getOrDefault(i, activationFunction);
            DEFAULT_LOSS_FUNCTIONS lossFunctionToUse = lossFunctionsMap.getOrDefault(i, lossFunction);

            int[] trueValueToUse = trueValuesMap.getOrDefault(i, trueValues);
            int[][] hotOneVecToUse = new int[0][0];
            if (trueValueToUse.length == 0) hotOneVecMap.getOrDefault(i, hotOneVec);

            if(i != 0) {
                double[][] weights = randn(numberOfNeuronsPerLayer[i], numberOfNeuronsPerLayer[i-1]);
                layerList.add(new DenseLayer(weights, layerList.get(i - 1).getBatchLayerOutputs(), activationFunctionToUse, lossFunctionToUse, hotOneVecToUse, trueValueToUse));
            } else {
                layerList.add(new DenseLayer(initialWeights, biases, initialInput, activationFunctionToUse, lossFunctionToUse, hotOneVecToUse, trueValueToUse));
            }
        });
    }


    public void printLayers(String label) {
        System.out.println(bold(green(":::: " + label + " ::::")));
        AtomicInteger counter = new AtomicInteger(1);
        this.getLayerList().forEach(i -> MathUtils.print(i.getBatchLayerOutputs(), bold(green("Layer " + (counter.getAndIncrement()))) + " Outputs Using Activation Function: " +  bold(purple((i.getActivationFunction().name()))) + "\n"));
    }

    public static double[][] randn(int rows, int cols) {
        return getDoubles(rows, cols);
    }

    static double[][] getDoubles(int rows, int cols) {
        double[][] output = new double[rows][cols];
        for (int i = 0; i < output.length; i++) {
            for (int j = 0; j < output[0].length; j++) {
                output[i][j] = (0.1 * random.nextGaussian());
            }
        }
        return output;
    }

    public DenseLayer getLayerAt(int index){
        return this.getLayerList().get(index);
    }
    public DenseLayer getFirstLayer() { return this.getLayerAt(0); }
    public DenseLayer getLastLayer() { return this.getLayerAt(this.getLayerList().size()-1);}
    public double[] generateLoss() { return this.getLastLayer().generateLoss(); }
    public double[][] generateFullLoss() {
        double[][] fullLoss = new double[this.getLayerList().size()][];
        IntStream.range(0, fullLoss.length).parallel().forEach(i -> fullLoss[i] = this.getLayerAt(i).generateLoss());
        return fullLoss;
    }

    public double[][] getBatchOutputs(){
        return getLayerAt(getLayerList().size()-1).getBatchLayerOutputs();
    }

    public ArrayList<Neuron> getNeuronsFromLayerAt(int index){
        return this.getLayerList().get(index).getNeuronList();
    }

    public Neuron getNeuronFromLayerAt(int layerIndex, int nueronIndex){
        return this.getLayerList().get(layerIndex).getNeuronList().get(nueronIndex);
    }

    @NoArgsConstructor
    public static class DenseLayersBuilder{
        private ArrayList<DenseLayer> layerList;
        private boolean isUsingListOfLayers = true;

        private DEFAULT_ACTIVATION_FUNCTIONS activationFunction = DEFAULT_ACTIVATION_FUNCTIONS.LINEAR_ACTIVATION_FUNCTION;
        private final Map<Integer, DEFAULT_ACTIVATION_FUNCTIONS> activationFunctionsMap = new HashMap<>();
        private boolean hasSetSpecificLayerActivationFunctions = false;

        private DEFAULT_LOSS_FUNCTIONS lossFunction = DEFAULT_LOSS_FUNCTIONS.NLL_LOSS_FUNCTION;
        private int[][] hotOneVec;
        private int[] trueValueIndex;
        private final Map<Integer, DEFAULT_LOSS_FUNCTIONS> lossFunctionsMap = new HashMap<>();
        private final Map<Integer, int[][]> hotOneMap = new HashMap<>();
        private final Map<Integer, int[]> trueValueIndexMap = new HashMap<>();
        private boolean hasSetSpecificLayerLossFunctions = false;
        private boolean isUsingHotOneMap = false;
        private boolean isUsingTrueValueIndexMap = false;

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
            this.trueValueIndex = trueValueIndex;
            return this;
        }
        public DenseLayersBuilder withHotOneVector(int[][] hotOneVec){
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
                    built = new DenseLayers(this.numberOfLayers, this.initialWeights, this.initialBiases, this.initialInput, this.activationFunction, this.activationFunctionsMap,
                            this.lossFunction, this.lossFunctionsMap, this.hotOneVec, this.hotOneMap, this.trueValueIndex, this.trueValueIndexMap);
                }
                else if(this.isUsingFileAsInput && this.isUsingSpecificWeights) {
                    built = new DenseLayers(this.numberOfLayers, this.initialWeights, this.initialInput, this.activationFunction, this.activationFunctionsMap,
                            this.lossFunction, this.lossFunctionsMap, this.hotOneVec, this.hotOneMap, this.trueValueIndex, this.trueValueIndexMap);
                }
                else if(this.isUsingFileAsInput) {
                    built = new DenseLayers(this.numberOfLayers, this.initialInput, this.activationFunction, this.activationFunctionsMap,
                            this.lossFunction, this.lossFunctionsMap, this.hotOneVec, this.hotOneMap, this.trueValueIndex, this.trueValueIndexMap);
                }
                else if(!this.isUsingFileAsInput && !this.isUsingSpecificWeights && !this.isUsingSpecificBiases) {
                    built = new DenseLayers(this.numberOfLayers, this.activationFunction, this.activationFunctionsMap,
                            this.lossFunction, this.lossFunctionsMap, this.hotOneVec, this.hotOneMap, this.trueValueIndex, this.trueValueIndexMap);
                }
            }
            else if(this.isUsingNumberOfNeurons && this.isUsingNumberOfLayers && !this.isUsingBatchInputs) {
                if(this.isUsingSpecificWeights && this.isUsingSpecificBiases) {
                    built = new DenseLayers(this.numberOfLayers, this.numberOfNeurons, this.initialWeights, this.initialBiases, this.initialInput, this.activationFunction, this.activationFunctionsMap,
                            this.lossFunction, this.lossFunctionsMap, this.hotOneVec, this.hotOneMap, this.trueValueIndex, this.trueValueIndexMap);
                }
                else if(this.isUsingFileAsInput  && this.isUsingSpecificWeights) {
                    built = new DenseLayers(this.numberOfLayers, this.numberOfNeurons, this.initialWeights, this.initialInput, this.activationFunction, this.activationFunctionsMap,
                            this.lossFunction, this.lossFunctionsMap, this.hotOneVec, this.hotOneMap, this.trueValueIndex, this.trueValueIndexMap);
                }
                else if(this.isUsingFileAsInput) {
                    built = new DenseLayers(this.numberOfLayers, this.numberOfNeurons, this.initialInput, this.activationFunction, this.activationFunctionsMap,
                            this.lossFunction, this.lossFunctionsMap, this.hotOneVec, this.hotOneMap, this.trueValueIndex, this.trueValueIndexMap);
                }
                else if(!this.isUsingFileAsInput && !this.isUsingSpecificWeights && !this.isUsingSpecificBiases) {
                    built = new DenseLayers(this.numberOfLayers, this.numberOfNeurons, this.activationFunction, this.activationFunctionsMap,
                            this.lossFunction, this.lossFunctionsMap, this.hotOneVec, this.hotOneMap, this.trueValueIndex, this.trueValueIndexMap);
                }
            }
            else if(this.isUsingSpecificNeurons && this.isUsingNumberOfLayers && !this.isUsingBatchInputs) {
                if(this.isUsingFileAsInput && this.isUsingSpecificWeights && this.isUsingSpecificBiases) {
                    built = new DenseLayers(this.numberOfLayers, this.numberOfNeuronsPerLayer, this.initialWeights, this.initialBiases, this.initialInput, this.activationFunction, this.activationFunctionsMap,
                            this.lossFunction, this.lossFunctionsMap, this.hotOneVec, this.hotOneMap, this.trueValueIndex, this.trueValueIndexMap);
                }
                else if(this.isUsingFileAsInput && this.isUsingSpecificWeights) {
                    built = new DenseLayers(this.numberOfLayers, this.numberOfNeuronsPerLayer, this.initialWeights, this.initialInput, this.activationFunction, this.activationFunctionsMap,
                            this.lossFunction, this.lossFunctionsMap, this.hotOneVec, this.hotOneMap, this.trueValueIndex, this.trueValueIndexMap);
                }
                else if(this.isUsingFileAsInput) {
                    built = new DenseLayers(this.numberOfLayers, this.numberOfNeuronsPerLayer, this.initialInput, this.activationFunction, this.activationFunctionsMap,
                            this.lossFunction, this.lossFunctionsMap, this.hotOneVec, this.hotOneMap, this.trueValueIndex, this.trueValueIndexMap);
                }
                else if(!this.isUsingFileAsInput && !this.isUsingSpecificWeights && !this.isUsingSpecificBiases) {
                    built = new DenseLayers(this.numberOfLayers, this.numberOfNeuronsPerLayer, this.activationFunction, this.activationFunctionsMap,
                            this.lossFunction, this.lossFunctionsMap, this.hotOneVec, this.hotOneMap, this.trueValueIndex, this.trueValueIndexMap);
                }
            }
            else if(!this.isUsingSpecificNeurons && !this.isUsingNumberOfNeurons && this.isUsingNumberOfLayers && !this.isUsingFileAsInput){
                if(this.isUsingBatchInputs && this.isUsingSpecificWeights && this.isUsingSpecificBiases) {
                    built = new DenseLayers(this.numberOfLayers, this.initialWeights, this.initialBiases, this.initialInput, this.activationFunction, this.activationFunctionsMap,
                            this.lossFunction, this.lossFunctionsMap, this.hotOneVec, this.hotOneMap, this.trueValueIndex, this.trueValueIndexMap);
                }
                else if(this.isUsingBatchInputs  && this.isUsingSpecificWeights) {
                    built = new DenseLayers(this.numberOfLayers, this.initialWeights, this.initialInput, this.activationFunction, this.activationFunctionsMap,
                            this.lossFunction, this.lossFunctionsMap, this.hotOneVec, this.hotOneMap, this.trueValueIndex, this.trueValueIndexMap);
                }
                else if(this.isUsingBatchInputs) {
                    built = new DenseLayers(this.numberOfLayers, this.initialInput, this.activationFunction, this.activationFunctionsMap,
                            this.lossFunction, this.lossFunctionsMap, this.hotOneVec, this.hotOneMap, this.trueValueIndex, this.trueValueIndexMap);
                }
                else if(!this.isUsingBatchInputs && !this.isUsingSpecificWeights && !this.isUsingSpecificBiases) {
                    built = new DenseLayers(this.numberOfLayers, this.activationFunction, this.activationFunctionsMap,
                            this.lossFunction, this.lossFunctionsMap, this.hotOneVec, this.hotOneMap, this.trueValueIndex, this.trueValueIndexMap);
                }
            }
            else if(this.isUsingNumberOfNeurons && this.isUsingNumberOfLayers && !this.isUsingFileAsInput) {
                if(this.isUsingBatchInputs && this.isUsingSpecificWeights && this.isUsingSpecificBiases) {
                    built = new DenseLayers(this.numberOfLayers, this.numberOfNeurons, this.initialWeights, this.initialBiases, this.initialInput, this.activationFunction, this.activationFunctionsMap,
                            this.lossFunction, this.lossFunctionsMap, this.hotOneVec, this.hotOneMap, this.trueValueIndex, this.trueValueIndexMap);
                }
                else if(this.isUsingBatchInputs  && this.isUsingSpecificWeights) {
                    built = new DenseLayers(this.numberOfLayers, this.numberOfNeurons, this.initialWeights, this.initialInput, this.activationFunction, this.activationFunctionsMap,
                            this.lossFunction, this.lossFunctionsMap, this.hotOneVec, this.hotOneMap, this.trueValueIndex, this.trueValueIndexMap);
                }
                else if(this.isUsingBatchInputs) {
                    built = new DenseLayers(this.numberOfLayers, this.numberOfNeurons, this.initialInput, this.activationFunction, this.activationFunctionsMap,
                            this.lossFunction, this.lossFunctionsMap, this.hotOneVec, this.hotOneMap, this.trueValueIndex, this.trueValueIndexMap);
                }
                else if(!this.isUsingBatchInputs && !this.isUsingSpecificWeights && !this.isUsingSpecificBiases) {
                    built = new DenseLayers(this.numberOfLayers, this.numberOfNeurons, this.activationFunction, this.activationFunctionsMap,
                            this.lossFunction, this.lossFunctionsMap, this.hotOneVec, this.hotOneMap, this.trueValueIndex, this.trueValueIndexMap);
                }
            }
            else if(this.isUsingSpecificNeurons && this.isUsingNumberOfLayers && !this.isUsingFileAsInput) {
                if(this.isUsingBatchInputs && this.isUsingSpecificWeights && this.isUsingSpecificBiases) {
                    built = new DenseLayers(this.numberOfLayers, this.numberOfNeuronsPerLayer, this.initialWeights, this.initialBiases, this.initialInput, this.activationFunction, this.activationFunctionsMap,
                            this.lossFunction, this.lossFunctionsMap, this.hotOneVec, this.hotOneMap, this.trueValueIndex, this.trueValueIndexMap);
                }
                else if(this.isUsingBatchInputs  && this.isUsingSpecificWeights) {
                    built = new DenseLayers(this.numberOfLayers, this.numberOfNeuronsPerLayer, this.initialWeights, this.initialInput, this.activationFunction, this.activationFunctionsMap,
                            this.lossFunction, this.lossFunctionsMap, this.hotOneVec, this.hotOneMap, this.trueValueIndex, this.trueValueIndexMap);
                }
                else if(this.isUsingBatchInputs) {
                    built = new DenseLayers(this.numberOfLayers, this.numberOfNeuronsPerLayer, this.initialInput, this.activationFunction, this.activationFunctionsMap,
                            this.lossFunction, this.lossFunctionsMap, this.hotOneVec, this.hotOneMap, this.trueValueIndex, this.trueValueIndexMap);
                }
                else if(!this.isUsingBatchInputs && !this.isUsingSpecificWeights && !this.isUsingSpecificBiases) {
                    built = new DenseLayers(this.numberOfLayers, this.numberOfNeuronsPerLayer, this.activationFunction, this.activationFunctionsMap,
                            this.lossFunction, this.lossFunctionsMap, this.hotOneVec, this.hotOneMap, this.trueValueIndex, this.trueValueIndexMap);
                }
            }
            else {
                built = new DenseLayers(2, DEFAULT_ACTIVATION_FUNCTIONS.LINEAR_ACTIVATION_FUNCTION, this.activationFunctionsMap,
                        this.lossFunction, this.lossFunctionsMap, this.hotOneVec, this.hotOneMap, this.trueValueIndex, this.trueValueIndexMap);
                if(this.isUsingBatchInputs && this.isUsingFileAsInput){
                    throw new RuntimeException(bold(red("Builder Not Configured Properly! Do Not Use File As Input and Batch Input Together!")));
                }
                throw new RuntimeException(bold(red("Builder Not Configured Properly!")));
            }
            return built;
        }
    }
}
