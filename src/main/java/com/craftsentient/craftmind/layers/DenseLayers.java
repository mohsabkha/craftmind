package com.craftsentient.craftmind.layers;

import com.craftsentient.craftmind.activation.ActivationFunctions;
import com.craftsentient.craftmind.activation.DEFAULT_ACTIVATION_FUNCTIONS;
import com.craftsentient.craftmind.layer.DenseLayer;
import com.craftsentient.craftmind.mathUtils.MathUtils;
import com.craftsentient.craftmind.mathUtils.MathUtils.Pair;
import com.craftsentient.craftmind.neuron.Neuron;
import lombok.Getter;

import java.util.ArrayList;
import java.util.Random;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;

@Getter
public class DenseLayers {
    private ArrayList<DenseLayer> layerList;
    private double[][] initialInput;
    public static final Random random = new Random(0);

    public static DenseLayers init(int numberOfLayers, int numberOfInputs, int numberOfNeurons, DEFAULT_ACTIVATION_FUNCTIONS[] activationFunction){
        double[][] initialInput = randn(1, numberOfInputs);
        double[][] initialWeights= randn(numberOfNeurons, numberOfInputs);
        double[] initialBiases = new double[numberOfNeurons];
        IntStream.range(0, numberOfNeurons).parallel().forEachOrdered(i -> initialBiases[i] = 0);
        return new DenseLayers(numberOfLayers, numberOfNeurons, initialWeights, initialBiases, initialInput, activationFunction);
    }

    public static DenseLayers init(int numberOfLayers, int numberOfInputs, int[] numberOfNeuronsPerLayer, DEFAULT_ACTIVATION_FUNCTIONS[] activationFunction){
        double[][] initialInput = randn(1, numberOfInputs);
        double[][] initialWeights= randn(numberOfNeuronsPerLayer[0], numberOfInputs);
        double[] initialBiases = new double[numberOfNeuronsPerLayer[0]];
        IntStream.range(0, numberOfNeuronsPerLayer[0]).parallel().forEachOrdered(i -> initialBiases[i] = 0);
        return new DenseLayers(numberOfLayers, numberOfNeuronsPerLayer, initialWeights, initialBiases, initialInput, activationFunction);
    }


    private DenseLayers(int layers, DEFAULT_ACTIVATION_FUNCTIONS[] activationFunction) {
        this.layerList = new ArrayList<>();
        this.initialInput = randn(layers,layers);
        IntStream.range(0, layers).forEach(i -> {
            double[][] weights = randn(layers, layers);
            if(i != 0) layerList.add(new DenseLayer(weights, layerList.get(i-1).getBatchLayerOutputs(), activationFunction[i]));
            else layerList.add(new DenseLayer(weights, initialInput,activationFunction[i]));
        });
    }

    private DenseLayers(int layers, double[][] initialInput, DEFAULT_ACTIVATION_FUNCTIONS[] activationFunction) {
        this.initialInput = initialInput;
        this.layerList = new ArrayList<>();
        IntStream.range(0, layers).forEach(i -> {
            double[][] weights = randn(initialInput.length, initialInput[0].length);
            if(i != 0) layerList.add(new DenseLayer(weights, layerList.get(i-1).getBatchLayerOutputs(),activationFunction[i]));
            else layerList.add(new DenseLayer(weights, initialInput, activationFunction[i]));
        });
    }

    private DenseLayers(int layers, double[][] initialWeights, double[][] initialInput, DEFAULT_ACTIVATION_FUNCTIONS[] activationFunction) {
        this.initialInput = initialInput;
        this.layerList = new ArrayList<>();
        IntStream.range(0, layers).forEach(i -> {
            if(i != 0) {
                double[][] weights = randn(initialWeights.length, initialInput[0].length);
                layerList.add(new DenseLayer(weights, layerList.get(i - 1).getBatchLayerOutputs(), activationFunction[i]));
            } else {
                layerList.add(new DenseLayer(initialWeights, initialInput, activationFunction[i]));
            }
        });
    }

    private DenseLayers(int layers, double[][] initialWeights, double[]biases, double[][] initialInput, DEFAULT_ACTIVATION_FUNCTIONS[] activationFunction) {
        this.layerList = new ArrayList<>();
        this.initialInput = initialInput;
        IntStream.range(0, layers).forEach(i -> {
            if(i != 0) {
                double[][] weights = randn(initialWeights.length, initialInput[0].length);
                layerList.add(new DenseLayer(weights, layerList.get(i - 1).getBatchLayerOutputs(), activationFunction[i]));
                System.out.println("");
            } else {
                layerList.add(new DenseLayer(initialWeights, biases, initialInput, activationFunction[i]));
            }
        });
    }

    private DenseLayers(int layers, int numberOfNeurons, DEFAULT_ACTIVATION_FUNCTIONS[] activationFunction) {
        this.layerList = new ArrayList<>();
        this.initialInput = randn(1,numberOfNeurons);

        IntStream.range(0, layers).forEach(i -> {
            double[][] weights = randn(numberOfNeurons, numberOfNeurons);
            if(i != 0) layerList.add(new DenseLayer(weights, layerList.get(i-1).getBatchLayerOutputs(),activationFunction[i]));
            else layerList.add(new DenseLayer(weights, initialInput, activationFunction[i]));
        });
    }

    private DenseLayers(int layers, int numberOfNeurons, double[][] initialInput, DEFAULT_ACTIVATION_FUNCTIONS[] activationFunction) {
        if(numberOfNeurons != initialInput.length) {
            throw new IllegalArgumentException("neuronsPerLayer of " + numberOfNeurons + " and initialInput size of " + initialInput.length + " do not match!");
        }
        this.initialInput = initialInput;
        this.layerList = new ArrayList<>();
        IntStream.range(0, layers).forEach(i -> {
            double[][] weights = randn(numberOfNeurons, initialInput[0].length);
            if(i != 0) layerList.add(new DenseLayer(weights, layerList.get(i-1).getBatchLayerOutputs(), activationFunction[i]));
            else layerList.add(new DenseLayer(weights, initialInput, activationFunction[i]));
        });
    }

    private DenseLayers(int layers, int numberOfNeurons, double[][] initialWeights, double[][] initialInput, DEFAULT_ACTIVATION_FUNCTIONS[] activationFunction) {
        this.initialInput = initialInput;
        this.layerList = new ArrayList<>();
        IntStream.range(0, layers).forEach(i -> {
            if(i != 0) {
                double[][] weights = randn(numberOfNeurons, numberOfNeurons);
                layerList.add(new DenseLayer(weights, layerList.get(i - 1).getBatchLayerOutputs(), activationFunction[i]));
            } else {
                layerList.add(new DenseLayer(initialWeights, initialInput, activationFunction[i]));
            }
        });
    }

    private DenseLayers(int layers, int numberOfNeurons, double[][] initialWeights, double[]biases, double[][] initialInput, DEFAULT_ACTIVATION_FUNCTIONS[] activationFunction) {
        this.layerList = new ArrayList<>();
        this.initialInput = initialInput;
        IntStream.range(0, layers).forEach(i -> {
            if(i != 0) {
                double[][] weights = randn(numberOfNeurons, numberOfNeurons);
                layerList.add(new DenseLayer(weights, layerList.get(i - 1).getBatchLayerOutputs(), activationFunction[i]));
                System.out.println("");
            } else {
                layerList.add(new DenseLayer(initialWeights, biases, initialInput, activationFunction[i]));
            }
        });
    }

    private DenseLayers(int layers, int[] numberOfNeuronsPerLayer, DEFAULT_ACTIVATION_FUNCTIONS[] activationFunction) {
        if(layers != numberOfNeuronsPerLayer.length) {
            throw new IllegalArgumentException(layers + " Layers given but only " + numberOfNeuronsPerLayer.length
                    + " layers described!\nAdjust neuronsPerLayer to be of same length as number of layers!");
        }
        this.layerList = new ArrayList<>();
        this.initialInput = randn(numberOfNeuronsPerLayer[0], numberOfNeuronsPerLayer[0]);
        IntStream.range(0, layers).forEach(i -> {
            double[][] weights;
            if(i != 0) {
                weights = randn(numberOfNeuronsPerLayer[i], numberOfNeuronsPerLayer[i-1]);
                layerList.add(new DenseLayer(weights, layerList.get(i-1).getBatchLayerOutputs(), activationFunction[i]));
            }
            else {
                weights = randn(numberOfNeuronsPerLayer[i], numberOfNeuronsPerLayer[i]);
                layerList.add(new DenseLayer(weights, initialInput, activationFunction[i]));
            }
        });
    }

    private DenseLayers(int layers, int[] numberOfNeuronsPerLayer, double[][] initialInput, DEFAULT_ACTIVATION_FUNCTIONS[] activationFunction) {
        this.initialInput = initialInput;
        this.layerList = new ArrayList<>();
        IntStream.range(0, layers).forEach(i -> {
            double[][] weights;
            if(i != 0) {
                weights = randn(numberOfNeuronsPerLayer[i], numberOfNeuronsPerLayer[i-1]);
                layerList.add(new DenseLayer(weights, layerList.get(i-1).getBatchLayerOutputs(), activationFunction[i]));
            }
            else {
                weights = randn(numberOfNeuronsPerLayer[i], initialInput[0].length);
                layerList.add(new DenseLayer(weights, initialInput, activationFunction[i]));
            }
        });
    }

    private DenseLayers(int layers, int[] numberOfNeuronsPerLayer, double[][] initialWeights, double[][] initialInput, DEFAULT_ACTIVATION_FUNCTIONS[] activationFunction) {
        this.layerList = new ArrayList<>();
        this.initialInput = initialInput;
        IntStream.range(0, layers).forEach(i -> {
            if(i != 0) {
                double[][] weights = randn(numberOfNeuronsPerLayer[i], numberOfNeuronsPerLayer[i-1]);
                layerList.add(new DenseLayer(weights, layerList.get(i - 1).getBatchLayerOutputs(), activationFunction[i]));
            } else {
                layerList.add(new DenseLayer(initialWeights, initialInput, activationFunction[i]));
            }
        });
    }

    private DenseLayers(int layers, int[] numberOfNeuronsPerLayer, double[][] initialWeights, double[]biases, double[][] initialInput, DEFAULT_ACTIVATION_FUNCTIONS[] activationFunction) {
        this.layerList = new ArrayList<>();
        this.initialInput = initialInput;
        IntStream.range(0, layers).forEach(i -> {
            if(i != 0) {
                double[][] weights = randn(numberOfNeuronsPerLayer[i], numberOfNeuronsPerLayer[i-1]);
                layerList.add(new DenseLayer(weights, layerList.get(i - 1).getBatchLayerOutputs(), activationFunction[i]));
                System.out.println("");
            } else {
                layerList.add(new DenseLayer(initialWeights, biases, initialInput, activationFunction[i]));
            }
        });
    }

    private DenseLayers(int layers, DEFAULT_ACTIVATION_FUNCTIONS activationFunction) {
        this.layerList = new ArrayList<>();
        this.initialInput = randn(layers,layers);
        IntStream.range(0, layers).forEach(i -> {
            double[][] weights = randn(layers, layers);
            if(i != 0) layerList.add(new DenseLayer(weights, layerList.get(i-1).getBatchLayerOutputs(), activationFunction));
            else layerList.add(new DenseLayer(weights, initialInput,activationFunction));
        });
    }

    private DenseLayers(int layers, double[][] initialInput, DEFAULT_ACTIVATION_FUNCTIONS activationFunction) {
        this.initialInput = initialInput;
        this.layerList = new ArrayList<>();
        IntStream.range(0, layers).forEach(i -> {
            double[][] weights = randn(initialInput.length, initialInput[0].length);
            if(i != 0) layerList.add(new DenseLayer(weights, layerList.get(i-1).getBatchLayerOutputs(),activationFunction));
            else layerList.add(new DenseLayer(weights, initialInput, activationFunction));
        });
    }

    private DenseLayers(int layers, double[][] initialWeights, double[][] initialInput, DEFAULT_ACTIVATION_FUNCTIONS activationFunction) {
        this.initialInput = initialInput;
        this.layerList = new ArrayList<>();
        IntStream.range(0, layers).forEach(i -> {
            if(i != 0) {
                double[][] weights = randn(initialWeights.length, initialInput[0].length);
                layerList.add(new DenseLayer(weights, layerList.get(i - 1).getBatchLayerOutputs(),activationFunction));
            } else {
                layerList.add(new DenseLayer(initialWeights, initialInput, activationFunction));
            }
        });
    }

    private DenseLayers(int layers, double[][] initialWeights, double[]biases, double[][] initialInput, DEFAULT_ACTIVATION_FUNCTIONS activationFunction) {
        this.layerList = new ArrayList<>();
        this.initialInput = initialInput;
        IntStream.range(0, layers).forEach(i -> {
            if(i != 0) {
                double[][] weights = randn(initialWeights.length, initialInput[0].length);
                layerList.add(new DenseLayer(weights, layerList.get(i - 1).getBatchLayerOutputs(), activationFunction));
                System.out.println("");
            } else {
                layerList.add(new DenseLayer(initialWeights, biases, initialInput, activationFunction));
            }
        });
    }

    private DenseLayers(int layers, int numberOfNeurons, DEFAULT_ACTIVATION_FUNCTIONS activationFunction) {
        this.layerList = new ArrayList<>();
        this.initialInput = randn(1,numberOfNeurons);

        IntStream.range(0, layers).forEach(i -> {
            double[][] weights = randn(numberOfNeurons, numberOfNeurons);
            if(i != 0) layerList.add(new DenseLayer(weights, layerList.get(i-1).getBatchLayerOutputs(), activationFunction));
            else layerList.add(new DenseLayer(weights, initialInput, activationFunction));
        });
    }

    private DenseLayers(int layers, int numberOfNeurons, double[][] initialInput, DEFAULT_ACTIVATION_FUNCTIONS activationFunction) {
        if(numberOfNeurons != initialInput.length) {
            throw new IllegalArgumentException("neuronsPerLayer of " + numberOfNeurons + " and initialInput size of " + initialInput.length + " do not match!");
        }
        this.initialInput = initialInput;
        this.layerList = new ArrayList<>();
        IntStream.range(0, layers).forEach(i -> {
            double[][] weights = randn(numberOfNeurons, initialInput[0].length);
            if(i != 0) layerList.add(new DenseLayer(weights, layerList.get(i-1).getBatchLayerOutputs(), activationFunction));
            else layerList.add(new DenseLayer(weights, initialInput, activationFunction));
        });
    }

    private DenseLayers(int layers, int numberOfNeurons, double[][] initialWeights, double[][] initialInput, DEFAULT_ACTIVATION_FUNCTIONS activationFunction) {
        this.initialInput = initialInput;
        this.layerList = new ArrayList<>();
        IntStream.range(0, layers).forEach(i -> {
            if(i != 0) {
                double[][] weights = randn(numberOfNeurons, numberOfNeurons);
                layerList.add(new DenseLayer(weights, layerList.get(i - 1).getBatchLayerOutputs(), activationFunction));
            } else {
                layerList.add(new DenseLayer(initialWeights, initialInput, activationFunction));
            }
        });
    }

    private DenseLayers(int layers, int numberOfNeurons, double[][] initialWeights, double[]biases, double[][] initialInput, DEFAULT_ACTIVATION_FUNCTIONS activationFunction) {
        this.layerList = new ArrayList<>();
        this.initialInput = initialInput;
        IntStream.range(0, layers).forEach(i -> {
            if(i != 0) {
                double[][] weights = randn(numberOfNeurons, numberOfNeurons);
                layerList.add(new DenseLayer(weights, layerList.get(i - 1).getBatchLayerOutputs(), activationFunction));
                System.out.println("");
            } else {
                layerList.add(new DenseLayer(initialWeights, biases, initialInput, activationFunction));
            }
        });
    }

    private DenseLayers(int layers, int[] numberOfNeuronsPerLayer, DEFAULT_ACTIVATION_FUNCTIONS activationFunction) {
        if(layers != numberOfNeuronsPerLayer.length) {
            throw new IllegalArgumentException(layers + " Layers given but only " + numberOfNeuronsPerLayer.length
                    + " layers described!\nAdjust neuronsPerLayer to be of same length as number of layers!");
        }
        this.layerList = new ArrayList<>();
        this.initialInput = randn(numberOfNeuronsPerLayer[0], numberOfNeuronsPerLayer[0]);
        IntStream.range(0, layers).forEach(i -> {
            double[][] weights;
            if(i != 0) {
                weights = randn(numberOfNeuronsPerLayer[i], numberOfNeuronsPerLayer[i-1]);
                layerList.add(new DenseLayer(weights, layerList.get(i-1).getBatchLayerOutputs(), activationFunction));
            }
            else {
                weights = randn(numberOfNeuronsPerLayer[i], numberOfNeuronsPerLayer[i]);
                layerList.add(new DenseLayer(weights, initialInput, activationFunction));
            }
        });
    }

    private DenseLayers(int layers, int[] numberOfNeuronsPerLayer, double[][] initialInput, DEFAULT_ACTIVATION_FUNCTIONS activationFunction) {
        this.initialInput = initialInput;
        this.layerList = new ArrayList<>();
        IntStream.range(0, layers).forEach(i -> {
            double[][] weights;
            if(i != 0) {
                weights = randn(numberOfNeuronsPerLayer[i], numberOfNeuronsPerLayer[i-1]);
                layerList.add(new DenseLayer(weights, layerList.get(i-1).getBatchLayerOutputs(), activationFunction));
            }
            else {
                weights = randn(numberOfNeuronsPerLayer[i], initialInput[0].length);
                layerList.add(new DenseLayer(weights, initialInput, activationFunction));
            }
        });
    }

    private DenseLayers(int layers, int[] numberOfNeuronsPerLayer, double[][] initialWeights, double[][] initialInput, DEFAULT_ACTIVATION_FUNCTIONS activationFunction) {
        this.layerList = new ArrayList<>();
        this.initialInput = initialInput;
        IntStream.range(0, layers).forEach(i -> {
            if(i != 0) {
                double[][] weights = randn(numberOfNeuronsPerLayer[i], numberOfNeuronsPerLayer[i-1]);
                layerList.add(new DenseLayer(weights, layerList.get(i - 1).getBatchLayerOutputs(), activationFunction));
            } else {
                layerList.add(new DenseLayer(initialWeights, initialInput, activationFunction));
            }
        });
    }

    private DenseLayers(int layers, int[] numberOfNeuronsPerLayer, double[][] initialWeights, double[]biases, double[][] initialInput, DEFAULT_ACTIVATION_FUNCTIONS activationFunction) {
        this.layerList = new ArrayList<>();
        this.initialInput = initialInput;
        IntStream.range(0, layers).forEach(i -> {
            if(i != 0) {
                double[][] weights = randn(numberOfNeuronsPerLayer[i], numberOfNeuronsPerLayer[i-1]);
                layerList.add(new DenseLayer(weights, layerList.get(i - 1).getBatchLayerOutputs(), activationFunction));
                System.out.println("");
            } else {
                layerList.add(new DenseLayer(initialWeights, biases, initialInput, activationFunction));
            }
        });
    }


//    public void setMultipleActivationFunctionsAcrossMultipleLayers(int starting, int ending, DEFAULT_ACTIVATION_FUNCTIONS[] activationFunctions) {
//        AtomicInteger activationFunctionIndex = new AtomicInteger(0);
//        IntStream.range(starting, ending+1).parallel().forEachOrdered(i -> {
//            try {
//                this.getLayerAt(i).setBatchLayerOutputs( ActivationFunctions.activationFunction(activationFunctions[activationFunctionIndex.getAndIncrement()], this.getLayerAt(starting).getBatchLayerOutputs()) );
//            } catch (Exception e) {
//                throw new RuntimeException(e);
//            }
//        });
//    }
//    public void setSingleActivationFunctionsAcrossMultipleLayers(int starting, int ending, DEFAULT_ACTIVATION_FUNCTIONS activationFunction) {
//        IntStream.range(starting, ending+1).parallel().forEachOrdered(i -> {
//            try {
//                this.getLayerAt(i).setBatchLayerOutputs( ActivationFunctions.activationFunction(activationFunction, this.getLayerAt(starting).getBatchLayerOutputs()) );
//            } catch (Exception e) {
//                throw new RuntimeException(e);
//            }
//        });
//    }
//    public void setSingleActivationFunctionsAcrossSingleLayer(int layer, DEFAULT_ACTIVATION_FUNCTIONS activationFunctions) throws Exception {
//        this.getLayerAt(layer).setBatchLayerOutputs( ActivationFunctions.activationFunction(activationFunctions, this.getLayerAt(layer).getBatchLayerOutputs()) );
//    }

    private void printLayers(String label) {
        System.out.println(":::: " + label + " ::::");
        AtomicInteger counter = new AtomicInteger(1);
        this.getLayerList().forEach(i -> {
            MathUtils.print(i.getBatchLayerOutputs(), "Layer " + (counter.getAndIncrement()) + " Outputs Using Activation Function: " + i.getActivationFunction().name());
        });
    }

    public static double[][] randn(int rows, int cols) {
        return getDoubles(rows, cols, random);
    }

    static double[][] getDoubles(int rows, int cols, Random random) {
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

    public ArrayList<Neuron> getNeuronsFromLayerAt(int index){
        return this.getLayerList().get(index).getNeuronList();
    }

    public Neuron getNeuronFromLayerAt(int layerIndex, int nueronIndex){
        return this.getLayerList().get(layerIndex).getNeuronList().get(nueronIndex);
    }

    public static class DenseLayersBuilder{
        private ArrayList<DenseLayer> layerList;
        private boolean isUsingListOfLayers = true;

        private Random random = new Random(0);

        private DEFAULT_ACTIVATION_FUNCTIONS activationFunction;
        private DEFAULT_ACTIVATION_FUNCTIONS[] activationFunctions;
        private ArrayList<Pair<Integer, DEFAULT_ACTIVATION_FUNCTIONS>> activationFunctionsPairs;


        private ArrayList<Neuron> neuronList;
        private boolean hasSetSpecificLayerActivationFunctions = false;
        private int startingLayer;
        private int endingLayer;

        private int numberOfLayers;
        private boolean isUsingNumberOfLayers = false;

        private int numberOfInputs;
        private double[][] initialInput;
        private boolean isUsingBatchInputs = false;

        private double[][] initialWeights;
        private boolean isUsingSpecificWeights = false;

        private double bias;
        private double[] initialBiases;
        private boolean isUsingSpecificBiases = false;
        private boolean isSettingSpecificBiases = false;

        private int[] numberOfNeuronsPerLayer;
        private boolean isUsingSpecificNeurons = false;
        private int numberOfNeurons;
        private boolean isUsingNumberOfNeurons = false;

        public DenseLayersBuilder withRandom(Random random){
            this.random = random;
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
            if(this.activationFunctions.length == 0 && this.numberOfLayers != 0)
                this.activationFunctions = new DEFAULT_ACTIVATION_FUNCTIONS[this.numberOfLayers];
            this.activationFunction = activationFunction;
            fillActivationFunctionList(this.activationFunctions, activationFunction);
            return this;
        }

        public DenseLayersBuilder withSingleActivationFunctionForSingleLayer(int layer, DEFAULT_ACTIVATION_FUNCTIONS activationFunction) {
            if(this.activationFunctions.length == 0 && this.numberOfLayers != 0)
                this.activationFunctions = new DEFAULT_ACTIVATION_FUNCTIONS[this.numberOfLayers];

            //this.layerList.get(layer).setActivationFunction(activationFunction);
            return this;
        }

        public DenseLayersBuilder withSingleActivationFunctionForMultipleLayers(int startingLayer, int endingLayer, DEFAULT_ACTIVATION_FUNCTIONS activationFunction) {
            if(this.activationFunctions.length == 0 && this.numberOfLayers != 0)
                this.activationFunctions = new DEFAULT_ACTIVATION_FUNCTIONS[numberOfLayers];
            this.hasSetSpecificLayerActivationFunctions = true;
            this.activationFunction = activationFunction;
            this.startingLayer = startingLayer;
            this.endingLayer = endingLayer;
            return this;
        }

//        public DenseLayersBuilder withMultipleActivationFunctionsForMultipleLayers(int startingLayer, int endingLayer, DEFAULT_ACTIVATION_FUNCTIONS[] activationFunctions){
//            this.hasSetSpecificLayerActivationFunctions = true;
//            this.activationFunctions = activationFunctions;
//            this.startingLayer = startingLayer;
//            this.endingLayer = endingLayer;
//            return this;
//        }
        private void fillActivationFunctionList(DEFAULT_ACTIVATION_FUNCTIONS[] funcitonsList, DEFAULT_ACTIVATION_FUNCTIONS function){
            IntStream.range(0, funcitonsList.length).parallel().forEachOrdered(i -> funcitonsList[i] = function);
        }

        public DenseLayers build(){
            DenseLayers built = null;
            if(!this.isUsingNumberOfLayers && !this.isUsingListOfLayers) throw new IllegalArgumentException("Please use the numberOfLayers() builder method to initialize, or provide an ArrayList<DenseLayer> using the withLayerList() builder method!");
            if(!this.isUsingNumberOfNeurons && this.isUsingNumberOfLayers){
                if(this.isUsingBatchInputs && this.isUsingSpecificWeights && this.isUsingSpecificBiases) {
                    built = new DenseLayers(this.numberOfLayers, this.initialWeights, this.initialBiases, this.initialInput, this.activationFunction);
                }
                else if(this.isUsingBatchInputs  && this.isUsingSpecificWeights) {
                    built = new DenseLayers(this.numberOfLayers, this.initialWeights, this.initialInput, this.activationFunction);
                }
                else if(this.isUsingBatchInputs) {
                    built = new DenseLayers(this.numberOfLayers, this.initialInput, this.activationFunction);
                }
                else if(!this.isUsingBatchInputs && !this.isUsingSpecificWeights && !this.isUsingSpecificBiases) {
                    built = new DenseLayers(this.numberOfLayers, this.activationFunction);
                }
            }
            else if(this.isUsingNumberOfNeurons) {
                if(this.isUsingBatchInputs && this.isUsingSpecificWeights && this.isUsingSpecificBiases) {
                    built = new DenseLayers(this.numberOfLayers, this.numberOfNeurons, this.initialWeights, this.initialBiases, this.initialInput, this.activationFunction);
                }
                else if(this.isUsingBatchInputs  && this.isUsingSpecificWeights) {
                    built = new DenseLayers(this.numberOfLayers, this.numberOfNeurons, this.initialWeights, this.initialInput, this.activationFunction);
                }
                else if(this.isUsingBatchInputs) {
                    built = new DenseLayers(this.numberOfLayers, this.numberOfNeurons, this.initialInput, this.activationFunction);
                }
                else if(!this.isUsingBatchInputs && !this.isUsingSpecificWeights && !this.isUsingSpecificBiases) {
                    built = new DenseLayers(this.numberOfLayers, this.numberOfNeurons, this.activationFunction);
                }
            }
            else if(this.isUsingSpecificNeurons) {
                if(this.isUsingBatchInputs && this.isUsingSpecificWeights && this.isUsingSpecificBiases) {
                    built = new DenseLayers(this.numberOfLayers, this.numberOfNeuronsPerLayer, this.initialWeights, this.initialBiases, this.initialInput, this.activationFunction);
                }
                else if(this.isUsingBatchInputs  && this.isUsingSpecificWeights) {
                    built = new DenseLayers(this.numberOfLayers, this.numberOfNeuronsPerLayer, this.initialWeights, this.initialInput, this.activationFunction);
                }
                else if(this.isUsingBatchInputs) {
                    built = new DenseLayers(this.numberOfLayers, this.numberOfNeuronsPerLayer, this.initialInput, this.activationFunction);
                }
                else if(!this.isUsingBatchInputs && !this.isUsingSpecificWeights && !this.isUsingSpecificBiases) {
                    built = new DenseLayers(this.numberOfLayers, this.numberOfNeuronsPerLayer, this.activationFunction);
                }
            }
            else {
                built = new DenseLayers(2, DEFAULT_ACTIVATION_FUNCTIONS.LINEAR_ACTIVATION_FUNCTION);
                throw new RuntimeException("Builder Not Configured Properly!");
            }

            return built;
        }
    }
}
