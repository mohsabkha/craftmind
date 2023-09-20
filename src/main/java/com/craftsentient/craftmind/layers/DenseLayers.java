package com.craftsentient.craftmind.layers;

import com.craftsentient.craftmind.layer.DenseLayer;
import com.craftsentient.craftmind.mathUtils.MathUtils;
import lombok.Getter;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;

@Getter
public class DenseLayers {
    private ArrayList<DenseLayer> layerList;
    private double[][] initialInput;
    public static final Random random = new Random(0);

    public static DenseLayers init(int numberOfLayers, int numberOfInputs, int numberOfNeurons){
        double[][] initialInput = randn(1, numberOfInputs);
        double[][] initialWeights= randn(numberOfNeurons, numberOfInputs);
        double[] initialBiases = new double[numberOfNeurons];
        IntStream.range(0, numberOfNeurons).parallel().forEachOrdered(i -> initialBiases[i] = 0);
        return new DenseLayers(numberOfLayers, numberOfNeurons, initialWeights, initialBiases, initialInput);
    }

    public static DenseLayers init(int numberOfLayers, int numberOfInputs, int[] numberOfNeuronsPerLayer){
        double[][] initialInput = randn(1, numberOfInputs);
        double[][] initialWeights= randn(numberOfNeuronsPerLayer[0], numberOfInputs);
        double[] initialBiases = new double[numberOfNeuronsPerLayer[0]];
        IntStream.range(0, numberOfNeuronsPerLayer[0]).parallel().forEachOrdered(i -> initialBiases[i] = 0);
        return new DenseLayers(numberOfLayers, numberOfNeuronsPerLayer, initialWeights, initialBiases, initialInput);
    }

    public DenseLayers(int layers) {
        this.layerList = new ArrayList<>();
        this.initialInput = randn(layers,layers);
        IntStream.range(0, layers).forEach(i -> {
            double[][] weights = randn(layers, layers);
            if(i != 0) layerList.add(new DenseLayer(weights, layerList.get(i-1).getBatchLayerOutputs()));
            else layerList.add(new DenseLayer(weights, initialInput));
        });
    }

    public DenseLayers(int layers, double[][] initialInput) {
        this.initialInput = initialInput;
        this.layerList = new ArrayList<>();
        IntStream.range(0, layers).forEach(i -> {
            double[][] weights = randn(initialInput.length, initialInput[0].length);
            if(i != 0) layerList.add(new DenseLayer(weights, layerList.get(i-1).getBatchLayerOutputs()));
            else layerList.add(new DenseLayer(weights, initialInput));
        });
    }

    public DenseLayers(int layers, double[][] initialWeights, double[][] initialInput) {
        this.initialInput = initialInput;
        this.layerList = new ArrayList<>();
        IntStream.range(0, layers).forEach(i -> {
            if(i != 0) {
                double[][] weights = randn(initialWeights.length, initialInput[0].length);
                layerList.add(new DenseLayer(weights, layerList.get(i - 1).getBatchLayerOutputs()));
            } else {
                layerList.add(new DenseLayer(initialWeights, initialInput));
            }
        });
    }

    public DenseLayers(int layers, double[][] initialWeights, double[]biases, double[][] initialInput) {
        this.layerList = new ArrayList<>();
        this.initialInput = initialInput;
        IntStream.range(0, layers).forEach(i -> {
            if(i != 0) {
                double[][] weights = randn(initialWeights.length, initialInput[0].length);
                layerList.add(new DenseLayer(weights, layerList.get(i - 1).getBatchLayerOutputs()));
                System.out.println("");
            } else {
                layerList.add(new DenseLayer(initialWeights, biases, initialInput));
            }
        });
    }

    public DenseLayers(int layers, int neuronsPerLayer) {
        this.layerList = new ArrayList<>();
        this.initialInput = randn(1,neuronsPerLayer);

        IntStream.range(0, layers).forEach(i -> {
            double[][] weights = randn(neuronsPerLayer, neuronsPerLayer);
            if(i != 0) layerList.add(new DenseLayer(weights, layerList.get(i-1).getBatchLayerOutputs()));
            else layerList.add(new DenseLayer(weights, initialInput));
        });
    }

    public DenseLayers(int layers, int[] neuronsPerLayer) {
        if(layers != neuronsPerLayer.length) {
            throw new IllegalArgumentException(layers + " Layers given but only " + neuronsPerLayer.length
                    + " layers described!\nAdjust neuronsPerLayer to be of same length as number of layers!");
        }
        this.layerList = new ArrayList<>();
        this.initialInput = randn(neuronsPerLayer[0],neuronsPerLayer[0]);
        IntStream.range(0, layers).forEach(i -> {
            double[][] weights;
            if(i != 0) {
                weights = randn(neuronsPerLayer[i], neuronsPerLayer[i-1]);
                layerList.add(new DenseLayer(weights, layerList.get(i-1).getBatchLayerOutputs()));
            }
            else {
                weights = randn(neuronsPerLayer[i], neuronsPerLayer[i]);
                layerList.add(new DenseLayer(weights, initialInput));
            }
        });
    }

    public DenseLayers(int layers, int neuronsPerLayer, double[][] initialInput) {
        if(neuronsPerLayer != initialInput.length) {
            throw new IllegalArgumentException("neuronsPerLayer of " + neuronsPerLayer + " and initialInput size of " + initialInput.length + " do not match!");
        }
        this.initialInput = initialInput;
        this.layerList = new ArrayList<>();
        IntStream.range(0, layers).forEach(i -> {
            double[][] weights = randn(neuronsPerLayer, initialInput[0].length);
            if(i != 0) layerList.add(new DenseLayer(weights, layerList.get(i-1).getBatchLayerOutputs()));
            else layerList.add(new DenseLayer(weights, initialInput));
        });
    }

    public DenseLayers(int layers, int[] neuronsPerLayer, double[][] initialInput) {
        this.initialInput = initialInput;
        this.layerList = new ArrayList<>();
        IntStream.range(0, layers).forEach(i -> {
            double[][] weights;
            if(i != 0) {
                weights = randn(neuronsPerLayer[i], neuronsPerLayer[i-1]);
                layerList.add(new DenseLayer(weights, layerList.get(i-1).getBatchLayerOutputs()));
            }
            else {
                weights = randn(neuronsPerLayer[i], initialInput[0].length);
                layerList.add(new DenseLayer(weights, initialInput));
            }
        });
    }

    public DenseLayers(int layers, int neuronsPerLayer, double[][] initialWeights, double[][] initialInput) {
        this.initialInput = initialInput;
        this.layerList = new ArrayList<>();
        IntStream.range(0, layers).forEach(i -> {
            if(i != 0) {
                double[][] weights = randn(neuronsPerLayer, neuronsPerLayer);
                layerList.add(new DenseLayer(weights, layerList.get(i - 1).getBatchLayerOutputs()));
            } else {
                layerList.add(new DenseLayer(initialWeights, initialInput));
            }
        });
    }

    public DenseLayers(int layers, int[] neuronsPerLayer, double[][] initialWeights, double[][] initialInput) {
        this.layerList = new ArrayList<>();
        this.initialInput = initialInput;
        IntStream.range(0, layers).forEach(i -> {
            if(i != 0) {
                double[][] weights = randn(neuronsPerLayer[i], neuronsPerLayer[i-1]);
                layerList.add(new DenseLayer(weights, layerList.get(i - 1).getBatchLayerOutputs()));
            } else {
                layerList.add(new DenseLayer(initialWeights, initialInput));
            }
        });
    }

    public DenseLayers(int layers, int neuronsPerLayer, double[][] initialWeights, double[]biases, double[][] initialInput) {
        this.layerList = new ArrayList<>();
        this.initialInput = initialInput;
        IntStream.range(0, layers).forEach(i -> {
            if(i != 0) {
                double[][] weights = randn(neuronsPerLayer, neuronsPerLayer);
                layerList.add(new DenseLayer(weights, layerList.get(i - 1).getBatchLayerOutputs()));
                System.out.println("");
            } else {
                layerList.add(new DenseLayer(initialWeights, biases, initialInput));
            }
        });
    }

    public DenseLayers(int layers, int[] neuronsPerLayer, double[][] initialWeights, double[]biases, double[][] initialInput) {
        this.layerList = new ArrayList<>();
        this.initialInput = initialInput;
        IntStream.range(0, layers).forEach(i -> {
            if(i != 0) {
                double[][] weights = randn(neuronsPerLayer[i], neuronsPerLayer[i-1]);
                layerList.add(new DenseLayer(weights, layerList.get(i - 1).getBatchLayerOutputs()));
                System.out.println("");
            } else {
                layerList.add(new DenseLayer(initialWeights, biases, initialInput));
            }
        });
    }

    public void printLayers(String label) {
        System.out.println(":::: " + label + " ::::");
        AtomicInteger counter = new AtomicInteger(1);
        this.getLayerList().forEach(i -> {
            MathUtils.print(i.getBatchLayerOutputs(), "Layer " + (counter.getAndIncrement()) + " Outputs");
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
}
