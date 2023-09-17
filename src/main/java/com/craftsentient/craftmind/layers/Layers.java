package com.craftsentient.craftmind.layers;

import com.craftsentient.craftmind.layer.Layer;
import com.craftsentient.craftmind.mathUtils.MathUtils;
import lombok.Getter;

import java.util.ArrayList;
import java.util.Random;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;

@Getter
public class Layers {
    private ArrayList<Layer> layerList;
    private double[][] initialInput;

    private  static final Random random = new Random(0);

    public Layers(int layers) {
        this.layerList = new ArrayList<>();
        this.initialInput = randn(3,3);
        IntStream.range(0, layers).forEach(i -> {
            double[][] weights = randn(3, 3);
            if(i != 0) layerList.add(new Layer(weights, layerList.get(i-1).getBatchLayerOutputs()));
            else layerList.add(new Layer(weights, initialInput));
        });
    }

    public Layers(int layers, double[][] initialInput) {
        this.initialInput = initialInput;
        this.layerList = new ArrayList<>();
        IntStream.range(0, layers).forEach(i -> {
            double[][] weights = randn(layers, layers);
            if(i != 0) layerList.add(new Layer(weights, layerList.get(i-1).getBatchLayerOutputs()));
            else layerList.add(new Layer(weights, initialInput));
        });
    }

    public Layers(int layers, double[][] initialWeights, double[][] initialInput) {
        this.initialInput = initialInput;
        this.layerList = new ArrayList<>();
        IntStream.range(0, layers).forEach(i -> {
            if(i != 0) {
                double[][] weights = randn(layers, layers);
                layerList.add(new Layer(weights, layerList.get(i - 1).getBatchLayerOutputs()));
            } else {
                layerList.add(new Layer(initialWeights, initialInput));
            }
        });
    }

    public Layers(int layers, double[][] initialWeights, double[]biases, double[][] initialInput) {
        this.layerList = new ArrayList<>();
        this.initialInput = initialInput;
        IntStream.range(0, layers).forEach(i -> {
            if(i != 0) {
                double[][] weights = randn(layers, layers);
                layerList.add(new Layer(weights, layerList.get(i - 1).getBatchLayerOutputs()));
                System.out.println("");
            } else {
                layerList.add(new Layer(initialWeights, biases, initialInput));
            }
        });
    }

    public Layers(int layers, int neuronsPerLayer) {
        this.layerList = new ArrayList<>();
        this.initialInput = randn(3,3);
        IntStream.range(0, layers).forEach(i -> {
            double[][] weights = randn(neuronsPerLayer, neuronsPerLayer);
            if(i != 0) layerList.add(new Layer(weights, layerList.get(i-1).getBatchLayerOutputs()));
            else layerList.add(new Layer(weights, initialInput));
        });
    }

    public Layers(int layers, int[] neuronsPerLayer) {
        this.layerList = new ArrayList<>();
        IntStream.range(0, layers).forEach(i -> {
            double[][] weights = randn(neuronsPerLayer[i], neuronsPerLayer[i-1]);
            if(i != 0) layerList.add(new Layer(weights, layerList.get(i-1).getBatchLayerOutputs()));
            else layerList.add(new Layer(weights, initialInput));
        });
    }

    public Layers(int layers, int neuronsPerLayer, double[][] initialInput) {
        this.initialInput = initialInput;
        this.layerList = new ArrayList<>();
        IntStream.range(0, layers).forEach(i -> {
            double[][] weights = randn(neuronsPerLayer, neuronsPerLayer);
            if(i != 0) layerList.add(new Layer(weights, layerList.get(i-1).getBatchLayerOutputs()));
            else layerList.add(new Layer(weights, initialInput));
        });
    }

    public Layers(int layers, int[] neuronsPerLayer, double[][] initialInput) {
        this.initialInput = initialInput;
        this.layerList = new ArrayList<>();
        IntStream.range(0, layers).forEach(i -> {
            double[][] weights = randn(neuronsPerLayer[i], neuronsPerLayer[i-1]);
            if(i != 0) layerList.add(new Layer(weights, layerList.get(i-1).getBatchLayerOutputs()));
            else layerList.add(new Layer(weights, initialInput));
        });
    }

    public Layers(int layers, int neuronsPerLayer, double[][] initialWeights, double[][] initialInput) {
        this.initialInput = initialInput;
        this.layerList = new ArrayList<>();
        IntStream.range(0, layers).forEach(i -> {
            if(i != 0) {
                double[][] weights = randn(neuronsPerLayer, neuronsPerLayer);
                layerList.add(new Layer(weights, layerList.get(i - 1).getBatchLayerOutputs()));
            } else {
                layerList.add(new Layer(initialWeights, initialInput));
            }
        });
    }

    public Layers(int layers, int[] neuronsPerLayer, double[][] initialWeights, double[][] initialInput) {
        this.layerList = new ArrayList<>();
        this.initialInput = initialInput;
        IntStream.range(0, layers).forEach(i -> {
            if(i != 0) {
                double[][] weights = randn(neuronsPerLayer[i], neuronsPerLayer[i-1]);
                layerList.add(new Layer(weights, layerList.get(i - 1).getBatchLayerOutputs()));
            } else {
                layerList.add(new Layer(initialWeights, initialInput));
            }
        });
    }

    public Layers(int layers, int[] neuronsPerLayer, double[][] initialWeights, double[]biases, double[][] initialInput) {
        this.layerList = new ArrayList<>();
        this.initialInput = initialInput;
        IntStream.range(0, layers).forEach(i -> {
            if(i != 0) {
                double[][] weights = randn(neuronsPerLayer[i], neuronsPerLayer[i-1]);
                layerList.add(new Layer(weights, layerList.get(i - 1).getBatchLayerOutputs()));
                System.out.println("");
            } else {
                layerList.add(new Layer(initialWeights, biases, initialInput));
            }
        });
    }

    public Layers(int layers, int neuronsPerLayer, double[][] initialWeights, double[]biases, double[][] initialInput) {
        this.layerList = new ArrayList<>();
        this.initialInput = initialInput;
        IntStream.range(0, layers).forEach(i -> {
            if(i != 0) {
                double[][] weights = randn(neuronsPerLayer, neuronsPerLayer);
                layerList.add(new Layer(weights, layerList.get(i - 1).getBatchLayerOutputs()));
                System.out.println("");
            } else {
                layerList.add(new Layer(initialWeights, biases, initialInput));
            }
        });
    }

    public void printLayers() {
        AtomicInteger counter = new AtomicInteger(1);
        this.getLayerList().forEach(i -> {
            MathUtils.print(i.getBatchLayerOutputs(), "Layer " + (counter.getAndIncrement()) + " Outputs");
        });

    }

    public double[][] randn(int rows, int cols) {
        double[][] output = new double[rows][cols];
        for (int i = 0; i < output.length; i++) {
            for (int j = 0; j < output[0].length; j++) {
                output[i][j] = 0.1 * random.nextGaussian();
            }
        }
        return output;
    }
}
