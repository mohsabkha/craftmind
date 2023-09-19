package com.craftsentient.craftmind.layers;

import org.junit.jupiter.api.Test;

public class LayersTest {
    @Test
    public void testAllConstructors() {

        double[][] inputs = {
                {1, 2, 3, 2.5},
                {2, 5, -1, 2},
                {-1.5, 2.7, 3.3, -0.8}};
        double[][] weights = {
                {0.2, 0.8, -0.5, 1.0},
                {0.5, -0.91, 0.26, -0.5},
                {-0.26, -0.27, 0.17, 0.87}};
        double[] biases = {2, 3, 0.5};

        DenseLayers layers1 = new DenseLayers(5);
        layers1.printLayers("Layer 1");

        DenseLayers layers2A = new DenseLayers(7, 3);
        layers2A.printLayers("Layer 2A");

        DenseLayers layers2B = new DenseLayers(7, new int[]{2, 5, 9, 2, 5, 4, 1});
        layers2B.printLayers("Layer 2B");

        DenseLayers layers3A = new DenseLayers(3, inputs);
        layers3A.printLayers("Layer 3A");

        DenseLayers layers3B = new DenseLayers(5, 3, inputs);
        layers3B.printLayers("Layer 3B");

        DenseLayers layers3C = new DenseLayers(3, new int[]{3, 5, 9, 4, 6, 3}, inputs);
        layers3C.printLayers("Layer 3C");

        DenseLayers layers4A = new DenseLayers(5, weights, inputs);
        layers4A.printLayers("Layer 4A");

        DenseLayers layers4B = new DenseLayers(5, 3, weights, inputs);
        layers4B.printLayers("Layer 4B");

        DenseLayers layers4C = new DenseLayers(5, new int[]{3, 5, 9, 2, 7}, weights, inputs);
        layers4C.printLayers("Layer 4C");

        DenseLayers layers5A = new DenseLayers(4, weights, biases, inputs);
        layers5A.printLayers("Layer 5A");

        DenseLayers layers5B = new DenseLayers(4, 3, weights, biases, inputs);
        layers5B.printLayers("Layer 5B");
    }
}
