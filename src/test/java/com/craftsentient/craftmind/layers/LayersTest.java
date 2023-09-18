package com.craftsentient.craftmind.layers;

import org.junit.jupiter.api.Assertions;
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

        Layers layers1 = new Layers(5);
        layers1.printLayers("Layer 1");

        Layers layers2A = new Layers(7, 3);
        layers2A.printLayers("Layer 2A");

        Layers layers2B = new Layers(7, new int[]{2, 5, 9, 2, 5, 4, 1});
        layers2B.printLayers("Layer 2B");

        Layers layers3A = new Layers(3, inputs);
        layers3A.printLayers("Layer 3A");

        Layers layers3B = new Layers(5, 3, inputs);
        layers3B.printLayers("Layer 3B");

        Layers layers3C = new Layers(3, new int[]{3, 5, 9, 4, 6, 3}, inputs);
        layers3C.printLayers("Layer 3C");

        Layers layers4A = new Layers(5, weights, inputs);
        layers4A.printLayers("Layer 4A");

        Layers layers4B = new Layers(5, 3, weights, inputs);
        layers4B.printLayers("Layer 4B");

        Layers layers4C = new Layers(5, new int[]{3, 5, 9, 2, 7}, weights, inputs);
        layers4C.printLayers("Layer 4C");

        Layers layers5A = new Layers(4, weights, biases, inputs);
        layers5A.printLayers("Layer 5A");

        Layers layers5B = new Layers(4, 3, weights, biases, inputs);
        layers5B.printLayers("Layer 5B");
    }

}
