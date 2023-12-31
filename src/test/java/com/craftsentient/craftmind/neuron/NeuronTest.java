package com.craftsentient.craftmind.neuron;

import com.craftsentient.craftmind.utils.PrintUtils;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;

public class NeuronTest {

    double[] inputs = new double[1];
    double[] weights = {1.0};
    Double BIAS = 1.0;

    @BeforeEach
    public void setUp() {
        inputs[0] = (1.0);
        weights = new double[]{0.87};
    }

    @Test
    public void generateOutputTest() {
        Neuron neuron = Neuron.builder()
                .weights(weights)
                .bias(BIAS)
                .output(0.0)
                .build();
        double output = neuron.generateOutput(inputs);

        Assertions.assertEquals(1.87, neuron.getOutput());
        Assertions.assertEquals(output, neuron.getOutput());
    }

    @Test
    public void createNeuronTest() {
        Neuron neuron = new Neuron(4);
        Assertions.assertNotNull(neuron);
        Assertions.assertEquals(0.0, neuron.getBias());
    }

    @Test
    public void generateNeuronTest() {
        Neuron neuron = new Neuron(4, 1.0);
        Assertions.assertNotNull(neuron);

    }
}
