package com.craftsentient.craftmind.layer;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;


public class LayerTest {

    @Test
    public void layerConstructorTest() {
        DenseLayer layer = new DenseLayer();
        Assertions.assertNotNull(layer);
        Assertions.assertNotNull(layer.getNeuronList());
        Assertions.assertNotNull(layer.getInputs());
    }

    @Test
    public void layerLombokConstructionTest() {
        DenseLayer layer = DenseLayer.builder()
                .layerOutputs(new double[0])
                .neuronList(new ArrayList<>())
                .inputs(new double[0])
                .build();
        Assertions.assertNotNull(layer);
        Assertions.assertNotNull(layer.getNeuronList());
        Assertions.assertNotNull(layer.getInputs());
        Assertions.assertNotNull(layer.getLayerOutputs());
        Assertions.assertNotNull(layer.toString());
    }

    @Test
    public void layerGeneratorTest() {
        DenseLayer layer = new DenseLayer();
        Assertions.assertTrue(layer.getNeuronList().isEmpty());
        layer.generateLayer(10);
        Assertions.assertEquals(layer.getNeuronList().size(), 10);
    }
}
