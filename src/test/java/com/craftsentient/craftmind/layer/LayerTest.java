package com.craftsentient.craftmind.layer;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;


public class LayerTest {

    @Test
    public void layerConstructorTest(){
        Layer layer = new Layer();
        Assertions.assertNotNull(layer);
        Assertions.assertNotNull(layer.getNeuronList());
        Assertions.assertNotNull(layer.getInputs());
    }

    @Test
    public void layerLombokConstructionTest(){
        Layer layer = Layer.builder()
                .layerOutputs(new ArrayList<>())
                .neuronList(new ArrayList<>())
                .inputs(new ArrayList<>())
                .build();
        Assertions.assertNotNull(layer);
        Assertions.assertNotNull(layer.getNeuronList());
        Assertions.assertNotNull(layer.getInputs());
        Assertions.assertNotNull(layer.getLayerOutputs());
        Assertions.assertNotNull(layer.toString());
    }

    @Test
    public void layerGeneratorTest(){
        Layer layer = Layer.builder()
                .layerOutputs(new ArrayList<>())
                .neuronList(new ArrayList<>())
                .inputs(new ArrayList<>())
                .build();

        Assertions.assertTrue(layer.getNeuronList().isEmpty());
        layer.generateLayer(10);
        Assertions.assertEquals(layer.getNeuronList().size(), 10);
    }
}
