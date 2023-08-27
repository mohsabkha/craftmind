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
                .layerOutput(new ArrayList<>())
                .neuronList(new ArrayList<>())
                .inputs(new ArrayList<>())
                .build();
        Assertions.assertNotNull(layer);
        Assertions.assertNotNull(layer.getNeuronList());
        Assertions.assertNotNull(layer.getInputs());
        Assertions.assertNotNull(layer.getLayerOutput());
    }
}
