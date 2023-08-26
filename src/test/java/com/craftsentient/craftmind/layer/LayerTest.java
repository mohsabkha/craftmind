package com.craftsentient.craftmind.layer;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;


public class LayerTest {

    @Test
    public void layerConstructorTest(){
        Layer layer = new Layer();
        Assertions.assertNotNull(layer);
    }

}
