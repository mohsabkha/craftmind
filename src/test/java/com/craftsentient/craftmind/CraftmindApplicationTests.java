package com.craftsentient.craftmind;

import com.craftsentient.craftmind.layers.DenseLayers;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.SpringBootTest;

@SpringBootTest
class CraftmindApplicationTests {
    @Test
    public void neuralNetworkTest() {

        // initial input layer
        double[][] inputs = {
                { 1,2,3,2.5 },
                { 2,5,-1,2 },
                { -1.5,2.7,3.3,-0.8 }};
        double[][] weights = {
                {    0.2, 0.8, -0.5, 1.0    },
                {    0.5,-0.91,0.26,-0.5    },
                {   -0.26,-0.27,0.17,0.87   }};
        double[] biases = { 2, 3, 0.5 };


        DenseLayers layers = new DenseLayers(4, new int[]{3,7,2,6}, weights, biases, inputs);
        layers.printLayers("Full Constructor Layer");

        Assertions.assertEquals(3, layers.getLayerList().get(0).getNeuronList().size());
        Assertions.assertEquals(3, layers.getLayerList().get(0).getLayerOutputs().length);
        Assertions.assertEquals(4.8, layers.getLayerList().get(0).getBatchLayerOutputs()[0][0]);
        Assertions.assertEquals(1.21, layers.getLayerList().get(0).getBatchLayerOutputs()[0][1]);
        Assertions.assertEquals(2.385, layers.getLayerList().get(0).getBatchLayerOutputs()[0][2]);
    }

    @Test
    public void mainTest() {
        String [] args = new String[0];
        CraftmindApplication.main(args);
    }

    @Test
    public void matrixTest(){

    }
}
