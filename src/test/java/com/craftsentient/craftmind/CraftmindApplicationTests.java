package com.craftsentient.craftmind;

import com.craftsentient.craftmind.activationFunctions.DEFAULT_ACTIVATION_FUNCTIONS;
import com.craftsentient.craftmind.layer.Layer;
import com.craftsentient.craftmind.layers.DenseLayers;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.SpringBootTest;

@SpringBootTest
class CraftmindApplicationTests {
    @Test
    public void neuralNetworkTest() {
        DenseLayers layers = DenseLayers.init(4,8, new int[]{1,8,8,1}, DEFAULT_ACTIVATION_FUNCTIONS.LINEAR_ACTIVATION_FUNCTION);
        layers.printLayers("INIT NETWORK");
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
