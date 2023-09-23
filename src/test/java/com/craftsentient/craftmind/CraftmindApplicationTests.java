package com.craftsentient.craftmind;

import com.craftsentient.craftmind.activation.DEFAULT_ACTIVATION_FUNCTIONS;
import com.craftsentient.craftmind.layers.DenseLayers;
import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.SpringBootTest;

@SpringBootTest
class CraftmindApplicationTests {
    @Test
    public void neuralNetworkTest() {
        DenseLayers layers = DenseLayers.init(4,8, new int[]{10,8,8,10}, DEFAULT_ACTIVATION_FUNCTIONS.GAUSSIAN_ACTIVATION_FUNCTION);
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
