package com.craftsentient.craftmind;

import com.craftsentient.craftmind.layer.Layer;
import com.craftsentient.craftmind.layers.DenseLayers;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.SpringBootTest;

@SpringBootTest
class CraftmindApplicationTests {
    @Test
    public void neuralNetworkTest() {
        DenseLayers layers = DenseLayers.init(3,5,4);
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
