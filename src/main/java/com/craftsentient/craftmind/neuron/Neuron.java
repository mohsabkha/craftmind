package com.craftsentient.craftmind.neuron;

import lombok.Data;

public interface Neuron {
    Long[] inputs[] = {};
    Long bias = 0L;
    Long output = 0L;
}
