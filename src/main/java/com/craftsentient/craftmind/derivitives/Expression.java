package com.craftsentient.craftmind.derivitives;

import java.util.function.Function;

public class Expression {
    public static final double ALPHA = 0.00001;
    private Function<Double, Double> function;

    // constructor
    public Expression(Function<Double, Double> func) {
        this.function = func;
    }

    public double evaluate(double x) {
        return function.apply(x);
    }

    // creates and returns derivative function expression
    public Expression derivative() {
        // Compute the derivative using the chain rule
        Function<Double, Double> derivativeFunc = (x) -> {
            double x1 = x + ALPHA;
            double fx1 = function.apply(x1);
            double fx2 = function.apply(x);
            return (fx1 - fx2) / (x1 - x);
        };

        return new Expression(derivativeFunc);
    }

    public Expression chainDerivative(Expression outerFunction) {
        // Compute the chain rule derivative of f(g(x))
        Function<Double, Double> chainDerivativeFunc = (x) -> {
            double g_x = outerFunction.evaluate(x);
            double f_prime_g_x = derivative().evaluate(g_x);
            return f_prime_g_x * outerFunction.derivative().evaluate(x);
        };

        return new Expression(chainDerivativeFunc);
    }

}
