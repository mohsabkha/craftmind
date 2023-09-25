package com.craftsentient.craftmind.errorLoss;

public class ErrorLossFunctions {
    public static void lossFunction(DEFAULT_LOSS_FUNCTIONS lossFunction, double[] trueValues, double[] predictedValues){
        switch (lossFunction){
            case BINARY_CROSS_ENTROPY_LOSS_FUNCTION -> {}
            case CATEGORICAL_CROSS_ENTROPY_LOSS_FUNCTION -> {}
            case CONTRASTIVE_LOSS_FUNCTION -> {}
            case COSINE_PROXIMITY_LOSS_FUNCTION -> {}
            case FOCAL_LOSS_FUNCTION -> {}
            case HINGE_LOSS_FUNCTION -> {}
            case HUBER_LOSS_FUNCTION -> {}
            case KL_DIVERGENCE_LOSS_FUNCTION -> {}
            case LOG_COSH_LOSS_FUNCTION -> {}
            case L1_LOSS_FUNCTION -> {}
            case MAPE_LOSS_FUNCTION -> {}
            case MSE_LOSS_FUNCTION -> {}
            case MSLE_LOSS_FUNCTION -> {}
            case NLL_LOSS_FUNCTION -> {}
            case RANKNET_LOSS_FUNCTION -> {}
            case SPARSE_CATEGORICAL_CROSS_ENTROPY_LOSS_FUNCTION -> {}
            case SQUARED_HINGE_LOSS_FUNCTION -> {}
            case SSIM_LOSS_FUNCTION -> {}
            case TRIPLET_MARGIN_LOSS_FUNCTION -> {}
        }
    }
    public static void lossFunction(DEFAULT_LOSS_FUNCTIONS lossFunction, double[][] trueValues, double[][] predictedValues){
        switch (lossFunction){
            case BINARY_CROSS_ENTROPY_LOSS_FUNCTION -> {}
            case CATEGORICAL_CROSS_ENTROPY_LOSS_FUNCTION -> {}
            case CONTRASTIVE_LOSS_FUNCTION -> {}
            case COSINE_PROXIMITY_LOSS_FUNCTION -> {}
            case FOCAL_LOSS_FUNCTION -> {}
            case HINGE_LOSS_FUNCTION -> {}
            case HUBER_LOSS_FUNCTION -> {}
            case KL_DIVERGENCE_LOSS_FUNCTION -> {}
            case LOG_COSH_LOSS_FUNCTION -> {}
            case L1_LOSS_FUNCTION -> {}
            case MAPE_LOSS_FUNCTION -> {}
            case MSE_LOSS_FUNCTION -> {}
            case MSLE_LOSS_FUNCTION -> {}
            case NLL_LOSS_FUNCTION -> {}
            case RANKNET_LOSS_FUNCTION -> {}
            case SPARSE_CATEGORICAL_CROSS_ENTROPY_LOSS_FUNCTION -> {}
            case SQUARED_HINGE_LOSS_FUNCTION -> {}
            case SSIM_LOSS_FUNCTION -> {}
            case TRIPLET_MARGIN_LOSS_FUNCTION -> {}
        }
    }


    private void binaryCrossEntropy(double[] values){}
    private void binaryCrossEntropy(double[][] values){}


    private void categoricalCrossEntropy(double[] values){}
    private void categoricalCrossEntropy(double[][] values){}


    private void contrastive(double[] values){}
    private void contrastive(double[][] values){}
}
