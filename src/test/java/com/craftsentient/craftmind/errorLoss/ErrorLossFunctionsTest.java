package com.craftsentient.craftmind.errorLoss;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

public class ErrorLossFunctionsTest {
//    @Test
//    public void errorLossFunctionWithDoubleTruthAndDoublePredicted() throws Exception {
//        double loss = ErrorLossFunctions.lossFunction(
//                DEFAULT_LOSS_FUNCTIONS.HINGE_LOSS_FUNCTION,
//                -2.0,
//                1.0);
//        Assertions.assertEquals(3.0, loss);
//    }
//
//    @Test
//    public void errorLossFunctionWithDoubleArrayTruthAndDoubleArrayPredictedWithBinary() throws Exception {
//        // cannot be one or zero -> must be between them
//        double loss = (double)ErrorLossFunctions.lossFunction(
//                DEFAULT_LOSS_FUNCTIONS.BINARY_CROSS_ENTROPY_LOSS_FUNCTION,
//                new double[]{0.5},
//                new double[]{0.5});
//        Assertions.assertEquals(0.6931471805599453, loss);
//    }
//
//    @Test
//    public void errorLossFunctionWithDoubleArrayTruthAndDoubleArrayPredictedWithCCE() throws Exception {
//        double loss = (double)ErrorLossFunctions.lossFunction(
//                DEFAULT_LOSS_FUNCTIONS.CATEGORICAL_CROSS_ENTROPY_LOSS_FUNCTION,
//                new double[]{1},
//                new double[]{1});
//        Assertions.assertEquals(0, loss);
//    }
//
////    @Test
////    public void fullErrorLossFunctionWithDoubleArrayTruthAndDoubleArrayPredictedWithCCE() throws Exception {
////        double loss = (double)ErrorLossFunctions.lossFunction(
////                DEFAULT_LOSS_FUNCTIONS.CATEGORICAL_CROSS_ENTROPY_LOSS_FUNCTION,
////                new double[]{0},
////                new double[]{1});
////        Assertions.assertEquals(1, loss);
////    }
//
//    @Test
//    public void errorLossFunctionWithDoubleArrayTruthAndDoubleArrayPredictedWithCosineProximity() throws Exception {
//        double loss = (double)ErrorLossFunctions.lossFunction(
//                DEFAULT_LOSS_FUNCTIONS.COSINE_PROXIMITY_LOSS_FUNCTION,
//                new double[]{1},
//                new double[]{1});
//        Assertions.assertEquals(-1, loss);
//    }
//
//    @Test
//    public void errorLossFunctionWithDoubleArrayTruthAndDoubleArrayPredictedWithHinge() throws Exception {
//        double loss = (double)ErrorLossFunctions.lossFunction(
//                DEFAULT_LOSS_FUNCTIONS.COSINE_PROXIMITY_LOSS_FUNCTION,
//                new double[]{1},
//                new double[]{1});
//        Assertions.assertEquals(-1, loss);
//    }

    @Test
    public void errorLossFunctionWithDoubleArrayTruthAndDoubleArrayPredictedWithKLDivergence(){

    }

    @Test
    public void errorLossFunctionWithDoubleArrayTruthAndDoubleArrayPredictedWithLogCosh(){

    }

    @Test
    public void errorLossFunctionWithDoubleArrayTruthAndDoubleArrayPredictedWithL1(){

    }

    @Test
    public void errorLossFunctionWithDoubleArrayTruthAndDoubleArrayPredictedWithMape(){

    }

    @Test
    public void errorLossFunctionWithDoubleArrayTruthAndDoubleArrayPredictedWithMSLE(){

    }

    @Test
    public void errorLossFunctionWithDoubleArrayTruthAndDoubleArrayPredictedWithQuadratic(){

    }

    @Test
    public void errorLossFunctionWithDoubleArrayTruthAndDoubleArrayPredictedWithSquaredHinge(){

    }



}
