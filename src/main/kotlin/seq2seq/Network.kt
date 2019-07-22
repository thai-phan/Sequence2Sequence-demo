package seq2seq
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.*

import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.conf.layers.*
import org.deeplearning4j.nn.conf.preprocessor.CnnToRnnPreProcessor
import org.deeplearning4j.nn.conf.preprocessor.RnnToCnnPreProcessor
import org.deeplearning4j.nn.weights.WeightInit

import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam

import org.nd4j.linalg.lossfunctions.LossFunctions;
import seq2seq.data.intersectSize

fun buildLSTMNetwork(learningRate: Double, lstmLayer: Int): MultiLayerNetwork {
    val conf = NeuralNetConfiguration.Builder()
        .seed(123)
        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
        .updater(Adam(learningRate))
        .list()
        .layer(0, DenseLayer.Builder()
            .nIn(6)
            .nOut(128)
            .weightInit(WeightInit.XAVIER)
            .activation(Activation.TANH)
            .build())
        .layer(1, LSTM.Builder()
            .nIn(128)
            .nOut(lstmLayer)
            .activation(Activation.TANH)
            .build()
        )
//        .layer(2, LSTM.Builder()
//            .nIn(lstmLayer)
//            .nOut(lstmLayer)
//            .activation(Activation.TANH)
//            .build()
//        )
        .layer(2, RnnOutputLayer.Builder()
            .nIn(lstmLayer)
            .nOut(6)
            .activation(Activation.TANH)
            .lossFunction(LossFunctions.LossFunction.SQUARED_LOSS)
            .build()
        )
//        .inputPreProcessor(0, RnnToCnnPreProcessor(6, 6, 5))
//        .inputPreProcessor(1, CnnToRnnPreProcessor(3, 3, cnnLayer1.toLong()))

    val net = MultiLayerNetwork(conf.build())
    net.init()
    return net
}
