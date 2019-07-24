package seq2seq
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.*
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.conf.layers.*
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.lossfunctions.LossFunctions;



fun buildLSTMNetwork(learningRate: Double, lstmLayer: Int, fullyConnectedLayer: Int): MultiLayerNetwork {
    val networkConfig = NeuralNetConfiguration.Builder()
        .seed(12345)
        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
        .updater(Adam(learningRate))
        .list()
        .layer(0, DenseLayer.Builder()
            .nIn(6)
            .nOut(fullyConnectedLayer)
            .weightInit(WeightInit.XAVIER)
            .activation(Activation.TANH)
            .build())
        .layer(1, LSTM.Builder()
            .nIn(fullyConnectedLayer)
            .nOut(lstmLayer)
            .activation(Activation.TANH)
            .build()
        )
        .layer(2, LSTM.Builder()
            .nIn(lstmLayer)
            .nOut(lstmLayer)
            .activation(Activation.TANH)
            .build()
        )
        .layer(3, OutputLayer.Builder()
            .nIn(lstmLayer)
            .nOut(1)
            .activation(Activation.TANH)
            .lossFunction(LossFunctions.LossFunction.SQUARED_LOSS)
            .build()
        )

    val net = MultiLayerNetwork(networkConfig.build())
    net.init()
    return net
}
