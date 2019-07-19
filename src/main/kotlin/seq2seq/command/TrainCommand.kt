package seq2seq.command

import org.bytedeco.opencv.opencv_text.DoubleVector
import kotlin.math.roundToInt
import org.nd4j.evaluation.regression.RegressionEvaluation

import picocli.CommandLine
import picocli.CommandLine.*

import seq2seq.data.loadDataFromFolder
import seq2seq.data.restoreDataNormalizer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.api.ndarray.BaseNDArray
import seq2seq.data.DataNormalizer
import seq2seq.data.dataNormalized
import java.io.*

private var intersectList: MutableSet<ArrayList<String>> = mutableSetOf();

fun setIntersetList(list: MutableSet<ArrayList<String>>) {
    intersectList = list
}

fun getIntersetList(): MutableSet<ArrayList<String>> {
    return intersectList
}



//  train -in data outModel.bin outNormalize.bin -testRatio 0.4
@CommandLine.Command(name = "train", description = ["Train"])
class TrainCommand: Runnable {
    @Option(names = ["--help"], usageHelp = true, description = ["display this help and exit"])
    var help: Boolean = false

    @Option(names = ["-e"], description = ["number of epoch to train"], required = false)
    private var epoch: Int = 1

    @Option(names = ["-ts"], description = ["number of time steps is used to predict one hour ahead"])
    private var ts: Int = 6

    @Option(names = ["-in"], description = ["input directory"], required = true)
    private lateinit var inputDirectory: File

    @Option(names = ["-model"], description = ["Pre-trained model for incremental learning"])
    private var inputModel: File? = null

    @Option(names = ["-normalizer"], description = ["Pre-trained normalizer"])
    private var inputNormalizer: File? = null

    @Option(names = ["-learningRate"], description = ["learning Rate"])
    private var learningRate: Double = 0.01

    @Option(names = ["-testRatio"], description = ["ratio to split test and train"], required = false)
    private var testRatio: Double = 0.05

    @Option(names = ["-lstmHiddenLayer"], description = ["Number of hidden neuron in LSTM layer"], required = false)
    private var lstmHiddenLayer: Int = 200

    @Option(names = ["-batchSize"], description = ["Batch size"], required = false)
    private var batchSize: Int = 150
    //
    @Option(names = ["-miniBatchSize"], description = ["Mini Batch size"], required = false)
    private var miniBatchSize: Int = 150

    @Option(names = ["-fullyConnLayer"], description = ["Fully Connected Layer"], required = false)
    private var fullyConnectedLayer = 128

    @Option(names = ["-monitor"], description = ["Enable graphical UI to monitor training process at http://localhost:9000"])
    private var monitor = false

    @Parameters(index = "0", description = ["output location (folder) for trained model and normalizer model"])
    private lateinit var outputModel: File

    @Parameters(index = "1", description = ["output train data normalizer"])
    private lateinit var outputNormalizer: File

    override fun run() {
        val dataset = loadDataFromFolder(inputDirectory, ts, false)
        val listMiniBatch = dataset.batchBy(miniBatchSize)
        val trainSet = listMiniBatch.subList(0, (listMiniBatch.size * (1-testRatio)).roundToInt())
        val testSet = listMiniBatch.subList((listMiniBatch.size * (1-testRatio)).roundToInt(), listMiniBatch.size)

        val model = seq2seq.buildLSTMNetwork(learningRate, lstmHiddenLayer)

        for (i in 1..epoch) {
            for (trainMiniBatch in trainSet) {
                model.fit(trainMiniBatch)
            }
            println(i.toString() + " / " + model.score())
        }
        model.save(outputModel)
        dataNormalized.save(outputNormalizer)
        val eval = RegressionEvaluation()

        val total: List<List<Double>> = listOf()
        for (testBatch in testSet) {
            model.rnnClearPreviousState()
            val output = model.rnnTimeStep(testBatch.features)
//            val result = ((output as Iterable<*>).first() as BaseNDArray).getColumn((ts -1).toLong()).toDoubleVector()
//            val result = ((output as Iterable<*>).first() as BaseNDArray).toDoubleVector()
            val result = output.toDoubleVector()
//            total.plus(result.toList())

            eval.eval(output, testBatch.labels)
        }
        println(eval.stats())
//        OutputStreamWriter(FileOutputStream("trainOutput.csv")).use {
//            total.forEachIndexed {
//                i, list ->
//                    list.forEachIndexed { index, d ->
//                        val value: Double
//                        if (d < 0) {
//                            value = -d
//                        } else {
//                            value = d
//                        }
//                        it.write((value.times(dataNormalized.stdArray.last())).plus(dataNormalized.mean.last()).toString())
//                        it.write("\n")
//
//                    }
//            }
//            it.flush()
//        }

//                it.write((result.times(dataNormalized.stdArray.last())).plus(dataNormalized.mean.last()).toString())
//                it.write("\n")
//                it.flush()

        val a = File("stat.csv")
        a.writeText(eval.stats())
    }
}
