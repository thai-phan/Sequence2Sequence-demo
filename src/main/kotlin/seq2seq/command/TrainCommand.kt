package seq2seq.command

import kotlin.math.roundToInt
import org.nd4j.evaluation.regression.RegressionEvaluation

import picocli.CommandLine
import picocli.CommandLine.*

import seq2seq.data.*
import java.io.*



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
    private var learningRate: Double = 0.001

    @Option(names = ["-testRatio"], description = ["ratio to split test and train"], required = false)
    private var testRatio: Double = 0.05

    @Option(names = ["-lstmHiddenLayer"], description = ["Number of hidden neuron in LSTM layer"], required = false)
    private var lstmHiddenLayer: Int = 200

    @Option(names = ["-batchSize"], description = ["Batch size"], required = false)
    private var batchSize: Int = 150
    //
    @Option(names = ["-miniBatchSize"], description = ["Mini Batch size"], required = false)
    private var miniBatchSize: Int = 50

    @Option(names = ["-fullyConnLayer"], description = ["Fully Connected Layer"], required = false)
    private var fullyConnectedLayer = 128

    @Option(names = ["-monitor"], description = ["Enable graphical UI to monitor training process at http://localhost:9000"])
    private var monitor = false

    @Parameters(index = "0", description = ["output location (folder) for trained model and normalizer model"])
    private lateinit var outputModel: File

    @Parameters(index = "1", description = ["output train data normalizer"])
    private lateinit var outputNormalizer: File

    override fun run() {
        val fileList = loadDataFromFolder(inputDirectory)
        val trainFiles = fileList.subList(0, (fileList.size * (1-testRatio)).roundToInt())
        val testFiles = fileList.subList((fileList.size * (1-testRatio)).roundToInt(), fileList.size)
        println("Number of train files with test ratio " + testRatio + " : " + trainFiles.size)
        println("Number of test files with test ratio " + testRatio + " : " + testFiles.size)

        val trainDataSet = loadDataSetFromFiles(trainFiles, false)
        val testDataSet = loadDataSetFromFiles(testFiles, true)
        val trainSet = trainDataSet.batchBy(miniBatchSize)
        val model = seq2seq.buildLSTMNetwork(learningRate, lstmHiddenLayer, fullyConnectedLayer)

        for (i in 1..epoch) {
            for (trainMiniBatch in trainSet) {
                model.fit(trainMiniBatch)
            }
            println(i.toString() + " / " + model.score())
        }
        model.save(outputModel)
        dataNormalized.save(outputNormalizer)

        if (testFiles.isNotEmpty()) {
            val eval = RegressionEvaluation()
            val output = model.rnnTimeStep(testDataSet.features)
            val result = output.toDoubleVector()
            eval.eval(output, testDataSet.labels)
            println(eval.stats())
            OutputStreamWriter(FileOutputStream("trainOutput.csv")).use {
                it.write("X|Y|Origin|Predict\n")
                result.forEachIndexed { index, d ->
                    val originData = locationFile.last()[index]
                    it.write(originData[2] + "|" + originData[3] + "|" + originData[10] + "|"+ (d.times(dataNormalized.stdArray.last())).plus(dataNormalized.meanArray.last()).toString())
                    it.write("\n")
                    it.flush()
                }
            }

            val a = File("stat.csv")
            a.writeText(eval.stats())
        }
    }
}
