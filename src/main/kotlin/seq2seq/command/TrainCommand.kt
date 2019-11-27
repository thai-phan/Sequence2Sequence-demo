package seq2seq.command

import kotlin.math.roundToInt
import org.nd4j.evaluation.regression.RegressionEvaluation

import picocli.CommandLine
import picocli.CommandLine.*

import seq2seq.data.*
import java.io.*

//  train -in data -testRatio 0.06 -stat stat.csv -e 1 outModel.bin outNormalize.bin trainOutput.csv -columns FLOW_POP_CNT_MON,FLOW_POP_CNT_TUS,FLOW_POP_CNT_WED,FLOW_POP_CNT_THU,FLOW_POP_CNT_FRI,FLOW_POP_CNT_SAT

@CommandLine.Command(name = "train", description = ["Train"])
class TrainCommand: Runnable {
    @Option(names = ["--help"], usageHelp = true, description = ["display this help and exit"])
    var help: Boolean = false

    @Option(names = ["-e"], description = ["number of epoch to train"], required = false)
    private var epoch: Int = 1000

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

    @Option(names = ["-miniBatchSize"], description = ["Mini Batch size"], required = false)
    private var miniBatchSize: Int = 50

    @Option(names = ["-fullyConnLayer"], description = ["Fully Connected Layer"], required = false)
    private var fullyConnectedLayer = 128

    @Option(names = ["-columns"], description = ["Select columns' index which are used to predict, index starts from 0"], required = false)
    private var columns: String = "mon,tus,wed,thu,fri,sat"

    @Option(names = ["-monitor"], description = ["Enable graphical UI to monitor training process at http://localhost:9000"])
    private var monitor = false

    @Option(names = ["-normalizeCoefficient"], description = ["Coefficient for standard deviation"])
    private var coefficient: Int = 10

    @Option(names = ["-stat"], description = ["results stats file"], required = true)
    private lateinit var statFile: File

    @Parameters(index = "0", description = ["output location (folder) for trained model and normalizer model"])
    private lateinit var outputModel: File

    @Parameters(index = "1", description = ["output train data normalizer"])
    private lateinit var outputNormalizer: File

    @Parameters(index = "2", description = ["output test data"])
    private lateinit var outputTest: File
    override fun run() {
        val fileList = loadDataFromFolder(inputDirectory)
        val trainFiles = fileList.subList(0, (fileList.size * (1-testRatio)).roundToInt())
        val testFiles = fileList.subList((fileList.size * (1-testRatio)).roundToInt(), fileList.size)
        println("Number of train files with test ratio " + testRatio + " : " + trainFiles.size)
        println("Number of test files with test ratio " + testRatio + " : " + testFiles.size)
        val columnList = columns.split(",")
        println("Number of column use to train: " + columnList.size)
        val trainDataSet = loadDataSetFromFiles(trainFiles, false, coefficient, columnList)
        val testDataSet = loadDataSetFromFiles(testFiles, true, coefficient, columnList)
        val trainSet = trainDataSet.batchBy(miniBatchSize)
        val model = seq2seq.buildLSTMNetwork(learningRate, lstmHiddenLayer, fullyConnectedLayer, columnList.size)

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
            val revertResult = output.mul(dataNormalized.stdArray.last() * dataNormalized.coefficientStd).add(dataNormalized.meanArray.last())
            val result = revertResult.toDoubleVector()
            val origin = testDataSet.labels.mul(dataNormalized.stdArray.last() * dataNormalized.coefficientStd).add(dataNormalized.meanArray.last())

            eval.eval(revertResult, origin)
            println(eval.stats())

            OutputStreamWriter(FileOutputStream(outputTest)).use {
                it.write("X|Y|Origin|Predict\n")
                locationFile.forEachIndexed { index, originData ->
                    val d = result[index]
                    it.write(originData[2] + "|" + originData[3] + "|" + originData[10] + "|"+ d)
                    it.write("\n")
                    it.flush()
                }
            }
            statFile.writeText(eval.stats())
        }
    }
}
