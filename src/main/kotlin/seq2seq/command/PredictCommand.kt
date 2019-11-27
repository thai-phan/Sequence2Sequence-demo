package seq2seq.command

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.evaluation.regression.RegressionEvaluation
import picocli.CommandLine
import java.io.File
import java.io.FileOutputStream
import java.io.OutputStreamWriter
import picocli.CommandLine.*
import seq2seq.data.*
import java.io.IOException

// predict -in dataIn -model outModel.bin -normalizer outNormalize.bin -stat stat_pre.csv result_predict.csv -columns FLOW_POP_CNT_MON,FLOW_POP_CNT_TUS,FLOW_POP_CNT_WED,FLOW_POP_CNT_THU,FLOW_POP_CNT_FRI,FLOW_POP_CNT_SAT
@CommandLine.Command(name = "predict", description = ["Predict"])
class PredictCommand: Runnable {
    @Option(names = ["-model"], description = ["Trained model which used to predict"], required = true)
    private lateinit var inputModel: File

    @Option(names = ["-normalizer"], description = ["Pre-trained normalizer"], required = true)
    private lateinit var inputNormalizer: File

    @Option(names = ["-in"], description = ["input directory"], required = true)
    private lateinit var inputDirectory: File

    @Option(names = ["-stat"], description = ["results stats file"])
    private lateinit var statFile: File

    @Parameters(index = "0", description = ["results output file"])
    private lateinit var outputFile: File

    @Option(names = ["-columns"], description = ["Select columns' index which are used to predict, index starts from 0"], required = false)
    private var columns: String = "mon,tus,wed,thu,fri,sat"

    override fun run() {
        dataNormalized = restoreDataNormalizer(inputNormalizer)
        setIntersetList(dataNormalized.list)
        val files = loadDataFromFolder(inputDirectory)
        val columnList = columns.split(",")
        println("Number of column use to predict: " + columnList.size)
        val dataset = loadDataSetFromFiles(files, true, dataNormalized.coefficientStd, columnList)
        val model = MultiLayerNetwork.load(inputModel, false)
        val indResult = model.rnnTimeStep(dataset.features)
        val eval = RegressionEvaluation()
        val revertResult = indResult.mul(dataNormalized.stdArray.last() * dataNormalized.coefficientStd).add(dataNormalized.meanArray.last())
        val result = revertResult.toDoubleVector()
        val origin = dataset.labels.mul(dataNormalized.stdArray.last() * dataNormalized.coefficientStd).add(dataNormalized.meanArray.last())
        eval.eval(revertResult, origin)
        println(eval.stats())
        if (outputFile.exists()) {
            outputFile.delete()
            try {
                outputFile.createNewFile()
            } catch (e: IOException) {
                e.printStackTrace()
            }
        }

        OutputStreamWriter(FileOutputStream(outputFile)).use {
            it.write("X|Y|Origin|Predict\n")
            locationFile.forEachIndexed { index, originData ->
                val d = result[index]
                it.write(originData[2] + "|" + originData[3] + "|" + originData[10] + "|"+ "%.2f".format(d))
                it.write("\n")
                it.flush()
            }
        }

        statFile.writeText(eval.stats())
    }
}
