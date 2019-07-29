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

// predict -in dataIn -model outModel.bin -normalizer outNormalize.bin -stat stat_pre.csv result_predict.csv
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

    override fun run() {
        dataNormalized = restoreDataNormalizer(inputNormalizer)
        setIntersetList(dataNormalized.list)
        val files = loadDataFromFolder(inputDirectory)
        val dataset = loadDataSetFromFiles(files, true, dataNormalized.coefficientStd)
        val model = MultiLayerNetwork.load(inputModel, false)
        val indResult = model.rnnTimeStep(dataset.features)
        val eval = RegressionEvaluation()
        val result = indResult.toDoubleVector()
        eval.eval(indResult, dataset.labels)
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
                it.write(originData[2] + "|" + originData[3] + "|" + originData[10] + "|"+ (d.times(dataNormalized.stdArray.last() * dataNormalized.coefficientStd)).plus(dataNormalized.meanArray.last()).toString())
                it.write("\n")
                it.flush()
            }
        }

        statFile.writeText(eval.stats())
    }
}
