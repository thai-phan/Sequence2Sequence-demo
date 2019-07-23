package seq2seq.command

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.evaluation.regression.RegressionEvaluation
import org.nd4j.linalg.api.ndarray.BaseNDArray
import picocli.CommandLine
import java.io.File
import java.io.FileOutputStream
import java.io.OutputStreamWriter
import picocli.CommandLine.*
import seq2seq.data.*
import java.io.IOException

// predict -in dataIn -model outModel.bin -normalizer outNormalize.bin result.csv
@CommandLine.Command(name = "predict", description = ["Predict"])
class PredictCommand: Runnable {
    @Option(names = ["-model"], description = ["Trained model which used to predict"], required = true)
    private lateinit var inputModel: File

    @Option(names = ["-normalizer"], description = ["Pre-trained normalizer"], required = true)
    private lateinit var inputNormalizer: File

    @Option(names = ["-in"], description = ["input directory"], required = true)
    private lateinit var inputDirectory: File

    @Option(names = ["-ts"], description = ["number of time steps, must match time steps from input model"])
    private var ts: Int = 5

    @Option(names = ["-stat"], description = ["results stats file"])
    private lateinit var statFile: File

    @Parameters(index = "0", description = ["results output file"])
    private lateinit var outputFile: File

    override fun run() {
        val normalizer = restoreDataNormalizer(inputNormalizer)
        setIntersetList(normalizer.list)
        val files = loadDataFromFolder(inputDirectory)
        val dataset = loadDataSetFromFiles(files, true)
        val model = MultiLayerNetwork.load(inputModel, false)
        val indResult = model.rnnTimeStep(dataset.features)


        val eval = RegressionEvaluation()
        val result = indResult.toDoubleVector()
        eval.eval(indResult, dataset.labels)

        if (outputFile.exists()) {
            outputFile.delete()
            try {
                outputFile.createNewFile()
            } catch (e: IOException) {
                e.printStackTrace()
            }
        }
        OutputStreamWriter(FileOutputStream("trainOutput.csv")).use {
            it.write("X|Y|Origin|Predict\n")
            result.forEachIndexed { index, d ->
                it.write(locationFile.last()[index][2] + "|" + locationFile.last()[index][3] + "|" + locationFile.last()[index][10] + "|"+ (d.times(dataNormalized.stdArray.first())).plus(dataNormalized.meanArray.first()).toString())
                it.write("\n")
                it.flush()
            }
        }
        statFile.writeText(eval.stats())
    }
}
