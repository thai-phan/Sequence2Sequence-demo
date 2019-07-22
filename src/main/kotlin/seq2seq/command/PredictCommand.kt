package seq2seq.command

import seq2seq.data.restoreDataNormalizer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.api.ndarray.BaseNDArray
import picocli.CommandLine
import java.io.File
import java.io.FileOutputStream
import java.io.OutputStreamWriter
import picocli.CommandLine.*
import seq2seq.data.loadDataFromFolder
import seq2seq.data.locationFile
import java.io.IOException


private var intersectPredictList: MutableSet<ArrayList<String>> = mutableSetOf();

fun setIntersetPredictList(list: MutableSet<ArrayList<String>>) {
    intersectPredictList = list
}

fun getIntersetPredictList(): MutableSet<ArrayList<String>> {
    return intersectPredictList
}

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

    @Parameters(index = "0", description = ["results output file"])
    private lateinit var outputFile: File


    override fun run() {

        val normalizer = restoreDataNormalizer(inputNormalizer)
        setIntersetPredictList(normalizer.list)
        val dataset = loadDataFromFolder(inputDirectory, ts, true)


        val model = MultiLayerNetwork.load(inputModel, false)
        val indResult = model.rnnTimeStep(dataset.features)
        val result = ((indResult as Iterable<*>).first() as BaseNDArray).getColumn((ts -1).toLong()).toDoubleVector()

        if (outputFile.exists()) {
            outputFile.delete()
            try {
                outputFile.createNewFile()
            } catch (e: IOException) {
                e.printStackTrace()
            }
        }

//        val intersectArray = getIntersetPredictList().toTypedArray()
        OutputStreamWriter(FileOutputStream(outputFile)).use {
            result.forEachIndexed { index, d ->
                it.write(locationFile[0][index][0] + "|" + locationFile[0][index][1] + "|" + (d.times(normalizer.stdArray.last())).plus(normalizer.mean.last()).toString())
                it.write("\n")
                it.flush()
            }
        }
    }
}
