package seq2seq.data

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import java.io.*

var dataNormalized: DataNormalizer = DataNormalizer()

data class DataNormalizer(val meanArray: List<Double> = listOf(),
                          val stdArray: List<Double> = listOf(),
                          val coefficientStd: Int = 0,
                          val list: MutableSet<ArrayList<String>> = mutableSetOf()): Serializable  {
    fun save(output: File) {
        ObjectOutputStream(FileOutputStream(output)).use {
            it.writeObject(this)
            it.flush()
        }
    }
}

fun normalizeZScore(dataNdArray: INDArray,
                    isPredict: Boolean): INDArray {
    if (isPredict) {
        val coefficientStd = dataNormalized.coefficientStd
        val meanArray = Nd4j.create(dataNormalized.meanArray)
        val stdArray = Nd4j.create(dataNormalized.stdArray)
        return dataNdArray.subRowVector(meanArray).divRowVector(stdArray).div(coefficientStd)
    } else {
        val coefficientStd = 3
        val meanArray = dataNdArray.mean(0)
        val stdArray = dataNdArray.std(0)
        dataNormalized = DataNormalizer(meanArray.toDoubleVector().toList(), stdArray.toDoubleVector().toList(), coefficientStd, getIntersetList())
        return dataNdArray.subRowVector(meanArray).divRowVector(stdArray).div(coefficientStd)
    }
}

fun restoreDataNormalizer(inputFile: File): DataNormalizer {
    val rd = ObjectInputStream(FileInputStream(inputFile))
    val ret = rd.readObject() as DataNormalizer
    rd.close()
    return ret
}
