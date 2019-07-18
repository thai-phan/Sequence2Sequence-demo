package seq2seq.data

import org.apache.commons.math3.stat.descriptive.moment.Mean
import seq2seq.command.getIntersetList
import java.io.*
import java.util.function.Consumer
import kotlin.math.pow
import kotlin.math.sqrt

data class DataNormalizer(val mean: ArrayList<Double> = ArrayList(),
                          val stdArray: ArrayList<Double> = ArrayList(),
                          val list: MutableSet<ArrayList<String>> = mutableSetOf()): Serializable  {
    fun save(output: File) {
        ObjectOutputStream(FileOutputStream(output)).use {
            it.writeObject(this)
            it.flush()
        }
    }
}

fun normalizerDataSet(dataList: ArrayList<ArrayList<Double>>,
                      normalizerFile: File,
                      isOutput: Boolean,
                      isPredict: Boolean): ArrayList<ArrayList<Double>> {
    val dataSize:Int = dataList.size;
    val meanArray: ArrayList<Double> = ArrayList()
    val stdArray: ArrayList<Double> = ArrayList()

    for (i in 0..5) {
        meanArray.add(0.0)
        stdArray.add(0.0)
    }

    for (data in dataList) {
        data.forEachIndexed { index, element ->
            meanArray[index] += element
        }
    }
    // Get mean
    meanArray.forEachIndexed { index, element ->
        meanArray[index] = element/dataSize
    }

    for (data in dataList) {
        data.forEachIndexed { index, element ->
            stdArray[index] += (element - meanArray[index]).pow(2)
        }
    }
    // Get standard deviation
    for (total in stdArray) {
        stdArray[stdArray.indexOf(total)] = sqrt(total/(dataSize-1))
    }

    // Normalize
    val dataListNormalized: ArrayList<ArrayList<Double>> = dataList
    dataListNormalized.forEachIndexed { indexSet, elementSet ->
        elementSet.forEachIndexed { indexItem, elementItem ->
            dataListNormalized[indexSet][indexItem] = (elementItem - meanArray[indexItem]) / stdArray[indexItem]
        }
    }
    if (isOutput && !isPredict) {
        val data = DataNormalizer(meanArray, stdArray, getIntersetList())
        data.save(normalizerFile)
    }
    for (data in dataList) {

    }
//    val list = getMean(dataList)
    return dataListNormalized
}

fun restoreDataNormalizer(inputFile: File): DataNormalizer {
    val rd = ObjectInputStream(FileInputStream(inputFile))
    val ret = rd.readObject() as DataNormalizer
    rd.close()
    return ret
}
