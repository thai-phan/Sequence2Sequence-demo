package seq2seq.data

import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import java.io.File
import kotlin.collections.ArrayList
import org.deeplearning4j.clustering.util.SetUtils
import org.nd4j.linalg.api.ndarray.INDArray

var intersectSize = 0
val locationFile = ArrayList<ArrayList<String>>()
private var intersectList: MutableSet<ArrayList<String>> = mutableSetOf();

fun setIntersetList(list: MutableSet<ArrayList<String>>) {
    intersectList = list
}

fun getIntersetList(): MutableSet<ArrayList<String>> {
    return intersectList
}

fun parseCSVtoMatrixObject(files: List<File>, isPredict: Boolean, coefficient: Int): INDArray {
    var intersectList: MutableSet<ArrayList<String>> = mutableSetOf()

    if (isPredict) {
        intersectList = getIntersetList()
        intersectSize = intersectList.size
    } else {
        val firstFile = files[0].bufferedReader()
        firstFile.readLine()
        val firstFileList = firstFile.readLines().map {
            it.split("|")
        }
        firstFileList.forEachIndexed { _, list ->
            intersectList.add(arrayListOf(list[2], list[3]))
        }
        for (file in files) {
            val reader = file.bufferedReader()
            reader.readLine()
            val lines = reader.readLines().map {
                it.split("|")
            }
            val listDay: MutableSet<ArrayList<String>> = mutableSetOf()

            lines.forEachIndexed { _, list ->
                listDay.add(arrayListOf(list[2], list[3]))
            }
            intersectList = SetUtils.intersection(intersectList, listDay)
        }
        setIntersetList(intersectList)
        intersectSize = intersectList.size
    }
    val objectSize = 7
    val dataNd = Nd4j.create(intArrayOf(files.size * intersectSize, objectSize), 'c')
    var countIndexForDataNd = 0
    files.forEachIndexed { indexFile, file ->
        val reader = file.bufferedReader()
        reader.readLine();
        val fileLines = reader.readLines().map {
            it.split("|")
        }
        fileLines.forEachIndexed { _, list ->
            val locationList: ArrayList<String> = arrayListOf()
            if (intersectList.contains(arrayListOf(list[2], list[3]))) {
                list.forEachIndexed { indexItem, item ->
                    locationList.add(item)
                    if (indexItem in 4..10) {
                        dataNd.putScalar(intArrayOf(countIndexForDataNd, indexItem-4), item.toDouble())
                    }
                }
                countIndexForDataNd += 1
                if (isPredict) {
                    locationFile.add(locationList)
                }
            }
        }
    }
    return normalizeZScore(dataNd, isPredict, coefficient);
}

fun loadDataFromFolder(location: File): List<File>  {
    val files = location.listFiles()
        .filter{ it.name.toLowerCase().endsWith(".csv") }
        .sortedBy { it.name }
    return files;
}

fun loadDataSetFromFiles(files: List<File>, isPredict: Boolean, coefficient: Int): DataSet {
    return splitFeatureAndLabel(parseCSVtoMatrixObject(files, isPredict, coefficient))
}

fun splitFeatureAndLabel(dataNd: INDArray): DataSet {
    val featureNd = dataNd.getColumns(0,1,2,3,4,5)
    val labelNd = dataNd.getColumn(6).reshape(dataNd.getColumn(6).shape()[0], 1)
    return DataSet(featureNd, labelNd)
}
