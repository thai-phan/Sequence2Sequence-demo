package seq2seq.data

import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.factory.Nd4j.ones
import java.io.File
import java.util.*
import kotlin.collections.ArrayList
import kotlin.collections.HashMap
import org.deeplearning4j.clustering.util.SetUtils
import seq2seq.command.getIntersetPredictList
import seq2seq.command.setIntersetList
import kotlin.collections.HashSet
import java.util.TreeSet

var intersectSize = 0
val locationFile = ArrayList<ArrayList<ArrayList<String>>>()

fun parseCSVtoMatrixObject(files: List<File>, isPredict: Boolean): Map<String, ArrayList<ArrayList<ArrayList<Double>>>> {
    val parseMap = HashMap<String, ArrayList<ArrayList<ArrayList<Double>>>>()
    val inputFile = ArrayList<ArrayList<ArrayList<Double>>>()
    val outputFile = ArrayList<ArrayList<ArrayList<Double>>>()

    var intersectList: MutableSet<ArrayList<String>> = mutableSetOf()

    if (isPredict) {
        intersectList = getIntersetPredictList()
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

    for (file in files) {
        val locationLists: ArrayList<ArrayList<String>> = ArrayList()
        val inputLists = ArrayList<ArrayList<Double>>()
        val outputLists = ArrayList<ArrayList<Double>>()
        val reader = file.bufferedReader()
        reader.readLine();
        val fileLines = reader.readLines().map {
            it.split("|")
        }
        var count = 0
        fileLines.forEachIndexed { indexLine, list ->
            val locationList: ArrayList<String> = ArrayList()
            val inputList: ArrayList<Double> = ArrayList()
            val outputList: ArrayList<Double> = ArrayList()

            if (intersectList.contains(arrayListOf(list[2], list[3]))) {
                locationList.add(list[2])
                locationList.add(list[3])

                list.forEachIndexed { indexItem, item ->
                    if (indexItem in 4..9) {
                        inputList.add(item.toDouble())
                    }
                    if (indexItem in 5..10) {
                        outputList.add(item.toDouble())
                    }
                }
            } else {
                count++
            }
            inputLists.add(inputList);
            outputLists.add(outputList);
            locationLists.add(locationList)
        }
        print("asdas" + count)
        inputFile.add(inputLists)
        outputFile.add(outputLists)
        locationFile.add(locationLists)
    }

//    val set = TreeSet(object : Comparator<ArrayList<Double>> {
//        override fun compare(one: ArrayList<Double>, other: ArrayList<Double>): Int {
//            return one[0].compareTo(other[0])
//        }
//    })

    parseMap["input"] = inputFile;
    parseMap["output"] = outputFile;
    return parseMap;
}

//fun getLocationFromFile(files: List<File>) {
//    val firstFile = files.get(0).bufferedReader()
//    firstFile.readLine()
//    val firstFileList = firstFile.readLines().map {
//        it.split("|")
//    }
//    val intersectList: ArrayList<ArrayList<Double>> = arrayListOf()
//    firstFileList.forEachIndexed { indexItem, list ->
//        intersectList.add(arrayListOf(list[2].toDouble(), list[3].toDouble()))
//    }
//}

fun loadDataFromFolder(location: File, timeStep: Int, normalizerFile: File, isPredict: Boolean): DataSet  {
    val files = location.listFiles()
        .filter{ it.name.toLowerCase().endsWith(".csv") }
        .sortedBy { it.name }
    return toDataSet(parseCSVtoMatrixObject(files, isPredict), timeStep, normalizerFile, isPredict)
}

fun toDataSet(dataList: Map<String, ArrayList<ArrayList<ArrayList<Double>>>>, timeStep: Int, normalizerFile: File, isPredict: Boolean): DataSet {

    val inputSet: ArrayList<ArrayList<ArrayList<Double>>> = dataList.get("input")!!

    val outputSet: ArrayList<ArrayList<ArrayList<Double>>> = dataList.get("output")!!
    inputSet.forEachIndexed { index, element ->
        inputSet[index] = normalizerDataSet(element, normalizerFile, false, isPredict)
    }
    outputSet.forEachIndexed { index, element ->
        outputSet[index] = normalizerDataSet(element, normalizerFile, true, isPredict)
    }

    val inputNd = Nd4j.create(intArrayOf(inputSet.size, intersectSize, inputSet.first().first().size), 'c')
    val outputNd = Nd4j.create(intArrayOf(outputSet.size, intersectSize, outputSet.first().first().size), 'c')

    inputSet.forEachIndexed { indexArray, arrayList ->
        for (i in 0..intersectSize -1) {
            arrayList[i].forEachIndexed { indexSet, set ->
                inputNd.putScalar(intArrayOf(indexArray, i, indexSet), set)
            }
        }
    }
    outputSet.forEachIndexed { indexArray, arrayList ->
        for (i in 0..intersectSize -1) {
            arrayList[i].forEachIndexed { indexSet, set ->
                outputNd.putScalar(intArrayOf(indexArray, i, indexSet), set)
            }
        }
    }
    return DataSet(inputNd, outputNd)
}
