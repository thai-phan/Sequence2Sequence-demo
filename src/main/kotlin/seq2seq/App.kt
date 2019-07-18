package seq2seq

import seq2seq.command.PredictCommand
import seq2seq.command.TrainCommand

import picocli.CommandLine
import picocli.CommandLine.*


//  train -in ./data -e 10 -ts 6 airpollution.bin normalizer.bin
@Command(subcommands = [TrainCommand::class, PredictCommand::class])
class App {
    fun run(args: Array<String>) {
        val command = CommandLine(App())
        command.parseWithHandler(RunLast(), args)
    }
}

fun main(args: Array<String>) {
    App().run(args)
}
