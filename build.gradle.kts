import org.jetbrains.kotlin.gradle.tasks.KotlinCompile

plugins {
    // Apply the Kotlin JVM plugin to add support for Kotlin on the JVM.
    id("org.jetbrains.kotlin.jvm").version("1.3.21")

    // Apply the application plugin to add support for building a CLI application.
    application
}

val dl4jVersion = "1.0.0-beta4"
val awsS3Version = "1.11.109"
val picocliVersion = "3.9.3"
val kotlinLoggerVersion = "1.6.22"

repositories {
    // Use jcenter for resolving your dependencies.
    // You can declare any Maven/Ivy/file repository here.
    jcenter()
}

dependencies {
    // Use the Kotlin JDK 8 standard library.
    implementation("org.jetbrains.kotlin:kotlin-stdlib-jdk8")

    // Use the Kotlin test library.
    testImplementation("org.jetbrains.kotlin:kotlin-test")

    // Use the Kotlin JUnit integration.
    testImplementation("org.jetbrains.kotlin:kotlin-test-junit")
    compile("org.slf4j", "slf4j-jdk14", "1.7.25")
    compile("org.deeplearning4j", "deeplearning4j-core", dl4jVersion)

//    compile("org.nd4j", "nd4j-cuda-10.1-platform", dl4jVersion)
    compile("org.nd4j", "nd4j-native-platform", dl4jVersion)

    compile("software.amazon.awssdk", "s3", "2.3.8")
    compile("com.natpryce", "konfig", "1.6.10.0")
    runtime("org.jetbrains.kotlin", "kotlin-stdlib-jdk8", "1.3.41")
    compile("jfree", "jfreechart", "1.0.13")
    compile("info.picocli", "picocli", picocliVersion)
    compile("org.jetbrains.kotlin", "kotlin-runtime", "1.2.71")
    compile("nl.dionsegijn", "konfetti", "1.1.2")
    compile(kotlin("stdlib-jdk8"))
}

application {
    // Define the main class for the application.
    mainClassName = "seq2seq.AppKt"
}
tasks.withType<KotlinCompile> {
    kotlinOptions.jvmTarget = "1.8"
}

val fatJar = task("fatJar", type = Jar::class) {
    manifest {
        attributes["Main-Class"] = "seq2seq.AppKt"
    }
    from(configurations.runtime.get().map { if (it.isDirectory) it else zipTree(it) })
    with(tasks["jar"] as CopySpec)
}
tasks {
    "build" {
        dependsOn(fatJar)
    }
}
val compileKotlin: KotlinCompile by tasks
compileKotlin.kotlinOptions {
    jvmTarget = "1.8"
}
val compileTestKotlin: KotlinCompile by tasks
compileTestKotlin.kotlinOptions {
    jvmTarget = "1.8"
}
