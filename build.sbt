name := "mastering-deep-learning"

version := "1.0"

scalaVersion := "2.10.4"

resolvers += "Sonatype OSS Snapshots" at "https://oss.sonatype.org/content/repositories/snapshots"
resolvers += "Sonatype Nexus" at "http://oss.sonatype.org/service/local/staging/deploy/maven2/"

// Scala dependencies
libraryDependencies += "org.scala-lang" % "scala-library" % "2.10.4"

libraryDependencies += "org.apache.httpcomponents" % "httpclient" % "4.5.6"

libraryDependencies += "org.scala-lang" % "scala-reflect" % "2.10.4"

// Spark dependencies
libraryDependencies += "org.apache.spark" % "spark-mllib_2.10" % "1.3.1"

libraryDependencies += "org.apache.spark" % "spark-core_2.10" % "1.3.1"

// N-Dimensional Arrays for Java dependencies
libraryDependencies += "org.nd4j" % "nd4j-api" % "1.0.0-beta3"

libraryDependencies += "org.nd4j" % "nd4j-native" % "1.0.0-beta3"

libraryDependencies += "org.nd4j" % "nd4j-native-platform" % "1.0.0-beta3"

// Deeplearning4j dependencies
libraryDependencies += "org.deeplearning4j" %% "dl4j-spark" % "1.0.0-beta3_spark_1"
libraryDependencies += "org.deeplearning4j" % "deeplearning4j-nlp" % "1.0.0-beta3"

// Typesafe dependencies
libraryDependencies += "com.typesafe" % "config" % "1.3.0"

libraryDependencies += "org.scalatest" %% "scalatest" % "3.0.5" % Test

libraryDependencies += "org.datavec" %% "datavec-spark" % "1.0.0-beta3_spark_1"

libraryDependencies += "org.datavec" % "datavec-api" % "1.0.0-beta3"

libraryDependencies += "org.datavec" % "datavec-data-codec" % "1.0.0-beta3"

// https://mvnrepository.com/artifact/com.twelvemonkeys.imageio/imageio-core
libraryDependencies += "com.twelvemonkeys.imageio" % "imageio-core" % "3.4.1"

// https://mvnrepository.com/artifact/org.deeplearning4j/deeplearning4j-play
libraryDependencies += "org.deeplearning4j" %% "deeplearning4j-play" % "1.0.0-beta3"





// JUnit
libraryDependencies += "junit" % "junit" % "4.12" % Test

envVars in Test := Map("dependency.platform" -> "macosx-x86_64")


// Cuda Dependencies
libraryDependencies += "org.bytedeco.javacpp-presets" % "cuda" % "10.0-7.3-1.4.3" classifier "macosx-x86_64-redist"
libraryDependencies += "org.bytedeco.javacpp-presets" % "cuda-platform" % "10.0-7.3-1.4.3"


libraryDependencies += "org.deeplearning4j" % "deeplearning4j-parallel-wrapper" % "1.0.0-beta3"

libraryDependencies += "org.nd4j" % "nd4j-cuda-9.0-platform" % "1.0.0-beta3" excludeAll ExclusionRule(organization = "org.bytedeco.javacpp-presets")
libraryDependencies += "org.nd4j" % "nd4j-cuda-9.2-platform" % "1.0.0-beta3" excludeAll ExclusionRule(organization = "org.bytedeco.javacpp-presets")
libraryDependencies += "org.nd4j" % "nd4j-cuda-10.0-platform" % "1.0.0-beta3" excludeAll ExclusionRule(organization = "org.bytedeco.javacpp-presets")

// https://mvnrepository.com/artifact/org.jcodec/jcodec
libraryDependencies += "org.jcodec" % "jcodec" % "0.1.5"
