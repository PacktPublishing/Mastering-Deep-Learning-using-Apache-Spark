package chapter1

import org.apache.spark.{SparkConf, SparkContext}
import org.scalatest.FunSuite

class SparkRDDAPITest extends FunSuite {
  val spark: SparkContext = SparkContext.getOrCreate(new SparkConf().setAppName("test-1").setMaster("local[*]"))


  test("should trigger computations using actions without reuse") {
    //given
    val input = spark.makeRDD(
      List(
        UserTransaction(userId = "A", amount = 1001),
        UserTransaction(userId = "A", amount = 100),
        UserTransaction(userId = "A", amount = 102),
        UserTransaction(userId = "A", amount = 1),
        UserTransaction(userId = "B", amount = 13)))

    //when apply transformation
    val rdd = input
      .filter(_.userId.contains("A"))
      .keyBy(_.userId)
      .map(_._2.amount)


    //then every call to action means that we are going up to the RDD chain
    //if we are loading data from external file-system (I.E.: HDFS), every action means
    //that we need to load it from FS.
    val start = System.currentTimeMillis()
    println(rdd.collect().toList)
    println(rdd.count())
    println(rdd.first())
    rdd.foreach(println(_))
    rdd.foreachPartition(t => t.foreach(println(_)))
    println(rdd.max())
    println(rdd.min())
    println(rdd.takeOrdered(1).toList)
    println(rdd.takeSample(false, 2).toList)
    val result = System.currentTimeMillis() - start

    println(s"time taken (no-cache): $result")


  }


  test("should trigger computations using actions with reuse") {
    //given
    val input = spark.makeRDD(
      List(
        UserTransaction(userId = "A", amount = 1001),
        UserTransaction(userId = "A", amount = 100),
        UserTransaction(userId = "A", amount = 102),
        UserTransaction(userId = "A", amount = 1),
        UserTransaction(userId = "B", amount = 13)))

    //when apply transformation
    val rdd = input
      .filter(_.userId.contains("A"))
      .keyBy(_.userId)
      .map(_._2.amount)
      .cache()


    //then every call to action means that we are going up to the RDD chain
    //if we are loading data from external file-system (I.E.: HDFS), every action means
    //that we need to load it from FS.
    val start = System.currentTimeMillis()
    println(rdd.collect().toList)
    println(rdd.count())
    println(rdd.first())
    rdd.foreach(println(_))
    rdd.foreachPartition(t => t.foreach(println(_)))
    println(rdd.max())
    println(rdd.min())
    println(rdd.takeOrdered(1).toList)
    println(rdd.takeSample(false, 2).toList)
    val result = System.currentTimeMillis() - start

    println(s"time taken(cache): $result")


  }
  case class UserTransaction(userId: String, amount: Int)
}

