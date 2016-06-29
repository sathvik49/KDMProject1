
import org.apache.spark.ml.feature.{StopWordsRemover, HashingTF, IDF, Tokenizer}
import org.apache.spark.sql.SparkSession


/**
  * Created by Mayanka on 17-Jun-16.
  */
object SparkNLPMain {

  def main(args: Array[String]) {
    val spark = SparkSession
      .builder
      .appName("TfIdfExample")
      .master("local[*]")
      .getOrCreate()

    // $example on$
    val sentenceData = spark.createDataFrame(Seq(
      (0, "Hi I heard about Spark"),
      (0, "I wish Java could use case classes"),
      (1, "Logistic regression models are neat")
    )).toDF("label", "sentence")

    val tokenizer = new Tokenizer().setInputCol("sentence").setOutputCol("words")
    val wordsData = tokenizer.transform(sentenceData)

    val remover = new StopWordsRemover()
      .setInputCol("words")
      .setOutputCol("filteredWords")
    val processedWordData= remover.transform(wordsData)

    val hashingTF = new HashingTF()
      .setInputCol("filteredWords").setOutputCol("rawFeatures").setNumFeatures(20)
    val featurizedData = hashingTF.transform(processedWordData)
    // alternatively, CountVectorizer can also be used to get term frequency vectors

    val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
    val idfModel = idf.fit(featurizedData)
    val rescaledData = idfModel.transform(featurizedData)
    rescaledData.select("filteredWords","features", "label").take(3).foreach(println)


    spark.stop()

  }

}
