import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.sql.SQLContext
import org.apache.spark.mllib.stat.Statistics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.regression.DecisionTreeRegressor
import org.apache.spark.mllib.util.MLUtils
import scala.math.sqrt
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.regression.RandomForestRegressor
import org.apache.spark.ml.regression.GBTRegressor
import org.apache.spark.mllib.regression.LinearRegressionWithSGD




object project {
  def main(args: Array[String]): Unit = {
    val sparkConfig = new SparkConf().setMaster("local[2]").setAppName("ClusterScore")
    val sc = new SparkContext(sparkConfig)

    val sqlCtx = new SQLContext(sc)
    import sqlCtx._
    import sqlCtx.implicits._

    //load weather dataset
    var weatherDF = sqlCtx.load("com.databricks.spark.csv", Map("path" -> "loudacre/cleanData/weatherData/part*")).toDF(
      "time", "city_name", "temp", "temp_min", "temp_max", "pressure", "humidity", "wind_speed", "wind_deg",
      "rain_1h", "rain_3h", "snow_3h", "clouds_all", "weather_id", "weather_main", "weather_description")
    weatherDF.createOrReplaceTempView("weather")
    weatherDF = sqlCtx.sql("""SELECT * FROM weather WHERE temp != "temp" """)
    weatherDF.createOrReplaceTempView("weather")

    //load energy dataset
    var energyDF = sqlCtx.load("com.databricks.spark.csv", Map("path" -> "loudacre/cleanData/energyData/part*")).toDF(
      "time", "total_load_forecast", "total_load_actual", "price_day_ahead", "price_actual")
    energyDF.createOrReplaceTempView("energy")
    energyDF = sqlCtx.sql("""SELECT * FROM energy WHERE time != "time" AND total_load_actual != "" """)
    energyDF.createOrReplaceTempView("energy")

    //join energy dataset and weather dataset
    var joinDF = sqlCtx.sql("""SELECT * FROM weather,energy WHERE weather.time = energy.time""")
    joinDF.createOrReplaceTempView("join")

    val total_load_actualDF = sqlCtx.sql("""SELECT total_load_actual FROM join """)
    val total_load_actualRDD = total_load_actualDF.rdd.map(field => {
      val str = field.toString(); val parsedStr = str.slice(1, str.length - 1); parsedStr.toDouble
    })

    val paraArray = Array("city_name","temp","temp_min","temp_max","pressure","humidity","wind_speed","wind_deg",
      "rain_1h","rain_3h","snow_3h","clouds_all","weather_main")

    //Calculate correlation coefficient between electricity demand and weather for feature selection
    val corrDemand = new Array[(String,Double)](13)
    for (i <- 0 until paraArray.length){
      val para = paraArray(i)
      val paraDF = sqlCtx.sql(s"""SELECT $para FROM join """)
      val paraRDD = paraDF.rdd.map(field =>
      {val str = field.toString(); val parsedStr = str.slice(1,str.length-1); parsedStr.toDouble})
      val correlation_para = Statistics.corr(total_load_actualRDD, paraRDD, "pearson")
      corrDemand(i) = (para,correlation_para)
    }

    //Calculate correlation coefficient between electricity demand and price
    val priceDF = sqlCtx.sql("""SELECT price_actual FROM join""")
    val priceRDD = priceDF.rdd.map(field =>
    {val str = field.toString(); val parsedStr = str.slice(1,str.length-1); parsedStr.toDouble})
    val correlation_price = Statistics.corr(total_load_actualRDD,priceRDD,"pearson")

    //Only consider parameters which has correlation coefficient larger than 0.1
    //(temp,temp_min,temp_max,humidity,wind_speed)

    //Find Average electricity demand for each weather category and save table for visualization
    val averageDesDF = sqlCtx.sql("""SELECT weather_description,AVG(total_load_actual) FROM join GROUP BY weather_description """)
    averageDesDF.show()
    averageDesDF.repartition(1).write.format("csv").save("loudacre/table")

    // Create training and testing data
    val inputDF = sqlCtx.sql("""SELECT total_load_actual,temp,temp_min,temp_max,humidity,wind_speed FROM join""")
    val inputRDD = inputDF.rdd
    val parsedRDD = inputRDD.map(array => LabeledPoint(array(0).toString.toDouble, Vectors.dense(array(1).toString.toDouble,
      array(2).toString.toDouble, array(3).toString.toDouble, array(4).toString.toDouble, array(5).toString.toDouble))).cache()

    val splitWeight = Array(0.2,0.8)
    val seed = 1
    val splitRDDs = parsedRDD.randomSplit(splitWeight,seed)
    val train = splitRDDs(1)
    val test = splitRDDs(0)
    MLUtils.saveAsLibSVMFile(train,"loudacre/train")
    MLUtils.saveAsLibSVMFile(test,"loudacre/test")

    val inputTrain = sqlCtx.read.format("libsvm").load("loudacre/train/part*")
    val inputTest = sqlCtx.read.format("libsvm").load("loudacre/test/part*")

    val tempDF = sqlCtx.sql("""SELECT total_load_actual,temp FROM join""")
    val tempRDD = sc.makeRDD(tempDF.rdd.takeSample(false,200))
    val tempParsedRDD =tempRDD.map(array => LabeledPoint(array(0).toString.toDouble, Vectors.dense(array(1).toString.toDouble)))

    // Liner Regression with temp as input
    val numIterations = 10000
    val stepSize = 0.0001
    val lrTempModel = LinearRegressionWithSGD.train(tempParsedRDD, numIterations, stepSize)
    val valuesAndPreds = tempParsedRDD.map { point =>
      val prediction = lrTempModel.predict(point.features)
      (point.label, prediction)
    }
    val RMSE = sqrt(valuesAndPreds.map{ case(x, y) => math.pow((x - y), 2) }.mean())
    println(RMSE)

    // Save the input, label, and prediction for visualization
    val inputTempRDD = tempRDD.map(array => array(1).toString.toDouble)
    val zipRDD = inputTempRDD.zip(valuesAndPreds)
    val outputRDD = zipRDD.map(array => (array._1,array._2._1,array._2._2))
    val outputDF = outputRDD.toDF()
    outputDF.repartition(1).write.format("csv").save("loudacre/graph")

    //Linear Regression
    val lr = new LinearRegression()
    val lrModel = lr.fit(inputTrain)
    val lrResult = lrModel.transform(inputTest).cache()
    val lrPredictions = lrResult.select("prediction").rdd.map(_.getDouble(0))
    val lrLabels = lrResult.select("label").rdd.map(_.getDouble(0))
    val lrZip = lrPredictions.zip(lrLabels)

    val summary = lrModel.summary
    val lrRMSE = summary.rootMeanSquaredError

    //Decision-Tree Regression
    val dt = new DecisionTreeRegressor()
    val dtModel = dt.fit(inputTrain)
    val dtResult = dtModel.transform(inputTest).cache()
    val dtPredictions = dtResult.select("prediction").rdd.map(_.getDouble(0))
    val dtLabels = dtResult.select("label").rdd.map(_.getDouble(0))
    val dtZip = dtPredictions.zip(dtLabels)

    val dtEvaluator = new RegressionEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("rmse")
    val dtRMSE = dtEvaluator.evaluate(dtResult)

    //Random Forest Regression
    val rf = new RandomForestRegressor()
    val rfModel = rf.fit(inputTrain)
    val rfResult = rfModel.transform(inputTest).cache()
    val rfPredictions = rfResult.select("prediction").rdd.map(_.getDouble(0))
    val rfLabels = rfResult.select("label").rdd.map(_.getDouble(0))
    val rfZip = rfPredictions.zip(rfLabels)

    val rfEvaluator = new RegressionEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("rmse")
    val rfRMSE = rfEvaluator.evaluate(rfResult)

    //Gradient-Boost-Tree Regression
    val gb = new GBTRegressor()
    val gbModel = gb.fit(inputTrain)
    val gbResult = gbModel.transform(inputTest).cache()
    val gbPredictions = gbResult.select("prediction").rdd.map(_.getDouble(0))
    val gbLabels = gbResult.select("label").rdd.map(_.getDouble(0))
    val gbZip = gbPredictions.zip(gbLabels)

    val gbEvaluator = new RegressionEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("rmse")
    val gbRMSE = gbEvaluator.evaluate(gbResult)

    for(i <- corrDemand){
      println(i)
    }
    println(("price",correlation_price))
    println("Linear Regression RMSE " + lrRMSE)
    println("Decision-Tree RMSE: " + dtRMSE)
    println("Random Forest RMSE : " + rfRMSE)
    println("Gradient-Boost-Tree RMSE: " + gbRMSE)

  }
}