
import org.apache.spark.SparkConf;
import org.apache.spark.ml.feature.HashingTF;
import org.apache.spark.ml.feature.IDF;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.ml.classification.NaiveBayes;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.feature.Tokenizer;
import static org.apache.spark.sql.functions.col;
import org.apache.spark.sql.types.StructType;

public class Naive_Bayes {
	SparkConf cnf = new SparkConf().setMaster("local").setAppName("Naive_Bayes").set("spark.driver.allowMultipleContexts", "true");
	
	//SparkContext sc =new SparkContext(cnf);
	//JavaSparkContext jsc =  JavaSparkContext.fromSparkContext(sc);
	//SQLContext sqlCon = new SQLContext(ss);
	
	public void Naive_Bayes() {
		SparkSession ss = SparkSession.builder().appName("Naive_Bayes").master("local").getOrCreate(); 
        StructType schema = new StructType().add("text", "string").add("classe", "integer");
        Dataset<Row> data = ss.read().option("mode", "DROPMALFORMED").schema(schema).csv("C:\\Users\\ASUS\\eclipse-workspace\\search_engine\\data.csv");
        //Dataset<Row> data = sqlCon.createDataFrame(dt, LabeledPoint.class); 

		
		Dataset<Row>[] split = data.randomSplit(new double[] {0.7,0.3});
        Dataset<Row> train = split[0];
        Dataset<Row> test = split[1];

        Tokenizer tokenizer = new Tokenizer().setInputCol("text").setOutputCol("words");
		HashingTF hashingTF = new HashingTF().setNumFeatures(1000).setInputCol("words").setOutputCol("rawFeatures");
		IDF idf = new IDF().setInputCol("rawFeatures").setOutputCol("features").setMinDocFreq(0);

        NaiveBayes nb = new NaiveBayes().setLabelCol("classe").setFeaturesCol("features");
        
        Pipeline pipeline = new Pipeline().setStages(new PipelineStage[]
						  {tokenizer, hashingTF,idf, nb});
        
		PipelineModel plModel = pipeline.fit(train);
		
		Dataset<Row> predictions = plModel.transform(test);
		predictions.show(20);

		System.out.println("Confusion Matrix :");
		predictions.groupBy(col("classe"), col("prediction")).count().show();
		
		//Accuracy computation
		MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
				  .setLabelCol("classe")
				  .setPredictionCol("prediction")
				  .setMetricName("accuracy");
				double accuracy = evaluator.evaluate(predictions);
				System.out.println("Accuracy = " + Math.round( accuracy * 100) + " %" );
				
	}
	public static void main (String[] args) {
		Naive_Bayes m = new Naive_Bayes();
		m.Naive_Bayes();
	}
}
