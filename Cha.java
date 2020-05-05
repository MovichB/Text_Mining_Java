import org.apache.spark.SparkConf;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.clustering.BisectingKMeans;
import org.apache.spark.ml.clustering.BisectingKMeansModel;
import org.apache.spark.ml.clustering.KMeans;
import org.apache.spark.ml.evaluation.ClusteringEvaluator;
import org.apache.spark.ml.feature.HashingTF;
import org.apache.spark.ml.feature.IDF;
import org.apache.spark.ml.feature.Tokenizer;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.StructType;


public class Cha {
	SparkConf cnf = new SparkConf().setMaster("local").setAppName("KMeans").set("spark.driver.allowMultipleContexts", "true");
	
	public void Hierarchical() {
		SparkSession ss = SparkSession.builder().appName("KMeans").master("local").getOrCreate(); 
        StructType schema = new StructType().add("text", "string").add("classe", "integer");
        Dataset<Row> data = ss.read().option("mode", "DROPMALFORMED").schema(schema).csv("C:\\Users\\ASUS\\eclipse-workspace\\search_engine\\data.csv");
		
		Dataset<Row>[] split = data.randomSplit(new double[] {0.7,0.3});
        Dataset<Row> train = split[0];
        Dataset<Row> test = split[1];

        Tokenizer tokenizer = new Tokenizer().setInputCol("text").setOutputCol("words");
		HashingTF hashingTF = new HashingTF().setNumFeatures(1000).setInputCol("words").setOutputCol("rawFeatures");
		IDF idf = new IDF().setInputCol("rawFeatures").setOutputCol("features").setMinDocFreq(0);

		BisectingKMeans bkm = new BisectingKMeans().setK(2).setSeed(1);
        
        Pipeline pipeline = new Pipeline().setStages(new PipelineStage[]
				  {tokenizer, hashingTF,idf, bkm});
        
        PipelineModel plModel = pipeline.fit(train);
        Dataset<Row> predictions = plModel.transform(test);
		predictions.show(30);
		
        double cost = plModel.computeCost(test);
        System.out.println("Within Set Sum of Squared Errors = " + cost);

        // Shows the result.
        System.out.println("Cluster Centers: ");
        Vector[] centers = plModel.clusterCenters();
        for (Vector center : centers) {
          System.out.println(center);
        }
 
	}
	public static void main (String[] args) {
		Cha m = new Cha();
		m.Hierarchical();
	}
}
