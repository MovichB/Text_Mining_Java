package search_engine;
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.Map.Entry;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.apache.lucene.analysis.PorterStemmer;

public class txt2hashmap{
	public File f;
	static Scanner sc;
	public static List<String> files = new ArrayList<String>();
    public static HashMap<List<String>,String> docs = new HashMap<List<String>,String>();
    
    public static String stemTerm (String term) {
    	PorterStemmer stemmer = new PorterStemmer();
        return stemmer.stem(term);
    }
    
	static String path = "F:\\M2 BDSaS\\S3\\Image, Multimedia & Web Mining\\NLP\\Text Mining\\20_newsgroups";
	private static BufferedWriter bufferedWriter;
	public static void extract_words() throws IOException,UnsupportedOperationException  {
    	try (Stream<Path> walk = Files.walk(Paths.get(path))) {
    		List<String> result = walk.filter(Files::isDirectory)
    				.map(x -> x.toString()).collect(Collectors.toList());
    		for (int i = 0; i < result.size(); i++) {
    			String d = result.get(i);
    			files.add(d);
    		}
    		List<String> stopwords = Files.readAllLines(Paths.get("C:\\Users\\ASUS\\eclipse-workspace\\search_engine\\stopwords.txt"));
    	    Iterator<String> it = files.iterator();
    	    List<String> newList = new LinkedList<String>();
    	    newList.addAll(stopwords);
    		while(it.hasNext()) {
    		try (Stream<Path> wlk = Files.walk(Paths.get(it.next()))) {
	    		List<String> res = wlk.filter(Files::isRegularFile)
	    				.map(x -> x.toString()).collect(Collectors.toList());
	    		for (int j = 0; j < res.size(); j++) {
	    			String f = res.get(j);
	    			String scan = new String(Files.readAllBytes(Paths.get(f)), StandardCharsets.UTF_8);
	    			
	    			String lines = stemTerm(scan.replaceAll("\\r|\\n", " ").replace(".", " ")
	    					.replaceAll("[0-9]", "").replaceAll("\\.", "").replaceAll("\\t", "").replaceAll("\\؛", "").replaceAll("\\:","")
	                        .replaceAll("\\n", "").replaceAll("\\،", "").replaceAll("\\-", "").replaceAll("<", "").replaceAll("]", "")
	                        .replaceAll("\\(", "").replaceAll("\\)", "").replaceAll("_", ""));
	    			String[] line = lines.split(" ");
	    			
	    			for (int i = 0; i < line.length; i++) {
	    				for (int k = 0; k < newList.size(); k++) {
		    				if (newList.get(k).contains(line.toString())) {
		    					line[i] = null;
		    			        break;
		                    }
	    			}}
	    			List<String> list = Arrays.asList(line);
	    			//list.removeAll(Arrays.asList("", null));
	    			docs.put(list,f.replace(path, "").replaceAll("[0-9]", "").replace("\\", ""));}
	    			}
	    		}
			}
	    	 catch (IOException e) {
	    		e.printStackTrace();
	    	 	}
    	}
	
	public static void main(String[] args) throws NullPointerException, IOException {
    	 extract_words();
    	 FileWriter fstream_train;
    	 fstream_train = new FileWriter("data.txt");
    	 bufferedWriter = new BufferedWriter(fstream_train); 
    	 for(Entry<List<String>, String> entry : docs.entrySet()){
    		 bufferedWriter.write(entry.getKey()+";;"+entry.getValue() + "\n"); 
         }
	}
}