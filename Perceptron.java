import java.io.*;
import java.util.Scanner;
import java.util.Map;
import java.util.Iterator;
import java.util.LinkedHashMap;

public class Perceptron{
	public static String trainFilePath = "spam_train.txt";
	public static String testFilePath = "spam_test.txt";
	
	public static LinkedHashMap<String, Integer> wordcounts;
	public static LinkedHashMap<String, Integer> wordMapTemplate;//zero'ed out wordcounts
	
	public static void main(String[] args) throws IOException{
		FileReader fileIn = new FileReader(trainFilePath);
		BufferedReader in = new BufferedReader(fileIn);
		
		Email[] trainingSet = new Email[4000];
		Email[] validateSet = new Email[1000];
		
		//filling training set
		String line = in.readLine();
		for(int i=0;i<4000 && line != null;i++){
			trainingSet[i] = new Email(line.substring(0,1).equals("1"),line.substring(2));
			line = in.readLine();
		}
		
		//filling validation set
		for(int i=0;i<1000 && line != null;i++){
			validateSet[i] =new Email(line.substring(0,1).equals("1"),line.substring(2));
			line = in.readLine();
		}
		
		//finding occurences of words in all emails (in the training set)
		wordcounts = new LinkedHashMap<String, Integer>();
		for(int i=0;i<trainingSet.length;i++){
			LinkedHashMap<String, Boolean> wordsExist = new LinkedHashMap<String,Boolean>();
			
			String content = trainingSet[i].content;
			Scanner scan = new Scanner(content);
			String word;
			
			while(scan.hasNext()){
				word = scan.next();
				
				Integer occurences = wordcounts.get(word);
				if(occurences == null){
					wordcounts.put(word,1);
				} else if(wordsExist.get(word) == null){
					wordcounts.put(word,occurences+1);
				}
				
				wordsExist.put(word,true);
			}
		}
		
		
		//deleting all entries with < 30 occurences and filling wordMapTemplate
		wordMapTemplate = new LinkedHashMap<String, Integer>();
		Iterator it = wordcounts.entrySet().iterator();
		while(it.hasNext()){
			Map.Entry pairs = (Map.Entry) it.next();
			if((Integer) pairs.getValue() < 30){
				it.remove();
			} else {
				wordMapTemplate.put((String) pairs.getKey(),0);
			}
		}
		
		
		//creating feature vectors for each email in training set
		for(int i=0;i<trainingSet.length;i++){
			trainingSet[i].setFeatures(wordcounts);
		}
		
		//creating feature vectors for each email in validation set
		for(int i=0;i<validateSet.length;i++){
			validateSet[i].setFeatures(wordcounts);
		}
		
		//set the weight vector to all 0's
		
		System.out.println("Running Perceptron Training Algorithm...");
		LinkedHashMap<String,Integer> w1 = perceptron_train(trainingSet);
		System.out.println();
		
		System.out.println("Running Perceptron Testing Algorithm on Training Set...");
		perceptron_test(trainingSet,w1);
		System.out.println();
		
		System.out.println("Running Perceptron Testing Algorithm on Validation Set...");
		perceptron_test(validateSet,w1);
		System.out.println();
		
		System.out.println("Top 15 Words with the most Positive Weights:");
		LinkedHashMap<String, Integer> top15positive = new LinkedHashMap<String, Integer>();
		for(int i=0;i<15;i++){
			int maxVal = 0;
			String maxKey = "";
			it = w1.entrySet().iterator();
			while(it.hasNext()){
				Map.Entry pairs = (Map.Entry) it.next();
				int val = (Integer) pairs.getValue();
				String key = (String) pairs.getKey();
				if(val > maxVal && top15positive.get(key) == null){
					maxVal = val;
					maxKey = key;
				}
			}
			
			top15positive.put(maxKey,maxVal);
		}
		
		it = top15positive.entrySet().iterator();
		while(it.hasNext()){
			Map.Entry pairs = (Map.Entry) it.next();
			System.out.println(pairs.getKey()+": "+pairs.getValue());
		}
		System.out.println();
		
		System.out.println("Top 15 Words with the most Negative Weights:");
		LinkedHashMap<String, Integer> top15negative = new LinkedHashMap<String, Integer>();
		for(int i=0;i<15;i++){
			int minVal = 0;
			String minKey = "";
			it = w1.entrySet().iterator();
			while(it.hasNext()){
				Map.Entry pairs = (Map.Entry) it.next();
				int val = (Integer) pairs.getValue();
				String key = (String) pairs.getKey();
				if(val < minVal && top15negative.get(key) == null){
					minVal = val;
					minKey = key;
				}
			}
			
			top15negative.put(minKey,minVal);
		}
		
		it = top15negative.entrySet().iterator();
		while(it.hasNext()){
			Map.Entry pairs = (Map.Entry) it.next();
			System.out.println(pairs.getKey()+": "+pairs.getValue());
		}
		
		System.out.println();
		
		System.out.println("Running Averaged Perceptron Algorithm...");
		LinkedHashMap<String,Integer> w2 = perceptron_averaged_train(trainingSet);
		System.out.println();
		
		System.out.println("Running Averaged Perceptron Testing Algorithm on Training Set...");
		perceptron_test(trainingSet,w2);
		System.out.println();
		
		System.out.println("Running Averaged Perceptron Testing Algorithm on Validation Set...");
		perceptron_test(validateSet,w2);
		System.out.println();
		
		//creating smaller data sets
		Email[] trainingSet100 = new Email[100];
		Email[] trainingSet200 = new Email[200];
		Email[] trainingSet400 = new Email[400];
		Email[] trainingSet800 = new Email[800];
		Email[] trainingSet2000 = new Email[2000];
		Email[] trainingSet4000 = new Email[4000];
		for(int i=0;i<trainingSet.length;i++){
			trainingSet4000[i] = trainingSet[i];
			if(i < 2000) trainingSet2000[i] = trainingSet[i];
			if(i < 800) trainingSet800[i] = trainingSet[i];
			if(i < 400) trainingSet400[i] = trainingSet[i];
			if(i < 200) trainingSet200[i] = trainingSet[i];
			if(i < 100) trainingSet100[i] = trainingSet[i];
		}
		
		System.out.println("Running Perception Algorithm on Different Amounts of Training Data");
		System.out.print("N=100: ");
		LinkedHashMap<String,Integer> w100 = perceptron_train(trainingSet100);
		System.out.print("N=200: ");
		LinkedHashMap<String,Integer> w200 = perceptron_train(trainingSet200);
		System.out.print("N=400: ");
		LinkedHashMap<String,Integer> w400 = perceptron_train(trainingSet400);
		System.out.print("N=800: ");
		LinkedHashMap<String,Integer> w800 = perceptron_train(trainingSet800);
		System.out.print("N=2000: ");
		LinkedHashMap<String,Integer> w2000 = perceptron_train(trainingSet2000);
		System.out.print("N=4000: ");
		LinkedHashMap<String,Integer> w4000 = perceptron_train(trainingSet4000);
		System.out.println();
		
		System.out.println("Running Perceptron Testing Algorithms on Validation Set...");
		System.out.print("N=100: ");
		perceptron_test(validateSet,w100);
		System.out.print("N=200: ");
		perceptron_test(validateSet,w200);
		System.out.print("N=400: ");
		perceptron_test(validateSet,w400);
		System.out.print("N=800: ");
		perceptron_test(validateSet,w800);
		System.out.print("N=2000: ");
		perceptron_test(validateSet,w2000);
		System.out.print("N=4000: ");
		perceptron_test(validateSet,w4000);
		System.out.println();
		
		System.out.println("Running Averaged Perception Algorithm on Different Amounts of Training Data");
		System.out.print("N=100: ");
		LinkedHashMap<String,Integer> aw100 = perceptron_averaged_train(trainingSet100);
		System.out.print("N=200: ");
		LinkedHashMap<String,Integer> aw200 = perceptron_averaged_train(trainingSet200);
		System.out.print("N=400: ");
		LinkedHashMap<String,Integer> aw400 = perceptron_averaged_train(trainingSet400);
		System.out.print("N=800: ");
		LinkedHashMap<String,Integer> aw800 = perceptron_averaged_train(trainingSet800);
		System.out.print("N=2000: ");
		LinkedHashMap<String,Integer> aw2000 = perceptron_averaged_train(trainingSet2000);
		System.out.print("N=4000: ");
		LinkedHashMap<String,Integer> aw4000 = perceptron_averaged_train(trainingSet4000);
		System.out.println();
		
		System.out.println("Running Averaged Perceptron Testing Algorithms on Validation Set...");
		System.out.print("N=100: ");
		perceptron_test(validateSet,aw100);
		System.out.print("N=200: ");
		perceptron_test(validateSet,aw200);
		System.out.print("N=400: ");
		perceptron_test(validateSet,aw400);
		System.out.print("N=800: ");
		perceptron_test(validateSet,aw800);
		System.out.print("N=2000: ");
		perceptron_test(validateSet,aw2000);
		System.out.print("N=4000: ");
		perceptron_test(validateSet,aw4000);
		System.out.println();
		
		System.out.println("Running Customized Perceptron Algorithms on Training Data");
		LinkedHashMap<String,Integer> cw1 = perceptron_custom_train(trainingSet,4,true);
		LinkedHashMap<String,Integer> cw2 = perceptron_custom_train(trainingSet,5,true);
		LinkedHashMap<String,Integer> cw3 = perceptron_custom_train(trainingSet,6,true);
		LinkedHashMap<String,Integer> cw4 = perceptron_custom_train(trainingSet,4,false);
		LinkedHashMap<String,Integer> cw5 = perceptron_custom_train(trainingSet,5,false);
		LinkedHashMap<String,Integer> cw6 = perceptron_custom_train(trainingSet,6,false);
		System.out.println();
		
		System.out.println("Testing Customized Perceptron Algorithms on Validation Set");
		System.out.print("Iteration Limit = 4, Averaged: ");
		perceptron_test(validateSet,cw1);
		System.out.print("Iteration Limit = 5, Averaged: ");
		perceptron_test(validateSet,cw2);
		System.out.print("Iteration Limit = 6, Averaged: ");
		perceptron_test(validateSet,cw3);
		System.out.print("Iteration Limit = 4, Not Averaged: ");
		perceptron_test(validateSet,cw4);
		System.out.print("Iteration Limit = 5, Not Averaged: ");
		perceptron_test(validateSet,cw5);
		System.out.print("Iteration Limit = 6, Not Averaged: ");
		perceptron_test(validateSet,cw6);
		System.out.println();
		
		//setting up test set;
		Email[] testSet = new Email[1000];
		
		fileIn = new FileReader(testFilePath);
		in = new BufferedReader(fileIn);
		
		line = in.readLine();
		for(int i=0;i<testSet.length && line != null;i++){
			testSet[i] = new Email(line.substring(0,1).equals("1"),line.substring(2));
			line = in.readLine();
		}
		for(int i=0;i<testSet.length;i++){
			testSet[i].setFeatures(wordcounts);
		}
		
		//comparing results from different types of perceptron algorithms
		
		System.out.println("Testing Regular Perceptron Algorithm on Test Set");
		perceptron_test(testSet,w1);
		System.out.println();
		
		System.out.println("Testing Averaged Perceptron Algorithm on Test Set");
		perceptron_test(testSet,w2);
		System.out.println();

		System.out.println("Testing Best Customized Perceptron Algorithm on Test Set");
		perceptron_test(testSet,cw1);
		System.out.println();



	}
	
	
	
	
	
	
	public static LinkedHashMap<String,Integer> perceptron_train(Email[] data){
		//setting up weight vector
		LinkedHashMap<String,Integer> weight = new LinkedHashMap<String,Integer>(wordMapTemplate);
		
		
		int iter = 0;
		int mistakes_total = 0;
		int mistakes = 0;
		int mistakes_prev = 0;
		while(true){
			mistakes = 0;
			iter ++;
			
			for(int i=0;i<data.length;i++){
				
				//make a guess and see if it's right
				int guess;
				if(dotProduct(weight,data[i].features) >= 0){
					guess = 1;
				} else {
					guess = -1;
				}
				//if wrong, update the weight vector
				if(data[i].isSpam && guess == -1 || !data[i].isSpam && guess == 1){
					mistakes ++;
					
					Iterator it = weight.entrySet().iterator();
					while(it.hasNext()){
						Map.Entry pairs = (Map.Entry) it.next();
						String key = (String) pairs.getKey();
						int yi = data[i].isSpam ? 1 : -1;
						//w = w + yi * f(xi)
						weight.put(key,weight.get(key)+yi*data[i].features.get(key));
					}
				}
			}
			
			//System.out.println("Iteration Complete. Number of updates this iteration: "+mistakes);
			mistakes_total += mistakes;
			//if no mistakes are made, stop
			if(mistakes == 0) {
				break;
			}
			
			mistakes_prev = mistakes;
		}
		System.out.println(iter+" total iterations. "+mistakes_total+" total updates.");
		return weight;
	}
	
	public static LinkedHashMap<String,Integer> perceptron_averaged_train(Email[] data){
		//setting up weight vector
		LinkedHashMap<String,Integer> weight = new LinkedHashMap<String,Integer>(wordMapTemplate);
		LinkedHashMap<String,Integer> averagedWeight = new LinkedHashMap<String, Integer>(wordMapTemplate);
		
		
		int iter = 0;
		int steps = 0;
		int mistakes_total = 0;
		int mistakes = 0;
		int mistakes_prev = 0;
		while(true){
			mistakes = 0;
			iter ++;
			
			for(int i=0;i<data.length;i++){
				steps ++;
				//make a guess and see if it's right
				int guess;
				if(dotProduct(weight,data[i].features) >= 0){
					guess = 1;
				} else {
					guess = -1;
				}
				//if wrong, update the weight vector
				if(data[i].isSpam && guess == -1 || !data[i].isSpam && guess == 1){
					mistakes ++;
					
					Iterator it = weight.entrySet().iterator();
					while(it.hasNext()){
						Map.Entry pairs = (Map.Entry) it.next();
						String key = (String) pairs.getKey();
						int yi = data[i].isSpam ? 1 : -1;
						//w = w + yi * f(xi)
						weight.put(key,weight.get(key)+yi*data[i].features.get(key));
					}
				}
				
				Iterator it = weight.entrySet().iterator();
				while(it.hasNext()){
					Map.Entry pairs = (Map.Entry) it.next();
					String key = (String) pairs.getKey();
					int yi = data[i].isSpam ? 1 : -1;
					//w = w + yi * f(xi)
					averagedWeight.put(key,averagedWeight.get(key)+weight.get(key));
				}

			}
			
			//System.out.println("Iteration Complete. Number of updates this iteration: "+mistakes);
			mistakes_total += mistakes;
			//if no mistakes are made or algorithm runs too long, stop
			if(mistakes == 0 || iter > 100) {
				break;
			}
			
			mistakes_prev = mistakes;
		}
		
		//calculate average weights
		Iterator it = averagedWeight.entrySet().iterator();
		while(it.hasNext()){
			Map.Entry pairs = (Map.Entry) it.next();
			String key = (String) pairs.getKey();
			int val = (Integer) pairs.getValue();
			
			averagedWeight.put(key,val/steps);
		}
		
		System.out.println(iter+" total iterations. "+mistakes_total+" total updates.");
		return averagedWeight;

	}
	
	public static LinkedHashMap<String,Integer> perceptron_custom_train(Email[] data,int iterLimit, boolean averaged){
		//setting up weight vector
		LinkedHashMap<String,Integer> weight = new LinkedHashMap<String,Integer>(wordMapTemplate);
		LinkedHashMap<String,Integer> averagedWeight = new LinkedHashMap<String, Integer>(wordMapTemplate);
		
		
		int iter = 0;
		int mistakes_total = 0;
		int mistakes = 0;
		int mistakes_prev = 0;
		while(true){
			mistakes = 0;
			iter ++;
			
			for(int i=0;i<data.length;i++){
				
				//make a guess and see if it's right
				int guess;
				if(dotProduct(weight,data[i].features) >= 0){
					guess = 1;
				} else {
					guess = -1;
				}
				//if wrong, update the weight vector
				if(data[i].isSpam && guess == -1 || !data[i].isSpam && guess == 1){
					mistakes ++;
					
					Iterator it = weight.entrySet().iterator();
					while(it.hasNext()){
						Map.Entry pairs = (Map.Entry) it.next();
						String key = (String) pairs.getKey();
						int yi = data[i].isSpam ? 1 : -1;
						//w = w + yi * f(xi)
						weight.put(key,weight.get(key)+yi*data[i].features.get(key));
						averagedWeight.put(key,averagedWeight.get(key)+weight.get(key));
					}
				}
			}
			
			//System.out.println("Iteration Complete. Number of updates this iteration: "+mistakes);
			mistakes_total += mistakes;
			//if no mistakes are made or algorithm runs too long, stop
			if(mistakes == 0 || iter >= iterLimit) {
				break;
			}
			
			mistakes_prev = mistakes;
		}
		
		//calculate average weights
		Iterator it = averagedWeight.entrySet().iterator();
		while(it.hasNext()){
			Map.Entry pairs = (Map.Entry) it.next();
			String key = (String) pairs.getKey();
			int val = (Integer) pairs.getValue();
			
			averagedWeight.put(key,val/mistakes_total);
		}
		
		System.out.println(iter+" total iterations. "+mistakes_total+" total updates.");
		return averaged ? averagedWeight : weight;
		
	}
	
	public static void perceptron_test(Email[] data, LinkedHashMap<String, Integer> weight){
		int testSize = data.length;
		int numMisclassified = 0;
		
		for(int i=0;i<testSize;i++){
			//make a guess and see if it's right
			int guess;
			if(dotProduct(weight,data[i].features) >= 0){
				guess = 1;
			} else {
				guess = -1;
			}
			
			//if wrong, record the error
			if(data[i].isSpam && guess == -1 || !data[i].isSpam && guess == 1){
				numMisclassified ++;
			}
		}
		
		System.out.println(numMisclassified+"/"+testSize+" emails misclassified, "+((double) numMisclassified)/testSize*100+"% test error");
	}
	
	//returns the dot product of two string hashmaps
	public static int dotProduct(LinkedHashMap<String,Integer> v1, LinkedHashMap<String,Integer> v2){
		int ret = 0;
		if(v1.size() != v2.size()){
			System.out.println("Warning! Hashmap sizes are different! v1 size: "+v1.size()+", v2 size: "+v2.size());
		}
		Iterator it = v1.entrySet().iterator();
		while(it.hasNext()){
			String key = (String) ((Map.Entry) it.next()).getKey();
			
			ret += v1.get(key)*v2.get(key);
		}
		
		return ret;
	}
}

class Email {
	public boolean isSpam;
	public String content;
	public LinkedHashMap<String,Integer> features;
	
	public Email(boolean iS, String c){
		isSpam = iS;
		content = c;
	}
	
	public void setFeatures(LinkedHashMap<String, Integer> words){
		//copying the hashmap and setting all values to 0
		features = new LinkedHashMap<String,Integer>(words);
		
		
		Iterator it = features.entrySet().iterator();
		while(it.hasNext()){
			Map.Entry pairs = (Map.Entry) it.next();
			features.put((String) pairs.getKey(),0);
		}

		//filling out feature vector
		Scanner scan = new Scanner(content);
		String word;
		
		while(scan.hasNext()){
			word = scan.next();
			if(features.get(word) != null) features.put(word,1);
		}
	}
}