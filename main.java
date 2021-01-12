import weka.classifiers.*;
import weka.classifiers.bayes.*;
import weka.classifiers.rules.*;
import weka.classifiers.trees.*;
import weka.core.*;
import weka.core.converters.ConverterUtils.DataSource;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;
import java.util.Random;

public class main {
	public static void main(String[] args) throws Exception {
		//load train data.
		DataSource source = new DataSource("C:/Users/user/Desktop/testCase_arff.arff");
		Instances trainSet = source.getDataSet();
		trainSet.setClassIndex(trainSet.numAttributes() - 1);
		
		//Model Instance.
		J48 tree = new J48();
		OneR oneR = new OneR();
		NaiveBayes nb = new NaiveBayes();
		
		//Set Attributes
		List<String> attr1 = new ArrayList<String>(5);//운전경력.
		attr1.add("Under 5");
		attr1.add("5~9");
		attr1.add("10~14");
		attr1.add("Over 15");
		attr1.add("Unknown");
		Attribute a1 = new Attribute("Career", attr1);
		
		List<String> attr2 = new ArrayList<String>(2);//주야.
		attr2.add("D");
		attr2.add("N");
		Attribute a2 = new Attribute("D/N", attr2);
		
		List<String> attr3 = new ArrayList<String>(3);//운전자 성별.
		attr3.add("M");
		attr2.add("W");
		attr3.add("Unknown");
		Attribute a3 = new Attribute("Sex", attr3);
		
		List<String> attr4 = new ArrayList<String>(5);//차종.
		attr4.add("Motorcycle");
		attr4.add("Truck");
		attr4.add("Car");
		attr4.add("Van");
		attr4.add("Unknown");
		Attribute a4 = new Attribute("Vehicle_type", attr4);
		
		List<String> attr5 = new ArrayList<String>(4);//도로형태.
		attr5.add("etc");
		attr5.add("Intersection");
		attr5.add("Single_route");
		attr5.add("Tunnel");
		Attribute a5 = new Attribute("Road_type", attr5);
		
		List<String> cls = new ArrayList<String>(3);//클래스 정의.
		cls.add("Injured");
		cls.add("Seriously_injured");
		cls.add("Killed");
		Attribute a6 = new Attribute("State", cls);
		
		ArrayList<Attribute> instanceAttributes = new ArrayList<Attribute>(5);
		instanceAttributes.add(a1);
		instanceAttributes.add(a2);
		instanceAttributes.add(a3);
		instanceAttributes.add(a4);
		instanceAttributes.add(a5);
		instanceAttributes.add(a6);
		
		//Training.
		tree.buildClassifier(trainSet);
		oneR.buildClassifier(trainSet);
		nb.buildClassifier(trainSet);
		System.out.println(oneR.toString());
		System.out.println(nb.toString());
		System.out.println(tree.toString());
		
		//Evaluation.
				Evaluation eval = new Evaluation(trainSet);
				eval.evaluateModel(oneR,  trainSet);
				System.out.println("OneR_accuracy: "+ eval.toSummaryString());
				//eval.evaluateModel(nb,  trainSet);
				//System.out.println("Naive_bayes_accuracy: " + eval.toSummaryString());
				eval.evaluateModel(tree,  trainSet);
				System.out.println("Decision_tree_accuracy: " + eval.toSummaryString());
		//Input.
		Instances testSet = new Instances("testSet", instanceAttributes, 0);
		testSet.setClassIndex(testSet.numAttributes() - 1);
		Scanner scan = new Scanner(System.in);
		double atr1, atr2, atr3, atr4, atr5;
		while(true) {
			String str;
			System.out.print("Career: ");
			str = scan.nextLine();
			if(str.equals("-1")) break;
			else if(str.equals("Under 5")) atr1 = 0.0;
			else if(str.equals("5~9")) atr1 = 1.0;
			else if(str.equals("10~14")) atr1 = 2.0;
			else if(str.equals("Over 15")) atr1 = 3.0;
			else if(str.equals("Unknown")) atr1 = 4.0;
			else {
				System.out.println("잘못된 입력입니다. 다시 입력하세요.");
				continue;
			}
			System.out.print("D/N: ");
			str = scan.nextLine();
			if(str.equals("D")) atr2 = 0.0;
			else if(str.equals("N")) atr2 = 1.0;
			else {
				System.out.println("잘못된 입력입니다. 다시 입력하세요.");
				continue;
			}
			System.out.print("Sex: ");
			str = scan.nextLine();
			if(str.equals("M")) atr3 = 0.0;
			else if(str.equals("W")) atr3 = 1.0;
			else if(str.equals("Unknown")) atr3 = 2.0;
			else {
				System.out.println("잘못된 입력입니다. 다시 입력하세요.");
				continue;
			}
			System.out.print("Vehicle_type: ");
			str = scan.nextLine();
			if(str.equals("Motorcycle")) atr4 = 0.0;
			else if(str.equals("Truck")) atr4 = 1.0;
			else if(str.equals("Car")) atr4 = 2.0;
			else if(str.equals("Van")) atr4 = 3.0;
			else if(str.equals("Unknown")) atr4 = 4.0;
			else {
				System.out.println("잘못된 입력입니다. 다시 입력하세요.");
				continue;
			}
			System.out.print("Road_type: ");
			str = scan.nextLine();
			if(str.equals("etc")) atr5 = 0.0;
			else if(str.equals("Intersection")) atr5 = 1.0;
			else if(str.equals("Single_route")) atr5 = 2.0;
			else if(str.equals("Tunnel")) atr5 = 3.0;
			else {
				System.out.println("잘못된 입력입니다. 다시 입력하세요.");
				continue;
			}
			double[] testData = new double[] {
					atr1, atr2, atr3, atr4, atr5
			};
			//Test.
			Instance testInstance = new DenseInstance(1.0, testData);
			testSet.add(testInstance);
			double result = oneR.classifyInstance(testSet.instance(testSet.size()-1));
			if(result == 0.0) System.out.println("OneR_output: Injured");
			else if(result == 1.0) System.out.println("OneR_output: Seriously_injured");
			else System.out.println("OneR_output: Killed");
			//result = nb.classifyInstance(testSet.instance(testSet.size()-1));
			//if(result == 0.0) System.out.println("Naive_bayes_output: Injured");
			//else if(result == 1.0) System.out.println("Naive_bayes_output: Seriously_injured");
			//else System.out.println("Naive_bayes_output: Killed");
			result = tree.classifyInstance(testSet.instance(testSet.size()-1));
			if(result == 0.0) System.out.println("Decision_tree_output: Injured");
			else if(result == 1.0) System.out.println("Decision_tree_output: Seriously_injured");
			else System.out.println("Decision_tree_output: Killed");
		}
	}

}
