package model;

/**
 * Description: CoForest is a semi-supervised algorithm, which exploits the power of ensemble learning and available
 *              large amount of unlabeled data to produce hypothesis with better performance.
 *
 * Reference:   M. Li, Z.-H. Zhou. Improve computer-aided diagnosis with machine learning techniques using undiagnosed
 *              samples. IEEE Transactions on Systems, Man and Cybernetics - Part A: Systems and Humans, 2007, 37(6).
 *
 * ATTN:        This package is free for academic usage. You can run it at your own risk.
 *	     	For other purposes, please contact Prof. Zhi-Hua Zhou (zhouzh@nju.edu.cn).
 *
 * Requirement: To use this package, the whole WEKA environment (ver 3.4) must be available.
 *	        refer: I.H. Witten and E. Frank. Data Mining: Practical Machine Learning
 *		Tools and Techniques with Java Implementations. Morgan Kaufmann,
 *		San Francisco, CA, 2000.
 *
 * Data format: Both the input and output formats are the same as those used by WEKA.
 *
 * ATTN2:       This package was developed by Mr. Ming Li (lim@lamda.nju.edu.cn). There
 *		is a ReadMe file provided for roughly explaining the codes. But for any
 *		problem concerning the code, please feel free to contact with Mr. Li.
 *
 */


import java.io.*;

import java.text.*;
import java.util.*;
import java.util.Date;

import weka.core.*;
import weka.classifiers.*;
import weka.classifiers.trees.*;

import weka.core.converters.ArffSaver;


import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.GreedyStepwise;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;

public class CoForest
{
  /** Random Forest */
  protected Classifier[] m_classifiers = null;

  /** The number component */
  protected int m_numClassifiers = 10;

  /** The random seed */
  protected int m_seed = 1;

  /** Number of features to consider in random feature selection.
      If less than 1 will use int(logM+1) ) */
  protected int m_numFeatures = 0;

  /** Final number of features that were considered in last build. */
  protected int m_KValue = 0;

  /** confidence threshold */
  protected double m_threshold = 0.75;

  private int m_numOriginalLabeledInsts = 0;



  /**
   * The constructor
   */
  public CoForest()
  {
  }


  /**
   * Set the seed for initiating the random object used inside this class
   *
   * @param s int -- The seed
   */
  public void setSeed(int s)
  {
    m_seed = s;
  }

  /**
   * Set the number of trees used in Random Forest.
   *
   * @param s int -- Value to assign to numTrees.
   */
  public void setNumClassifiers(int n)
  {
    m_numClassifiers = n;
  }

  /**
   * Get the number of trees used in Random Forest
   *
   * @return int -- The number of trees.
   */
  public int getNumClassifiers()
  {
    return m_numClassifiers;
  }

  /**
   * Set the number of features to use in random selection.
   *
   * @param n int -- Value to assign to m_numFeatures.
   */
  public void setNumFeatures(int n)
  {
    m_numFeatures = n;
  }

  /**
   * Get the number of featrues to use in random selection.
   *
   * @return int -- The number of features
   */
  public int getNumFeatures()
  {
    return m_numFeatures;
  }

  /**
   * Resample instances w.r.t the weight
   *
   * @param data Instances -- the original data set
   * @param random Random -- the random object
   * @param sampled boolean[] -- the output parameter, indicating whether the instance is sampled
   * @return Instances
   */
  public final Instances resampleWithWeights(Instances data,
                                             Random random,
                                             boolean[] sampled)
  {

    double[] weights = new double[data.numInstances()];
    for (int i = 0; i < weights.length; i++) {
      weights[i] = data.instance(i).weight();
    }
    Instances newData = new Instances(data, data.numInstances());
    if (data.numInstances() == 0) {
      return newData;
    }
    double[] probabilities = new double[data.numInstances()];
    double sumProbs = 0, sumOfWeights = Utils.sum(weights);
    for (int i = 0; i < data.numInstances(); i++) {
      sumProbs += random.nextDouble();
      probabilities[i] = sumProbs;
    }
    Utils.normalize(probabilities, sumProbs / sumOfWeights);

    // Make sure that rounding errors don't mess things up
    probabilities[data.numInstances() - 1] = sumOfWeights;
    int k = 0; int l = 0;
    sumProbs = 0;
    while ((k < data.numInstances() && (l < data.numInstances()))) {
      if (weights[l] < 0) {
        throw new IllegalArgumentException("Weights have to be positive.");
      }
      sumProbs += weights[l];
      while ((k < data.numInstances()) &&
             (probabilities[k] <= sumProbs)) {
        newData.add(data.instance(l));
        sampled[l] = true;
        newData.instance(k).setWeight(1);
        k++;
      }
      l++;
    }
    return newData;
  }

  /**
   * Returns the probability label of a given instance
   *
   * @param inst Instance -- The instance
   * @return double[] -- The probability label
   * @throws Exception -- Some exception
   */
  public double[] distributionForInstance(Instance inst) throws Exception
  {
    double[] res = new double[inst.numClasses()];
    for(int i = 0; i < m_classifiers.length; i++)
    {
      double[] distr = m_classifiers[i].distributionForInstance(inst);
      for(int j = 0; j < res.length; j++)
        res[j] += distr[j];
    }
    Utils.normalize(res);
    return res;
  }

  /**
   * Classifies a given instance
   *
   * @param inst Instance -- The instance
   * @return double -- The class value
   * @throws Exception -- Some Exception
   */
  public double classifyInstance(Instance inst) throws Exception
  {
    double[] distr = distributionForInstance(inst);
    return Utils.maxIndex(distr);
  }
  
  public void evaluate(int it, Instances test) throws Exception
  {
	     double TP=0,FP=0,FN=0,TN=0;
	     for(int i = 0; i < test.numInstances(); i++)
	     {
	       if(this.classifyInstance(test.instance(i)) == test.instance(i).classValue())
	       {
	    	   if(test.instance(i).classValue() == 0) TP++;
	    	   else TN++;
	       }
	       else {
	    	   if(test.instance(i).classValue() == 0) FN++;
	    	   else FP++;
	       }
	         //err++;
	     }
	     System.out.println("第" + it + "轮准确率 = " + ((TP+TN)/(TP+TN+FP+FN)));
  }

  /**
   * Build the classifiers using Co-Forest algorithm
   *
   * @param labeled Instances -- Labeled training set
   * @param unlabeled Instances -- unlabeled training set
   * @throws Exception -- certain exception
   */
  public void buildClassifier(Instances labeled, Instances unlabeled, Instances test) throws Exception
  {
    double[] err = new double[m_numClassifiers];
    double[] err_prime = new double[m_numClassifiers];
    double[] s_prime = new double[m_numClassifiers];

    boolean[][] inbags = new boolean[m_numClassifiers][];

    Random rand = new Random(m_seed);
    m_numOriginalLabeledInsts = labeled.numInstances();

    RandomTree rTree = new RandomTree();

    // set up the random tree options
    m_KValue = m_numFeatures;
    if (m_KValue < 1) m_KValue = (int) Utils.log2(labeled.numAttributes())+1;
    rTree.setKValue(m_KValue);
    
    //rTree.setMaxDepth(3);
    rTree.setNumFolds(5);

    m_classifiers = rTree.makeCopies(rTree, m_numClassifiers);
    Instances[] labeleds = new Instances[m_numClassifiers];
    int[] randSeeds = new int[m_numClassifiers];

    for(int i = 0; i < m_numClassifiers; i++)
    {
      randSeeds[i] = rand.nextInt();
      ((RandomTree)m_classifiers[i]).setSeed(randSeeds[i]);
      inbags[i] = new boolean[labeled.numInstances()];
      labeleds[i] = resampleWithWeights(labeled, rand, inbags[i]);
      m_classifiers[i].buildClassifier(labeleds[i]);
      err_prime[i] = 0.5;
      s_prime[i] = 0;
    }

    boolean bChanged = true;
    int it = 0;
    while(bChanged)
    {
      it++;
      evaluate(it,test);
      bChanged = false;
      boolean[] bUpdate = new boolean[m_classifiers.length];
      Instances[] Li = new Instances[m_numClassifiers];

      for(int i = 0; i < m_numClassifiers; i++)
      {
        err[i] = measureError(labeled, inbags, i);
        //System.out.println("第" + it + "轮迭代的第" + i + "个分类器的袋外错误率为" + err[i]);
        Li[i] = new Instances(labeled, 0);
        
        /** if (e_i < e'_i) */
        if(err[i] < err_prime[i])
        {
          if(s_prime[i] == 0)
            s_prime[i] = Math.min(unlabeled.sumOfWeights() / 10, 100);

          /** Subsample U for each hi */
          double weight = 0;
          unlabeled.randomize(rand);
          int numWeightsAfterSubsample = (int) Math.ceil(err_prime[i] * s_prime[i] / err[i] - 1);
          for(int k = 0; k < unlabeled.numInstances(); k++)
          {
            weight += unlabeled.instance(k).weight();
            if (weight > numWeightsAfterSubsample)
             break;
           Li[i].add((Instance)unlabeled.instance(k).copy());
          }

          /** for every x in U' do */
          for(int j = Li[i].numInstances() - 1; j > 0; j--)
          {
            Instance curInst = Li[i].instance(j);
            if(!isHighConfidence(curInst, i))       //in which the label is assigned
              Li[i].delete(j);
          }//end of j

          if(s_prime[i] < Li[i].numInstances())
          {
            if(err[i] * Li[i].sumOfWeights() < err_prime[i] * s_prime[i])
              bUpdate[i] = true;
          }
        }
      }//end of for i

      //update
      Classifier[] newClassifier = rTree.makeCopies(rTree, m_numClassifiers);
      for(int i = 0; i < m_numClassifiers; i++)
      {
        if(bUpdate[i])
        {
          double size = Li[i].sumOfWeights();

          bChanged = true;
          m_classifiers[i] = newClassifier[i];
          ((RandomTree)m_classifiers[i]).setSeed(randSeeds[i]);
          m_classifiers[i].buildClassifier(combine(labeled, Li[i]));
          err_prime[i] = err[i];
          s_prime[i] = size;
        }
      }
    }//end of while
  }


  /**
   * To judege whether the confidence for a given instance of H* is high enough,
   * which is affected by the onfidence threshold. Meanwhile, if the example is
   * the confident one, assign label to it and weigh the example with the confidence
   *
   * @param inst Instance -- The instance
   * @param idExcluded int -- the index of the individual should be excluded from H*
   * @return boolean -- true for high
   * @throws Exception - some exception
   */
  protected boolean isHighConfidence(Instance inst, int idExcluded) throws Exception
  {
    double[] distr = distributionForInstanceExcluded(inst, idExcluded);
    double confidence = getConfidence(distr);
    if(confidence > m_threshold)
    {
      double classval = Utils.maxIndex(distr);
      inst.setClassValue(classval);    //assign label
      inst.setWeight(confidence);      //set instance weight
      return true;
    }
    else return false;
  }


  private Instances combine(Instances L, Instances Li)
  {
    for(int i = 0; i < L.numInstances(); i++)
      Li.add(L.instance(i));

    return Li;
  }

  private double measureError(Instances data, boolean[][] inbags, int id) throws Exception
   {
     double err = 0;
     double count = 0;
     for(int i = 0; i < data.numInstances() && i < m_numOriginalLabeledInsts; i++)
     {
       Instance inst = data.instance(i);
       double[] distr = outOfBagDistributionForInstanceExcluded(inst, i, inbags, id);

       if(getConfidence(distr) > m_threshold)
       {
         count += inst.weight();
         if(Utils.maxIndex(distr) != inst.classValue())
           err += inst.weight();
       }
     }
     err /= count;
     return err;
  }

  private double getConfidence(double[] p)
  {
    int maxIndex = Utils.maxIndex(p);
//    if(p[maxIndex]!=0&&p[maxIndex]!=1)System.out.println(p[maxIndex]);
    return p[maxIndex];
  }

  private double[] distributionForInstanceExcluded(Instance inst, int idExcluded) throws Exception
  {
    double[] distr = new double[inst.numClasses()];
    for(int i = 0; i < m_numClassifiers; i++)
    {
      if(i == idExcluded)
        continue;

      double[] d = m_classifiers[i].distributionForInstance(inst);
      for(int iClass = 0; iClass < inst.numClasses(); iClass++)
        distr[iClass] += d[iClass];
    }

    Utils.normalize(distr);
    return distr;
  }

  private double[] outOfBagDistributionForInstanceExcluded(Instance inst, int idxInst, boolean[][] inbags, int idExcluded) throws Exception
  {
    double[] distr = new double[inst.numClasses()];
    for(int i = 0; i < m_numClassifiers; i++)
    {
      if(inbags[i][idxInst] == true || i == idExcluded)
        continue;

      double[] d = m_classifiers[i].distributionForInstance(inst);
      for(int iClass = 0; iClass < inst.numClasses(); iClass++)
        distr[iClass] += d[iClass];
    }
    if(Utils.sum(distr) != 0)
      Utils.normalize(distr);
    return distr;
  }




  /**
   * The main method only for demonstrating the simple use of this package
   *
   * @param args String[]
   */
  public static void main(String[] args)
  {
    try
    {
    	
    	String testPath = "data/test3Feature.arff";
    	String trainPath = "data/train3Feature.arff";
    	
     int seed = 0;
     Date date = new Date();
     int numFeatures = 0;
     Random rand = new Random(seed);
     final int NUM_CLASSIFIERS = 6;

     BufferedReader r = new BufferedReader(new FileReader(trainPath));
     Instances train = new Instances(r);
     train.setClassIndex(train.numAttributes()-1);
     r.close();
     
/*//1.
		
     trainingSet.randomize(new Random(date.getTime()));
     int unlabelSize = (int) Math.round(trainingSet.numInstances() * 0.70);
     int labelSize = trainingSet.numInstances() - unlabelSize;
     Instances unlabeled = new Instances(trainingSet, 0, unlabelSize);
     Instances labeled = new Instances(trainingSet, unlabelSize, labelSize);
     labeled.setClassIndex(labeled.numAttributes()-1);
     unlabeled.setClassIndex(labeled.numAttributes()-1);
         */
     
   
     r = new BufferedReader(new FileReader(testPath));
     Instances testSet = new Instances(r);
     testSet.randomize(new Random(rand.nextLong()));
     testSet.setClassIndex(testSet.numAttributes()-1);
     r.close();
     
/*//4.全部混在一起
     BufferedReader r = new BufferedReader(new FileReader(testPath));
     Instances trainingSet = new Instances(r);		
     r.close();
     
     r = new BufferedReader(new FileReader(testPath));
     Instances testSet = new Instances(r);
     trainingSet.addAll(testSet);
     //trainingSet.addAll(testSet1);
     trainingSet.randomize(new Random(date.getTime()));
  // save ARFF
	    //ArffSaver saver = new ArffSaver();
	    //saver.setInstances(trainingSet);
	    //saver.setFile(new File("data\\big.arff"));
	    //saver.writeBatch();

     int testSize = (int) Math.round(trainingSet.numInstances() * 0.30);
     int unlabelSize = (int)Math.round((trainingSet.numInstances() - testSize)*0.70);
     int labelSize = trainingSet.numInstances() - unlabelSize - testSize;
     Instances unlabeled = new Instances(trainingSet, 0, unlabelSize);
     Instances labeled = new Instances(trainingSet, unlabelSize, labelSize);
     Instances test = new Instances(trainingSet, unlabelSize+labelSize,testSize);
     test.setClassIndex(labeled.numAttributes()-1);
     labeled.setClassIndex(labeled.numAttributes()-1);
     unlabeled.setClassIndex(labeled.numAttributes()-1);
     r.close();*/
     
/*//2.从train中取一部分当test，准确率接近1
     BufferedReader r = new BufferedReader(new FileReader(trainPath));
     Instances trainingSet = new Instances(r);		
     trainingSet.randomize(new Random(date.getTime()));
     int testSize = (int) Math.round(trainingSet.numInstances() * 0.30);
     int unlabelSize = (int)Math.round((trainingSet.numInstances() - testSize)*0.70);
     int labelSize = trainingSet.numInstances() - unlabelSize - testSize;
     Instances unlabeled = new Instances(trainingSet, 0, unlabelSize);
     Instances labeled = new Instances(trainingSet, unlabelSize, labelSize);
     Instances test = new Instances(trainingSet, unlabelSize+labelSize,testSize);
     test.setClassIndex(labeled.numAttributes()-1);
     labeled.setClassIndex(labeled.numAttributes()-1);
     unlabeled.setClassIndex(labeled.numAttributes()-1);
     r.close();*/
     

/*     r =  new BufferedReader(new FileReader("data/normal_data_39k_buptFeature_useful.arff"));
     Instances testSet = new Instances(r);
     r.close();*/
/*//3.从test中取出30%加到unlabeled，测试结果全部为xss，和1结合
     r = new BufferedReader(new FileReader(testPath));
     //testSet.addAll(new Instances(r));
     Instances testSet = new Instances(r);
     testSet.randomize(new Random(date.getTime()));
     int testSize = (int) Math.round(testSet.numInstances() * 0.95);
     int labelSize2 = testSet.numInstances() - testSize;
     //Instances test = new Instances(testSet, 0, testSize);
     Instances test = new Instances(testSet, 0, testSize);;
     Instances labeled2 = new Instances(testSet, testSize, labelSize2);
     labeled.addAll(labeled2);
     labeled.randomize(rand);
     test.setClassIndex(labeled.numAttributes()-1);
     r.close();*/    

		AttributeSelection filter=new AttributeSelection();
		CfsSubsetEval eval=new CfsSubsetEval();
		GreedyStepwise search=new GreedyStepwise();
		search.setSearchBackwards(true);
		filter.setEvaluator(eval);
		filter.setSearch(search);
		filter.setInputFormat(train);
		
		Instances trainingSet=Filter.useFilter(train, filter);//使用特征选择方法，生成新的训练数据
		/*trainingSet.deleteAttributeAt(2);	
		trainingSet.deleteAttributeAt(2);
		trainingSet.deleteAttributeAt(5);
		trainingSet.deleteAttributeAt(6);
		trainingSet.deleteAttributeAt(7);
		trainingSet.deleteAttributeAt(5);
		trainingSet.deleteAttributeAt(5);*/
		//trainingSet.deleteAttributeAt(10);
		trainingSet.deleteAttributeAt(8);
		trainingSet.deleteAttributeAt(8);
		trainingSet.deleteAttributeAt(8);
		Enumeration<Attribute> oldDataAttributes = train.enumerateAttributes();
		
		int index=0;
		while(oldDataAttributes.hasMoreElements()){
			Attribute oldAttribute=oldDataAttributes.nextElement();
			String thisAttributeString=oldAttribute.toString();
			Enumeration<Attribute> trainingSetAttributes = trainingSet.enumerateAttributes();
			
			int exist=0;
			while(trainingSetAttributes.hasMoreElements()){
				Attribute newAttribute=trainingSetAttributes.nextElement();
				String newString=newAttribute.toString();
				System.out.println(newString);
				if(newString.equals(thisAttributeString))
					exist=1;
			}
			if(exist==0){
				testSet.deleteAttributeAt(index);
				index--;
			}
			index++;
		}
		
     //Instances trainingSet = train;
	     trainingSet.randomize(new Random(rand.nextLong()));
	     int unlabelSize = (int) Math.round(trainingSet.numInstances() * 0.70);
	     int labelSize = trainingSet.numInstances() - unlabelSize;
	     Instances unlabeled = new Instances(trainingSet, 0, unlabelSize);
	     Instances labeled = new Instances(trainingSet, unlabelSize, labelSize);
	     labeled.setClassIndex(labeled.numAttributes()-1);
	     unlabeled.setClassIndex(labeled.numAttributes()-1);
	     Instances test = testSet;
	     
/*//4.全部混在一起
         trainingSet.addAll(testSet);
	     trainingSet.randomize(new Random(rand.nextLong()));
	  // save ARFF
		    //ArffSaver saver = new ArffSaver();
		    //saver.setInstances(trainingSet);
		    //saver.setFile(new File("data\\big.arff"));
		    //saver.writeBatch();*/

	/*     int testSize = (int) Math.round(trainingSet.numInstances() * 0.30);
	     int unlabelSize = (int)Math.round((trainingSet.numInstances() - testSize)*0.70);
	     int labelSize = trainingSet.numInstances() - unlabelSize - testSize;
	     Instances unlabeled = new Instances(trainingSet, 0, unlabelSize);
	     Instances labeled = new Instances(trainingSet, unlabelSize, labelSize);
	     Instances test = new Instances(trainingSet, unlabelSize+labelSize,testSize);
	     test.setClassIndex(labeled.numAttributes()-1);
	     labeled.setClassIndex(labeled.numAttributes()-1);
	     unlabeled.setClassIndex(labeled.numAttributes()-1);
	     r.close();*/

     CoForest forest = new CoForest();
     forest.setNumClassifiers(NUM_CLASSIFIERS);
     forest.setNumFeatures(numFeatures);
     forest.setSeed(rand.nextInt());
     forest.buildClassifier(labeled, unlabeled, test);

     double TP=0,FP=0,FN=0,TN=0;
     for(int i = 0; i < test.numInstances(); i++)
     {
       if(forest.classifyInstance(test.instance(i)) == test.instance(i).classValue())
       {
    	   if(test.instance(i).classValue() == 0) TP++;
    	   else TN++;
       }
       else {
    	   if(test.instance(i).classValue() == 0) FN++;
    	   else { FP++;
    	   //System.out.println(test.instance(i));
    	   }
       }
         //err++;
     }
     double pre = TP/(TP+FP);
     double recall = TP/(TP+FN);
     System.out.println("TP=" + TP + " TN=" + TN + " FP=" + FP +" FN=" +FN);
     System.out.println("准确率 = " + ((TP+TN)/(TP+TN+FP+FN)));
     System.out.println("召回率 = " + recall);
     System.out.println("精确率 = " + pre);
     System.out.println("F1 = " + 2*pre*recall/(pre+recall));
     System.out.println("误报率 = " + (FP/(TN+FP)));
     System.out.println("漏报率 = " + (1-recall));

   }
   catch(Exception e)
   {
     e.printStackTrace();
   }
 }
}
