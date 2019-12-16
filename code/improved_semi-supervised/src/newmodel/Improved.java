package newmodel;

import java.io.*;
import java.text.*;
import java.util.*;

import weka.core.*;
import weka.classifiers.*;
import weka.classifiers.trees.*;

import org.apache.commons.math3.distribution.BetaDistribution;
import org.apache.commons.math3.special.Gamma;

public class Improved
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
  public Improved()
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
    
    rTree.setMaxDepth(3);

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
      err_prime[i] = 0.01;
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
   * get a random list of 0 ~ N-1
   *
   * @param N int
   */
  public static int[] randperm(int N) throws Exception
  {
	  int arr[] = new int[N];
	  for(int i=0;i<N;i++)
	  {
		  arr[i]=i;
	  }
	  for(int i=0;i<N;i++)
	  {
		  int x = (int)(Math.random()*1000000)%N;
		  int temp = arr[i];
		  arr[i] = arr[x];
		  arr[x] = temp;
	  }
	return arr;
	  
  }
  
   /**
   * The Mixup method  for data augmentation
   *
   * @param trainingSet Instances
   * @param batchsize int
   * @param alpha double
   */
  public static void Mixup(Instances trainingSet, int batchsize, double alpha) throws Exception
  {
	  for(int i=0;i<trainingSet.numInstances()/10*10;i+=batchsize)
	  {
		  double lam = betasampler(alpha,alpha);
		  int[] indexes = randperm(batchsize);
		  Instance inst = trainingSet.instance(0);
		  for(int j=0;j<batchsize;j++)
		  {
			  for(int k=0;k<trainingSet.numAttributes();k++)
			  {
				  double mixed = inst.value(k)*lam + trainingSet.instance(indexes[j]+i).value(k)*(1-lam);
				  inst.setValue(k, mixed);
			  }
			  if(lam>0.5) inst.setClassValue(trainingSet.instance(i+j).classValue());
			  else inst.setClassValue(trainingSet.instance(indexes[j]+i).classValue());
		  }
	  }
    return;
  }
  
   /**
   * get a sample of beta distribution
   *
   * @param alpha double
   * @param beta double  
   */
  public static double betasampler(double alpha,double beta)
  {
      BetaDistribution bd =new BetaDistribution(alpha,beta);
      return bd.sample();
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
    	
    	String testPath = "data/test.arff";
    	String trainPath = "data/train.arff";
    	
     int seed = 0;
     int numFeatures = 0;
     int batchsize = 10;
     Random rand = new Random(seed);
     final int NUM_CLASSIFIERS = 10;

     BufferedReader r = new BufferedReader(new FileReader(trainPath));
     Instances trainingSet = new Instances(r);		
     trainingSet.randomize(new Random(rand.nextLong()));
     trainingSet.setClassIndex(trainingSet.numAttributes()-1);
     Mixup(trainingSet,batchsize,1.0);  //alpha=0.1
     trainingSet.randomize(new Random(rand.nextLong()));
     int unlabelSize = (int) Math.round(trainingSet.numInstances() * 0.70);
     int labelSize = trainingSet.numInstances() - unlabelSize;
     Instances unlabeled = new Instances(trainingSet, 0, unlabelSize);
     Instances labeled = new Instances(trainingSet, unlabelSize, labelSize);
     labeled.setClassIndex(labeled.numAttributes()-1);
     unlabeled.setClassIndex(labeled.numAttributes()-1);
     r.close();    

     r = new BufferedReader(new FileReader(testPath));
     Instances test = new Instances(r);
     test.randomize(new Random(rand.nextLong()));
     test.setClassIndex(labeled.numAttributes()-1);
     r.close();

     Improved forest = new Improved();
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
    	   else FP++;
       }
     }
     double pre = TP/(TP+FP);
     double recall = TP/(TP+FN);
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
