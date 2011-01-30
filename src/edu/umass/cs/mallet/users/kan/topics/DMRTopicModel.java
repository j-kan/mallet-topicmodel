/*=============================================================================
  DMRTopicModel.java
                                    Created by jkan on Apr 21, 2010
                                    Copyright (c)2010 Essbare Weichware, GmbH
                                    All rights reserved.
  =============================================================================*/

package edu.umass.cs.mallet.users.kan.topics;

import java.io.File;
import java.io.IOException;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Arrays;

import cc.mallet.classify.MaxEnt;
import cc.mallet.optimize.LimitedMemoryBFGS;
import cc.mallet.optimize.OptimizationException;
import cc.mallet.optimize.Optimizer;
import cc.mallet.optimize.StochasticMetaAscent;
import cc.mallet.pipe.Noop;
import cc.mallet.pipe.Pipe;
import cc.mallet.types.FeatureCounter;
import cc.mallet.types.FeatureVector;
import cc.mallet.types.Instance;
import cc.mallet.types.InstanceList;
import cc.mallet.types.MatrixOps;
import cc.mallet.util.Randoms;


public class DMRTopicModel extends ParallelTopicModel {

    private static final long serialVersionUID = 1;
	
	MaxEnt dmrParameters = null;
    int numFeatures;
    int defaultFeatureIndex;

    Pipe parameterPipe = null;
    
	double[][] alphaCache;
    double[] alphaSumCache;

    boolean useStochasticMetaAscent = true;
    int     numBatches              = 4;
    double  initialStep             = 0.002;
    double  metaStep                = 0.005;
    

	public DMRTopicModel(int numberOfTopics) {
		super(numberOfTopics);
	}

    /**
         @param useStochasticMetaAscent 
                     if true, use {@link StochasticMetaAscent} 
                     instead of {@link LimitedMemoryBFGS} to optimize
     **/
    public void setUseStochasticMetaAscent(boolean useStochasticMetaAscent) {
        this.useStochasticMetaAscent = useStochasticMetaAscent;
        if (!useStochasticMetaAscent)
            this.numBatches = 1;    // L-BFGS is not batched
    }

    /**
         @param numBatches  number of batches for {@link StochasticMetaAscent}
     **/
    public void setNumBatches(int numBatches) {
        this.numBatches              = numBatches;
        this.useStochasticMetaAscent = true;
    }
    
    /**
         @param initialStep  initial step parameter for {@link StochasticMetaAscent}
         @see StochasticMetaAscent#setInitialStep(double)
     **/
    public void setInitialStep(double initialStep) {
        this.initialStep = initialStep;
    }
    
    /**
         @param metaStep    mu parameter for {@link StochasticMetaAscent}
         @see StochasticMetaAscent#setMu(double)
     **/
    public void setMetaStep(double metaStep) {
        this.metaStep = metaStep;
    }
    
    
	@Override
	public void addInstances(InstanceList training) {

		super.addInstances(training);
		
        numFeatures = data.get(0).instance.getTargetAlphabet().size() + 1;
        defaultFeatureIndex = numFeatures - 1;

		int numDocs = data.size(); // TODO consider beginning by sub-sampling?

		alphaCache    = new double[numDocs][numTopics];
        alphaSumCache = new double[numDocs];
        
        // Create a "fake" pipe with the features in the data and 
        //  a trove int-int hashmap of topic counts in the target.
        
        if (parameterPipe == null) {
            parameterPipe = new Noop();

            parameterPipe.setDataAlphabet(data.get(0).instance.getTargetAlphabet());
            parameterPipe.setTargetAlphabet(topicAlphabet);
        }
	}
	
	@Override
    protected WorkerRunnable makeWorkerRunnable(int offset, int docsPerThread)
    {
        TypeTopicCounts typeTopicCounts = this.typeTopicCounts;
        int[]           tokensPerTopic  = this.tokensPerTopic;
        double[]        alpha           = this.alpha;
        
        // If there is only one thread, copy the typeTopicCounts
        //  arrays directly, rather than allocating new memory.
        // DMR also needs to copy the alpha array, since they are modified on a per-document basis.

        if (numThreads > 1)    // otherwise, make a copy for the thread
        {
            typeTopicCounts = new TypeTopicCounts(typeTopicCounts);
            tokensPerTopic  = Arrays.copyOf(tokensPerTopic, tokensPerTopic.length);
            alpha           = Arrays.copyOf(alpha, alpha.length);
        }
        
        WorkerRunnable runnable = new DMRWorkerRunnable(
                                          alpha, beta,
                                          makeRandom(), data,
                                          typeTopicCounts, tokensPerTopic,
                                          offset, docsPerThread);

        runnable.initializeAlphaStatistics(docLengthCounts.length);
        
        return runnable;
	}
	
	
	/**
	 * For the DMRTopicModel, optimizing alphas means training regression parameters
	 * (was "learnParameters()")
	 *	
	 *  @see edu.umass.cs.mallet.users.kan.topics.ParallelTopicModel#optimizeAlpha(edu.umass.cs.mallet.users.kan.topics.WorkerRunnable[])
	 **/
	@Override
	public void optimizeAlpha() {

        InstanceList parameterInstances = new InstanceList(parameterPipe);

        if (dmrParameters == null) {
            dmrParameters = new MaxEnt(parameterPipe, new double[numFeatures * numTopics]);
        }
        
        for (TopicAssignment ta: data) {
            
            if (ta.instance.getTarget() == null) {
                continue;
            }

			FeatureCounter counter = new FeatureCounter(topicAlphabet);

			for (int topic : ta.topicSequence.getFeatures()) {
				counter.increment(topic);
            }

            // Put the real target in the data field, and the
            //  topic counts in the target field
            parameterInstances.add( new Instance(ta.instance.getTarget(), counter.toFeatureVector(), null, null) );
        }

        DMROptimizable optimizable = new DMROptimizable(parameterInstances, this.numBatches, this.dmrParameters);
        optimizable.setRegularGaussianPriorVariance(0.5);
        optimizable.setInterceptGaussianPriorVariance(100.0);

        Optimizer optimizer = makeOptimizer(optimizable);

		// Optimize once
		try {
			optimizer.optimize();
		} catch (OptimizationException e) {
			// step size too small
		}

		// Restart with a fresh initialization to improve likelihood
		try {
			optimizer.optimize();
		} catch (OptimizationException e) {
			// step size too small
		}
        dmrParameters = optimizable.getClassifier();

        cacheAlphas();
    }

    /**
         @param optimizable
         @param optimizer
         @return
     **/
    private Optimizer makeOptimizer(DMROptimizable optimizable)
    {
        if (this.useStochasticMetaAscent)
        {
            StochasticMetaOptimizer optimizer = new StochasticMetaOptimizer(optimizable, this.numBatches, optimizable.trainingList.size(), this.makeRandom());

            optimizer.setInitialStep(this.initialStep);
            optimizer.setMu(this.metaStep);
            
            return optimizer;
        }
        else
        {
            return new LimitedMemoryBFGS(optimizable);
        }
    }


	private void cacheAlphas() {
		
		for (int doc=0; doc < data.size(); doc++) {
            Instance instance = data.get(doc).instance;
            //FeatureSequence tokens = (FeatureSequence) instance.getData();
            if (instance.getTarget() == null) { continue; }
            //int numTokens = tokens.getLength();

            alphaSumCache[doc] = setAlphasFromDocFeatures(alphaCache[doc], instance);
        }
	}
	
	/**
	 *  Set alpha based on features in an instance
	 *  	     
	 *  @param alpha		the array of alpha values to set (out parameter)
	 *  @param instance		the instance from which to read the features
	 *  @return the sum of the resulting alphas
	 */
	double setAlphasFromDocFeatures(double[] alpha, Instance instance) {
	    
        // we can't use the standard score functions from MaxEnt,
        //  since our features are currently in the Target.
        FeatureVector features = (FeatureVector) instance.getTarget();
        if (features == null) { return setAlphasWithoutDocFeatures(alpha); }
        
        double[] parameters = dmrParameters.getParameters();
        double   alphaSum   = 0.0;
        
        for (int topic = 0; topic < numTopics; topic++) {
            alpha[topic] = parameters[topic*numFeatures + defaultFeatureIndex]
                + MatrixOps.rowDotProduct (parameters,
                                           numFeatures,
                                           topic, features,
                                           defaultFeatureIndex,
                                           null);
            
            alpha[topic] = Math.exp(alpha[topic]);
            alphaSum += alpha[topic];
    	}
        return alphaSum;
	}

	/**
	 *  Use only the default features to set the topic prior (use no document features)
	 *  	     
	 *  @param alpha		the array of alpha values to set (out parameter)
	 *  @return the sum of the resulting alphas
	 */
	double setAlphasWithoutDocFeatures(double[] alpha) {

        double[] parameters = dmrParameters.getParameters();
        double   alphaSum   = 0.0;

        // Use only the default features to set the topic prior (use no document features)
        for (int topic=0; topic < numTopics; topic++) {
            alpha[topic] = Math.exp( parameters[ (topic * numFeatures) + defaultFeatureIndex ] );
            alphaSum += alpha[topic];
        }
        return alphaSum;
    }
	
	public void printTopWords (PrintStream out, int numWords, boolean usingNewLines) {
		if (dmrParameters != null) { setAlphasWithoutDocFeatures(alpha); }
		super.printTopWords(out, numWords, usingNewLines);
	}

	public void writeParameters(File parameterFile) throws IOException {
		if (dmrParameters != null) {
			PrintStream out = new PrintStream(parameterFile);
			dmrParameters.print(out);
			out.close();
		}
	}

	public MaxEnt getRegressionParameters() { return dmrParameters; }
	
	class DMRWorkerRunnable extends WorkerRunnable {

		DMRWorkerRunnable(double[] alpha, double beta, 
		           Randoms random,
				   ArrayList<TopicAssignment> data,
				   TypeTopicCounts runnableCounts, 
				   int[] tokensPerTopic,
				   int startDoc, int numDocs) {
			
			super(alpha, beta, random, data, runnableCounts, tokensPerTopic, startDoc, numDocs);
		}

		@Override
		protected void prepareToSample() {
			if (dmrParameters == null) { // before we start doing regression, behave like normal LDA
				super.prepareToSample();
			}
			// in normal LDA, this recalculates the smoothingOnlyMass and cachedCoefficients
			// but after we start regression, we have to do this with every doc since the alphas
			//   are different for every doc, so we can skip it here.
		}
		
		@Override
		protected void sampleTopicsForOneDoc(TopicAssignment document, boolean readjustTopicsAndStats) {
			
			if (dmrParameters != null) {
				// set the alphas for each doc before sampling
				setAlphasFromDocFeatures(this.alpha, document.instance);
				initSmoothingOnlyMassAndCachedCoefficients();
			}
			super.sampleTopicsForOneDoc(document, readjustTopicsAndStats);
		}
	}
}
