package com.mycompany.lstm;

import org.apache.commons.io.FileUtils;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.nio.charset.Charset;
import java.util.Random;

public class App 
{

    static {
	System.load("/usr/local/lib/nd4j/nd4j-backends/nd4j-backend-impls/nd4j-native/target/classes/org/nd4j/nativeblas/linux-x86_64/libjnind4jcpu.so");
	
	System.load("/root/lstm/java/dl4j/javacpp-presets/openblas/target/classes/org/bytedeco/javacpp/linux-x86_64/libjniopenblas.so");
	System.load("/usr/lib/gcc/x86_64-linux-gnu/5/libgfortran.so");
    }

    public static void main( String[] args ) throws Exception {
		int lstmLayerSize = 200;			//Number of units in each GravesLSTM layer
		int miniBatchSize = 32;				//Size of mini batch to use when  training
		int exampleLength = 1000;
		System.out.println("... lstm ...");

		int tbpttLength = 50;                       
		int numEpochs = 1;							
                int generateSamplesEveryNMinibatches = 10;  
		int nSamplesToGenerate = 4;					
		int nCharactersToSample = 300;				
		String generationInitialization = null;		
		
		Random rng = new Random(12345);

		CharacterIterator iter = getShakespeareIterator(miniBatchSize,exampleLength);
		int nOut = iter.totalOutcomes();

		//Set up network configuration:
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
			.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
			.learningRate(0.1)
			.rmsDecay(0.95)
			.seed(12345)
			.regularization(true)
			.l2(0.001)
            		.weightInit(WeightInit.XAVIER)
            		.updater(Updater.RMSPROP)
			.list()
			.layer(0, new GravesLSTM.Builder().nIn(iter.inputColumns()).nOut(lstmLayerSize)
					.activation("tanh").build())
			.layer(1, new GravesLSTM.Builder().nIn(lstmLayerSize).nOut(lstmLayerSize)
					.activation("tanh").build())
			.layer(2, new RnnOutputLayer.Builder(LossFunction.MCXENT).activation("softmax")        //MCXENT + softmax for classification
					.nIn(lstmLayerSize).nOut(nOut).build())
            		.backpropType(BackpropType.TruncatedBPTT).tBPTTForwardLength(tbpttLength).tBPTTBackwardLength(tbpttLength)
			.pretrain(false).backprop(true)
			.build();

		MultiLayerNetwork net = new MultiLayerNetwork(conf);
		net.init();
		net.setListeners(new ScoreIterationListener(1));

                //Print the  number of parameters in the network (and for each layer)
		Layer[] layers = net.getLayers();
		int totalNumParams = 0;
		for( int i=0; i<layers.length; i++ ){
			int nParams = layers[i].numParams();
			System.out.println("Number of parameters in layer " + i + ": " + nParams);
			totalNumParams += nParams;
		}
		
		System.out.println("Total number of network parameters: " + totalNumParams);
		System.out.println(" Wait for a while ");
		//Do training, and then generate and print samples from network
                int miniBatchNumber = 0;
		
                int test_set_max = 10;
		for( int i=0; i<numEpochs; i++ ){
			
			int test_set_counter = 0;
            		while(iter.hasNext()){
				
				if(miniBatchNumber > test_set_max){
					System.out.print("test_set_max is reached.");
					break;
				}
                		DataSet ds = iter.next();
                		net.fit(ds);
                		if(++miniBatchNumber % generateSamplesEveryNMinibatches == 0){
                    			System.out.println("--------------------");
                    			System.out.println("Completed " + miniBatchNumber + " minibatches of size " + miniBatchSize + "x" + exampleLength + " characters" );
                    			System.out.println("Sampling characters from network given initialization \"" + (generationInitialization == null ? "" : generationInitialization) + "\"");
                    			String[] samples = sampleCharactersFromNetwork(generationInitialization,net,iter,rng,nCharactersToSample,nSamplesToGenerate);

                    			for( int j=0; j<samples.length; j++ ){
                        			System.out.println("----- Sample " + j + " -----");
                        			System.out.println(samples[j]);
                        			System.out.println();
                    			}
                		}
				
            		}

			iter.reset();	//Reset iterator for another epoch
		}

		System.out.println("\n\nExample complete");
			
    }

    public static CharacterIterator getShakespeareIterator(int miniBatchSize, int sequenceLength) throws Exception{
		//The Complete Works of William Shakespeare
		//5.3MB file in UTF-8 Encoding, ~5.4 million characters
		//https://www.gutenberg.org/ebooks/100
		String url = "https://s3.amazonaws.com/dl4j-distribution/pg100.txt";
		String tempDir = System.getProperty("java.io.tmpdir");
		String fileLocation = tempDir + "/Shakespeare.txt";	//Storage location from downloaded file
		File f = new File(fileLocation);
		if( !f.exists() ){
			FileUtils.copyURLToFile(new URL(url), f);
			System.out.println("File downloaded to " + f.getAbsolutePath());
		} else {
			System.out.println("Using existing text file at " + f.getAbsolutePath());
		}

		if(!f.exists()) throw new IOException("File does not exist: " + fileLocation);	//Download problem?

		char[] validCharacters = CharacterIterator.getMinimalCharacterSet();	
		return new CharacterIterator(fileLocation, Charset.forName("UTF-8"),
				miniBatchSize, sequenceLength, validCharacters, new Random(12345));
    }

    private static String[] sampleCharactersFromNetwork(String initialization, MultiLayerNetwork net,
                                                        CharacterIterator iter, Random rng, int charactersToSample, int numSamples ){
		//Set up initialization. If no initialization: use a random character
		if( initialization == null ){
			initialization = String.valueOf(iter.getRandomCharacter());
		}

		//Create input for initialization
		INDArray initializationInput = Nd4j.zeros(numSamples, iter.inputColumns(), initialization.length());
		char[] init = initialization.toCharArray();
		for( int i=0; i<init.length; i++ ){
			int idx = iter.convertCharacterToIndex(init[i]);
			for( int j=0; j<numSamples; j++ ){
				initializationInput.putScalar(new int[]{j,idx,i}, 1.0f);
			}
		}

		StringBuilder[] sb = new StringBuilder[numSamples];
		for( int i=0; i<numSamples; i++ ) sb[i] = new StringBuilder(initialization);

		//Sample from network (and feed samples back into input) one character at a time (for all samples)
		//Sampling is done in parallel here
		net.rnnClearPreviousState();
		INDArray output = net.rnnTimeStep(initializationInput);
		output = output.tensorAlongDimension(output.size(2)-1,1,0);	//Gets the last time step output

		for( int i=0; i<charactersToSample; i++ ){
			//Set up next input (single time step) by sampling from previous output
			INDArray nextInput = Nd4j.zeros(numSamples,iter.inputColumns());
			//Output is a probability distribution. Sample from this for each example we want to generate, and add it to the new input
			for( int s=0; s<numSamples; s++ ){
				double[] outputProbDistribution = new double[iter.totalOutcomes()];
				for( int j=0; j<outputProbDistribution.length; j++ ) outputProbDistribution[j] = output.getDouble(s,j);
				int sampledCharacterIdx = sampleFromDistribution(outputProbDistribution,rng);

				nextInput.putScalar(new int[]{s,sampledCharacterIdx}, 1.0f);		//Prepare next time step input
				sb[s].append(iter.convertIndexToCharacter(sampledCharacterIdx));	//Add sampled character to StringBuilder (human readable output)
			}

			output = net.rnnTimeStep(nextInput);	//Do one time step of forward pass
		}

		String[] out = new String[numSamples];
		for( int i=0; i<numSamples; i++ ) out[i] = sb[i].toString();
		return out;
    }

    public static int sampleFromDistribution( double[] distribution, Random rng ){
		double d = rng.nextDouble();
		double sum = 0.0;
		for( int i=0; i<distribution.length; i++ ){
			sum += distribution[i];
			if( d <= sum ) return i;
		}
		//Should never happen if distribution is a valid probability distribution
		throw new IllegalArgumentException("Distribution is invalid? d="+d+", sum="+sum);
    }
}
