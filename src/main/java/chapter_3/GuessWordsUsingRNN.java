package chapter_3;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration.ListBuilder;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.jetbrains.annotations.NotNull;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.indexaccum.IMax;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.RmsProp;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;

public class GuessWordsUsingRNN {

	// define a sentence to learn.
    // Add a special character at the beginning so the RNN learns the complete string and ends with the marker.
	private static final char[] LEARN_STRING = "* The pilot is flying using a plane.".toCharArray();

	// a list of all possible characters
	private static final List<Character> LEARN_STRING_CHARS_LIST = new ArrayList<>();

	// RNN dimensions
	private static final int HIDDEN_LAYER_WIDTH = 50;
	private static final int HIDDEN_LAYER_CONT = 2;

	public static void main(String[] args) {

		LinkedHashSet<Character> LEARN_STRING_CHARS = createListOfPossibleCharacters();

		NeuralNetConfiguration.Builder builder = parametrizeNeuralNetwork();

		ListBuilder listBuilder = builder.list();

		buildRNN(LEARN_STRING_CHARS, listBuilder);

		RnnOutputLayer.Builder outputLayerBuilder = normalizeOutputOfNeurons(LEARN_STRING_CHARS);

		listBuilder.layer(HIDDEN_LAYER_CONT, outputLayerBuilder.build());

		// finish builder
		listBuilder.pretrain(false);
		listBuilder.backprop(true);

		// create network
		MultiLayerNetwork net = initializeNetwork(listBuilder);

		DataSet trainingData = createTrainingData();

		// some epochs
		for (int epoch = 0; epoch < 500; epoch++) {

			System.out.println("Epoch " + epoch);

			// train the data
			net.fit(trainingData);

			// clear current stance from the last example
			net.rnnClearPreviousState();

			// put the first character into the rrn as an initialisation
			INDArray testInit = Nd4j.zeros(1, LEARN_STRING_CHARS_LIST.size(), 1);
			testInit.putScalar(LEARN_STRING_CHARS_LIST.indexOf(LEARN_STRING[0]), 1);


			INDArray output = predictWhatShouldBeNext(net, testInit);

			// NeuralNetwork will guess other characters
            for (char dummy : LEARN_STRING) {
				int sampledCharacterIdx = getHighestScoreNeuron(output);

				System.out.print(LEARN_STRING_CHARS_LIST.get(sampledCharacterIdx));

				output = useLastOutputAsAnInputToNextIteration(net, sampledCharacterIdx);

			}
			System.out.print("\n");
		}
	}

	private static INDArray predictWhatShouldBeNext(MultiLayerNetwork net, INDArray testInit) {
		return net.rnnTimeStep(testInit);
	}

	private static INDArray useLastOutputAsAnInputToNextIteration(MultiLayerNetwork net, int sampledCharacterIdx) {
		INDArray output;
		INDArray nextInput = Nd4j.zeros(1, LEARN_STRING_CHARS_LIST.size(), 1);
		nextInput.putScalar(sampledCharacterIdx, 1);
		output = net.rnnTimeStep(nextInput);
		return output;
	}

	private static int getHighestScoreNeuron(INDArray output) {
		// first process the last output of the network to a concrete
		// neuron, the neuron with the highest output has the highest
		// chance to get chosen
		return Nd4j.getExecutioner().exec(new IMax(output), 1).getInt(0);
	}

	@NotNull
	private static DataSet createTrainingData() {
		/*
		 * CREATE OUR TRAINING DATA
		 */
		// create input and output arrays: SAMPLE_INDEX, INPUT_NEURON,
		// SEQUENCE_POSITION
		INDArray input = Nd4j.zeros(1, LEARN_STRING_CHARS_LIST.size(), LEARN_STRING.length);
		INDArray labels = Nd4j.zeros(1, LEARN_STRING_CHARS_LIST.size(), LEARN_STRING.length);
		int samplePos = 0;
		for (char currentChar : LEARN_STRING) {
			char nextChar = LEARN_STRING[(samplePos + 1) % (LEARN_STRING.length)];
			input.putScalar(new int[] { 0, LEARN_STRING_CHARS_LIST.indexOf(currentChar), samplePos }, 1);
			labels.putScalar(new int[] { 0, LEARN_STRING_CHARS_LIST.indexOf(nextChar), samplePos }, 1);
			samplePos++;
		}
		return new DataSet(input, labels);
	}

	@NotNull
	private static MultiLayerNetwork initializeNetwork(ListBuilder listBuilder) {
		MultiLayerConfiguration conf = listBuilder.build();
		MultiLayerNetwork net = new MultiLayerNetwork(conf);
		net.init();
		net.setListeners(new ScoreIterationListener(1));
		return net;
	}

	@NotNull
	private static RnnOutputLayer.Builder normalizeOutputOfNeurons(LinkedHashSet<Character> LEARN_STRING_CHARS) {
		// we need to use RnnOutputLayer for our RNN
		RnnOutputLayer.Builder outputLayerBuilder = new RnnOutputLayer.Builder(LossFunction.MCXENT);
		// softmax normalizes the output neurons, the sum of all outputs is 1
		// this is required for our sampleFromDistribution-function
		outputLayerBuilder.activation(Activation.SOFTMAX);
		outputLayerBuilder.nIn(HIDDEN_LAYER_WIDTH);
		outputLayerBuilder.nOut(LEARN_STRING_CHARS.size());
		return outputLayerBuilder;
	}

	private static void buildRNN(LinkedHashSet<Character> LEARN_STRING_CHARS, ListBuilder listBuilder) {
		System.out.println(LEARN_STRING_CHARS);
		for (int i = 0; i < HIDDEN_LAYER_CONT; i++) {
			LSTM.Builder hiddenLayerBuilder = new LSTM.Builder();
			hiddenLayerBuilder.nIn(i == 0 ? LEARN_STRING_CHARS.size() : HIDDEN_LAYER_WIDTH);
			hiddenLayerBuilder.nOut(HIDDEN_LAYER_WIDTH);
			hiddenLayerBuilder.activation(Activation.TANH);
			listBuilder.layer(i, hiddenLayerBuilder.build());
		}
	}

	@NotNull
	private static NeuralNetConfiguration.Builder parametrizeNeuralNetwork() {
		NeuralNetConfiguration.Builder builder = new NeuralNetConfiguration.Builder();
		builder.seed(123);
		builder.biasInit(0);
		builder.miniBatch(false);
		builder.updater(new RmsProp(0.001));
		builder.weightInit(WeightInit.XAVIER);
		return builder;
	}

	@NotNull
	private static LinkedHashSet<Character> createListOfPossibleCharacters() {
		LinkedHashSet<Character> LEARN_STRING_CHARS = new LinkedHashSet<>();
		for (char c : LEARN_STRING)
			LEARN_STRING_CHARS.add(c);
		LEARN_STRING_CHARS_LIST.addAll(LEARN_STRING_CHARS);
		return LEARN_STRING_CHARS;
	}
}