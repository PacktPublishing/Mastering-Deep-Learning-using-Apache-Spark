package chapter_5;

import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.RmsProp;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;

public class ClassifyNewsIntoCategories {
    private static String userDirectory = "";
    private static String DATA_PATH = "";
    private static String WORD_VECTORS_PATH = "";

    public static void main(String[] args) throws Exception {
        userDirectory = new ClassPathResource("NewsData").getFile().getAbsolutePath() + File.separator;
        DATA_PATH = userDirectory + "LabelledNews";
        WORD_VECTORS_PATH = userDirectory + "NewsWordVector.txt";

        int batchSize = 50;
        int nEpochs = 1000;
        int truncateReviewsToLength = 300;

        WordVectors wordVectors = transformTextIntoFeatureVector();

        TokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
        tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());

        NewsIterator iTrain = trainNewsIterator(batchSize, truncateReviewsToLength, wordVectors, tokenizerFactory);

        NewsIterator iTest = testNewsIterator(batchSize, truncateReviewsToLength, wordVectors, tokenizerFactory);

        int inputNeurons = wordVectors.getWordVector(wordVectors.vocab().wordAtIndex(0)).length;
        int outputs = iTrain.getLabels().size();

        MultiLayerConfiguration conf = configureMultiLayer(inputNeurons, outputs);

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        net.setListeners(new ScoreIterationListener(1));

        assignNewsIntoClassesUsingDL(nEpochs, iTrain, iTest, net);

        ModelSerializer.writeModel(net, userDirectory + "NewsModel.net", true);
        System.out.println("----- Example complete -----");
    }

    private static void assignNewsIntoClassesUsingDL(int nEpochs, NewsIterator iTrain, NewsIterator iTest, MultiLayerNetwork net) {
        System.out.println("Starting training");
        for (int i = 0; i < nEpochs; i++) {
            net.fit(iTrain);
            iTrain.reset();
            System.out.println("Epoch " + i + " complete. Starting evaluation:");

            Evaluation evaluation = net.evaluate(iTest);

            System.out.println(evaluation.stats());
        }
    }

    private static MultiLayerConfiguration configureMultiLayer(int inputNeurons, int outputs) {
        TokenizerFactory tokenizerFactory;
        tokenizerFactory = new DefaultTokenizerFactory();
        tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());
        //Set up network configuration
        return new NeuralNetConfiguration.Builder()
            .updater(new RmsProp(0.0018))
            .l2(1e-5)
            .weightInit(WeightInit.XAVIER)
            .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue).gradientNormalizationThreshold(1.0)
            .list()
            .layer(0, new LSTM.Builder().nIn(inputNeurons).nOut(200)
                .activation(Activation.SOFTSIGN).build())
            .layer(1, new RnnOutputLayer.Builder().activation(Activation.SOFTMAX)
                .lossFunction(LossFunctions.LossFunction.MCXENT).nIn(200).nOut(outputs).build())
            .build();
    }

    private static NewsIterator testNewsIterator(int batchSize, int truncateReviewsToLength, WordVectors wordVectors, TokenizerFactory tokenizerFactory) {
        return new NewsIterator.Builder()
                  .dataDirectory(DATA_PATH)
                  .wordVectors(wordVectors)
                  .batchSize(batchSize)
                  .tokenizerFactory(tokenizerFactory)
                  .truncateLength(truncateReviewsToLength)
                  .train(false)
                  .build();
    }

    private static NewsIterator trainNewsIterator(int batchSize, int truncateReviewsToLength, WordVectors wordVectors, TokenizerFactory tokenizerFactory) {
        return new NewsIterator.Builder()
                  .dataDirectory(DATA_PATH)
                  .wordVectors(wordVectors)
                  .batchSize(batchSize)
                  .truncateLength(truncateReviewsToLength)
                  .tokenizerFactory(tokenizerFactory)
                  .train(true)
                  .build();
    }

    private static WordVectors transformTextIntoFeatureVector() {
        return WordVectorSerializer.readWord2VecModel(new File(WORD_VECTORS_PATH));
    }

}
