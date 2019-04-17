package chapter_2;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;
import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.split.InputSplit;
import org.datavec.api.split.NumberedFileInputSplit;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.AsyncDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.conf.preprocessor.CnnToFeedForwardPreProcessor;
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToRnnPreProcessor;
import org.deeplearning4j.nn.conf.preprocessor.RnnToCnnPreProcessor;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.AdaGrad;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.datavec.api.conf.Configuration;
import org.datavec.codec.reader.CodecRecordReader;


import java.io.File;
import java.util.HashMap;
import java.util.Map;

/**
 * Classification Video with 4 different shapes into 4 output classes using DL
 * run with -Xmx16G or -Xmx8G
 */
public class VideoClassificationExample {

  private static final int N_VIDEOS_TO_GENERATE = 25;
  private static final int V_WIDTH = 130;
  private static final int V_HEIGHT = 130;
  private static final int V_NFRAMES = 150;

  public static void main(String[] args) throws Exception {

    int miniBatchSize = 10;
    boolean generateData = true;

    String tempDir = System.getProperty("java.io.tmpdir");
    String dataDirectory = FilenameUtils.concat(tempDir, "chapter_2_video_DL/");

    //Generate data: number of .mp4 videos for input, plus .txt files for the labels
    generateVideoDataIfNeeded(generateData, dataDirectory);

    //Set up network architecture:
    MultiLayerConfiguration conf = configureNeuralNetwork();

    MultiLayerNetwork net = new MultiLayerNetwork(conf);
    net.init();
    net.setListeners(new ScoreIterationListener(1));

    printNeuralNetworkParameters(net);

    int testStartIdx = (int) (0.9 * N_VIDEOS_TO_GENERATE);  //90% in train, 10% in test
    int nTest = N_VIDEOS_TO_GENERATE - testStartIdx;

    performLearning(miniBatchSize, dataDirectory, conf, net, testStartIdx, nTest);
  }

  private static void performLearning(int miniBatchSize, String dataDirectory, MultiLayerConfiguration conf, MultiLayerNetwork net, int testStartIdx, int nTest) throws Exception {
    System.out.println("Starting training...");
    int nTrainEpochs = 15;
    for (int i = 0; i < nTrainEpochs; i++) {
      DataSetIterator trainData = getDataSetIterator(dataDirectory, 0, testStartIdx - 1, miniBatchSize);
      while (trainData.hasNext())
        net.fit(trainData.next());
      Nd4j.saveBinary(net.params(), new File("videomodel.bin"));
      FileUtils.writeStringToFile(new File("videoconf.json"), conf.toJson());
      System.out.println("Epoch " + i + " complete");

      performCrossValidation(net, testStartIdx, nTest, dataDirectory);
    }
  }

  private static void printNeuralNetworkParameters(MultiLayerNetwork net) {
    System.out.println("Number of parameters in network: " + net.numParams());
    for (int i = 0; i < net.getnLayers(); i++) {
      System.out.println("Layer " + i + " nParams = " + net.getLayer(i).numParams());
    }
  }

  private static MultiLayerConfiguration configureNeuralNetwork() {
    return new NeuralNetConfiguration.Builder()
        .seed(12345)
        .l2(0.001) //l2 regularization on all layers
        .updater(new AdaGrad(0.04))
        .list()
        .layer(0, new ConvolutionLayer.Builder(10, 10)
            .nIn(3) //3 channels: RGB
            .nOut(30)
            .stride(4, 4)
            .activation(Activation.RELU)
            .weightInit(WeightInit.RELU)
            .build())   //Output: (130-10+0)/4+1 = 31 -> 31*31*30
        .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
            .kernelSize(3, 3)
            .stride(2, 2).build())   //(31-3+0)/2+1 = 15
        .layer(2, new ConvolutionLayer.Builder(3, 3)
            .nIn(30)
            .nOut(10)
            .stride(2, 2)
            .activation(Activation.RELU)
            .weightInit(WeightInit.RELU)
            .build())   //Output: (15-3+0)/2+1 = 7 -> 7*7*10 = 490
        .layer(3, new DenseLayer.Builder()
            .activation(Activation.RELU)
            .nIn(490)
            .nOut(50)
            .weightInit(WeightInit.RELU)
            .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
            .gradientNormalizationThreshold(10)
            .updater(new AdaGrad(0.01))
            .build())
        .layer(4, new LSTM.Builder()
            .activation(Activation.SOFTSIGN)
            .nIn(50)
            .nOut(50)
            .weightInit(WeightInit.XAVIER)
            .updater(new AdaGrad(0.008))
            .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
            .gradientNormalizationThreshold(10)
            .build())
        .layer(5, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
            .activation(Activation.SOFTMAX)
            .nIn(50)
            .nOut(4)    //4 possible shapes: circle, square, arc, line
            .weightInit(WeightInit.XAVIER)
            .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
            .gradientNormalizationThreshold(10)
            .build())
        .inputPreProcessor(0, new RnnToCnnPreProcessor(V_HEIGHT, V_WIDTH, 3))
        .inputPreProcessor(3, new CnnToFeedForwardPreProcessor(7, 7, 10))
        .inputPreProcessor(4, new FeedForwardToRnnPreProcessor())
        .backpropType(BackpropType.TruncatedBPTT)
        .tBPTTForwardLength(V_NFRAMES / 5)
        .tBPTTBackwardLength(V_NFRAMES / 5)
        .build();
  }

  private static void generateVideoDataIfNeeded(boolean generateData, String dataDirectory) throws Exception {
    if (generateData) {
      System.out.println("Starting data generation...");
      generateData(dataDirectory);
      System.out.println("Data generation complete");
    }
  }

  private static void generateData(String path) throws Exception {
    File f = new File(path);
    if (!f.exists()) f.mkdir();

    VideoGenerator.generateVideoData(
        path,
        "shapes",
        N_VIDEOS_TO_GENERATE,
        V_NFRAMES, V_WIDTH, V_HEIGHT,
        3,
        false,
        0,
        12345L);
  }

  private static void performCrossValidation(MultiLayerNetwork net, int testStartIdx, int nExamples, String outputDirectory) throws Exception {
    Map<Integer, String> labelMap = new HashMap<>();
    labelMap.put(0, "circle");
    labelMap.put(1, "square");
    labelMap.put(2, "arc");
    labelMap.put(3, "line");
    Evaluation evaluation = new Evaluation(labelMap);

    DataSetIterator testData = getDataSetIterator(outputDirectory, testStartIdx, nExamples, 10);
    while (testData.hasNext()) {
      DataSet dsTest = testData.next();
      INDArray predicted = net.output(dsTest.getFeatures(), false);
      INDArray actual = dsTest.getLabels();
      evaluation.evalTimeSeries(actual, predicted);
    }

    System.out.println(evaluation.stats());
  }

  private static DataSetIterator getDataSetIterator(String dataDirectory, int startIdx, int nExamples, int miniBatchSize) throws Exception {
    SequenceRecordReader featuresTrain = getFeaturesReader(dataDirectory, startIdx, nExamples);
    SequenceRecordReader labelsTrain = getLabelsReader(dataDirectory, startIdx, nExamples);

    SequenceRecordReaderDataSetIterator sequenceIter =
        new SequenceRecordReaderDataSetIterator(featuresTrain, labelsTrain, miniBatchSize, 4, false);
    sequenceIter.setPreProcessor(new VideoPreProcessor());

    return new AsyncDataSetIterator(sequenceIter, 1);
  }

  private static SequenceRecordReader getFeaturesReader(String path, int startIdx, int num) throws Exception {
    InputSplit is = new NumberedFileInputSplit(path + "shapes_%d.mp4", startIdx, startIdx + num - 1);

    Configuration conf = new Configuration();
    conf.set(CodecRecordReader.RAVEL, "true");
    conf.set(CodecRecordReader.START_FRAME, "0");
    conf.set(CodecRecordReader.TOTAL_FRAMES, String.valueOf(V_NFRAMES));
    conf.set(CodecRecordReader.ROWS, String.valueOf(V_WIDTH));
    conf.set(CodecRecordReader.COLUMNS, String.valueOf(V_HEIGHT));
    CodecRecordReader crr = new CodecRecordReader();
    crr.initialize(conf, is);
    return crr;
  }

  private static SequenceRecordReader getLabelsReader(String path, int startIdx, int num) throws Exception {
    InputSplit isLabels = new NumberedFileInputSplit(path + "shapes_%d.txt", startIdx, startIdx + num - 1);
    CSVSequenceRecordReader csvSeq = new CSVSequenceRecordReader();
    csvSeq.initialize(isLabels);
    return csvSeq;
  }

  private static class VideoPreProcessor implements DataSetPreProcessor {
    @Override
    public void preProcess(org.nd4j.linalg.dataset.api.DataSet toPreProcess) {
      toPreProcess.getFeatures().divi(255);
    }
  }
}
