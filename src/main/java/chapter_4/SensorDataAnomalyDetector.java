package chapter_4;


import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.jetbrains.annotations.NotNull;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.dataset.api.preprocessor.serializer.NormalizerSerializer;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.RmsProp;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Queue;
import java.util.Stack;

/**
 * Detection anomaly data from sequence data from sensors based on multiple features
 */
public class SensorDataAnomalyDetector {

  private static int trainBatchSize = 64;
  private static int testBatchSize = 1;
  private static int numEpochs = 10;

  public static void main(String[] args) throws Exception {

    String dataPath = new ClassPathResource("/anomaly_sensor_data").getFile().getPath();
    File modelFile = new File(dataPath + File.separatorChar + "anomalyDetectionModel.gz");
    DataSetIterator trainIterator = trainDataSetIterator(dataPath);
    DataSetIterator testIterator = testDataSetIterator(dataPath);

    MultiLayerNetwork net = createModel(trainIterator.inputColumns(), trainIterator.totalOutcomes());

    prepareTraining(dataPath, trainIterator, testIterator);


    for (int i = 0; i < numEpochs; i++) {
      System.out.println("=============numEpochs==========================" + i);
      net.fit(trainIterator);
    }
    ModelSerializer.writeModel(net, modelFile, true);

    Stack<String> anomalyData = findOutliers(testIterator, net);
    printAnomalyData(anomalyData);

  }

  private static void printAnomalyData(Stack<String> anomalyData) {
    for (int i = anomalyData.size(); i > 0; i--) {
      System.out.println(anomalyData.pop());
    }
  }

  @NotNull
  private static Stack<String> findOutliers(DataSetIterator testIterator, MultiLayerNetwork net) {
    List<Pair<Double, String>> evalList = new ArrayList<>();
    Queue<String> queue = ((AnomalyDataSetIterator) testIterator).getCurrentLines();
    double totalScore = 0;
    while (testIterator.hasNext()) {
      DataSet ds = testIterator.next();
      double score = net.score(ds);
      String currentLine = queue.poll();
      totalScore += score;
      evalList.add(new ImmutablePair<>(score, currentLine));
    }

    Collections.sort(evalList, Comparator.comparing(Pair::getLeft));
    Stack<String> anomalyData = new Stack<>();
    double threshold = totalScore / evalList.size();
    for (Pair<Double, String> pair : evalList) {
      double s = pair.getLeft();
      if (s > threshold) {
        anomalyData.push(pair.getRight());
      }
    }
    return anomalyData;
  }

  private static void prepareTraining(String dataPath, DataSetIterator trainIterator, DataSetIterator testIterator) throws IOException {
    DataNormalization normalizer = new NormalizerStandardize();
    normalizer.fit(trainIterator);
    trainIterator.reset();
    trainIterator.setPreProcessor(normalizer);
    testIterator.setPreProcessor(normalizer);
    NormalizerSerializer.getDefault().write(normalizer, dataPath + File.separatorChar + "anomalyDetectionNormlizer.ty");
  }

  @NotNull
  private static void configureUI(MultiLayerNetwork net) {
    UIServer uiServer = UIServer.getInstance();
    StatsStorage statsStorage = new InMemoryStatsStorage();
    uiServer.attach(statsStorage);
    net.setListeners(new StatsListener(statsStorage), new ScoreIterationListener(10));
  }

  @NotNull
  private static AnomalyDataSetIterator testDataSetIterator(String dataPath) {
    return new AnomalyDataSetIterator(dataPath + File.separatorChar + "test.csv", testBatchSize);
  }

  @NotNull
  private static AnomalyDataSetIterator trainDataSetIterator(String dataPath) {
    return new AnomalyDataSetIterator(dataPath + File.separatorChar + "ads.csv", trainBatchSize);
  }

  public static MultiLayerNetwork createModel(int inputNum, int outputNum) {
    MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
        .trainingWorkspaceMode(WorkspaceMode.ENABLED).inferenceWorkspaceMode(WorkspaceMode.ENABLED)
        .seed(123456)
        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
        .updater(new RmsProp.Builder().learningRate(0.05).rmsDecay(0.002).build())
        .l2(0.0005)
        .weightInit(WeightInit.XAVIER)
        .activation(Activation.TANH)
        .list()
        .layer(0, new LSTM.Builder().name("encoder0").nIn(inputNum).nOut(100).activation(Activation.TANH).build())
        .layer(1, new LSTM.Builder().name("encoder1").nOut(80).activation(Activation.TANH).build())
        .layer(2, new LSTM.Builder().name("encoder2").nOut(5).activation(Activation.TANH).build())
        .layer(3, new LSTM.Builder().name("decoder1").nOut(80).activation(Activation.TANH).build())
        .layer(4, new LSTM.Builder().name("decoder2").nOut(100).activation(Activation.TANH).build())
        .layer(5, new RnnOutputLayer.Builder().name("output").nOut(outputNum)
            .activation(Activation.IDENTITY).lossFunction(LossFunctions.LossFunction.MSE).build())
        .build();
    MultiLayerNetwork net = new MultiLayerNetwork(conf);
    net.init();
    return net;
  }

}
