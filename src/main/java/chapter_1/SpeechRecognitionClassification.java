package chapter_1;

import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.paragraphvectors.ParagraphVectors;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.text.documentiterator.FileLabelAwareIterator;
import org.deeplearning4j.text.documentiterator.LabelAwareIterator;
import org.deeplearning4j.text.documentiterator.LabelledDocument;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.jetbrains.annotations.NotNull;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.primitives.Pair;

import java.io.IOException;
import java.util.List;

//todo

/**
 * Build a model for labeled text data
 * Next classify documents from different business domains into those classes
 */
public class SpeechRecognitionClassification {

  private ParagraphVectors paragraphVectors;
  private LabelAwareIterator iterator;
  private TokenizerFactory tokenizerFactory;


  public static void main(String[] args) throws Exception {

    SpeechRecognitionClassification app = new SpeechRecognitionClassification();
    app.makeParagraphVectors();
    app.classifyUnlabeledDataIntoClasses();
  }

  void makeParagraphVectors() throws Exception {
    ClassPathResource resource = new ClassPathResource("text_from_business_domains/labeled");
    System.out.println("load from: " + resource.getPath());
    loadLabeledData(resource);

    tokenizerFactory = new DefaultTokenizerFactory();
    tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());

    extractFeatureVectorFromSpeechData();

    paragraphVectors.fit();
  }

  private void extractFeatureVectorFromSpeechData() {
    paragraphVectors = new ParagraphVectors.Builder()
        .learningRate(0.025)
        .minLearningRate(0.001)
        .batchSize(1000)
        .epochs(20)
        .iterate(iterator)
        .trainWordVectors(true)
        .tokenizerFactory(tokenizerFactory)
        .build();
  }

  private void loadLabeledData(ClassPathResource resource) throws IOException {
    iterator = new FileLabelAwareIterator.Builder()
        .addSourceFolder(resource.getFile())
        .build();
  }

  void classifyUnlabeledDataIntoClasses() throws IOException {
    ClassPathResource unClassifiedResource = new ClassPathResource("text_from_business_domains/unlabeled");
    FileLabelAwareIterator unClassifiedIterator = new FileLabelAwareIterator.Builder()
        .addSourceFolder(unClassifiedResource.getFile())
        .build();


    CentroidVectorBuilder centroidVectorBuilder = buildFeatureVectorForUnlabeledTestData();
    NearestLabelFinder seeker = new NearestLabelFinder(
        iterator.getLabelsSource().getLabels(),
        (InMemoryLookupTable<VocabWord>) paragraphVectors.getLookupTable()
    );

    while (unClassifiedIterator.hasNextDocument()) {
      assignDocumentIntoClasses(unClassifiedIterator, centroidVectorBuilder, seeker);
    }

  }

  private void assignDocumentIntoClasses(FileLabelAwareIterator unClassifiedIterator, CentroidVectorBuilder centroidVectorBuilder, NearestLabelFinder seeker) {
    LabelledDocument document = unClassifiedIterator.nextDocument();
    INDArray documentAsCentroid = centroidVectorBuilder.documentAsVector(document);
    List<Pair<String, Double>> scores = seeker.getScores(documentAsCentroid);
    System.out.println("Document '" + document.getLabels() + "' falls into the following categories: ");
    for (Pair<String, Double> score : scores) {
      System.out.println("        " + score.getFirst() + ": " + score.getSecond());
    }
  }

  @NotNull
  private CentroidVectorBuilder buildFeatureVectorForUnlabeledTestData() {
    return new CentroidVectorBuilder(
        (InMemoryLookupTable<VocabWord>) paragraphVectors.getLookupTable(),
        tokenizerFactory);
  }
}
