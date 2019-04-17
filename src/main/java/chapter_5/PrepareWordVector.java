package chapter_5;

import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;

public class PrepareWordVector {

    private static Logger log = LoggerFactory.getLogger(PrepareWordVector.class);

    public static void main(String[] args) throws Exception {
        String classPathResource = new ClassPathResource("NewsData").getFile().getAbsolutePath() + File.separator;
        String filePath = new File(classPathResource + File.separator + "RawNewsToGenerateWordVector.txt").getAbsolutePath();
        System.out.println("loading news from: " + filePath);

        log.info("Load & Vectorize Sentences....");

        SentenceIterator iter = new BasicLineIterator(filePath);
        TokenizerFactory t = new DefaultTokenizerFactory();
        removeSpecialCharactersAndWhitespaces(t);

        Word2Vec vec = buildWord2Vec(iter, t);

        log.info("Fitting Word2Vec model....");
        vec.fit();

        log.info("Writing word vectors to text file....");

        writeVectorToFile(classPathResource, vec);
    }

    private static void removeSpecialCharactersAndWhitespaces(TokenizerFactory t) {
        t.setTokenPreProcessor(new CommonPreprocessor());
    }

    private static Word2Vec buildWord2Vec(SentenceIterator iter, TokenizerFactory t) {
        log.info("Building model....");
        return new Word2Vec.Builder()
            .minWordFrequency(2)
            .iterations(5)
            .layerSize(100)
            .seed(42)
            .windowSize(20)
            .iterate(iter)
            .tokenizerFactory(t)
            .build();
    }

    private static void writeVectorToFile(String classPathResource, Word2Vec vec) throws java.io.IOException {
        String outputPath = classPathResource + "NewsWordVector.txt";
        System.out.println("saving news transformed into feature vector to outputPath: " + outputPath);
        WordVectorSerializer.writeWordVectors(vec.lookupTable(), outputPath);
    }
}
