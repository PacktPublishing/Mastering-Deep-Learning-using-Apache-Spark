package chapter1;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class ND4JTest {
  private static int NUMBER_OF_ROWS = 5;
  private static int NUMBER_OF_COLUMNS = 6;
  private static int[] shape = new int[]{NUMBER_OF_ROWS, NUMBER_OF_COLUMNS};

  @Test
  public void shouldPopulateArray() {
    int nRows = 4;
    int nColumns = 10;
    INDArray myArray = Nd4j.zeros(nRows, nColumns);
    double val0 = myArray.getDouble(0, 1);
    System.out.println("\nValue at (0,1):     " + val0);

    INDArray myArray2 = myArray.add(1.0);
    System.out.println("\nNew INDArray, after adding 1.0 to each entry:");
    System.out.println(myArray2);

    INDArray myArray3 = myArray2.mul(2.0);
    System.out.println("\nNew INDArray, after multiplying each entry by 2.0:");
    System.out.println(myArray3);
  }


  @Test
  public void shouldCreateRandomArray() {
    INDArray uniformRandom = Nd4j.rand(shape);
    System.out.println("\n\n\nUniform random array:");
    System.out.println(uniformRandom);
    System.out.println("Full precision of random value at position (0,0): " + uniformRandom.getDouble(0, 0));

    INDArray gaussianMeanZeroUnitVariance = Nd4j.randn(shape);
    System.out.println("\nN(0,1) random array:");
    System.out.println(gaussianMeanZeroUnitVariance);
  }

  private static INDArray THREE_BY_TWO_RANDOM = Nd4j.rand(new int[]{3, 2});

  @Test
  public void shouldCalculateMeanOnDimensionZero() {
    INDArray mean = Nd4j.mean(THREE_BY_TWO_RANDOM, 0);
    System.out.println("Mean on dimension zero: " + mean);
  }

  @Test
  public void shouldCombineArrays() {
    //We can create INDArrays by combining other INDArrays, too:
    INDArray rowVector1 = Nd4j.create(new double[]{1, 2, 3});
    INDArray rowVector2 = Nd4j.create(new double[]{4, 5, 6});

    INDArray vStack = Nd4j.vstack(rowVector1, rowVector2);      //Vertical stack:   [1,3]+[1,3] to [2,3]
    INDArray hStack = Nd4j.hstack(rowVector1, rowVector2);      //Horizontal stack: [1,3]+[1,3] to [1,6]
    System.out.println("\n\n\nCreating INDArrays from other INDArrays, using hstack and vstack:");
    System.out.println("vStack:\n" + vStack);
    System.out.println("hStack:\n" + hStack);
  }

}
