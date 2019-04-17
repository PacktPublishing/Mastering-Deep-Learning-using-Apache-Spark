package chapter_2;

import org.apache.commons.io.FilenameUtils;
import org.jcodec.api.SequenceEncoder;

import java.awt.*;
import java.awt.geom.Arc2D;
import java.awt.geom.Ellipse2D;
import java.awt.geom.Line2D;
import java.awt.geom.Rectangle2D;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.Random;


class VideoGenerator {

  private static final int NUM_SHAPES = 4;  //0=circle, 1=square, 2=arc, 3=line
  private static final int MAX_VELOCITY = 3;
  private static final int SHAPE_SIZE = 25;
  private static final int SHAPE_MIN_DIST_FROM_EDGE = 15;
  private static final int DISTRACTOR_MIN_DIST_FROM_EDGE = 0;
  private static final int LINE_STROKE_WIDTH = 6;  //Width of line (line shape only)
  private static final BasicStroke lineStroke = new BasicStroke(LINE_STROKE_WIDTH);
  private static final int MIN_FRAMES = 10;    //Minimum number of frames the target shape to be present
  private static final float MAX_NOISE_VALUE = 0.5f;

  private static int[] generateVideo(String path, int nFrames, int width, int height, int numShapes, Random r,
                                     boolean backgroundNoise, int numDistractorsPerFrame) throws Exception {


    int[] startFrames = findTranstionsBetweenShapesCoordinates(nFrames, numShapes, r);

    ShapePositionsGenerator shapePositionsGenerator = new ShapePositionsGenerator(width, height, numShapes, r).invoke();

    return generateAndSaveImagesReturningLabels(path, nFrames, width, height, numShapes, r, backgroundNoise, numDistractorsPerFrame, startFrames, shapePositionsGenerator);
  }

  private static int[] generateAndSaveImagesReturningLabels(String path, int nFrames, int width, int height, int numShapes, Random r, boolean backgroundNoise, int numDistractorsPerFrame, int[] startFrames, ShapePositionsGenerator shapePositionsGenerator) throws IOException {
    SequenceEncoder enc = new SequenceEncoder(new File(path));
    int currShape = 0;
    int[] labels = new int[nFrames];
    for (int i = 0; i < nFrames; i++) {
      if (currShape < numShapes - 1 && i >= startFrames[currShape + 1]) currShape++;

      BufferedImage bi = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
      Graphics2D g2d = bi.createGraphics();
      g2d.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
      g2d.setBackground(Color.BLACK);

      if (backgroundNoise) {
        for (int x = 0; x < width; x++) {
          for (int y = 0; y < height; y++) {
            bi.setRGB(x, y, new Color(r.nextFloat() * MAX_NOISE_VALUE, r.nextFloat() * MAX_NOISE_VALUE, r.nextFloat() * MAX_NOISE_VALUE).getRGB());
          }
        }
      }

      g2d.setColor(shapePositionsGenerator.color[currShape]);

      //Position of shape this frame
      int currX = (int) (shapePositionsGenerator.initialX[currShape] + (i - startFrames[currShape]) * shapePositionsGenerator.velocityX[currShape] * MAX_VELOCITY);
      int currY = (int) (shapePositionsGenerator.initialY[currShape] + (i - startFrames[currShape]) * shapePositionsGenerator.velocityY[currShape] * MAX_VELOCITY);

      //Render the shape
      switch (shapePositionsGenerator.shapeTypes[currShape]) {
        case 0:
          //Circle
          g2d.fill(new Ellipse2D.Double(currX, currY, SHAPE_SIZE, SHAPE_SIZE));
          break;
        case 1:
          //Square
          g2d.fill(new Rectangle2D.Double(currX, currY, SHAPE_SIZE, SHAPE_SIZE));
          break;
        case 2:
          //Arc
          g2d.fill(new Arc2D.Double(currX, currY, SHAPE_SIZE, SHAPE_SIZE, 315, 225, Arc2D.PIE));
          break;
        case 3:
          //Line
          g2d.setStroke(lineStroke);
          g2d.draw(new Line2D.Double(currX, currY, currX + SHAPE_SIZE, currY + SHAPE_SIZE));
          break;
        default:
          throw new RuntimeException();
      }

      //Add some distractor shapes, which are present for one frame only
      for (int j = 0; j < numDistractorsPerFrame; j++) {
        int distractorShapeIdx = r.nextInt(NUM_SHAPES);

        int distractorX = DISTRACTOR_MIN_DIST_FROM_EDGE + r.nextInt(width - SHAPE_SIZE);
        int distractorY = DISTRACTOR_MIN_DIST_FROM_EDGE + r.nextInt(height - SHAPE_SIZE);

        g2d.setColor(new Color(r.nextFloat(), r.nextFloat(), r.nextFloat()));

        switch (distractorShapeIdx) {
          case 0:
            g2d.fill(new Ellipse2D.Double(distractorX, distractorY, SHAPE_SIZE, SHAPE_SIZE));
            break;
          case 1:
            g2d.fill(new Rectangle2D.Double(distractorX, distractorY, SHAPE_SIZE, SHAPE_SIZE));
            break;
          case 2:
            g2d.fill(new Arc2D.Double(distractorX, distractorY, SHAPE_SIZE, SHAPE_SIZE, 315, 225, Arc2D.PIE));
            break;
          case 3:
            g2d.setStroke(lineStroke);
            g2d.draw(new Line2D.Double(distractorX, distractorY, distractorX + SHAPE_SIZE, distractorY + SHAPE_SIZE));
            break;
          default:
            throw new RuntimeException();
        }
      }

      enc.encodeImage(bi);
      g2d.dispose();
      labels[i] = shapePositionsGenerator.shapeTypes[currShape];
    }
    enc.finish();   //write .mp4
    return labels;
  }

  private static int[] findTranstionsBetweenShapesCoordinates(int nFrames, int numShapes, Random r) {
    double[] rns = new double[numShapes];
    double sum = 0;
    for (int i = 0; i < numShapes; i++) {
      rns[i] = r.nextDouble();
      sum += rns[i];
    }
    for (int i = 0; i < numShapes; i++) rns[i] /= sum;

    int[] startFrames = new int[numShapes];
    startFrames[0] = 0;
    for (int i = 1; i < numShapes; i++) {
      startFrames[i] = (int) (startFrames[i - 1] + MIN_FRAMES + rns[i] * (nFrames - numShapes * MIN_FRAMES));
    }
    return startFrames;
  }

  public static void generateVideoData(String outputFolder, String filePrefix, int nVideos, int nFrames,
                                       int width, int height, int numShapesPerVideo, boolean backgroundNoise,
                                       int numDistractorsPerFrame, long seed) throws Exception {
    Random r = new Random(seed);

    for (int i = 0; i < nVideos; i++) {
      String videoPath = generateVideoPath(outputFolder, filePrefix, i, ".mp4");
      String labelsPath = generateVideoPath(outputFolder, filePrefix, i, ".txt");
      System.out.println("video path: " + videoPath);
      System.out.println("labels path: " + labelsPath);
      int[] labels = generateVideo(videoPath, nFrames, width, height, numShapesPerVideo, r, backgroundNoise, numDistractorsPerFrame);

      writeLabelsToTextFile(labelsPath, labels);
    }
  }

  private static void writeLabelsToTextFile(String labelsPath, int[] labels) throws IOException {
    StringBuilder sb = new StringBuilder();
    for (int j = 0; j < labels.length; j++) {
      sb.append(labels[j]);
      if (j != labels.length - 1) sb.append("\n");
    }
    Files.write(Paths.get(labelsPath), sb.toString().getBytes("utf-8"), StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING);
  }

  private static String generateVideoPath(String outputFolder, String filePrefix, int i, String s) {
    return FilenameUtils.concat(outputFolder, filePrefix + "_" + i + s);
  }

  private static class ShapePositionsGenerator {
    private int width;
    private int height;
    private int numShapes;
    private Random r;
    private int[] shapeTypes;
    private int[] initialX;
    private int[] initialY;
    private double[] velocityX;
    private double[] velocityY;
    private Color[] color;

    public ShapePositionsGenerator(int width, int height, int numShapes, Random r) {
      this.width = width;
      this.height = height;
      this.numShapes = numShapes;
      this.r = r;
    }

    public int[] getShapeTypes() {
      return shapeTypes;
    }

    public int[] getInitialX() {
      return initialX;
    }

    public int[] getInitialY() {
      return initialY;
    }

    public double[] getVelocityX() {
      return velocityX;
    }

    public double[] getVelocityY() {
      return velocityY;
    }

    public Color[] getColor() {
      return color;
    }

    public ShapePositionsGenerator invoke() {
      shapeTypes = new int[numShapes];
      initialX = new int[numShapes];
      initialY = new int[numShapes];
      velocityX = new double[numShapes];
      velocityY = new double[numShapes];
      color = new Color[numShapes];
      for (int i = 0; i < numShapes; i++) {
        shapeTypes[i] = r.nextInt(NUM_SHAPES);
        initialX[i] = SHAPE_MIN_DIST_FROM_EDGE + r.nextInt(width - SHAPE_SIZE - 2 * SHAPE_MIN_DIST_FROM_EDGE);
        initialY[i] = SHAPE_MIN_DIST_FROM_EDGE + r.nextInt(height - SHAPE_SIZE - 2 * SHAPE_MIN_DIST_FROM_EDGE);
        velocityX[i] = -1 + 2 * r.nextDouble();
        velocityY[i] = -1 + 2 * r.nextDouble();
        color[i] = new Color(r.nextFloat(), r.nextFloat(), r.nextFloat());
      }
      return this;
    }
  }
}
