package com.example.faceverification;

import org.bytedeco.opencv.presets.opencv_core;
import org.datavec.image.loader.NativeImageLoader;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.io.ClassPathResource;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class LFWDatasetIterator implements DataSetIterator {
    private File labelPath;
    private final File dataPath = new File(Paths.get(System.getProperty("user.home"), "data", "lfw").toString());

    private int batchSize;
    private int currentIteration=0;
    private int iteration;
    private int count;

    private List<List<String>> samePersonList = new ArrayList();
    private List<List<String>> diffPersonList = new ArrayList();

    private Random rand = new Random();


    private void readFile() throws IOException {

        BufferedReader reader = new BufferedReader(new FileReader(labelPath));
        reader.readLine(); // this will read the first line
        count = 0;
        String line;
        while ((line = reader.readLine()) != null) {
            String[] splits = line.split("\t");
            Path path1, path2;
            if (splits.length == 3) {
                path1 = Paths.get(dataPath.toString(), splits[0], splits[0] + "_000" + splits[1] + ".jpg");
                path2 = Paths.get(dataPath.toString(), splits[0], splits[0] + "_000" + splits[2] + ".jpg");
                if (Files.exists(path1)) {
                    List<String> samePerson = Arrays.asList(path1.toString(), path2.toString(), "1");
                    samePersonList.add(samePerson);
                }
            } else if (splits.length == 4) {
                path1 = Paths.get(dataPath.toString(), splits[0], splits[0] + "_000" + splits[1] + ".jpg");
                path2 = Paths.get(dataPath.toString(), splits[2], splits[2] + "_000" + splits[3] + ".jpg");
                if (Files.exists(path1)) {
                    List<String> diffPerson = Arrays.asList(path1.toString(), path2.toString(), "0");
                    diffPersonList.add(diffPerson);
                }
            }
            count++;
        }
    }


    public LFWDatasetIterator(File labelPath, int batchSize, int iteration) throws IOException {
        this.labelPath = labelPath;
        this.batchSize = batchSize;
        this.iteration = iteration;
        readFile();

        System.out.println(count);
    }

    private INDArray loadImage(String path) throws IOException {

        NativeImageLoader loader = new NativeImageLoader(224, 224, 3);
        INDArray image = loader.asMatrix(path);
        return image;
    }


    @Override
    public DataSet next(int bs) {
        List<List<String>> batch = new ArrayList<List<String>>();
        boolean toggle = true;
        while(batch.size() != batchSize){
            if(toggle){
                List<String> samePersonSample = samePersonList.get(rand.nextInt(samePersonList.size()));
                batch.add(samePersonSample);
            }else{
                List<String> diffPersonSample = diffPersonList.get(rand.nextInt(diffPersonList.size()));
                batch.add(diffPersonSample);
            }
            toggle = !toggle;
        }

        INDArray image1 = Nd4j.zeros(bs, 3, 224, 224);
        INDArray image2 = Nd4j.zeros(bs, 3, 224, 224);
        INDArray label = Nd4j.zeros(bs,1);

        for (int i = 0; i < bs; i++) {
            try {
                image1.put(
                        new INDArrayIndex[]{NDArrayIndex.point(i),NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all()},
                        loadImage( batch.get(i).get(0))
                );
                image2.put(
                        new INDArrayIndex[]{NDArrayIndex.point(i),NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all()},
                        loadImage( batch.get(i).get(1))
                );
                label.put(
                        new INDArrayIndex[]{NDArrayIndex.point(i),NDArrayIndex.all()},
                        Nd4j.valueArrayOf(1,1, Integer.valueOf(batch.get(i).get(2)))
                );

            } catch (IOException e) {
                e.printStackTrace();
            }
        }


        return new DataSet(image1, image2);
    }

    @Override
    public int inputColumns() {
        return 0;
    }

    @Override
    public int totalOutcomes() {
        return 0;
    }

    @Override
    public boolean resetSupported() {
        return true;
    }

    @Override
    public boolean asyncSupported() {
        return false;
    }

    @Override
    public void reset() {
        currentIteration=0;
    }

    @Override
    public int batch() {
        return batchSize;
    }

    @Override
    public void setPreProcessor(DataSetPreProcessor dataSetPreProcessor) {

    }

    @Override
    public DataSetPreProcessor getPreProcessor() {
        return null;
    }

    @Override
    public List<String> getLabels() {
        return null;
    }

    @Override
    public boolean hasNext() {
        return currentIteration<iteration;
    }

    @Override
    public DataSet next() {
        return next(batchSize);
    }
}
