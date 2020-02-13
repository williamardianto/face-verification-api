package com.example.faceverification;

import org.bytedeco.javacv.FrameFilter;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.io.ClassPathResource;

import java.io.File;
import java.io.IOException;
import java.nio.DoubleBuffer;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

public class LoadONNX {
    private static int[] inputShape = new int[]{3, 224, 224};
    public static void main(String args[]) throws IOException, UnsupportedKerasConfigurationException, InvalidKerasConfigurationException, Exception {

        String PATH = new ClassPathResource("/model/vggface-resnet.h5").getFile().getAbsolutePath();
        ComputationGraph net = KerasModelImport.importKerasModelAndWeights(PATH);

        String files = new ClassPathResource("/face").getFile().getAbsolutePath();
        File face = new File(files);

        File[] faces = face.listFiles();

        List<Double> mean = new ArrayList<>();
        List<Double> std = new ArrayList<>();
        List<Double> min = new ArrayList<>();
        List<Double> max = new ArrayList<>();

        for (int i = 0; i < faces.length; i++) {
            try {
                INDArray input =  preProcessInput(loadImage(faces[i].getAbsolutePath()));
                INDArray output = getEmbedding(net, input);
                mean.add(output.meanNumber().doubleValue());
                std.add(output.stdNumber().doubleValue());
                min.add(output.minNumber().doubleValue());
                max.add(output.maxNumber().doubleValue());
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        System.out.println(calculateAverage(mean));
        System.out.println(calculateAverage(std));
        System.out.println(min);
        System.out.println(max);

//        4.048700594902039 -> mean
//        4.445333131154379  -> std
        //max = 53
        //min = 0




//        INDArray sample = loadImage(new ClassPathResource("/mnist/four.png").getFile().getAbsolutePath());
//        String PATH = new ClassPathResource("/model/mnist_test.pb").getFile().getAbsolutePath();
//
//        SameDiff graph = TFGraphMapper.importGraph(new File(PATH));
//
//        graph.summary();
//        String PATH = new ClassPathResource("/model/vggface-resnet.h5").getFile().getAbsolutePath();
//        ComputationGraph model = KerasModelImport.importKerasModelAndWeights(PATH);
//        System.out.println(model.summary());

    }

    private static double calculateAverage(List <Double> marks) {
        double sum = 0;
        if(!marks.isEmpty()) {
            for (Double mark : marks) {
                sum += mark;
            }
            return sum / marks.size();
        }
        return sum;
    }

    private static INDArray loadImage(String path) throws IOException {

        NativeImageLoader loader = new NativeImageLoader(inputShape[1], inputShape[2], inputShape[0]);
        INDArray image = loader.asMatrix(path);
        return image;
    }

    public static INDArray getEmbedding(ComputationGraph net, INDArray feature){
        Map<String, INDArray> feedForward = net.feedForward(feature, false);
        return feedForward.get("global_average_pooling2d_1");
    }


    private static INDArray preProcessInput(INDArray input){
        INDArray getChannel1 = input.get(NDArrayIndex.all(), NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.all());
        INDArray getChannel2 = input.get(NDArrayIndex.all(), NDArrayIndex.point(1), NDArrayIndex.all(), NDArrayIndex.all());
        INDArray getChannel3 = input.get(NDArrayIndex.all(), NDArrayIndex.point(2), NDArrayIndex.all(), NDArrayIndex.all());

        input.put(
                new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.all()},
                getChannel1.subi(93.5940)
        );
        input.put(
                new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.point(1), NDArrayIndex.all(), NDArrayIndex.all()},
                getChannel2.subi(104.7624)
        );
        input.put(
                new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.point(2), NDArrayIndex.all(), NDArrayIndex.all()},
                getChannel3.subi(129.1863)
        );

        return  input;
    }
}
