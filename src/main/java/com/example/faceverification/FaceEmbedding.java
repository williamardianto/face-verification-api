package com.example.faceverification;

import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.io.ClassPathResource;

import java.io.IOException;
import java.nio.file.Paths;
import java.util.Map;

import static com.example.faceverification.Utils.distance;
import static com.example.faceverification.Utils.loadImage;

public class FaceEmbedding {
    private static String modelLocation = Paths.get(System.getProperty("user.home"), ".deeplearning4j", "models", "faceverification", "vggface-resnet.h5").toString();

    public static void main(String args[]) throws IOException, UnsupportedKerasConfigurationException, InvalidKerasConfigurationException {
        INDArray face1 = loadImage(new ClassPathResource("/face/m1.jpg").getFile());
        INDArray face2 = loadImage(new ClassPathResource("/face/k1.jpg").getFile());
        INDArray face3 = loadImage(new ClassPathResource("/face/m3.jpg").getFile());
        INDArray face4 = loadImage(new ClassPathResource("/face/y1.jpg").getFile());

        ComputationGraph net = KerasModelImport.importKerasModelAndWeights(modelLocation);
        System.out.println(net.summary());

        INDArray faceEmbedding1 = Utils.getEmbedding(net, face1);
        INDArray faceEmbedding2 = Utils.getEmbedding(net, face2);
        INDArray faceEmbedding3 = Utils.getEmbedding(net, face3);
        INDArray faceEmbedding4 = Utils.getEmbedding(net, face4);

        System.out.println(distance(faceEmbedding1, faceEmbedding1));
        System.out.println(distance(faceEmbedding1, faceEmbedding2));
        System.out.println(distance(faceEmbedding1, faceEmbedding3));
        System.out.println(distance(faceEmbedding1, faceEmbedding4));
        System.out.println(distance(faceEmbedding3, faceEmbedding4));
    }



//    public static INDArray standardizeOutput(INDArray input){
//        double mean = 4.048700594902039;
//        double std = 4.445333131154379;
//
//        return input.sub(mean).div(std);
//    }
//    public static INDArray minMaxNorm(INDArray input){
//        double min = 0.0;
//        double max = 53.0;
//
//        INDArray top = input.sub(min);
//        double bot = max-min;
//        return top.div(bot);
//    }
//
//    private static INDArray normalize(INDArray read) {
//        return read.div(255.0);
//    }
//
//

}
