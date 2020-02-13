package com.example.faceverification;

import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.vertex.GraphVertex;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.reduce3.EuclideanDistance;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.ops.transforms.Transforms;

import javax.security.auth.login.Configuration;
import java.io.File;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.Map;

public class FaceEmbedding {
    private static File modelLocation = new File(Paths.get(System.getProperty("user.home"), ".deeplearning4j", "models", "facenet3.zip").toString());
    private static int[] inputShape = new int[]{3, 224, 224};
    private static int embeddingSize = Model.embeddingSize;

    public static void main(String args[]) throws IOException, UnsupportedKerasConfigurationException, InvalidKerasConfigurationException {
        INDArray face1 = loadImage(new ClassPathResource("/face/m1.jpg").getFile().getAbsolutePath());
        INDArray face2 = loadImage(new ClassPathResource("/face/k1.jpg").getFile().getAbsolutePath());
        INDArray face3 = loadImage(new ClassPathResource("/face/j1.jpg").getFile().getAbsolutePath());
        INDArray face4 = loadImage(new ClassPathResource("/face/y1.jpg").getFile().getAbsolutePath());

//        ComputationGraph net = new ComputationGraph(Model.networkConfig()) ;
//        net.load(modelLocation, true);

//        ComputationGraph net = ModelSerializer.restoreComputationGraph(modelLocation);
        String PATH = new ClassPathResource("/model/vggface-resnet.h5").getFile().getAbsolutePath();
        ComputationGraph net = KerasModelImport.importKerasModelAndWeights(PATH);
        System.out.println(net.summary());

        INDArray faceEmbedding1 = getEmbedding(net, preProcessInput(face1));
        INDArray faceEmbedding2 = getEmbedding(net, preProcessInput(face2));
        INDArray faceEmbedding3 = getEmbedding(net, preProcessInput(face3));
        INDArray faceEmbedding4 = getEmbedding(net, preProcessInput(face4));


//        System.out.println(distance(faceEmbedding1, faceEmbedding1));
        System.out.println(distance(faceEmbedding1, faceEmbedding2));
        System.out.println(distance(faceEmbedding1, faceEmbedding3));
        System.out.println(distance(faceEmbedding1, faceEmbedding4));
//        System.out.println(distance(faceEmbedding3, faceEmbedding4));
    }

    private static INDArray loadImage(String path) throws IOException {

        NativeImageLoader loader = new NativeImageLoader(inputShape[1], inputShape[2], inputShape[0]);
        INDArray image = loader.asMatrix(path);
        return image;
    }

    public static INDArray getEmbedding(ComputationGraph net, INDArray feature){
        Map<String, INDArray> feedForward = net.feedForward(feature, false);
        return feedForward.get("global_average_pooling2d_2");
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

    public static INDArray standardizeOutput(INDArray input){
        double mean = 4.048700594902039;
        double std = 4.445333131154379;

        return input.sub(mean).div(std);
    }
    public static INDArray minMaxNorm(INDArray input){
        double min = 0.0;
        double max = 53.0;

        INDArray top = input.sub(min);
        double bot = max-min;
        return top.div(bot);
    }

//    private INDArray forwardPass(INDArray indArray) {
//        ComputationGraph computationGraph = new ComputationGraph(Model.networkConfig());
//        Map<String, INDArray> output = computationGraph.feedForward(indArray, false);
//        GraphVertex embeddings = computationGraph.getVertex("encodings");
//        INDArray dense = output.get("dense");
//        embeddings.setInputs(dense);
//        INDArray embeddingValues = embeddings.doForward(false, LayerWorkspaceMgr.builder().defaultNoWorkspace().build());
//        log.debug("dense =                 " + dense);
//        log.debug("encodingsValues =                 " + embeddingValues);
//        return embeddingValues;
//    }
    private static INDArray normalize(INDArray read) {
        return read.div(255.0);
    }


    private static double distance(INDArray a, INDArray b) {
        return a.distance2(b) / 100.0;
//        return Math.pow((a.distance2(b)/100.0)+1, -1);
//        return Transforms.cosineSim(a, b);
    }
}
