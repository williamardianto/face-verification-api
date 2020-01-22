package com.example.faceverification;

import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.vertex.GraphVertex;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.reduce3.EuclideanDistance;
import org.nd4j.linalg.io.ClassPathResource;

import javax.security.auth.login.Configuration;
import java.io.File;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.Map;

public class FaceEmbedding {
    private static File modelLocation = new File(Paths.get(System.getProperty("user.home"), ".deeplearning4j", "models", "facenet.zip").toString());
    private static int[] inputShape = Model.inputShape;
    private static int embeddingSize = Model.embeddingSize;

    public static void main(String args[]) throws IOException {
        INDArray face1 = loadImage(new ClassPathResource("/face/Yao_Ming_0002.jpg").getFile().getAbsolutePath());
        INDArray face2 = loadImage(new ClassPathResource("/face/Yao_Ming_0008.jpg").getFile().getAbsolutePath());
        INDArray face3 = loadImage(new ClassPathResource("/face/Megawati_Sukarnoputri_0023.jpg").getFile().getAbsolutePath());
        INDArray face4 = loadImage(new ClassPathResource("/face/Megawati_Sukarnoputri_0028.jpg").getFile().getAbsolutePath());

//        ComputationGraph net = new ComputationGraph(Model.networkConfig()) ;
//        net.load(modelLocation, true);

        ComputationGraph net = ModelSerializer.restoreComputationGraph(modelLocation);

        INDArray faceEmbedding1 = getEmbedding(net, normalize(face1));
        INDArray faceEmbedding2 = getEmbedding(net, normalize(face2));
        INDArray faceEmbedding3 = getEmbedding(net, normalize(face3));
        INDArray faceEmbedding4 = getEmbedding(net, normalize(face4));


        System.out.println(distance(faceEmbedding1, faceEmbedding2));
        System.out.println(distance(faceEmbedding1, faceEmbedding3));
        System.out.println(distance(faceEmbedding1, faceEmbedding4));
    }

    private static INDArray loadImage(String path) throws IOException {

        NativeImageLoader loader = new NativeImageLoader(inputShape[1], inputShape[2], inputShape[0]);
        INDArray image = loader.asMatrix(path);
        return image;
    }

    public static INDArray getEmbedding(ComputationGraph net, INDArray feature){
        Map<String, INDArray> feedForward = net.feedForward(feature, false);
        return feedForward.get("embeddings");
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
        return a.distance2(b);
    }
}
