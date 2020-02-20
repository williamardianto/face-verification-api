package com.example.faceverification;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.springframework.stereotype.Service;

import java.io.File;
import java.io.IOException;
import java.nio.file.Paths;

@Service
public class FaceService {
    private static String modelLocation = Paths.get(System.getProperty("user.home"), ".deeplearning4j", "models", "faceverification", "vggface-resnet.h5").toString();
    ComputationGraph net = KerasModelImport.importKerasModelAndWeights(modelLocation);

    public FaceService() throws InvalidKerasConfigurationException, IOException, UnsupportedKerasConfigurationException {
    }

    public double getDistance(File f1, File f2) throws IOException {
        INDArray array1 = Utils.loadImage(f1);
        INDArray array2 = Utils.loadImage(f2);

        double distance = Utils.distance(
                Utils.getEmbedding(net, array1),
                Utils.getEmbedding(net, array2)
        );

        return distance;
    }
}
