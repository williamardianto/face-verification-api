package com.example.faceverification;

import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.iterator.impl.LFWDataSetIterator;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.graph.L2NormalizeVertex;
import org.deeplearning4j.nn.conf.layers.CenterLossOutputLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.InvocationType;
import org.deeplearning4j.optimize.listeners.EvaluativeListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;

import java.io.File;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.Random;

import org.datavec.image.loader.LFWLoader;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.SqueezeNet;
import org.deeplearning4j.zoo.model.VGG16;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;

public class Train {
    private static final Logger log = org.slf4j.LoggerFactory.getLogger(Train.class);

    private static File modelLocation = new File(Paths.get(System.getProperty("user.home"), ".deeplearning4j", "models", "facenet.zip").toString());
    private static int batchSize = 8; // depending on your hardware, you will want to increase or decrease
    private static int numExamples = LFWLoader.NUM_IMAGES;
    private static int outputNum = 187; // number of "identities" in the dataset
    private static double splitTrainTest = 1.0;
    private static int randomSeed = 123;

    private static int[] inputShape = new int[]{3, 96, 96};

    private static int[] inputWHC = new int[]{inputShape[2], inputShape[1], inputShape[0]};

    //https://github.com/eclipse/deeplearning4j-examples/blob/master/tutorials/07.%20Convolutions-%20Train%20FaceNet%20Using%20Center%20Loss.zepp.ipynb

    public static void main(String args[]) throws IOException {
//        File trainSet = new ClassPathResource("/pairsDevTrain.txt").getFile();
//        LFWDatasetIterator iterator = new LFWDatasetIterator(trainSet,5, 10);
//
//        iterator.next();
        Nd4j.setDefaultDataTypes(DataType.DOUBLE, DataType.DOUBLE);

        LFWDataSetIterator trainIter = new LFWDataSetIterator(batchSize, numExamples, inputWHC, outputNum, false, true, splitTrainTest, new Random(randomSeed));
        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
        trainIter.setPreProcessor(scaler);

        ComputationGraph net = Model.getNetwork();

        log.info(net.summary());

        UIServer server = UIServer.getInstance();
        StatsStorage storage = new InMemoryStatsStorage();
        server.attach(storage);
        net.setListeners(
                new ScoreIterationListener(2),
                new StatsListener(storage),
                new EvaluativeListener(trainIter, 1, InvocationType.EPOCH_END)
        );

//        System.out.println(net.summary());

        for (int i = 0; i < 100; i++) {
            net.fit(trainIter);

            log.info("Epoch: {}", (i+1));
//            log.info("Accuracy: " + squeezeNet.evaluate(trainIter).stats());
//            log.info("Precision: " + squeezeNet.evaluate(trainIter).precision());
        }

        ComputationGraph snippedNet = snipNetwork(net);
        saveModel(snippedNet, modelLocation);

    }

    private static ComputationGraph snipNetwork(ComputationGraph net){
        ComputationGraph snipped = new TransferLearning.GraphBuilder(net)
                .setFeatureExtractor("embeddings") // the L2Normalize vertex and layers below are frozen
                .removeVertexAndConnections("lossLayer")
                .setOutputs("embeddings")
                .build();

        return snipped;
    }

    private static void saveModel(ComputationGraph net, File location) throws IOException {
        ModelSerializer.writeModel(net, location, true);

//        net.save(location, true);
        log.info("Model saved");
    }





}

