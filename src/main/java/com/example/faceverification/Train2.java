package com.example.faceverification;

import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.iterator.impl.LFWDataSetIterator;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.optimize.api.InvocationType;
import org.deeplearning4j.optimize.listeners.EvaluativeListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;

import java.io.File;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class Train2 {
    private static final Logger log = org.slf4j.LoggerFactory.getLogger(Train2.class);

    private static File modelLocation = new File(Paths.get(System.getProperty("user.home"), ".deeplearning4j", "models", "facenet3.zip").toString());
    private static int batchSize = 8; // depending on your hardware, you will want to increase or decrease

    private static int outputNum = 30; // number of "identities" in the dataset
    private static int numExamples = outputNum*6;

    private static double splitTrainTest = 1.0;
    private static int randomSeed = 123;

    private static int[] inputShape = new int[]{3, 96, 96};

    private static int[] inputWHC = new int[]{inputShape[2], inputShape[1], inputShape[0]};

    static DataSet lfwNext;
    static SplitTestAndTrain trainTest;
    static DataSet trainInput;
    static List<INDArray> testInput = new ArrayList<>();
    static List<INDArray> testLabels = new ArrayList<>();
    static int splitTrainNum = (int) (6 * .8);

    //https://github.com/eclipse/deeplearning4j-examples/blob/master/tutorials/07.%20Convolutions-%20Train%20FaceNet%20Using%20Center%20Loss.zepp.ipynb

    public static void main(String args[]) throws IOException {
//        File trainSet = new ClassPathResource("/pairsDevTrain.txt").getFile();
//        LFWDatasetIterator iterator = new LFWDatasetIterator(trainSet,5, 10);
//
//        iterator.next();
        Nd4j.setDefaultDataTypes(DataType.DOUBLE, DataType.DOUBLE);

        LFWDataSetIterator trainIter = new LFWDataSetIterator(batchSize, numExamples, inputWHC, 30, false, true, 1.0, new Random(randomSeed));
//        LFWDataSetIterator testIter = new LFWDataSetIterator(batchSize, numExamples, inputWHC, 6, false, false, 1-splitTrainTest, new Random(randomSeed));

        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
        trainIter.setPreProcessor(scaler);

        ComputationGraph net = Model.getNetwork();

        log.info(net.summary());

        for (int j = 0; j < 100;j++) {
            while (trainIter.hasNext()) {
                lfwNext = trainIter.next();
                trainTest = lfwNext.splitTestAndTrain(splitTrainNum, new Random(randomSeed)); // train set that is the result
                trainInput = trainTest.getTrain(); // get feature matrix and labels for training
                testInput.add(trainTest.getTest().getFeatures());
                testLabels.add(trainTest.getTest().getLabels());
                net.fit(trainInput);
            }

            log.info("Evaluate model....");
            Evaluation eval = new Evaluation(trainIter.getLabels());
            for (int i = 0; i < testInput.size(); i++) {
                DataSet ds = new DataSet(testInput.get(i), testLabels.get(i));
                INDArray output = net.feedForward( ds.getFeatures(), false).get("lossLayer");
                eval.eval(testLabels.get(i), output);
            }
//        INDArray output = net.output(testInput.get(0));
//        eval.eval(testLabels.get(0), output);
            log.info(eval.stats());
        }


        log.info("****************Example finished********************");

//        UIServer server = UIServer.getInstance();
//        StatsStorage storage = new InMemoryStatsStorage();
//        server.attach(storage);
//        net.setListeners(
//                new ScoreIterationListener(5),
//                new StatsListener(storage),
//                new EvaluativeListener(trainIter, 1, InvocationType.EPOCH_END)
////                new EvaluativeListener(testIter, 1, InvocationType.EPOCH_END)
//        );

//        System.out.println(net.summary());
// 650
//        for (int i = 0; i < 650; i++) {
//            net.fit(trainIter);
//
//            log.info("Epoch: {}", (i+1));
////            log.info("Accuracy: " + squeezeNet.evaluate(trainIter).stats());
////            log.info("Precision: " + squeezeNet.evaluate(trainIter).precision());
//        }
//
//        ComputationGraph snippedNet = snipNetwork(net);
//        saveModel(snippedNet, modelLocation);

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

