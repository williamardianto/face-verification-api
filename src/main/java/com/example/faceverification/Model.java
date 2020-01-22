package com.example.faceverification;

import org.datavec.image.loader.LFWLoader;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.graph.L2NormalizeVertex;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.SqueezeNet;
import org.deeplearning4j.zoo.model.helper.FaceNetHelper;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.learning.config.RmsProp;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.IOException;

public class Model {

    static int embeddingSize = 128;
    static int[] inputShape = new int[]{3, 96, 96};
    private static int outputNum = 187;

    public static ComputationGraph getNetwork() throws IOException {

        ZooModel zooModel = SqueezeNet.builder().build();
        ComputationGraph squeezeNet = (ComputationGraph) zooModel.initPretrained();

        FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
                .activation(Activation.LEAKYRELU)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(1e-2))
                .build();

        ComputationGraph squeezeNetFineTune = new TransferLearning.GraphBuilder(squeezeNet)
                .fineTuneConfiguration(fineTuneConf)
                .removeVertexAndConnections("loss")
                .addLayer("bottleneck",new DenseLayer.Builder().activation(Activation.RELU).nIn(1000).nOut(128).build(),"global_average_pooling2d_5")
                .addVertex("embeddings", new L2NormalizeVertex(new int[]{}, 1e-6), "bottleneck")
                .addLayer("lossLayer", new CenterLossOutputLayer.Builder()
                        .lossFunction(LossFunctions.LossFunction.SQUARED_LOSS)
                        .activation(Activation.SOFTMAX)
                        .nIn(128)
                        .nOut(outputNum)
                        .lambda(1e-2)
                        .alpha(0.5)
                        .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                        .build(), "embeddings")
                .setOutputs("lossLayer")
                .build();

        return squeezeNetFineTune;
    }
}
