package com.example.faceverification;

import org.apache.commons.net.util.Base64;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.springframework.web.multipart.MultipartFile;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.Map;

class Utils {
    static boolean isNullOrEmpty(String str) {
        return str == null || str.isEmpty();
    }

    static File convert(MultipartFile file) throws IOException {
        File convFile = File.createTempFile("temp", ".tmp");
        FileOutputStream fos = new FileOutputStream(convFile);
        fos.write(file.getBytes());
        fos.close();
        return convFile;
    }

    static File base64ToFile(String imageString) throws IOException {
        byte[] data = Base64.decodeBase64(imageString);

        File convFile = File.createTempFile("temp", ".tmp");
        FileOutputStream fos = new FileOutputStream(convFile);
        fos.write(data);
        fos.close();
        return convFile;
    }

    static INDArray loadImage(File file) throws IOException {

        NativeImageLoader loader = new NativeImageLoader(224, 224, 3);
        return loader.asMatrix(file);
    }

    static INDArray getEmbedding(ComputationGraph net, INDArray input) {
        Map<String, INDArray> feedForward = net.feedForward(preProcessInput(input), false);
        return feedForward.get("global_average_pooling2d_2");
    }


    static double distance(INDArray a, INDArray b) {
        return (a.distance2(b)) / 100.0;
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
