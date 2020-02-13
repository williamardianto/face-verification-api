package com.example.faceverification;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

import java.io.File;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;

@RestController
public class Controller {
    private static String UPLOADED_FOLDER = new File(Paths.get(System.getProperty("user.home"), "data", "face").toString()).toString();
    private static String modelLocation = Paths.get(System.getProperty("user.home"), ".deeplearning4j", "models", "faceverification", "vggface-resnet.h5").toString();
    ComputationGraph net = KerasModelImport.importKerasModelAndWeights(modelLocation);

    public Controller() throws InvalidKerasConfigurationException, IOException, UnsupportedKerasConfigurationException {
    }



    @PostMapping(value = "/verify")
    public ResponseEntity verify(
            @RequestParam(required = false) MultipartFile image1_file,
            @RequestParam(required = false) MultipartFile image2_file,
            @RequestParam(required = false) String image1_base64,
            @RequestParam(required = false) String image2_base64
    ) throws IOException {

//        byte[] bytes = file1.getBytes();
//        Path path = Paths.get(UPLOADED_FOLDER, file1.getOriginalFilename());
//        Files.write(path, bytes);
        File f1, f2;

        if (image1_file != null && image2_file != null){
            f1 = Utils.convert(image1_file);
            f2 = Utils.convert(image2_file);
        }else if(!Utils.isNullOrEmpty(image1_base64) && !Utils.isNullOrEmpty(image2_base64)){
            f1 = Utils.base64ToFile(image1_base64);
            f2 = Utils.base64ToFile(image2_base64);
        }else{
            Map<String, String> body = new HashMap<>();
            body.put("message", "incorrect input data");
            return ResponseEntity.badRequest().body(body);
        }

        INDArray array1 = Utils.loadImage(f1);
        INDArray array2 = Utils.loadImage(f2);

        double distance = Utils.distance(
                Utils.getEmbedding(net, array1),
                Utils.getEmbedding(net, array2)
        );

        Map<String , Double > response = new HashMap<>();
        response.put("distance", distance);

        return new ResponseEntity<>(response, HttpStatus.OK);
    }


}
