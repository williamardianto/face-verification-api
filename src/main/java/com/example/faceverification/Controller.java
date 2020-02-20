package com.example.faceverification;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
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
    @Autowired
    FaceService faceService;

    @PostMapping(value = "/verify")
    public ResponseEntity verify(
            @RequestParam(required = false) MultipartFile image1_file,
            @RequestParam(required = false) MultipartFile image2_file,
            @RequestBody(required = false) Map<String, String> image_base64
    ) throws IOException {

        File f1, f2;

        if (image1_file != null && image2_file != null) {
            f1 = Utils.convert(image1_file);
            f2 = Utils.convert(image2_file);
        }else if(image_base64 != null && image_base64.get("image1_base64") != null &&
                image_base64.get("image2_base64") != null){
            f1 = Utils.base64ToFile(image_base64.get("image1_base64"));
            f2 = Utils.base64ToFile(image_base64.get("image2_base64"));
        }else{
            Map<String, String> body = new HashMap<>();
            body.put("message", "incorrect input data");
            return ResponseEntity.badRequest().body(body);
        }

        double distance = faceService.getDistance(f1, f2);

        Map<String , Double > response = new HashMap<>();
        response.put("distance", distance);

        return new ResponseEntity<>(response, HttpStatus.OK);
    }


}
