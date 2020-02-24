package com.example.faceverification;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

import java.io.File;
import java.io.IOException;
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
            @RequestParam(required = false) String image1_base64,
            @RequestParam(required = false) String image2_base64
    ) throws IOException {
        File f1, f2;

        if (image1_file != null && image2_file != null) {
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

        double distance = faceService.getDistance(f1, f2);

        Map<String , Double > response = new HashMap<>();
        response.put("distance", distance);

        return new ResponseEntity<>(response, HttpStatus.OK);
    }


}
