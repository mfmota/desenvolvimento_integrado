package com.example.demo.controller;

import com.example.demo.dto.ReconstructionResponse;
import com.example.demo.service.ReconstructionService;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RequestPart;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;

@RestController
public class ReconstructionController {

    private final ReconstructionService reconstructionService;

    public ReconstructionController(ReconstructionService reconstructionService) {
        this.reconstructionService = reconstructionService;
    }

    @PostMapping("/compiledServe/reconstruct")
    public ResponseEntity<ReconstructionResponse> reconstruct(
            @RequestParam("algorithm") String algorithm,
            @RequestParam("model_id") String modelId,
            @RequestPart("g_file") MultipartFile gFile) {

        try {
            if (gFile.isEmpty()) {
                throw new IllegalArgumentException("O arquivo 'g_file' está vazio.");
            }

            ReconstructionResponse response = reconstructionService.reconstruct(
                algorithm,
                modelId,
                gFile.getInputStream().readAllBytes()
            );

            return ResponseEntity.ok(response);

        } catch (IOException e) {
            throw new RuntimeException("Falha ao ler o arquivo 'g_file' enviado.", e);
        }
    }
}