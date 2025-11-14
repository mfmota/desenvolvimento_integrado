package com.example.demo.controller;

import com.example.demo.dto.ImageResult;
import com.example.demo.dto.ReconstructionRequest;
import com.example.demo.service.ReconstructionService;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RestController;

import java.time.format.DateTimeFormatter;

@RestController
public class ReconstructionController {

    private final ReconstructionService reconstructionService;
    
    private final DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");


    public ReconstructionController(ReconstructionService reconstructionService) {
        this.reconstructionService = reconstructionService;
    }
    @GetMapping("/ping")
    public ResponseEntity<String> ping() {
        return ResponseEntity.ok("OK");
    }
    
    @PostMapping(value = "/compiledServer/reconstruct", consumes = MediaType.APPLICATION_JSON_VALUE)
    public ResponseEntity<byte[]> reconstruct(
            @RequestBody ReconstructionRequest request) { 

        ImageResult result = reconstructionService.reconstruct(request);

        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.IMAGE_PNG);
        headers.add("X-Algoritmo", result.algoritmo());
        headers.add("X-Inicio", result.startTime().format(formatter));
        headers.add("X-Fim", result.endTime().format(formatter));
        headers.add("X-Tamanho", result.tamanho());
        headers.add("X-Iteracoes", String.valueOf(result.iteracoes()));
        headers.add("X-Tempo", String.format("%.6f", result.tempoMs() / 1000.0)); // Em segundos
        headers.add("X-Cpu", String.format("%.1f", result.cpuPercent()));
        headers.add("X-Mem", String.format("%.1f", result.memPercent()));

        return ResponseEntity.ok()
                .headers(headers)
                .body(result.pngData());
    }
}