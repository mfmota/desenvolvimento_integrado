package com.example.demo.controller;

import com.example.demo.dto.ImageResult;
import com.example.demo.service.ReconstructionService;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

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

    @PostMapping(
            value = "/compiledServer/reconstruct",
            consumes = MediaType.APPLICATION_OCTET_STREAM_VALUE
    )
    public ResponseEntity<byte[]> reconstruct(
            @RequestBody byte[] rawSignal,
            @RequestHeader("X-Modelo") String modelName,
            @RequestHeader("X-Alg") String algorithm,
            @RequestHeader(value = "X-Tamanho", required = false) Integer tamanho,
            @RequestHeader(value = "X-Ganho", required = false) String ganho
    ) {
        ImageResult result = reconstructionService.reconstruct(
                rawSignal, modelName, algorithm, tamanho, ganho
        );

        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.IMAGE_PNG);
        headers.add("X-Algoritmo", result.algoritmo());
        headers.add("X-Inicio", result.startTime().format(formatter));
        headers.add("X-Fim", result.endTime().format(formatter));
        headers.add("X-Tamanho", result.tamanho());
        headers.add("X-Iteracoes", String.valueOf(result.iteracoes()));
        headers.add("X-Tempo", String.valueOf(result.tempoSegundos()));
        headers.add("X-Cpu", String.format("%.1f", result.cpuPercent()));
        headers.add("X-Mem", String.format("%.1f", result.memPercent()));

        if (ganho != null) {
            headers.add("X-Ganho", ganho);
        }

        return ResponseEntity.ok()
                .headers(headers)
                .body(result.pngData());
    }
}