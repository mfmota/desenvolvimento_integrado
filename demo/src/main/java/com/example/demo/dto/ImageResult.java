package com.example.demo.dto;

import java.time.LocalDateTime;

public record ImageResult(
        byte[] pngData,
        String algoritmo,
        String tamanho,
        int iteracoes,
        double tempoSegundos,
        double cpuPercent,
        double memPercent,
        LocalDateTime startTime,
        LocalDateTime endTime
) {}
