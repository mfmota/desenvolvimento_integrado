package com.example.demo.dto;

import java.time.ZonedDateTime;

public record ImageResult(
    byte[] pngData,
    String algoritmo,
    ZonedDateTime startTime,
    ZonedDateTime endTime,
    long tempoMs,
    String tamanho,
    int iteracoes,
    double cpuPercent,
    double memPercent
) {}