package com.example.demo.dto;

import java.time.ZonedDateTime;

public record ReconstructionResponse(String algorithmUsed,
    ZonedDateTime startTime,
    ZonedDateTime endTime,
    long reconstructionTimeMs,
    String pixelSize,
    int iterationsExecuted,
    double[][] imageReconstructed) {
}
