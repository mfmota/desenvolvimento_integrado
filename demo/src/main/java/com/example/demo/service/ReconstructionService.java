package com.example.demo.service;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.serde.binary.BinarySerde;
import org.nd4j.linalg.api.buffer.DataType;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

import com.example.demo.dto.ReconstructionRequest;
import com.example.demo.dto.ReconstructionResponse;

import java.io.ByteArrayInputStream;
import java.io.InputStream;
import java.time.ZonedDateTime;

@Service
public class ReconstructionService {

    private final ModelCacheService modelCacheService;

    private static final Logger logger = LoggerFactory.getLogger(ReconstructionService.class);
    private static final int MAX_ITERATIONS = 10;
    private static final double ERROR_TOLERANCE = 1e-4;

    public ReconstructionService(ModelCacheService modelCacheService) {
        this.modelCacheService = modelCacheService;
    }

    public ReconstructionResponse reconstruct(String algorithm, String modelId, byte[] gFileData) {
        logger.info("Iniciando reconstrução com o algoritmo: {}", algorithm);

        INDArray H = modelCacheService.getModel(modelId);

        INDArray g;
        try (InputStream is = new ByteArrayInputStream(gFileData)) {
            g = Nd4j.createFromNpy(is);
        } catch (Exception e) {
            throw new RuntimeException("Falha ao processar o arquivo .npy.", e);
        }

        g = g.castTo(DataType.DOUBLE);

        if (H.rows() != g.rows()) {
            throw new IllegalArgumentException(
                    "Incompatibilidade de dimensão: H tem " + H.rows() + " linhas, g tem " + g.rows() + " linhas.");
        }

        if ("CGNE".equalsIgnoreCase(algorithm)) {
            return executeCgne(H, g);
        } else if ("CGNR".equalsIgnoreCase(algorithm)) {
            return executeCgnr(H, g);
        } else {
            throw new IllegalArgumentException("Algoritmo desconhecido: " + algorithm);
        }
    }

    private ReconstructionResponse executeCgne(INDArray H, INDArray g) {
        ZonedDateTime startTime = ZonedDateTime.now();
        long startNanos = System.nanoTime();

        INDArray f = Nd4j.zeros(DataType.DOUBLE, H.columns(), 1);
        INDArray r = g.sub(H.mmul(f));
        INDArray p = H.transpose().mmul(r);
        double r_norm_sq_old = Math.pow(r.norm2Number().doubleValue(), 2);
        int iterations = 0;

        for (int i = 0; i < MAX_ITERATIONS; i++) {
            iterations = i + 1;

            double p_norm_sq = Math.pow(p.norm2Number().doubleValue(), 2);
            if (Math.abs(p_norm_sq) < 1e-12) {
                logger.warn("CGNE: Instabilidade numérica detectada (norma de p próxima de zero).");
                break;
            }
            double alpha = r_norm_sq_old / p_norm_sq;

            f.addi(p.mul(alpha));
            r.subi(H.mmul(p).mul(alpha));

            double r_norm_sq_new = Math.pow(r.norm2Number().doubleValue(), 2);

            double epsilon = r_norm_sq_new - r_norm_sq_old;
            if (Math.abs(epsilon) < ERROR_TOLERANCE) {
                logger.info("CGNE: Convergência atingida em {} iterações.", iterations);
                break;
            }

            if (Math.abs(r_norm_sq_old) < 1e-12) {
                logger.warn("CGNE: Instabilidade numérica detectada (norma de r próxima de zero).");
                break;
            }
            double beta = r_norm_sq_new / r_norm_sq_old;
            p = H.transpose().mmul(r).addi(p.mul(beta));
            r_norm_sq_old = r_norm_sq_new;
        }

        if (iterations == MAX_ITERATIONS) {
            logger.warn("CGNE: Máximo de {} iterações atingido.", MAX_ITERATIONS);
        }

        long endNanos = System.nanoTime();
        ZonedDateTime endTime = ZonedDateTime.now();
        long durationMs = (endNanos - startNanos) / 1_000_000;

        return new ReconstructionResponse(
                "CGNE",
                startTime,
                endTime,
                durationMs,
                getPixelSize(f),
                iterations,
                f.toDoubleMatrix());
    } 

    private ReconstructionResponse executeCgnr(INDArray H, INDArray g) {
        ZonedDateTime startTime = ZonedDateTime.now();
        long startNanos = System.nanoTime();

        INDArray f = Nd4j.zeros(DataType.DOUBLE, H.columns(), 1);
        INDArray r = g.dup(); // r0 = g - Hf0, com f0=0
        INDArray z = H.transpose().mmul(r);
        INDArray p = z.dup();

        double z_norm_sq_old = Math.pow(z.norm2Number().doubleValue(), 2);
        double r_norm_sq_old = Math.pow(r.norm2Number().doubleValue(), 2);
        int iterations = 0;

        for (int i = 0; i < MAX_ITERATIONS; i++) {
            iterations = i + 1;
            INDArray w = H.mmul(p);

            double alpha = z_norm_sq_old / Math.pow(w.norm2Number().doubleValue(), 2);
            f.addi(p.mul(alpha));
            r.subi(w.mul(alpha));

            double r_norm_sq_new = Math.pow(r.norm2Number().doubleValue(), 2);
            ;

            // Cálculo do erro (epsilon)
            double epsilon = r_norm_sq_new - r_norm_sq_old;
            if (Math.abs(epsilon) < ERROR_TOLERANCE) {
                logger.info("CGNR: Convergência atingida em {} iterações.", iterations);
                break;
            }

            z = H.transpose().mmul(r);
            double z_norm_sq_new = Math.pow(z.norm2Number().doubleValue(), 2);
            double beta = z_norm_sq_new / z_norm_sq_old;
            p = z.add(p.mul(beta));

            z_norm_sq_old = z_norm_sq_new;
            r_norm_sq_old = r_norm_sq_new;
        }

        if (iterations == MAX_ITERATIONS) {
            logger.warn("CGNR: Máximo de {} iterações atingido.", MAX_ITERATIONS);
        }

        long endNanos = System.nanoTime();
        ZonedDateTime endTime = ZonedDateTime.now();

        long durationMs = (endNanos - startNanos) / 1_000_000;

        return new ReconstructionResponse(
                "CGNR",
                startTime,
                endTime,
                durationMs,
                getPixelSize(f),
                iterations,
                f.toDoubleMatrix());
    }

    // Calcula o tamanho em pixels da imagem
    private String getPixelSize(INDArray f) {
        long nPixels = f.length();
        int side = (int) Math.sqrt(nPixels);
        if (side * side != nPixels) {
            return nPixels + " pixels (não quadrada)";
        }
        return side + "x" + side;
    }

}