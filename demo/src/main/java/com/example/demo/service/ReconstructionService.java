package com.example.demo.service;

import com.example.demo.dto.ImageResult;
import com.example.demo.dto.ReconstructionRequest;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import javax.imageio.ImageIO;

import oshi.SystemInfo;
import oshi.hardware.CentralProcessor;
import oshi.hardware.GlobalMemory;

import java.io.IOException;
import java.time.ZonedDateTime;

@Service
public class ReconstructionService {

    private final DataCacheService dataCacheService; // <- Modificado
    private final SystemInfo systemInfo; // <- Novo
    private final CentralProcessor processor; // <- Novo

    private static final Logger logger = LoggerFactory.getLogger(ReconstructionService.class);
    private static final int MAX_ITERATIONS = 10;
    private static final double ERROR_TOLERANCE = 1e-4;

    public ReconstructionService(DataCacheService dataCacheService) {
        this.dataCacheService = dataCacheService;
        // Inicializa o OSHI para monitoramento
        this.systemInfo = new SystemInfo();
        this.processor = systemInfo.getHardware().getProcessor();
    }

    // DTO interno para retornar resultado do algoritmo
    private record AlgorithmResult(INDArray f, int iterations) {
    }

    public ImageResult reconstruct(ReconstructionRequest request) {
        logger.info("Iniciando reconstrução: {}", request);
        ZonedDateTime startTime = ZonedDateTime.now();
        long startNanos = System.nanoTime();

        // 1. Obter dados do cache (SUPER RÁPIDO)
        INDArray H = dataCacheService.getData(request.modelo());
        INDArray g = dataCacheService.getData(request.sinal());

        // 2. Executar algoritmo
        AlgorithmResult algResult;
        if ("CGNE".equalsIgnoreCase(request.algoritmo())) {
            algResult = executeCgne(H, g);
        } else if ("CGNR".equalsIgnoreCase(request.algoritmo())) {
            algResult = executeCgnr(H, g);
        } else {
            throw new IllegalArgumentException("Algoritmo desconhecido: " + request.algoritmo());
        }

        INDArray f_final = algResult.f();
        int iterations = algResult.iterations();

        // 3. Obter status de CPU/Memória (como psutil)
        // (Nota: Medir CPU é complexo, isso é uma aproximação)
        double cpuPercent = getCpuUsage();
        GlobalMemory memory = systemInfo.getHardware().getMemory();
        double memPercent = (memory.getTotal() - memory.getAvailable()) * 100.0 / memory.getTotal();

        // 4. Normalização de Imagem ("Meio-Termo" com Sqrt)
        // Correto
        INDArray f_clipped = Transforms.relu(f_final); // Zera negativos (Rectified Linear Unit)
        double f_max = f_clipped.maxNumber().doubleValue();

        INDArray f_norm;
        if (f_max > 1e-12) {
            f_norm = f_clipped.div(f_max).mul(255);
        } else {
            f_norm = Nd4j.zeros(f_final.shape());
        }

        // 5. Reshape e Criação da Imagem PNG (com order 'F')
        int lado = (int) Math.sqrt(f_norm.length());

        byte[] pngData = createPng(f_norm, lado, lado);
        String tamanho = lado + "x" + lado;

        long endNanos = System.nanoTime();
        ZonedDateTime endTime = ZonedDateTime.now();
        long durationMs = (endNanos - startNanos) / 1_000_000;

        return new ImageResult(
                pngData,
                request.algoritmo(),
                startTime,
                endTime,
                durationMs,
                tamanho,
                iterations,
                cpuPercent,
                memPercent);
    }

    // --- Funções de Algoritmo (REESCRITAS PARA BATER COM PYTHON) ---

    // Normaliza um array (z-score)
    private INDArray normalize(INDArray A, double[] outStats) {
        double mean = A.meanNumber().doubleValue();
        double std = A.stdNumber().doubleValue();

        // Salva stats para de-normalização
        outStats[0] = mean;
        outStats[1] = std;

        if (std > 1e-12) {
            return A.sub(mean).div(std);
        }
        logger.warn("Desvio padrão muito pequeno, aplicando normalização parcial.");
        return A.sub(mean);
    }

    private AlgorithmResult executeCgne(INDArray H, INDArray g) {
        double[] hStats = new double[2];
        double[] gStats = new double[2];

        INDArray H_norm = normalize(H, hStats);
        INDArray g_norm = normalize(g, gStats);
        double H_std = hStats[1];
        double g_std = gStats[1];

        long n = H_norm.columns();
        INDArray f = Nd4j.zeros(DataType.DOUBLE, n, 1);
        INDArray r = g_norm.dup(); // r = g_norm.copy()
        INDArray p = H_norm.transpose().mmul(r);
        double r_norm_sq_old = r.norm2Number().doubleValue() * r.norm2Number().doubleValue();

        int i = 0;
        for (i = 0; i < MAX_ITERATIONS; i++) {
            double p_norm_sq = p.norm2Number().doubleValue() * p.norm2Number().doubleValue();

            if (p_norm_sq < 1e-20)
                break;

            double alpha = r_norm_sq_old / p_norm_sq;
            INDArray f_next = f.add(p.mul(alpha));
            INDArray q = H_norm.mmul(p);
            INDArray r_next = r.sub(q.mul(alpha));

            double error_absolute = r_next.norm2Number().doubleValue();
            double error_relative = Math.abs(error_absolute - r.norm2Number().doubleValue());

            if (error_absolute < ERROR_TOLERANCE || error_relative < ERROR_TOLERANCE) {
                f = f_next;
                break;
            }

            double r_norm_sq_new = r_next.norm2Number().doubleValue() * r_next.norm2Number().doubleValue();
            if (r_norm_sq_old < 1e-20)
                break;

            double beta = r_norm_sq_new / r_norm_sq_old;
            INDArray p_next = H_norm.transpose().mmul(r_next).add(p.mul(beta));

            f = f_next;
            r = r_next;
            p = p_next;
            r_norm_sq_old = r_norm_sq_new;
        }

        // De-normalização (como no Python)
        INDArray f_final = (H_std > 1e-12) ? f.mul(g_std / H_std) : f;

        logger.info("CGNE concluído em {} iterações", i + 1);
        return new AlgorithmResult(f_final, i + 1);
    }

    private AlgorithmResult executeCgnr(INDArray H, INDArray g) {
        double[] hStats = new double[2];
        double[] gStats = new double[2];

        INDArray H_norm = normalize(H, hStats);
        INDArray g_norm = normalize(g, gStats);
        double H_std = hStats[1];
        double g_std = gStats[1];

        long n = H_norm.columns();
        INDArray f = Nd4j.zeros(DataType.DOUBLE, n, 1);
        INDArray r = g_norm.sub(H_norm.mmul(f));
        INDArray z = H_norm.transpose().mmul(r);
        INDArray p = z.dup();

        double r_norm_old = r.norm2Number().doubleValue();
        double z_norm_sq_old = z.norm2Number().doubleValue() * z.norm2Number().doubleValue();

        int i = 0;
        for (i = 0; i < MAX_ITERATIONS; i++) {
            INDArray w = H_norm.mmul(p);
            double w_norm_sq = w.norm2Number().doubleValue() * w.norm2Number().doubleValue();

            if (w_norm_sq < 1e-20)
                break;

            double alpha = z_norm_sq_old / w_norm_sq;
            INDArray f_next = f.add(p.mul(alpha));
            INDArray r_next = r.sub(w.mul(alpha));

            double error_absolute = r_next.norm2Number().doubleValue();
            double error_relative = Math.abs(error_absolute - r_norm_old);

            if (error_absolute < ERROR_TOLERANCE || error_relative < ERROR_TOLERANCE) {
                f = f_next;
                break;
            }

            INDArray z_next = H_norm.transpose().mmul(r_next);
            double z_norm_sq_new = z_next.norm2Number().doubleValue() * z_next.norm2Number().doubleValue();

            if (z_norm_sq_old < 1e-20)
                break;

            double beta = z_norm_sq_new / z_norm_sq_old;
            INDArray p_next = z_next.add(p.mul(beta));

            f = f_next;
            r = r_next;
            z = z_next;
            p = p_next;
            r_norm_old = error_absolute;
            z_norm_sq_old = z_norm_sq_new;
        }

        // De-normalização (como no Python)
        INDArray f_final = (H_std > 1e-12) ? f.mul(g_std / H_std) : f;

        logger.info("CGNR concluído em {} iterações", i + 1);
        return new AlgorithmResult(f_final, i + 1);
    }

    // --- Funções Utilitárias ---

    private byte[] createPng(INDArray f_norm_vector, int width, int height) {
        BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);
        long length = f_norm_vector.length();
        for (int k = 0; k < length; k++) {
            // Mapeamento de índice 1D (ordem 'F') para 2D (x, y)
            // O k-ésimo item preenche as colunas primeiro.
            int x = k / height; // Índice da Coluna
            int y = k % height; // Índice da Linha

            // Pega o valor do vetor 1D
            int gray = (int) Math.round(f_norm_vector.getDouble(k));

            // Garante o clip (redundante se f_norm já foi clipado, mas seguro)
            gray = Math.max(0, Math.min(255, gray));

            // Define o pixel na coordenada (x, y) correta
            image.getRaster().setSample(x, y, 0, gray);
        }

        try (ByteArrayOutputStream baos = new ByteArrayOutputStream()) {
            ImageIO.write(image, "PNG", baos);
            return baos.toByteArray();
        } catch (IOException e) {
            throw new RuntimeException("Falha ao converter imagem para PNG.", e);
        }
    }

    // Método para simular o psutil.cpu_percent(interval=1)
    // É uma aproximação que bloqueia por 500ms
    private double getCpuUsage() {
        long[] prevTicks = processor.getSystemCpuLoadTicks();
        try {
            Thread.sleep(500); // Intervalo de medição
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
        return processor.getSystemCpuLoadBetweenTicks(prevTicks) * 100;
    }
}