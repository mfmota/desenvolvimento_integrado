package com.example.demo.service;

import com.example.demo.dto.ImageResult;
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
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.time.LocalDateTime;
import java.util.concurrent.locks.ReentrantLock;

@Service
public class ReconstructionService {

    private final DataCacheService dataCacheService;
    private final SystemInfo systemInfo;
    private final CentralProcessor processor;

    private final ReentrantLock cpuLock = new ReentrantLock();
    private long[] prevTicks;

    private static final Logger logger = LoggerFactory.getLogger(ReconstructionService.class);
    private static final int MAX_ITERATIONS = 10;
    private static final double ERROR_TOLERANCE = 1e-4;

    public ReconstructionService(DataCacheService dataCacheService) {
        this.dataCacheService = dataCacheService;
        this.systemInfo = new SystemInfo();
        this.processor = systemInfo.getHardware().getProcessor();
        this.prevTicks = processor.getSystemCpuLoadTicks();
    }

    private record AlgorithmResult(INDArray f, int iterations) {
    }

    public ImageResult reconstruct(byte[] rawSignal, String modelName, String algorithm, Integer tamanho, String ganho) {
        logger.info("Iniciando reconstrução. Modelo={}, Alg={}, TamanhoHeader={}, Ganho={}",
                modelName, algorithm, tamanho, ganho);
        LocalDateTime startTime = LocalDateTime.now();
        long startNanos = System.nanoTime();

        Object modelDataObj = dataCacheService.getData(modelName);
        if (!(modelDataObj instanceof DataCacheService.PrecalculatedModel modelData)) {
            throw new IllegalArgumentException("Modelo " + modelName + " não encontrado no cache ou tipo inválido.");
        }
        INDArray H_norm = modelData.H_norm();
        INDArray H_norm_T = modelData.H_norm_T();
        double H_std = modelData.H_std();
        double c_factor = modelData.c_factor();

        float[] floatArr = bytesToFloatArrayLE(rawSignal);
        if (tamanho != null && tamanho.intValue() != floatArr.length) {
            throw new IllegalArgumentException("Tamanho incorreto do sinal: esperado " + tamanho + ", recebido " + floatArr.length);
        }

        INDArray gFloat = Nd4j.createFromArray(floatArr); 
        INDArray g = gFloat.castTo(DataType.DOUBLE).reshape(gFloat.length(), 1); 

        double g_mean = g.meanNumber().doubleValue();
        double g_std = g.stdNumber().doubleValue();
        INDArray g_norm;
        if (g_std > 1e-12) {
            g_norm = g.sub(g_mean).div(g_std);
        } else {
            logger.warn("Desvio padrão muito pequeno no sinal; usando normalização parcial.");
            g_norm = g.sub(g_mean);
        }

        double lambda_reg = Transforms.abs(H_norm_T.mmul(g_norm)).maxNumber().doubleValue() * 0.10;
        logger.info("[CÁLCULO] Coeficiente λ: {}", String.format("%.4e", lambda_reg));
        logger.info("[CÁLCULO] Fator c (pré-calculado): {}", String.format("%.4e", c_factor));

        AlgorithmResult algResult;
        String algLower = algorithm.trim().toLowerCase();
        if ("cgne".equals(algLower)) {
            algResult = executeCgne(H_norm, H_norm_T, g_norm);
        } else if ("cgnr".equals(algLower)) {
            algResult = executeCgnr(H_norm, H_norm_T, g_norm);
        } else {
            throw new IllegalArgumentException("Algoritmo desconhecido: " + algorithm);
        }

        INDArray f = algResult.f();
        int iterations = algResult.iterations();

        double cpuPercent = getCpuUsage();
        GlobalMemory memory = systemInfo.getHardware().getMemory();
        double memPercent = (memory.getTotal() - memory.getAvailable()) * 100.0 / memory.getTotal();

        INDArray f_final;
        if (H_std > 1e-12) {
            f_final = f.mul(g_std / H_std);
        } else {
            f_final = f;
        }

        INDArray f_clipped = Transforms.relu(f_final);
        double f_max = f_clipped.maxNumber().doubleValue();
        INDArray f_norm;
        if (f_max > 1e-12) {
            f_norm = f_clipped.div(f_max).mul(255.0);
        } else {
            f_norm = Nd4j.zeros(f_clipped.shape());
        }

        long n_pixels = f_norm.length();
        int lado = (int) Math.floor(Math.sqrt(n_pixels));
        if (lado * lado <= 0) {
            throw new RuntimeException("Tamanho do vetor de reconstrução inválido: " + n_pixels);
        }

        double[] pixels = f_norm.data().asDouble(); 
        byte[] pngData = createPngFromColumnMajorDouble(pixels, lado, lado);

        LocalDateTime endTime = LocalDateTime.now();
        long endNanos = System.nanoTime();
        double durationSeconds = (endNanos - startNanos) / 1_000_000_000.0;

        ImageResult result = new ImageResult(
                pngData,
                algorithm,
                lado + "x" + lado,
                iterations,
                durationSeconds,
                cpuPercent,
                memPercent,
                startTime,
                endTime
        );

        return result;
    }

    private AlgorithmResult executeCgne(INDArray H_norm, INDArray H_norm_T, INDArray g_norm) {
        int n = (int) H_norm.columns();
        INDArray f = Nd4j.zeros(DataType.DOUBLE, n, 1);
        INDArray r = g_norm.dup();
        INDArray p = H_norm_T.mmul(r);
        double r_norm_old = r.norm2Number().doubleValue();

        for (int i = 0; i < MAX_ITERATIONS; i++) {
            double p_norm_sq = Math.pow(p.norm2Number().doubleValue(), 2.0);
            if (p_norm_sq < 1e-20) break;

            double alpha = (r_norm_old * r_norm_old) / p_norm_sq;
            INDArray f_next = f.add(p.mul(alpha));
            INDArray q = H_norm.mmul(p);
            INDArray r_next = r.sub(q.mul(alpha));

            double r_norm_new = r_next.norm2Number().doubleValue();
            double epsilon = Math.abs(r_norm_new - r_norm_old);

            if (epsilon < ERROR_TOLERANCE || r_norm_new < ERROR_TOLERANCE) {
                return new AlgorithmResult(f_next, i + 1);
            }

            double r_norm_sq_new = r_norm_new * r_norm_new;
            if ((r_norm_old * r_norm_old) < 1e-20) {
                return new AlgorithmResult(f_next, i + 1);
            }

            double beta = r_norm_sq_new / (r_norm_old * r_norm_old);
            INDArray p_next = H_norm_T.mmul(r_next).add(p.mul(beta));

            f = f_next;
            r = r_next;
            p = p_next;
            r_norm_old = r_norm_new;
        }

        return new AlgorithmResult(f, MAX_ITERATIONS);
    }

    private AlgorithmResult executeCgnr(INDArray H_norm, INDArray H_norm_T, INDArray g_norm) {
        int n = (int) H_norm.columns();
        INDArray f = Nd4j.zeros(DataType.DOUBLE, n, 1);
        INDArray r = g_norm.sub(H_norm.mmul(f));
        INDArray z = H_norm_T.mmul(r);
        INDArray p = z.dup();

        double r_norm_old = r.norm2Number().doubleValue();
        double z_norm_sq_old = Math.pow(z.norm2Number().doubleValue(), 2.0);

        for (int i = 0; i < MAX_ITERATIONS; i++) {
            INDArray w = H_norm.mmul(p);
            double w_norm_sq = Math.pow(w.norm2Number().doubleValue(), 2.0);
            if (w_norm_sq < 1e-20) break;

            double alpha = z_norm_sq_old / w_norm_sq;
            INDArray f_next = f.add(p.mul(alpha));
            INDArray r_next = r.sub(w.mul(alpha));

            double r_norm_new = r_next.norm2Number().doubleValue();
            double epsilon = Math.abs(r_norm_new - r_norm_old);
            if (epsilon < ERROR_TOLERANCE || r_norm_new < ERROR_TOLERANCE) {
                return new AlgorithmResult(f_next, i + 1);
            }

            INDArray z_next = H_norm_T.mmul(r_next);
            double z_norm_sq_new = Math.pow(z_next.norm2Number().doubleValue(), 2.0);

            if (z_norm_sq_old < 1e-20) {
                return new AlgorithmResult(f_next, i + 1);
            }

            double beta = z_norm_sq_new / z_norm_sq_old;
            INDArray p_next = z_next.add(p.mul(beta));

            f = f_next;
            r = r_next;
            z = z_next;
            p = p_next;
            r_norm_old = r_norm_new;
            z_norm_sq_old = z_norm_sq_new;
        }

        return new AlgorithmResult(f, MAX_ITERATIONS);
    }

    private float[] bytesToFloatArrayLE(byte[] raw) {
        ByteBuffer bb = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN);
        FloatBuffer fb = bb.asFloatBuffer();
        float[] arr = new float[fb.remaining()];
        fb.get(arr);
        return arr;
    }

    private byte[] createPngFromColumnMajorDouble(double[] pixelsColumnMajor, int width, int height) {
        BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);

        int total = width * height;
        for (int k = 0; k < Math.min(total, pixelsColumnMajor.length); k++) {
            int col = k / height;   
            int row = k % height;   
            int gray = (int) Math.round(pixelsColumnMajor[k]);
            gray = Math.max(0, Math.min(255, gray));
            image.getRaster().setSample(col, row, 0, gray);
        }

        try (ByteArrayOutputStream baos = new ByteArrayOutputStream()) {
            ImageIO.write(image, "PNG", baos);
            return baos.toByteArray();
        } catch (IOException e) {
            throw new RuntimeException("Falha ao converter imagem para PNG.", e);
        }
    }

    private double getCpuUsage() {
        cpuLock.lock();
        try {
            double load = processor.getSystemCpuLoadBetweenTicks(this.prevTicks) * 100.0;
            this.prevTicks = processor.getSystemCpuLoadTicks();
            return load;
        } finally {
            cpuLock.unlock();
        }
    }
}