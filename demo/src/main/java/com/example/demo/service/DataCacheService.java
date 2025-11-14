package com.example.demo.service;

import jakarta.annotation.PostConstruct;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.api.buffer.DataType;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
// NENHUMA importação de SVD é necessária
import org.springframework.core.io.ClassPathResource;
import org.springframework.stereotype.Service;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;


@Service
public class DataCacheService {
    private static final Logger logger = LoggerFactory.getLogger(DataCacheService.class);
    
    private final Map<String, Object> dataCache = new HashMap<>();

    public record PrecalculatedModel(
            INDArray H_norm,
            INDArray H_norm_T,
            double H_mean,
            double H_std
    ) {}

    private final List<String> MODEL_FILES = List.of("H_60x60.csv", "H_30x30.csv");
    private final List<String> SIGNAL_FILES = List.of(
            "sinal_1_60x60.csv", "sinal_2_60x60.csv", "sinal_3_60x60.csv",
            "sinal_1_30x30.csv", "sinal_2_30x30.csv", "sinal_3_30x30.csv"
    );

    @PostConstruct
    public void loadAllData() {
        logger.info("=== INICIANDO PRÉ-CARREGAMENTO DE DADOS E CÁLCULOS ===");

        for (String filename : SIGNAL_FILES) {
            loadSignalFile(filename);
        }

        for (String filename : MODEL_FILES) {
            loadModelFile(filename);
        }

        logger.info("=== PRÉ-CARREGAMENTO CONCLUÍDO ===");
    }

    private void loadSignalFile(String filename) {
        try {
            File file = new ClassPathResource(filename).getFile();
            INDArray data = Nd4j.readNumpy(file.getPath(), ",").castTo(DataType.DOUBLE);
            dataCache.put(filename, data);
            logger.info(" - [CACHE SINAL] {} carregado. Dimensões: {}", filename, Arrays.toString(data.shape()));
        } catch (IOException e) {
            logger.error("[ERRO CACHE] Falha ao carregar sinal {}. Erro: {}", filename, e.getMessage());
        }
    }

    private void loadModelFile(String filename) {
        try {
            logger.info(" - [CACHE MODELO] Processando {}...", filename);
            File file = new ClassPathResource(filename).getFile();
            INDArray H = Nd4j.readNumpy(file.getPath(), ",").castTo(DataType.DOUBLE);

            double H_mean = H.meanNumber().doubleValue();
            double H_std = H.stdNumber().doubleValue();
            INDArray H_norm;
            if (H_std > 1e-12) {
                H_norm = H.sub(H_mean).div(H_std);
            } else {
                H_norm = H.sub(H_mean);
            }

            INDArray H_norm_T = H_norm.transpose();

            PrecalculatedModel modelData = new PrecalculatedModel(H_norm, H_norm_T, H_mean, H_std);
            dataCache.put(filename, modelData);
            logger.info(" - [CACHE MODELO] {} processado e armazenado.", filename);

        } catch (IOException e) {
            logger.error("[ERRO CACHE] Falha ao carregar modelo {}. Erro: {}", filename, e.getMessage());
        }
    }

    public Object getData(String fileName) {
        Object data = dataCache.get(fileName);
        if (data == null) {
            throw new IllegalArgumentException("Arquivo '" + fileName + "' não encontrado no cache.");
        }
        return data;
    }
}