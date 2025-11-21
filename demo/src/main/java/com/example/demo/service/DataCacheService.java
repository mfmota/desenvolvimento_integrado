package com.example.demo.service;

import jakarta.annotation.PostConstruct;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.api.buffer.DataType;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.ClassPathResource;
import org.springframework.stereotype.Service;

import java.io.File;
import java.io.IOException;
import java.util.*;

@Service
public class DataCacheService {

    private static final Logger logger = LoggerFactory.getLogger(DataCacheService.class);

    private final Map<String, Object> dataCache = new HashMap<>();

    /**
     * Contém todos os dados pré-calculados necessários para reconstrução
     */
    public record PrecalculatedModel(
            INDArray H_norm,       // matriz H normalizada
            INDArray H_norm_T,     // transposta de H_norm
            double H_mean,         // média original da H
            double H_std,          // desvio padrão original da H
            double c_factor        // fator de reconstrução derivado de H_std
    ) {}

    private final List<String> MODEL_FILES = List.of("H_60x60.csv", "H_30x30.csv");

    private final List<String> SIGNAL_FILES = List.of(
            "sinal_1_60x60.csv", "sinal_2_60x60.csv", "sinal_3_60x60.csv",
            "sinal_1_30x30.csv", "sinal_2_30x30.csv", "sinal_3_30x30.csv"
    );

    @PostConstruct
    public void loadAllData() {
        logger.info("=== INICIANDO PRÉ-CARREGAMENTO DE DADOS ===");

        // Carrega todos os sinais
        for (String filename : SIGNAL_FILES) {
            loadSignalFile(filename);
        }

        // Carrega todos os modelos Hxx.csv e faz pré-cálculos
        for (String filename : MODEL_FILES) {
            loadModelFile(filename);
        }

        logger.info("=== TODOS OS DADOS PRÉ-CARREGADOS COM SUCESSO ===");
    }

    private void loadSignalFile(String filename) {
        try {
            File file = new ClassPathResource(filename).getFile();
            INDArray data = Nd4j.readNumpy(file.getPath(), ",").castTo(DataType.DOUBLE);
            dataCache.put(filename, data);

            logger.info(" - [SINAL] {} carregado. Shape: {}", filename, Arrays.toString(data.shape()));
        } catch (IOException e) {
            logger.error("[ERRO] Falha ao carregar sinal {}: {}", filename, e.getMessage());
        }
    }

    /**
     * Carrega o modelo H, normaliza, pré-calcula matriz transposta e fator C
     */
    private void loadModelFile(String filename) {
        try {
            logger.info(" - [MODELO] Pré-processando {}...", filename);

            File file = new ClassPathResource(filename).getFile();
            INDArray H = Nd4j.readNumpy(file.getPath(), ",").castTo(DataType.DOUBLE);

            // Estatísticas para normalização
            double H_mean = H.meanNumber().doubleValue();
            double H_std = H.stdNumber().doubleValue();

            INDArray H_norm;

            if (H_std > 1e-12) {
                H_norm = H.sub(H_mean).div(H_std);
            } else {
                logger.warn(" - [MODELO] std muito pequeno em {}. Normalização reduzida.", filename);
                H_norm = H.sub(H_mean);
            }

            // Transposta
            INDArray H_norm_T = H_norm.transpose();

            // c = 1 / std — usado em reconstrução para reverter normalização
            double c_factor = (H_std > 1e-12) ? 1.0 / H_std : 1.0;

            // Guarda tudo no cache
            PrecalculatedModel modelData = new PrecalculatedModel(
                    H_norm,
                    H_norm_T,
                    H_mean,
                    H_std,
                    c_factor
            );

            dataCache.put(filename, modelData);

            logger.info(" - [MODELO] {} carregado. (mean={:.4f}, std={:.4f}, c={:.4f})",
                    filename, H_mean, H_std, c_factor);

        } catch (IOException e) {
            logger.error("[ERRO] Falha ao carregar modelo {}: {}", filename, e.getMessage());
        }
    }

    /**
     * Recupera qualquer arquivo (sinal ou modelo) do cache
     */
    public Object getData(String fileName) {
        Object data = dataCache.get(fileName);
        if (data == null) {
            throw new IllegalArgumentException("Arquivo '" + fileName + "' não está carregado no cache.");
        }
        return data;
    }
}
