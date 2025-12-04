package com.example.demo.service;

import jakarta.annotation.PostConstruct;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
//operações de tranposta sem usar mais memória
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

    public record PrecalculatedModel(
            INDArray H_norm,       
            INDArray H_norm_T,     
            double H_mean,         
            double H_std,          
            double c_factor        
    ) {}

    private final List<String> MODEL_FILES = List.of("H_60x60.csv", "H_30x30.csv");

    @PostConstruct
    public void loadAllData() {
        logger.info("=== INICIANDO PRÉ-CARREGAMENTO DE DADOS ===");

        for (String filename : MODEL_FILES) {
            loadModelFile(filename);
        }

        logger.info("=== TODOS OS DADOS PRÉ-CARREGADOS COM SUCESSO ===");
    }

    private void loadModelFile(String filename) {
        try {
            logger.info(" - [MODELO] Pré-processando {}...", filename);

            File file = new ClassPathResource(filename).getFile();
            INDArray H = Nd4j.readNumpy(file.getPath(), ",").castTo(DataType.DOUBLE);

            double H_mean = H.meanNumber().doubleValue();
            double H_std = H.stdNumber().doubleValue();

            INDArray H_norm;

            if (H_std > 1e-12) {
                H_norm = H.sub(H_mean).div(H_std);
            } else {
                logger.warn(" - [MODELO] std muito pequeno em {}. Normalização reduzida.", filename);
                H_norm = H.sub(H_mean);
            }

            INDArray H_norm_T = H_norm.transpose();

            double c_factor = (H_std > 1e-12) ? 1.0 / H_std : 1.0;

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

    public Object getData(String fileName) {
        Object data = dataCache.get(fileName);
        if (data == null) {
            throw new IllegalArgumentException("Arquivo '" + fileName + "' não está carregado no cache.");
        }
        return data;
    }
}