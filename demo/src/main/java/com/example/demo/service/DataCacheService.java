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
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;

@Service
public class DataCacheService {
    private static final Logger logger = LoggerFactory.getLogger(DataCacheService.class);
    private final Map<String, INDArray> dataCache = new HashMap<>();

    // Listas idênticas às do Python
    private final List<String> MODEL_FILES = List.of("H_60x60.csv", "H_30x30.csv");
    private final List<String> SIGNAL_FILES = List.of(
        "sinal_1_60x60.csv", "sinal_2_60x60.csv", "sinal_3_60x60.csv",
        "sinal_1_30x30.csv", "sinal_2_30x30.csv", "sinal_3_30x30.csv"
    );

    @PostConstruct
    public void loadAllData() {
        logger.info("=== INICIANDO PRÉ-CARREGAMENTO DE DADOS ===");
        
        // Combina as duas listas e remove duplicatas
        Stream.concat(MODEL_FILES.stream(), SIGNAL_FILES.stream())
              .distinct()
              .forEach(this::loadDataFile);

        logger.info("=== PRÉ-CARREGAMENTO CONCLUÍDO ===");
    }

    private void loadDataFile(String filename) {
        try {
            logger.info(" - [CACHE] Carregando arquivo '{}'...", filename);
            File file = new ClassPathResource(filename).getFile();
            
            // ATENÇÃO: Use readTxt para CSV, não readNumpy.
            // O delimitador é passado como argumento.
            INDArray data = Nd4j.readNumpy(file.getPath(), ",");
            
            // Converte para Double, como no Python
            data = data.castTo(DataType.DOUBLE);
            
            dataCache.put(filename, data);
            logger.info(" - [CACHE] {} carregado com sucesso. Dimensões: {}", filename, Arrays.toString(data.shape()));

        } catch (IOException e) {
            logger.error("[ERRO CACHE] Falha ao carregar {}. Servidor pode falhar. Erro: {}", filename, e.getMessage());
        }
    }

    public INDArray getData(String fileName) {
        INDArray data = dataCache.get(fileName);
        if (data == null) {
            throw new IllegalArgumentException("Arquivo '" + fileName + "' não encontrado no cache.");
        }
        return data;
    }
}