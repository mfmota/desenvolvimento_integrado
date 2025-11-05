package com.example.demo.service;

import jakarta.annotation.PostConstruct;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.ClassPathResource;
import org.springframework.stereotype.Service;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

@Service
public class ModelCacheService {
    private static final Logger logger = LoggerFactory.getLogger(ModelCacheService.class);
    private final Map<String, INDArray> modelCache = new HashMap<>();

    @PostConstruct
    public void loadModels() {
        logger.info("Iniciando pré-carregamento dos modelos H...");
        loadModel("60x60", "H_60x60.csv");
        loadModel("30x30", "H_30x30.csv");
        logger.info("Modelos H pré-carregados com sucesso!");
    }

    private void loadModel(String modelId, String filename) {
        try {
            logger.info("Carregando modelo '{}' do arquivo '{}'...", modelId, filename);
            File modelFile = new ClassPathResource(filename).getFile();
            INDArray H = Nd4j.readNumpy(modelFile.getPath(), ",");
            modelCache.put(modelId, H);
            H = H.castTo(DataType.DOUBLE);
            logger.info("Modelo '{}' carregado. Dimensões: {}x{}", modelId, H.rows(), H.columns());
        } catch (IOException e) {
            logger.error("FALHA AO CARREGAR O MODELO: {} a partir do arquivo {}", modelId, filename, e);
        }
    }

    public INDArray getModel(String modelId) {
        INDArray model = modelCache.get(modelId);
        if (model == null) {
            throw new IllegalArgumentException("Model ID '" + modelId + "' não encontrado no cache.");
        }
        return model;
    }
}