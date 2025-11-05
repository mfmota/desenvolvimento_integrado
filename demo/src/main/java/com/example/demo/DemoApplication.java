package com.example.demo;

import org.nd4j.linalg.api.buffer.DataType; 
import org.nd4j.linalg.factory.Nd4j;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
public class DemoApplication {

	public static void main(String[] args) {
		Nd4j.setDefaultDataTypes(DataType.DOUBLE, DataType.DOUBLE);
		SpringApplication.run(DemoApplication.class, args);
	}

}
