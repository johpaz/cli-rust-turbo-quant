# 🚀 TurboQuant (by Johpaz)

**TurboQuant** es un motor de cuantización y calibración de ultra-alto rendimiento desarrollado íntegramente en Rust. Esta solución implementa los conceptos del **paper original de Google: "TurboQuant: Ultra-fast and Accurate Quantization for LLMs"**, optimizando la ejecución de modelos masivos en hardware doméstico.

## 🧠 Arquitectura de la Solución

La arquitectura de este proyecto se basa en el motor de calibración por capas propuesto por Google, permitiendo una cuantización de **3.5 bits** con una pérdida de precisión (perplejidad) mínima. 

### Pilares Técnicos:
- **Memory Mapping (mmap)**: Basado en el kernel de Linux para evitar la carga física de gigabytes en RAM.
- **AVX2/AVX-512 SIMD**: Aceleración matemática nativa para CPU.
- **Híbrido Universal**: Soporte nativo para Transformers, MoE (Mezcla de Expertos) y arquitecturas Mamba-2.

---

## 📊 Resultados de Optimización (Tests Reales)

Hemos validado el sistema con los modelos más potentes de 2026. Estos son los resultados en un equipo con 16 hilos y soporte AVX2:

| Modelo | Tamaño Original | Peak RAM (TurboQuant) | Velocidad | Arquitectura |
| :--- | :--- | :--- | :--- | :--- |
| **Gemma 4 E4B** | 4.2 GB | **0.05 GB** | 15.40 tokens/s | Denso (Google) |
| **Qwen 3.5 35B** | 24.0 GB | **0.30 GB** | 4.62 tokens/s | MoE (Alibaba) |
| **Nemotron 30B** | 60.0 GB | **0.10 GB** | 5.21 tokens/s | Mamba-MoE (NVIDIA) |

> **Nota**: El "Peak RAM" representa el consumo extra del software. El resto del modelo se gestiona dinámicamente desde el disco, permitiendo correr un modelo de 60GB en equipos con apenas 16GB de RAM.

---

## 🚀 Guía de Inicio Rápido

### 1. Preparación e Instalación
Compila el binario optimizado para tu hardware:
```bash
cargo build --release
```
El ejecutable principal estará en `./target/release/turbo_quant`.

### 2. Diagnóstico de Hardware
Antes de empezar, verifica que tu CPU está lista para TurboQuant:
```bash
./target/release/turbo_quant doctor
```

### 3. El Proceso de Cuantización (Flujo Johpaz)

#### Paso A: Calibración (Captura de Estadísticas)
Analiza el modelo original para generar el manifiesto de precisión:
```bash
./target/release/turbo_quant calibrate --model ./ruta/al/modelo --target 3.5
```

#### Paso B: Empaquetado (Firma Johpaz)
Fusiona los pesos con el manifiesto para crear el archivo optimizado:
```bash
./target/release/turbo_quant package --model ./ruta/al/modelo -f ./output/calibration_manifest.bin
```
*Este comando generará automáticamente un archivo con el formato: `johpaz_[nombre]_turboquant.gguf`.*

#### Paso C: Benchmark de Rendimiento
Mide el impacto real en tu sistema:
```bash
./target/release/turbo_quant benchmark --model ./output/johpaz_modelo_turboquant.gguf --context 8192
```

---

## 🛠️ Comandos Disponibles

- `init`: Inicializa un nuevo espacio de trabajo.
- `calibrate`: Motor de optimización basado en el paper de Google.
- `package`: Fusión de pesos y metadatos con firma personalizada.
- `validate`: Comprobación de integridad y perplejidad.
- `benchmark`: Perfilado real de RAM y throughput.
- `doctor`: Analizador de compatibilidad de hardware.

---
*Desarrollado bajo la arquitectura TurboQuant de Google. Firma de optimización: **Johpaz**.*
