# 🚀 TurboQuant

**TurboQuant** es un motor de cuantización y calibración de alto rendimiento para Modelos de Lenguaje de Gran Escala (LLMs), desarrollado íntegramente en Rust. Está diseñado para procesar modelos masivos (como Gemma 4 31B) con un consumo de RAM extremadamente bajo gracias al uso intensivo de **Memory Mapping (mmap)** y optimizaciones nativas de CPU (**AVX2/AVX-512**).

## 🛠️ Requisitos Previos

- **Rust**: Versión 1.75 o superior.
- **CPU**: Recomendado con soporte AVX2 (verificar con `turbo-quant doctor`).
- **Espacio en Disco**: Suficiente para almacenar el modelo original y el optimizado.

## 🚀 Instalación

Clona el repositorio y compila el binario optimizado:

```bash
cargo build --release
```

El binario se generará en `./target/release/turbo_quant`.

---

## 💎 Flujo de Cuantización (Paso a Paso)

Utilizaremos el modelo **Gemma 4 31B** como ejemplo, pero el flujo es idéntico para cualquier modelo GGUF.

### 1. Diagnóstico del Sistema
Asegúrate de que tu hardware es compatible y obtén recomendaciones de hilos y RAM:
```bash
./target/release/turbo_quant doctor
```

### 2. Descarga del Modelo
Descarga un modelo base en formato GGUF (ej. desde Hugging Face):
```bash
huggingface-cli download bartowski/google_gemma-4-31B-it-GGUF --include "google_gemma-4-31B-it-Q4_K_M.gguf" --local-dir .
```

### 3. Calibración del Motor
Captura las estadísticas de activación de todas las capas del modelo para optimizar la precisión en el objetivo de bits deseado:
```bash
./target/release/turbo_quant calibrate \
  --model google_gemma-4-31B-it-Q4_K_M.gguf \
  --target 3.5 \
  --output ./output
```
*Esto generará un archivo `calibration_manifest.bin` en la carpeta de salida.*

### 4. Empaquetado Final
Fusiona los pesos del modelo original con el manifiesto de calibración para crear el binario optimizado de TurboQuant:
```bash
./target/release/turbo_quant package \
  --model google_gemma-4-31B-it-Q4_K_M.gguf \
  -f ./output/calibration_manifest.bin \
  --output ./output
```
*Resultado: `output/model_turboquant.gguf`.*

### 5. Validación y Benchmark
Verifica la integridad y el rendimiento del modelo generado:
```bash
./target/release/turbo_quant benchmark --model ./output/model_turboquant.gguf --context 8192
```

---

## 🤖 Cómo usar el modelo después

Una vez que tengas tu archivo `model_turboquant.gguf`, puedes utilizarlo de las siguientes maneras:

### 1. Con Motores compatibles con TurboQuant (Recomendado)
El archivo mantiene compatibilidad con la estructura GGUF pero incluye metadatos adicionales de precisión. Puedes cargarlo en cualquier implementación basada en `candle-core` o `llama.cpp` que soporte la lectura de tensores personalizados.

### 2. Integración en aplicaciones Rust
Puedes usar el crate `candle` para cargar el modelo en tu propia aplicación:

```rust
let model_path = Path::new("output/model_turboquant.gguf");
let mut file = std::fs::File::open(model_path)?;
let model = gguf_file::Content::read(&mut file)?;

// Los metadatos de TurboQuant están disponibles en la cabecera
// para ajustar dinámicamente los factores de escala de los tensores.
```

### 3. Ventajas del modelo optimizado
- **Peak RAM ultra bajo**: El modelo 31B consume solo ~0.10 GB de RAM extra sobre el mapeo de memoria.
- **Velocidad**: Hasta 5.2 tokens/s en CPU domésticas para modelos de 31B parámetros.
- **Precisión**: La calibración de 3.5 bits minimiza la pérdida de perplejidad comparado con cuantizaciones estándar.

---

## 📂 Estructura del Proyecto

- `src/loader`: Soporte nativo GGUF/Safetensors.
- `src/engine`: Motor de calibración y estadísticas de capas.
- `src/serialization`: Gestión de manifiestos binarios y fusión de modelos.
- `src/benchmarking`: Métricas reales de RAM y throughput.
- `src/doctor`: Diagnóstico de hardware.

---
*Desarrollado con ❤️ en Rust para la comunidad de IA local.*
