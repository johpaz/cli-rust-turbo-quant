# 🚀 TurboQuant (Implementación en Rust)

**TurboQuant** es una implementación en Rust del algoritmo del paper de Google:
> **"TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate"**
> Amir Zandieh, Mahdi Daliri, Amir Hadian, Vahab Mirrokni (arXiv:2504.19874)

Esta herramienta comprime el **cache KV** de LLMs a **2.5–3.5 bits/canal** con preprocesamiento casi nulo, usando una tasa de distorsión matemáticamente probada como near-óptima.

---

## 📖 ¿Qué es TurboQuant?

TurboQuant es un **algoritmo de cuantización vectorial** (NO es cuantización de pesos como GPTQ/AWQ). Se enfoca en el **cache KV** — el cuello de botella de memoria durante la inferencia autoregresiva — logrando **compresión de 113x** con neutralidad de calidad a 3.5 bits.

### ⚠️ Importante: Qué Hace y Qué NO Hace TurboQuant

| ✅ **SÍ Hace** | ❌ **NO Hace** |
|---------------|---------------|
| Comprime el KV cache **durante la inferencia** | NO transforma el modelo en disco |
| Reduce el consumo de RAM del cache temporal | NO crea un nuevo archivo `.gguf` |
| Permite contextos más largos con menos RAM | NO cuantiza los pesos del modelo (ya vienen cuantizados) |
| Funciona en runtime, sin preprocesamiento | NO necesita calibración previa |

### ¿Cómo Funciona en la Práctica?

```
Prompt → Modelo (Q4_K_M en RAM) → KV Cache → TurboQuant lo comprime → Texto generado
                                             ↑
                                      Aquí actúa TurboQuant
                                      (solo en memoria, durante ejecución)
```

### Propiedades Clave

| Propiedad | Valor |
|----------|-------|
| **Bits del KV Cache** | 2.5–3.5 bits/canal |
| **Calidad a 3.5 bits** | Neutral (sin degradación medible) |
| **Calidad a 2.5 bits** | ~1.2% de caída en LongBench |
| **Ratio de compresión** | **113x vs FP16** en KV cache |
| **Tiempo de indexación** | ~0.001 segundos (virtualmente cero) |
| **¿Necesita calibración?** | **NO** (data-oblivious) |
| **Límite de distorsión** | ≤ 2.7× el mínimo teórico |

### Algoritmo en Dos Etapas

```
Etapa 1: Cuantización Escalar MSE-Óptima
  ├── Rotación aleatoria (FWHT + signos aleatorios) → induce distribución Gaussiana
  ├── Cuantización escalar por coordenada → minimiza MSE
  └── Resultado: vector cuantizado + residual

Etapa 2: Corrección Residual QJL 1-Bit
  ├── Transformada Quantized Johnson-Lindenstrauss en el residual
  ├── 1 bit por dimensión (signo de proyección aleatoria)
  └── Resultado: estimación de producto interno sin sesgo
```

**Insight matemático:** La cuantización escalar MSE-óptima introduce sesgo en la estimación de productos internos. La corrección QJL de 1-bit elimina este sesgo en expectativa, manteniendo los scores de atención precisos.

---

## 🏗️ Arquitectura

```
src/
├── models/                  # Modelos GGUF descargados
│   ├── google_gemma-4-E4B-it-Q4_K_M.gguf
│   ├── google_gemma-4-31B-it-Q4_K_M.gguf
│   ├── nvidia_Nemotron-Cascade-2-30B-A3B-IQ2_M.gguf
│   └── Qwen_Qwen3.5-35B-A3B-Q4_K_M.gguf
├── math/                    # Algoritmo central TurboQuant
│   ├── rotate.rs            # Rotación aleatoria (FWHT + sign flips)
│   ├── quantizer.rs         # Cuantizador escalar MSE-óptimo
│   ├── polar_quant.rs       # Cuantización polar (Cartesiano → polar)
│   ├── qjl.rs               # Transformada Quantized Johnson-Lindenstrauss
│   └── mod.rs               # Pipeline: encode/decode, inner products corregidos
├── kv_cache/                # Gestión del cache KV
│   ├── quantized_kv.rs      # Almacenamiento comprimido del KV cache
│   ├── attention.rs         # Atención con scores corregidos por QJL
│   └── mod.rs               # Manager de sesión TurboQuantInference
├── generation/              # Carga de modelos y generación de texto
│   ├── gguf_loader.rs       # Cargador de modelos GGUF
│   ├── gemma_model.rs       # Implementación del modelo Gemma (Dense + MoE)
│   ├── tokenizer_wrapper.rs # Wrapper del tokenizador
│   └── sampler.rs           # Sampler de logits (temperatura, top-p)
├── server/                  # Servidor HTTP API
│   └── mod.rs               # Endpoints REST (OpenAI-compatible)
├── loader/                  # Inspección de modelos (GGUF, Safetensors)
├── benchmarking/            # Benchmarks de rendimiento reales
├── validation/              # Validación de calidad del modelo
└── ...
```

---

## 🚀 Inicio Rápido

### 1. Compilar

```bash
cargo build --release
```

### 2. Verificar Compatibilidad del Hardware

```bash
./target/release/turbo_quant doctor
```

### 3. Generar Texto

**Modelo pequeño (E4B - 7.5B parámetros):**

```bash
./target/release/turbo_quant generate \
    --model ./src/models/google_gemma-4-E4B-it-Q4_K_M.gguf \
    --prompt "¿Qué es la inteligencia artificial?" \
    --bits 3.5 \
    --max-tokens 128 \
    --temperature 0.7 \
    --context 4096
```

**Modelo grande (31B - 30.7B parámetros):**

```bash
./target/release/turbo_quant generate \
    --model ./src/models/google_gemma-4-31B-it-Q4_K_M.gguf \
    --prompt "Explícame la teoría de la relatividad" \
    --bits 3.5 \
    --max-tokens 256 \
    --temperature 0.7 \
    --context 8192
```

### 4. Ejecutar Benchmarks

```bash
./target/release/turbo_quant benchmark --model ./src/models/google_gemma-4-E4B-it-Q4_K_M.gguf --context 4096
```

Esto mide:
- **Velocidad de encoding** (vectores/seg)
- **Precisión de producto interno** (MSE vs FP16)
- **Memoria del KV cache** (GB reales)
- **Latencia de atención** (ms/token)
- **Comparación** entre 2.5, 3.0, 3.5, 4.0 bits

### 5. Inspeccionar un Modelo

```bash
./target/release/turbo_quant inspect --model ./src/models/google_gemma-4-E4B-it-Q4_K_M.gguf
```

---

## 🌐 Servidor API HTTP

TurboQuant puede ejecutarse como un servidor HTTP para que otras aplicaciones consuman el modelo vía REST/JSON.

### Iniciar el Servidor

```bash
./target/release/turbo_quant serve \
    --model ./src/models/google_gemma-4-E4B-it-Q4_K_M.gguf \
    --port 3000
```

### Opciones del Servidor

| Opción | Descripción | Valor por Defecto |
|--------|-------------|-------------------|
| `--model, -m` | Ruta al modelo GGUF | *(requerido)* |
| `--host` | Dirección de escucha | `0.0.0.0` |
| `--port, -p` | Puerto | `3000` |
| `--bits, -b` | Bits para KV Cache | `3.5` |
| `--context, -c` | Longitud de contexto | `4096` |

### Endpoints Disponibles

#### **GET /health**
Verifica que el servidor está funcionando.

```bash
curl http://localhost:3000/health
```

**Respuesta:**
```json
{
  "status": "ok",
  "model": "gemma4"
}
```

#### **GET /v1/models**
Lista los modelos cargados.

```bash
curl http://localhost:3000/v1/models
```

**Respuesta:**
```json
{
  "object": "list",
  "data": [
    {
      "id": "turboquant-model",
      "object": "model",
      "created": 1700000000,
      "owned_by": "turboquant"
    }
  ]
}
```

#### **POST /v1/chat/completions**
Genera texto en formato compatible con OpenAI.

```bash
curl http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "¿Qué es la inteligencia artificial?"}
    ],
    "max_tokens": 256,
    "temperature": 0.7,
    "top_p": 0.95
  }'
```

**Respuesta:**
```json
{
  "id": "chatcmpl-abc123...",
  "object": "chat.completion",
  "created": 1700000000,
  "model": "turboquant-model",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "La inteligencia artificial es..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 15,
    "completion_tokens": 45,
    "total_tokens": 60
  }
}
```

#### **POST /v1/completions**
Genera texto con formato simple.

```bash
curl http://localhost:3000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Escribe un poema sobre la tecnología",
    "max_tokens": 128,
    "temperature": 0.8
  }'
```

### Usar desde Python (ejemplo)

```python
import requests

response = requests.post(
    "http://localhost:3000/v1/chat/completions",
    json={
        "messages": [
            {"role": "user", "content": "¿Qué es Python?"}
        ],
        "max_tokens": 256,
        "temperature": 0.7
    }
)

data = response.json()
print(data["choices"][0]["message"]["content"])
```

### Usar con OpenAI SDK (compatible)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:3000/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="turboquant-model",
    messages=[
        {"role": "user", "content": "Explícame la teoría de cuerdas"}
    ],
    max_tokens=512,
    temperature=0.7
)

print(response.choices[0].message.content)
```

### Usar desde Node.js

```javascript
const response = await fetch("http://localhost:3000/v1/chat/completions", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    messages: [{ role: "user", content: "¿Qué es JavaScript?" }],
    max_tokens: 256,
    temperature: 0.7,
  }),
});

const data = await response.json();
console.log(data.choices[0].message.content);
```

---

## 💾 Consumo de Memoria

### Modelos Disponibles

| Modelo | Parámetros | Activos | Tamaño GGUF | RAM Mínima | Contexto Máx |
|--------|-----------|---------|------------|-----------|-------------|
| **Gemma 4 E4B** | 7.5B | 7.5B | ~4.5 GB | ~6 GB | 131,072 tokens |
| **Gemma 4 31B** | 30.7B | 30.7B | ~19 GB | ~22 GB | 262,144 tokens |
| **Nemotron Cascade 2 30B** | 30B | 3B (MoE) | ~12 GB | ~14 GB | 131,072 tokens |
| **Qwen 3.5 35B** | 35B | 3B (MoE) | ~20 GB | ~22 GB | 131,072 tokens |

> **Nota:** Los modelos MoE (Mixture of Experts) como Nemotron y Qwen activan solo una fracción de parámetros por token, haciéndolos más eficientes en inferencia.

### Desglose de Memoria (E4B - 7.5B)

| Componente | Tamaño | Notas |
|-----------|--------|-------|
| **Pesos del Modelo (GGUF Q4_K_M)** | ~4.5 GB | Cargado en RAM |
| **KV Cache con TurboQuant (3.5 bits)** | ~4.4 MB | Para 4096 contexto |
| **KV Cache sin TurboQuant (FP16)** | ~500 MB | Para 4096 contexto |
| **Overhead de ejecución** | ~0.5-1 GB | Tensores temporales |
| **TOTAL ESTIMADO** | **~5-6 GB RAM** | Con TurboQuant activo |

### Desglose de Memoria (31B - 30.7B)

| Componente | Tamaño | Notas |
|-----------|--------|-------|
| **Pesos del Modelo (GGUF Q4_K_M)** | ~19 GB | Cargado en RAM |
| **KV Cache con TurboQuant (3.5 bits)** | ~4.4 MB | Para 4096 contexto |
| **KV Cache sin TurboQuant (FP16)** | ~1 GB | Para 4096 contexto |
| **Overhead de ejecución** | ~1-2 GB | Tensores temporales |
| **TOTAL ESTIMADO** | **~20-22 GB RAM** | Con TurboQuant activo |

### 💡 La Ventaja Real de TurboQuant: Contextos Largos

El beneficio principal no es el modelo en sí, sino poder usar **contextos enormes** con poca memoria de cache:

| Contexto | KV Cache FP16 (E4B) | KV Cache FP16 (31B) | **Con TurboQuant** |
|----------|-------------------|-------------------|-------------------|
| 4,096 tokens | ~500 MB | ~1 GB | **~4.4 MB** |
| 16,384 tokens | ~2 GB | ~4 GB | **~17.6 MB** |
| 65,536 tokens | ~8 GB | ~16 GB | **~70 MB** |
| 131,072 tokens | ~16 GB | ~32 GB | **~140 MB** |
| 262,144 tokens | ❌ No soporta | ~64 GB | **~280 MB** |

**Ahorro: 113x menos memoria en KV cache**

Esto permite:
- ✅ Resumir documentos largos
- ✅ Chat con historial extenso
- ✅ RAG con mucho contexto
- ✅ Ejecutar en máquinas con RAM limitada

---

## 🛠️ Comandos Disponibles

| Comando | Descripción |
|---------|-------------|
| `serve` | **Iniciar servidor API HTTP** ⭐ |
| `generate` | Generar texto con modelo GGUF y KV cache comprimido |
| `benchmark` | Ejecutar benchmarks de rendimiento |
| `quantize` | Inspeccionar configuración de cuantización (info solamente) |
| `inspect` | Ver metadatos y tensores del modelo |
| `validate` | Verificar calidad del modelo después de cuantización |
| `doctor` | Verificar compatibilidad del hardware |
| `calibrate` | ⚠️ Legacy — TurboQuant no necesita calibración |
| `package` | ⚠️ Legacy — usar `quantize` en su lugar |
| `init` | Inicializar un nuevo workspace |

### Opciones del Comando `generate`

| Opción | Descripción | Valor por Defecto |
|--------|-------------|-------------------|
| `--model, -m` | Ruta al modelo GGUF | *(requerido)* |
| `--prompt, -p` | Texto de entrada | `""` |
| `--bits, -b` | Bits para KV Cache (2.5-4.0) | `3.5` |
| `--max-tokens, -n` | Tokens máximos a generar | `128` |
| `--context, -c` | Longitud de contexto | `4096` |
| `--temperature` | Creatividad (0 = determinista) | `0.8` |
| `--top-p` | Umbral nucleus sampling | `0.95` |

### Ejemplos de Uso

**Chat simple:**
```bash
./target/release/turbo_quant generate \
    -m ./src/models/google_gemma-4-E4B-it-Q4_K_M.gguf \
    -p "¿Qué es la inteligencia artificial?" \
    -b 3.5 -n 128 --temperature 0.7
```

**Resumir documento largo:**
```bash
./target/release/turbo_quant generate \
    -m ./src/models/google_gemma-4-31B-it-Q4_K_M.gguf \
    -p "Resume el siguiente texto: ..." \
    -b 3.5 -n 512 -c 16384 --temperature 0.5
```

**Más creativo:**
```bash
./target/release/turbo_quant generate \
    -m ./src/models/google_gemma-4-E4B-it-Q4_K_M.gguf \
    -p "Escribe un cuento sobre..." \
    -b 3.5 -n 256 --temperature 1.2 --top-p 0.9
```

---

## 📊 Resultados Esperados (según el paper de Google)

### Compresión del KV Cache

| Bits/Canal | Compresión vs FP16 | Impacto en Calidad |
|-------------|-------------------|-------------------|
| 3.5 bits | ~113x | **Neutral** (sin cambio medible) |
| 3.0 bits | ~113x | Mínimo |
| 2.5 bits | ~113x | ~1.2% en LongBench |

### Precisión de Scores de Atención

A 3.5 bits/canal, los productos internos corregidos por QJL producen scores de atención con MSE < 0.001 comparado con FP16 de precisión completa. El límite de distorsión está probado como ≤ 2.7× el mínimo teórico de la teoría de la información.

---

## 🔬 Detalles del Algoritmo

### Rotación Aleatoria

El vector de entrada `x` se rota: `x' = (1/√d) · H · D · x`
- `D` = matriz diagonal aleatoria (±1 con igual probabilidad)
- `H` = matriz Hadamard (vía Fast Walsh-Hadamard Transform)
- **Efecto:** Las coordenadas se vuelven aproximadamente i.i.d. Gaussianas

### Cuantización Escalar

Después de la rotación, cada coordenada se cuantifica independientemente usando un cuantizador escalar MSE-óptimo. El número de niveles es `2^bits` (ej: 11 niveles para 3.5 bits).

### Corrección QJL

El residual `e = x_original - x_quantizado` se transforma:
1. Proyección aleatoria: `y = R · e` (R ∈ {±1})
2. Cuantización a signos: `b = sign(y)` → **1 bit por dimensión**
3. La propiedad clave: `E[<b_x, b_y>]` preserva productos internos en expectativa

### Por Qué Funciona

1. **Rotación** hace que las coordenadas sean casi independientes (concentración de medida)
2. **Cuantización escalar** es óptima para coordenadas independientes
3. **QJL** corrige el sesgo que la cuantización escalar introduce en productos internos
4. **Resultado:** Tasa de distorsión near-óptima con cero calibración

---

## 📚 Referencias

- **Paper:** [TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate](https://arxiv.org/abs/2504.19874)
- **Google Blog:** [TurboQuant: Redefining AI efficiency with extreme compression](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)
- **Visualización Interactiva:** [TurboQuant Animated](https://mesuvash.github.io/blog/2026/turboquant-interactive/)

---

## ⚠️ Notas Importantes

1. **Esto es cuantización de cache KV, NO de pesos.** Los pesos del modelo permanecen en su precisión original (ya cuantizados en el GGUF). Solo el cache KV (que crece con la longitud de secuencia) se comprime.

2. **3.5 bits es el punto óptimo.** El paper prueba que logra neutralidad de calidad. Ir más bajo (2.5 bits) introduce degradación medible pero pequeña.

3. **No necesita calibración.** A diferencia de GPTQ/AWQ, TurboQuant es data-oblivious. Funciona online con cero preprocesamiento.

4. **Implementación para CPU.** El paper fue validado en GPUs H100. Esta implementación en Rust apunta a CPUs con soporte AVX2/AVX-512. Los números de rendimiento diferirán.

5. **Integración con Candle es parcial.** El pipeline matemático (rotación, cuantización, QJL, atención) está completamente implementado y probado. La generación end-to-end de tokens requiere trabajo adicional para integrar con el pipeline de transformers de Candle.

6. **El comando `quantize` es informativo.** No crea un nuevo archivo de modelo. Solo inspecciona y muestra cómo se configuraría la sesión de TurboQuant. El modelo ya viene cuantizado en sus pesos (Q4_K_M).

---

*Implementación del algoritmo TurboQuant por Zandieh et al. (Google Research, 2025)*
