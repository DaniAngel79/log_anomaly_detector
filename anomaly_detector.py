import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- CONFIGURACIÓN ---
MODEL_NAME = "log_anomaly_detector_model" # Nombre ficticio para un modelo de logs entrenado

# Suponemos un dataset muy pequeño para el ejemplo (en un entorno real cargarías un CSV grande)
sample_logs = [
    "ERROR: System reboot failed, kernel panic detected.",
    "INFO: User danilo logged in successfully.",
    "WARNING: Low disk space on /var/log, 10% remaining.",
    "ERROR: Fatal exception in main process, resource timeout.",
    "INFO: Database backup completed without issues."
]

# --- FUNCIÓN PRINCIPAL DE DETECCIÓN ---
def detect_anomalies(logs):
    """
    Carga el modelo y clasifica cada línea de log.
    0: Normal, 1: Anómalo (ERROR o WARNING)
    """
    print(f"--- Inicializando Modelo: {MODEL_NAME} ---")

    # 1. Cargar el Tokenizador y el Modelo (Esto usa los recursos de Colab)
    # NOTA: Usamos un modelo público de ejemplo para esta prueba.
    try:
        tokenizer = AutoTokenizer.from_pretrained("d4rk-lucif3r/autotrain-log_anomaly_detection-881525996")
        model = AutoModelForSequenceClassification.from_pretrained("d4rk-lucif3r/autotrain-log_anomaly_detection-881525996")
    except Exception as e:
        print(f"\n[AVISO IMPORTANTE]: No se pudo cargar el modelo remoto. Ejecutando lógica simple.")
        # Si la carga del modelo falla (p. ej., sin conexión a internet o token faltante), 
        # usamos una lógica de detección basada en palabras clave como fallback.
        return detect_anomalies_fallback(logs)


    results = []
    for log_line in logs:
        # Tokenizar la línea de log
        inputs = tokenizer(log_line, return_tensors="pt", truncation=True, padding=True)

        # Predecir
        with torch.no_grad():
            outputs = model(**inputs)

        # Obtener el resultado (índice 0 = Normal, índice 1 = Anómalo)
        prediction = torch.argmax(outputs.logits, dim=1).item()

        status = "ANOMALÍA (MODELO)" if prediction == 1 else "Normal (MODELO)"
        results.append((log_line, status))

    return results

# --- FUNCIÓN DE FALLBACK (si el modelo no carga) ---
def detect_anomalies_fallback(logs):
    results = []
    keywords = ["ERROR", "FATAL", "PANIC", "TIMEOUT"]
    for log_line in logs:
        is_anomaly = any(keyword in log_line.upper() for keyword in keywords)
        status = "ANOMALÍA (KEYWORDS)" if is_anomaly else "Normal (KEYWORDS)"
        results.append((log_line, status))
    return results


if __name__ == "__main__":
    print("--- INICIO DE LA DETECCIÓN DE ANOMALÍAS DE LOGS ---")

    anomalies = detect_anomalies(sample_logs)

    print("\n--- RESULTADOS ---")
    for log, status in anomalies:
        print(f"[{status.ljust(20)}] {log}")

    print("\n--- DETECCIÓN FINALIZADA ---")
