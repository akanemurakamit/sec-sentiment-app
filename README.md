# Analizador de Sentimiento SEC con Qwen2-0.5B-Instruct

![Captura de la app](assets/app.png)

Esta aplicación permite analizar el sentimiento financiero de un documento de la SEC usando un modelo local con Instruct: `Qwen/Qwen2-0.5B-Instruct`.

## Objetivo

La app permite entender el sentimiento de un archivo dado de la SEC, ya sea analizando el documento completo o una sección específica.

## Funciones principales

- Subir un archivo de la SEC.
- Pegar una URL de un documento de la SEC.
- Analizar todo el documento.
- Buscar una sección específica por palabra clave.
- Pegar una sección manualmente.
- Clasificar el sentimiento como:
  - Positivo
  - Negativo
  - Neutro/Mixto
- Generar un resumen ejecutivo.
- Descargar los resultados en formato JSON.
- Ejecutarse localmente sin usar API keys externas.

## Modelo utilizado

La aplicación usa el modelo:

```text
Qwen/Qwen2-0.5B-Instruct
