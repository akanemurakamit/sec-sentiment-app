import io
import json
import re
from collections import Counter

import requests
import streamlit as st
import torch
from bs4 import BeautifulSoup
from pypdf import PdfReader
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "Qwen/Qwen2-0.5B-Instruct"


st.set_page_config(
    page_title="Analizador SEC con Qwen",
    page_icon="📈",
    layout="wide"
)


def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def html_to_text(html):
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript"]):
        tag.extract()

    return clean_text(soup.get_text(" "))


def pdf_to_text(data):
    reader = PdfReader(io.BytesIO(data))
    pages = []

    for page in reader.pages:
        pages.append(page.extract_text() or "")

    return clean_text("\n".join(pages))


def bytes_to_text(data, filename="", content_type=""):
    name = filename.lower()
    content_type = content_type.lower()

    if name.endswith(".pdf") or "pdf" in content_type:
        return pdf_to_text(data)

    sample = data[:3000].lower()

    if name.endswith((".html", ".htm")) or b"<html" in sample or b"<!doctype html" in sample:
        return html_to_text(data.decode("utf-8", errors="ignore"))

    return clean_text(data.decode("utf-8", errors="ignore"))


def fetch_url(url, user_agent):
    if not url.lower().startswith(("http://", "https://")):
        raise ValueError("La URL debe empezar con http:// o https://")

    headers = {
        "User-Agent": user_agent,
        "Accept-Encoding": "gzip, deflate",
    }

    response = requests.get(url, headers=headers, timeout=45)
    response.raise_for_status()

    return bytes_to_text(
        response.content,
        filename=url,
        content_type=response.headers.get("Content-Type", "")
    )


def extract_section(text, query, window_chars):
    if not query.strip():
        return text, False

    lower_text = text.lower()
    lower_query = query.lower().strip()

    idx = lower_text.find(lower_query)

    if idx == -1:
        words = [word for word in re.split(r"\W+", lower_query) if len(word) > 3]

        for word in words:
            idx = lower_text.find(word)

            if idx != -1:
                break

    if idx == -1:
        return text, False

    start = max(0, idx - window_chars)
    end = min(len(text), idx + len(query) + window_chars)

    return text[start:end], True


def split_text(text, chunk_chars):
    text = clean_text(text)
    return [text[i:i + chunk_chars] for i in range(0, len(text), chunk_chars)]


@st.cache_resource(show_spinner="Cargando Qwen2-0.5B-Instruct localmente...")
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        dtype = torch.float16
    else:
        dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        trust_remote_code=True
    )

    model.to(device)
    model.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return tokenizer, model


def generate_response(tokenizer, model, messages, max_new_tokens=300):
    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    except Exception:
        prompt = ""

        for message in messages:
            prompt += f"{message['role'].upper()}: {message['content']}\n"

        prompt += "ASSISTANT:"

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=4096
    ).to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    generated = output_ids[0][inputs["input_ids"].shape[-1]:]

    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def analyze_chunk(tokenizer, model, chunk, section_context):
    messages = [
        {
            "role": "system",
            "content": (
                "Eres un analista financiero especializado en documentos de la SEC. "
                "Tu tarea es clasificar el sentimiento financiero de un texto como "
                "Positivo, Negativo o Neutro/Mixto. Responde en español."
            )
        },
        {
            "role": "user",
            "content": f"""
Analiza el sentimiento financiero del siguiente fragmento de un documento de la SEC.

Contexto de análisis:
{section_context}

Criterios:
- Positivo: crecimiento, oportunidades, mejora de ingresos, expansión, menor riesgo.
- Negativo: pérdidas, incertidumbre, litigios, deuda, caída de ventas, riesgos altos.
- Neutro/Mixto: información descriptiva, legal o balanceada.

Devuelve SOLO este formato:

Sentimiento: Positivo | Negativo | Neutro/Mixto
Confianza: número de 0 a 1
Explicación: explicación breve
Señales clave: 3 bullets breves

Texto:
\"\"\"{chunk}\"\"\"
"""
        }
    ]

    return generate_response(tokenizer, model, messages)


def extract_label(response):
    text = response.lower()

    match = re.search(r"sentimiento\s*:\s*([a-záéíóúüñ/ -]+)", text)

    if match:
        candidate = match.group(1).strip()
    else:
        candidate = text

    if any(word in candidate for word in ["negativo", "negative", "desfavorable"]):
        return "Negativo"

    if any(word in candidate for word in ["positivo", "positive", "favorable"]):
        return "Positivo"

    if any(word in candidate for word in ["neutro", "neutral", "mixto", "mixed"]):
        return "Neutro/Mixto"

    return "No claro"


def summarize_results(tokenizer, model, results):
    joined = "\n\n".join(
        [
            f"Fragmento {result['fragmento']} - {result['sentimiento']}:\n{result['respuesta'][:900]}"
            for result in results
        ]
    )

    messages = [
        {
            "role": "system",
            "content": (
                "Eres un analista financiero. Resume resultados de sentimiento "
                "de documentos SEC en español, de forma clara y ejecutiva."
            )
        },
        {
            "role": "user",
            "content": f"""
Con base en estos análisis por fragmento, genera una conclusión final.

Incluye:
1. Sentimiento general.
2. Justificación.
3. Riesgos o señales principales.
4. Nota sobre limitaciones del análisis.

Resultados:
{joined}
"""
        }
    ]

    return generate_response(tokenizer, model, messages, max_new_tokens=350)


st.title("📈 Analizador de Sentimiento de Documentos SEC con Qwen2-0.5B-Instruct")

st.markdown(
    """
Esta app corre localmente y analiza el sentimiento financiero de un archivo o URL de la SEC.

Puedes analizar todo el documento o una sola sección.
"""
)

with st.sidebar:
    st.header("Configuración")

    user_agent = st.text_input(
        "User-Agent para SEC",
        value="SEC Sentiment App contacto@ejemplo.com",
        help="Cambia este texto por tu nombre/app y correo si vas a consultar la SEC."
    )

    chunk_chars = st.slider(
        "Tamaño de fragmento",
        min_value=1500,
        max_value=6000,
        value=3500,
        step=500
    )

    max_chunks = st.slider(
        "Máximo de fragmentos a analizar",
        min_value=1,
        max_value=15,
        value=6
    )

    st.caption("Entre más fragmentos analices, más tardará la ejecución local.")


source_mode = st.radio(
    "Selecciona la fuente del documento",
    ["Subir archivo", "Pegar URL"]
)

document_text = ""

if source_mode == "Subir archivo":
    uploaded_file = st.file_uploader(
        "Sube un archivo de la SEC",
        type=["txt", "html", "htm", "pdf"]
    )

    if uploaded_file is not None:
        try:
            document_text = bytes_to_text(
                uploaded_file.getvalue(),
                filename=uploaded_file.name
            )
            st.success(f"Archivo cargado: {uploaded_file.name}")
        except Exception as exc:
            st.error(f"No se pudo leer el archivo: {exc}")

else:
    url = st.text_input("Pega la URL del archivo SEC")

    if st.button("Descargar URL"):
        try:
            with st.spinner("Descargando documento..."):
                st.session_state["document_text"] = fetch_url(url, user_agent)

            st.success("Documento descargado correctamente.")

        except Exception as exc:
            st.error(f"No se pudo descargar la URL: {exc}")

    document_text = st.session_state.get("document_text", "")


if document_text:
    st.subheader("Vista previa del documento")

    col1, col2, col3 = st.columns(3)

    col1.metric("Caracteres", f"{len(document_text):,}")
    col2.metric("Palabras aprox.", f"{len(document_text.split()):,}")
    col3.metric("Modelo", MODEL_NAME)

    with st.expander("Ver texto extraído"):
        st.write(document_text[:10000])

    section_mode = st.radio(
        "¿Qué quieres analizar?",
        [
            "Todo el documento",
            "Buscar una sección por palabra clave",
            "Pegar una sección manualmente"
        ]
    )

    selected_text = document_text
    section_context = "Se analiza el documento completo."

    if section_mode == "Buscar una sección por palabra clave":
        query = st.text_input(
            "Palabra clave o título de sección",
            placeholder="Ejemplo: Risk Factors, Management Discussion, Liquidity"
        )

        window_chars = st.slider(
            "Caracteres alrededor de la sección encontrada",
            min_value=3000,
            max_value=20000,
            value=10000,
            step=1000
        )

        selected_text, found = extract_section(document_text, query, window_chars)

        if query:
            if found:
                st.success("Se encontró una sección relacionada.")
                section_context = f"Se analiza la sección relacionada con: {query}"
            else:
                st.warning("No se encontró la sección exacta. Se analizará el documento completo.")

    elif section_mode == "Pegar una sección manualmente":
        manual_text = st.text_area(
            "Pega aquí la sección que quieres analizar",
            height=250
        )

        if manual_text.strip():
            selected_text = manual_text
            section_context = "Se analiza una sección pegada manualmente."

    st.subheader("Texto seleccionado para análisis")

    with st.expander("Ver texto seleccionado"):
        st.write(selected_text[:12000])

    if st.button("Analizar sentimiento", type="primary"):
        chunks = split_text(selected_text, chunk_chars)[:max_chunks]

        if not chunks:
            st.error("No hay texto suficiente para analizar.")
        else:
            tokenizer, model = load_model()

            st.info(f"Se analizarán {len(chunks)} fragmento(s).")

            progress = st.progress(0)
            results = []

            for i, chunk in enumerate(chunks, start=1):
                with st.spinner(f"Analizando fragmento {i}/{len(chunks)}..."):
                    response = analyze_chunk(
                        tokenizer,
                        model,
                        chunk,
                        section_context
                    )

                label = extract_label(response)

                results.append(
                    {
                        "fragmento": i,
                        "sentimiento": label,
                        "respuesta": response
                    }
                )

                progress.progress(i / len(chunks))

            labels = [result["sentimiento"] for result in results]
            counts = Counter(labels)
            final_label = counts.most_common(1)[0][0]

            st.subheader("Resultado general")
            st.metric("Sentimiento dominante", final_label)

            st.write("Distribución de fragmentos:")
            st.json(dict(counts))

            with st.spinner("Generando resumen ejecutivo final..."):
                final_summary = summarize_results(tokenizer, model, results)

            st.subheader("Resumen ejecutivo")
            st.write(final_summary)

            st.subheader("Detalle por fragmento")

            for result in results:
                with st.expander(
                    f"Fragmento {result['fragmento']} - {result['sentimiento']}"
                ):
                    st.write(result["respuesta"])

            output = {
                "modelo": MODEL_NAME,
                "sentimiento_dominante": final_label,
                "distribucion": dict(counts),
                "resumen_final": final_summary,
                "resultados_por_fragmento": results
            }

            st.download_button(
                label="Descargar resultados en JSON",
                data=json.dumps(output, ensure_ascii=False, indent=2),
                file_name="sentimiento_sec.json",
                mime="application/json"
            )

else:
    st.info("Sube un archivo o pega una URL para comenzar.")
