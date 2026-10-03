# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # 01 · Extracción estructurada de fichas de búsqueda con LLM multimodal
# MAGIC
# MAGIC **Proyecto Integrador MNA — Cotejo ante-mortem / post-mortem (Querétaro)**
# MAGIC
# MAGIC **Objetivo.** Convertir cada ficha de búsqueda (PDF o imagen: texto de media filiación + fotografía en vida)
# MAGIC en un registro con **esquema fijo** que alimente el EDA, la vectorización y el índice de Databricks Vector Search.
# MAGIC
# MAGIC **Señales de identificación (según el perito forense) y cómo las trata este notebook:**
# MAGIC
# MAGIC | # | Señal | Fuente | ¿La extrae el LLM? |
# MAGIC |---|-------|--------|--------------------|
# MAGIC | 1 | Datos de la persona / media filiación (sexo, edad, piel, cara, cabello, ojos, estatura…) | Texto de la ficha + foto | Sí |
# MAGIC | 2 | Vestimenta | Texto de la ficha (+ foto si es visible) | Sí |
# MAGIC | 3 | Tatuajes y señas particulares (cicatrices, lunares, perforaciones…) | Texto de la ficha + foto | Sí, con detalle |
# MAGIC | 4 | Perfil genético | Laboratorio de genética | **No.** Se carga como tabla estructurada aparte (celda 12) |
# MAGIC | 5 | Huellas dactilares | AFIS / dactiloscopia | **No.** Solo referencia de disponibilidad (celda 12) |
# MAGIC
# MAGIC Las señales 4 y 5 son identificadores **primarios** (confirman identidad). Las 1–3 son **secundarias** y son las que
# MAGIC usa el retrieval para proponer candidatos. El LLM nunca debe inventar datos genéticos ni dactiloscópicos.
# MAGIC
# MAGIC **Reglas de privacidad aplicadas aquí**
# MAGIC - El LLM **no transcribe el nombre** de la persona. El vínculo con el expediente es el folio, que se seudonimiza (`id_ficha`).
# MAGIC - Las tablas `bronze`/`silver` contienen datos sensibles: deben vivir en un esquema con permisos restringidos (Unity Catalog).
# MAGIC - Verificar con el convenio de la Fiscalía qué endpoint de modelo está autorizado para procesar estos datos.
# MAGIC
# MAGIC **Flujo:** Volume (PDF/JPG/PNG) → render a imagen → LLM con esquema JSON → validación Pydantic → `bronze` (JSON crudo)
# MAGIC → `silver` (tablas planas) → texto canónico por faceta (insumo para embeddings).

# COMMAND ----------

# MAGIC %pip install -q pymupdf pydantic>=2 openai databricks-sdk --upgrade
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md ## 1. Parámetros

# COMMAND ----------

dbutils.widgets.text("catalog", "proyecto_mna", "Catálogo")
dbutils.widgets.text("schema", "fichas_busqueda", "Esquema")
dbutils.widgets.text("volume", "fichas_raw", "Volume con PDFs/imágenes")
dbutils.widgets.text("endpoint", "proyecto_mna.fichas_busqueda.gema", "Modelo (ruta de AI Gateway)")
dbutils.widgets.text("prompt_version", "v1", "Versión del prompt")
dbutils.widgets.text("dpi", "150", "DPI para renderizar PDF")
dbutils.widgets.text("max_workers", "4", "Llamadas concurrentes")
dbutils.widgets.text("limit", "0", "Límite de páginas (0 = todas)")
dbutils.widgets.dropdown("strict_schema", "false", ["true", "false"], "Forzar json_schema estricto")

# COMMAND ----------



CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
VOLUME = dbutils.widgets.get("volume")
ENDPOINT = dbutils.widgets.get("endpoint")
PROMPT_VERSION = dbutils.widgets.get("prompt_version")
DPI = int(dbutils.widgets.get("dpi"))
MAX_WORKERS = int(dbutils.widgets.get("max_workers"))
LIMIT = int(dbutils.widgets.get("limit"))
STRICT_SCHEMA = dbutils.widgets.get("strict_schema") == "true"

VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"
T_BRONZE = f"{CATALOG}.{SCHEMA}.fichas_extraccion_bronze"
T_PERSONA = f"{CATALOG}.{SCHEMA}.fichas_persona_silver"
T_TATUAJES = f"{CATALOG}.{SCHEMA}.fichas_tatuajes_silver"
T_SENAS = f"{CATALOG}.{SCHEMA}.fichas_senas_silver"
T_VESTIMENTA = f"{CATALOG}.{SCHEMA}.fichas_vestimenta_silver"
T_TEXTO = f"{CATALOG}.{SCHEMA}.fichas_texto_embedding"
T_GENETICO = f"{CATALOG}.{SCHEMA}.perfil_genetico"
T_HUELLAS = f"{CATALOG}.{SCHEMA}.huellas_referencia"

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.{VOLUME}")
print("Volume de entrada:", VOLUME_PATH)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Esquema fijo de salida
# MAGIC
# MAGIC Vocabularios controlados (enums) en vez de texto libre: bajan la cardinalidad para el EDA, hacen comparables
# MAGIC ambos lados (ficha vs. libro de necropsias) y reducen ruido en los embeddings. `SIN_DATO` distingue
# MAGIC "no se menciona / no se ve" de un valor real, lo que permite analizar patrones de faltantes.

# COMMAND ----------

from typing import List, Literal, Optional
from pydantic import BaseModel, Field

SIN_DATO = "SIN_DATO"

Sexo = Literal["HOMBRE", "MUJER", "SIN_DATO"]
Complexion = Literal["DELGADA", "MEDIANA", "ROBUSTA", "OBESA", "ATLETICA", "SIN_DATO"]
ColorPiel = Literal["BLANCO", "MORENO_CLARO", "MORENO", "MORENO_OSCURO", "NEGRO", "SIN_DATO"]
FormaCara = Literal["OVALADA", "REDONDA", "CUADRADA", "ALARGADA", "TRIANGULAR", "SIN_DATO"]
ColorCabello = Literal["NEGRO", "CASTANO_OSCURO", "CASTANO_CLARO", "RUBIO", "PELIRROJO",
                       "ENTRECANO", "CANO", "TENIDO", "SIN_CABELLO", "SIN_DATO"]
LargoCabello = Literal["RAPADO", "CORTO", "MEDIANO", "LARGO", "SIN_DATO"]
TipoCabello = Literal["LACIO", "ONDULADO", "RIZADO", "CRESPO", "SIN_DATO"]
ColorOjos = Literal["NEGROS", "CAFES_OSCUROS", "CAFES_CLAROS", "VERDES", "AZULES", "GRISES", "SIN_DATO"]
TamanoOjos = Literal["CHICOS", "MEDIANOS", "GRANDES", "SIN_DATO"]
Nariz = Literal["RECTA", "AGUILENA", "CHATA", "ANCHA", "RESPINGADA", "SIN_DATO"]
Boca = Literal["PEQUENA", "MEDIANA", "GRANDE", "SIN_DATO"]
Labios = Literal["DELGADOS", "MEDIANOS", "GRUESOS", "SIN_DATO"]
VelloFacial = Literal["NINGUNO", "BIGOTE", "BARBA", "BIGOTE_Y_BARBA", "SIN_DATO"]

Region = Literal["CABEZA", "CARA", "CUELLO", "HOMBRO", "PECHO", "ABDOMEN", "ESPALDA", "BRAZO",
                 "ANTEBRAZO", "MUNECA", "MANO", "DEDOS", "CADERA", "GLUTEO", "MUSLO", "RODILLA",
                 "PIERNA", "TOBILLO", "PIE", "OTRA", "SIN_DATO"]
Lado = Literal["IZQUIERDO", "DERECHO", "CENTRAL", "BILATERAL", "SIN_DATO"]
Origen = Literal["TEXTO_FICHA", "FOTO", "AMBOS"]

CategoriaTatuaje = Literal["NOMBRE_O_TEXTO", "FECHA_O_NUMERO", "RELIGIOSO", "ANIMAL", "FLORA",
                           "CALAVERA", "FIGURA_HUMANA", "PERSONAJE", "SIMBOLO", "TRIBAL_ABSTRACTO",
                           "OTRO", "SIN_DATO"]
ColorTinta = Literal["NEGRO_GRIS", "COLOR", "TINTA_BLANCA", "SIN_DATO"]
TipoSena = Literal["CICATRIZ", "LUNAR", "MANCHA", "PERFORACION", "AMPUTACION", "MALFORMACION",
                   "PROTESIS_O_IMPLANTE", "CIRUGIA", "RASGO_DENTAL", "OTRA"]
TipoPrenda = Literal["CAMISA", "PLAYERA", "BLUSA", "SUDADERA", "CHAMARRA", "CHALECO", "SUETER",
                     "PANTALON", "SHORT", "FALDA", "VESTIDO", "ROPA_INTERIOR", "CALZADO",
                     "GORRA_O_SOMBRERO", "ACCESORIO", "OTRA"]
CalidadFoto = Literal["BUENA", "REGULAR", "MALA", "SIN_FOTO"]


class MediaFiliacion(BaseModel):
    complexion: Complexion = SIN_DATO
    color_piel: ColorPiel = SIN_DATO
    forma_cara: FormaCara = SIN_DATO
    cabello_color: ColorCabello = SIN_DATO
    cabello_largo: LargoCabello = SIN_DATO
    cabello_tipo: TipoCabello = SIN_DATO
    ojos_color: ColorOjos = SIN_DATO
    ojos_tamano: TamanoOjos = SIN_DATO
    nariz: Nariz = SIN_DATO
    boca: Boca = SIN_DATO
    labios: Labios = SIN_DATO
    vello_facial: VelloFacial = SIN_DATO
    estatura_cm: Optional[float] = Field(None, description="Solo si está escrita en la ficha")
    peso_kg: Optional[float] = Field(None, description="Solo si está escrito en la ficha")


class Tatuaje(BaseModel):
    region: Region
    lado: Lado
    categoria: CategoriaTatuaje
    motivo: str = Field(description="Descripción breve del diseño, máx. 20 palabras")
    texto_legible: Optional[str] = Field(None, description="Letras/números del tatuaje tal cual, si los hay")
    color_tinta: ColorTinta = SIN_DATO
    tamano_cm: Optional[float] = None
    origen: Origen


class SenaParticular(BaseModel):
    tipo: TipoSena
    region: Region
    lado: Lado
    descripcion: str = Field(description="Descripción breve, máx. 20 palabras")
    tamano_cm: Optional[float] = None
    origen: Origen


class Prenda(BaseModel):
    tipo: TipoPrenda
    color: str = Field(description="Color principal en minúsculas, o 'sin_dato'")
    descripcion: Optional[str] = Field(None, description="Estampado, material, logotipos; máx. 15 palabras")
    marca: Optional[str] = None
    origen: Origen


class FichaExtraida(BaseModel):
    es_ficha_busqueda: bool = Field(description="False si la página no es una ficha de búsqueda")
    folio_unico: Optional[str] = Field(None, description="Folio Único de Identificación (FI..), tal cual")
    sexo: Sexo = SIN_DATO
    edad_desaparicion: Optional[int] = None
    fecha_hechos: Optional[str] = Field(None, description="Formato YYYY-MM-DD")
    estado_hechos: Optional[str] = None
    municipio_hechos: Optional[str] = None
    autoridad_emisora: Optional[str] = None
    media_filiacion: MediaFiliacion
    tatuajes: List[Tatuaje]
    senas_particulares: List[SenaParticular]
    vestimenta: List[Prenda]
    menciona_perfil_genetico: bool = Field(False, description="True solo si la ficha lo menciona explícitamente")
    menciona_huellas: bool = Field(False, description="True solo si la ficha lo menciona explícitamente")
    calidad_foto: CalidadFoto = "SIN_FOTO"
    notas_extraccion: Optional[str] = Field(None, description="Ambigüedades o contradicciones texto vs. foto")


def json_schema_estricto(model) -> dict:
    """Schema JSON apto para response_format strict: sin defaults y con todos los campos requeridos."""
    schema = model.model_json_schema()

    def _fix(node):
        if isinstance(node, dict):
            if node.get("type") == "object" and "properties" in node:
                node["required"] = list(node["properties"].keys())
                node["additionalProperties"] = False
            node.pop("default", None)
            node.pop("title", None)
            for v in node.values():
                _fix(v)
        elif isinstance(node, list):
            for v in node:
                _fix(v)

    _fix(schema)
    return schema


SCHEMA_JSON = json_schema_estricto(FichaExtraida)
print(f"Campos de primer nivel: {len(SCHEMA_JSON['properties'])}")

# COMMAND ----------

# MAGIC %md ## 3. Prompt (versionado)

# COMMAND ----------

import hashlib
import json

SYSTEM_PROMPT = """Eres un asistente de extracción de datos para peritos forenses en México.
Recibes la imagen de una FICHA DE BÚSQUEDA DE PERSONA DESAPARECIDA (texto + fotografía en vida).
Devuelves ÚNICAMENTE un JSON que cumple el esquema dado.

Reglas:
1. Prioridad de fuentes: el TEXTO de la ficha es la fuente principal de la media filiación.
   La FOTO solo complementa lo que el texto no dice (p. ej. tatuajes visibles, vello facial, tipo de cabello).
   Si texto y foto se contradicen, usa el texto y explica la diferencia en `notas_extraccion`.
2. No infieras ni adivines. Si un dato no está escrito ni es claramente visible, usa SIN_DATO o null.
   Edad, estatura, peso, fecha y lugar SOLO se toman del texto.
3. NO transcribas el nombre de la persona ni datos de contacto en ningún campo.
4. Normaliza al vocabulario del esquema (ej. "aguileña" -> AGUILENA, "café" -> CAFES_OSCUROS si no se precisa,
   "chamorro"/"pantorrilla" -> PIERNA, "porlicue" -> MANO).
5. Tatuajes y señas particulares: crea UN elemento por cada tatuaje/seña distinta. Separa región corporal y lado.
   En `texto_legible` copia letras o números del tatuaje exactamente como aparecen (son muy útiles para el cotejo).
   `origen` indica si lo obtuviste del texto, de la foto o de ambos.
6. Vestimenta: un elemento por prenda. Si la ficha dice "SIN DATO", deja la lista vacía.
7. Perfil genético y huellas: solo marca True si la ficha los menciona explícitamente. Nunca generes valores.
8. Si la página no es una ficha de búsqueda, pon es_ficha_busqueda=false y deja el resto vacío/SIN_DATO.
"""

USER_PROMPT = "Extrae la información de esta ficha de búsqueda según el esquema."

# Huella estable del prompt + esquema: si cambian, se reprocesa todo automáticamente.
_huella = hashlib.sha256((SYSTEM_PROMPT + json.dumps(SCHEMA_JSON, sort_keys=True)).encode()).hexdigest()[:8]
PROMPT_FINGERPRINT = f"{PROMPT_VERSION}:{_huella}"
print("Prompt:", PROMPT_FINGERPRINT)

# COMMAND ----------

# MAGIC %md ## 4. Carga de archivos y render a imagen
# MAGIC Cada página de un PDF se trata como una unidad (las compilaciones publicadas traen varias fichas por archivo).

# COMMAND ----------

import base64
import hashlib
import os

import pymupdf  # PyMuPDF

EXT_IMAGEN = {".png", ".jpg", ".jpeg", ".webp"}


def listar_archivos(raiz: str) -> list[str]:
    rutas = []
    for dirpath, _, files in os.walk(raiz):
        for f in files:
            if os.path.splitext(f)[1].lower() in EXT_IMAGEN | {".pdf"}:
                rutas.append(os.path.join(dirpath, f))
    return sorted(rutas)


def paginas_como_png(ruta: str, dpi: int = 150) -> list[dict]:
    """Devuelve [{ruta, pagina, sha256, mime, b64}] por cada página/imagen."""
    ext = os.path.splitext(ruta)[1].lower()
    salida = []
    if ext == ".pdf":
        with pymupdf.open(ruta) as doc:
            for i, page in enumerate(doc):
                png = page.get_pixmap(dpi=dpi).tobytes("png")
                salida.append({"ruta": ruta, "pagina": i + 1, "mime": "image/png", "bytes": png})
    else:
        with open(ruta, "rb") as fh:
            data = fh.read()
        mime = "image/jpeg" if ext in {".jpg", ".jpeg"} else f"image/{ext[1:]}"
        salida.append({"ruta": ruta, "pagina": 1, "mime": mime, "bytes": data})
    for p in salida:
        p["sha256"] = hashlib.sha256(p["bytes"]).hexdigest()
        p["b64"] = base64.b64encode(p.pop("bytes")).decode()
    return salida


archivos = listar_archivos(VOLUME_PATH)
print(f"{len(archivos)} archivos en {VOLUME_PATH}")

# COMMAND ----------

import unicodedata

# Sinónimos frecuentes en fichas/necropsias -> valor del vocabulario controlado.
# Se aplican solo si el resultado es válido para ESE campo.
SINONIMOS = {
    # cabello
    "LISO": "LACIO", "CHINO": "RIZADO", "QUEBRADO": "ONDULADO", "CALVO": "SIN_CABELLO",
    "CANOSO": "CANO", "BLANCO": "CANO", "CAFE": "CASTANO_OSCURO", "CASTANO": "CASTANO_OSCURO",
    "PINTADO": "TENIDO", "TENIDO": "TENIDO", "PELON": "RAPADO",
    # ojos
    "CAFES": "CAFES_OSCUROS", "CAFE_OSCURO": "CAFES_OSCUROS", "CAFE_CLARO": "CAFES_CLAROS",
    "CHICOS": "CHICOS", "PEQUENOS": "CHICOS", "CHICA": "PEQUENA", "CHICO": "PEQUENA",
    # piel / cara / nariz
    "TRIGUENO": "MORENO_CLARO", "APIÑONADO": "MORENO_CLARO", "APINONADO": "MORENO_CLARO",
    "OVALO": "OVALADA", "OVAL": "OVALADA", "REDONDO": "REDONDA", "CHATO": "CHATA",
    "GRUESA": "ANCHA", "RESPINGADO": "RESPINGADA", "DELGADO": "DELGADA",
    # región corporal
    "MEJILLA": "CARA", "POMULO": "CARA", "FRENTE": "CARA", "CEJA": "CARA", "OJO": "CARA",
    "LABIO": "CARA", "MENTON": "CARA", "NARIZ": "CARA", "OREJA": "CABEZA", "NUCA": "CUELLO",
    "CUERO_CABELLUDO": "CABEZA", "TORAX": "PECHO", "SENO": "PECHO", "CLAVICULA": "PECHO",
    "ESTOMAGO": "ABDOMEN", "VIENTRE": "ABDOMEN", "OMBLIGO": "ABDOMEN", "COSTADO": "ABDOMEN",
    "COSTILLA": "ABDOMEN", "OMOPLATO": "ESPALDA", "ESPALDA_BAJA": "ESPALDA", "BICEP": "BRAZO",
    "BICEPS": "BRAZO", "CODO": "BRAZO", "PALMA": "MANO", "PORLICUE": "MANO", "NUDILLOS": "DEDOS",
    "DEDO": "DEDOS", "NALGA": "GLUTEO", "PANTORRILLA": "PIERNA", "PANTORILLA": "PIERNA",
    "CHAMORRO": "PIERNA", "ESPINILLA": "PIERNA", "TALON": "PIE", "INGLE": "CADERA",
    # lado
    "AMBOS": "BILATERAL", "AMBOS_LADOS": "BILATERAL", "CENTRO": "CENTRAL", "MEDIO": "CENTRAL",
    # vestimenta
    "TENIS": "CALZADO", "ZAPATOS": "CALZADO", "ZAPATO": "CALZADO", "BOTAS": "CALZADO",
    "HUARACHES": "CALZADO", "SANDALIAS": "CALZADO", "CHANCLAS": "CALZADO", "JEANS": "PANTALON",
    "PANTALON_DE_MEZCLILLA": "PANTALON", "MEZCLILLA": "PANTALON", "PANTS": "PANTALON",
    "BERMUDA": "SHORT", "SHORTS": "SHORT", "POLO": "PLAYERA", "CAMISETA": "PLAYERA",
    "CHAQUETA": "CHAMARRA", "ABRIGO": "CHAMARRA", "JERSEY": "SUETER", "SUETER": "SUETER",
    "GORRA": "GORRA_O_SOMBRERO", "SOMBRERO": "GORRA_O_SOMBRERO", "CACHUCHA": "GORRA_O_SOMBRERO",
    "CINTURON": "ACCESORIO", "COLLAR": "ACCESORIO", "PULSERA": "ACCESORIO", "RELOJ": "ACCESORIO",
    "ANILLO": "ACCESORIO", "ARETES": "ACCESORIO", "LENTES": "ACCESORIO", "MOCHILA": "ACCESORIO",
    "BOXER": "ROPA_INTERIOR", "CALZON": "ROPA_INTERIOR", "BRASIER": "ROPA_INTERIOR",
    "SOSTEN": "ROPA_INTERIOR", "CALCETINES": "ROPA_INTERIOR",
    # tatuajes
    "LETRAS": "NOMBRE_O_TEXTO", "LETRA": "NOMBRE_O_TEXTO", "TEXTO": "NOMBRE_O_TEXTO",
    "NOMBRE": "NOMBRE_O_TEXTO", "FRASE": "NOMBRE_O_TEXTO", "FECHA": "FECHA_O_NUMERO",
    "NUMEROS": "FECHA_O_NUMERO", "NUMERO": "FECHA_O_NUMERO", "CRUZ": "RELIGIOSO",
    "VIRGEN": "RELIGIOSO", "ROSARIO": "RELIGIOSO", "SANTA_MUERTE": "RELIGIOSO", "ROSA": "FLORA",
    "FLOR": "FLORA", "CRANEO": "CALAVERA", "ESTRELLA": "SIMBOLO", "CORAZON": "SIMBOLO",
    "TRIBAL": "TRIBAL_ABSTRACTO", "ABSTRACTO": "TRIBAL_ABSTRACTO", "NEGRO": "NEGRO_GRIS",
    "GRIS": "NEGRO_GRIS", "NEGRA": "NEGRO_GRIS", "COLORES": "COLOR", "A_COLOR": "COLOR",
    "BLANCA": "TINTA_BLANCA",
    # genéricos de faltante
    "NO_ESPECIFICADO": "SIN_DATO", "NINGUNO": "SIN_DATO", "DESCONOCIDO": "SIN_DATO",
    "NO_VISIBLE": "SIN_DATO", "N/A": "SIN_DATO", "NA": "SIN_DATO", "NULL": "SIN_DATO",
}

def _canon(txt: str) -> str:
    t = unicodedata.normalize("NFKD", str(txt)).encode("ascii", "ignore").decode().upper().strip()
    return "_".join(t.replace("-", " ").replace("/", " ").split())


def _variantes(v: str):
    """Variantes de género/número (IZQUIERDA->IZQUIERDO, GRANDE->GRANDES) y su sinónimo."""
    formas = [v, v + "S", v + "ES"]
    if v.endswith("S"):
        formas.append(v[:-1])
    if v[-1:] in "AO":
        swap = v[:-1] + ("O" if v[-1] == "A" else "A")
        formas += [swap, swap + "S"]
    for f in formas:
        yield f
        if f in SINONIMOS:
            yield SINONIMOS[f]


def _norm_enum(valor, enum, ruta, log):
    if valor is None:
        return "SIN_DATO" if "SIN_DATO" in enum else valor
    v = _canon(valor)
    for cand in _variantes(v):
        if cand in enum:
            if cand != v:  # no registra cambios solo de mayúsculas/acentos
                log.append(f"{ruta}: {valor}->{cand}")
            return cand
    destino = "OTRA" if "OTRA" in enum else ("SIN_DATO" if "SIN_DATO" in enum else enum[0])
    log.append(f"{ruta}: {valor}->{destino} (fuera de vocabulario)")
    return destino


def _resolver(s):
    return SCHEMA_JSON["$defs"][s["$ref"].split("/")[-1]] if "$ref" in s else s


def _norm(valor, s, ruta, log):
    s = _resolver(s)
    if "anyOf" in s:
        if valor is None:
            return None
        no_nulos = [_resolver(o) for o in s["anyOf"] if o.get("type") != "null"]
        s = no_nulos[0] if no_nulos else s
    if "enum" in s:
        return _norm_enum(valor, s["enum"], ruta, log)
    if s.get("type") == "object" and isinstance(valor, dict):
        props = s.get("properties", {})
        return {k: (_norm(v, props[k], f"{ruta}.{k}".lstrip("."), log) if k in props else v)
                for k, v in valor.items()}
    if s.get("type") == "array" and isinstance(valor, list):
        return [_norm(v, s["items"], f"{ruta}[{i}]", log) for i, v in enumerate(valor)]
    return valor


def normalizar_salida(datos: dict) -> tuple[dict, list[str]]:
    """Ajusta la salida del LLM al vocabulario controlado y devuelve la lista de cambios aplicados."""
    log: list[str] = []
    return _norm(datos, SCHEMA_JSON, "", log), log

# COMMAND ----------

# MAGIC %md ## 5. Cliente del LLM y llamada con validación

# COMMAND ----------

import time
from datetime import datetime, timezone

from openai import OpenAI
from pydantic import ValidationError

# Host y token desde el contexto del notebook (no se pega el token)
_ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
HOST = _ctx.apiUrl().get().rstrip("/")
client = OpenAI(base_url=f"{HOST}/ai-gateway/mlflow/v1", api_key=_ctx.apiToken().get())


def _formato_texto():
    """Formato de salida en la Responses API (parámetro `text`)."""
    if STRICT_SCHEMA:
        return {"format": {"type": "json_schema", "name": "ficha_extraida",
                           "schema": SCHEMA_JSON, "strict": True}}
    return None


def _limpiar_json(texto: str) -> str:
    """Quita ``` y cualquier texto antes/después del objeto JSON."""
    t = (texto or "").strip()
    ini, fin = t.find("{"), t.rfind("}")
    return t[ini:fin + 1] if ini != -1 and fin > ini else t


def extraer_pagina(pag: dict, reintentos: int = 3) -> dict:
    """Llama al LLM; devuelve un dict listo para la tabla bronze (status ok/error)."""
    instrucciones = SYSTEM_PROMPT
    if not STRICT_SCHEMA:
        instrucciones += ("\nResponde SOLO con un objeto JSON (sin texto adicional ni ```) que cumpla este esquema:\n"
                          + json.dumps(SCHEMA_JSON, ensure_ascii=False))
    entrada = [
        {"role": "system", "content": [{"type": "input_text", "text": instrucciones}]},
        {"role": "user", "content": [
            {"type": "input_text", "text": USER_PROMPT},
            {"type": "input_image", "image_url": f"data:{pag['mime']};base64,{pag['b64']}"},
        ]},
    ]
    kwargs = {"text": _formato_texto()} if STRICT_SCHEMA else {}
    base = {"ruta": pag["ruta"], "pagina": pag["pagina"], "sha256": pag["sha256"],
            "modelo": ENDPOINT, "prompt": PROMPT_FINGERPRINT}
    ultimo_error = None
    for intento in range(1, reintentos + 1):
        t0 = time.time()
        try:
            resp = client.responses.create(
                model=ENDPOINT, input=entrada, temperature=0, max_output_tokens=4000, **kwargs,
            )
            datos = json.loads(_limpiar_json(resp.output_text))
            datos, cambios = normalizar_salida(datos)
            if cambios:
                datos["notas_extraccion"] = " | ".join(
                filter(None, [datos.get("notas_extraccion"), "normalizado: " + "; ".join(cambios)]))
            ficha = FichaExtraida.model_validate(datos)
            return {**base, "status": "ok", "error": None, "json": ficha.model_dump_json(),
                    "tokens_in": getattr(resp.usage, "input_tokens", None),
                    "tokens_out": getattr(resp.usage, "output_tokens", None),
                    "latencia_s": round(time.time() - t0, 2), "intentos": intento,
                    "ts": datetime.now(timezone.utc)}
        except ValidationError as e:
            ultimo_error = f"validacion: {e.errors()[:3]}"
        except Exception as e:
            ultimo_error = f"{type(e).__name__}: {str(e)[:500]}"
            time.sleep(2 ** intento)
        except json.JSONDecodeError as e:
            ultimo_error = f"json_invalido: {e}"
    return {**base, "status": "error", "error": ultimo_error, "json": None, "tokens_in": None,
            "tokens_out": None, "latencia_s": None, "intentos": reintentos, "ts": datetime.now(timezone.utc)}

# COMMAND ----------

# MAGIC %md ## 6. Prueba con una sola ficha
# MAGIC Revisa visualmente el resultado contra la ficha antes de correr el lote completo.

# COMMAND ----------

r_ping = client.responses.create(
    model=ENDPOINT, max_output_tokens=50,
    input=[{"role": "user", "content": [{"type": "input_text", "text": "Responde solo: OK"}]}],
)
print("Conexión:", r_ping.output_text)

# COMMAND ----------

if archivos:
    prueba = paginas_como_png(archivos[0], DPI)[0]
    r = extraer_pagina(prueba)
    print(r["status"], r["error"] or "", f"| {r['latencia_s']} s | tokens {r['tokens_in']}/{r['tokens_out']}")
    if r["json"]:
        print(json.dumps(json.loads(r["json"]), indent=2, ensure_ascii=False))

# COMMAND ----------

# MAGIC %md ## 7. Procesamiento por lote → `bronze`
# MAGIC Idempotente: no reprocesa páginas ya extraídas con el mismo hash de imagen + modelo + versión de prompt.

# COMMAND ----------

# DBTITLE 1,7. Procesamiento por lote → bronze
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {T_BRONZE} (
  ruta STRING, pagina INT, sha256 STRING, modelo STRING, prompt STRING,
  status STRING, error STRING, json STRING,
  tokens_in INT, tokens_out INT, latencia_s DOUBLE, intentos INT, ts TIMESTAMP
) COMMENT 'Salida cruda del LLM por página de ficha. CONTIENE DATOS SENSIBLES.'
""")

ya = {(r.sha256, r.modelo, r.prompt) for r in
      spark.table(T_BRONZE).where("status = 'ok'").select("sha256", "modelo", "prompt").collect()}

pendientes = []
for ruta in archivos:
    for pag in paginas_como_png(ruta, DPI):
        if (pag["sha256"], ENDPOINT, PROMPT_FINGERPRINT) not in ya:
            pendientes.append(pag)
if LIMIT > 0:
    pendientes = pendientes[:LIMIT]
print(f"Páginas pendientes: {len(pendientes)}")

resultados = []
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
    futuros = [pool.submit(extraer_pagina, p) for p in pendientes]
    for i, f in enumerate(as_completed(futuros), 1):
        resultados.append(f.result())
        if i % 25 == 0:
            print(f"  {i}/{len(pendientes)}")

if resultados:
    df_res = pd.DataFrame(resultados)
    # pandas convierte None → NaN (float64) en columnas INT; Spark no puede castear NaN a INT
    for c in ("tokens_in", "tokens_out"):
        df_res[c] = df_res[c].astype("Int32")
    spark.createDataFrame(df_res, schema=spark.table(T_BRONZE).schema).write.mode("overwrite").saveAsTable(T_BRONZE)
    print(df_res["status"].value_counts().to_string())

# COMMAND ----------

display(spark.sql(f"""
  SELECT status, count(*) n, round(avg(latencia_s),1) lat_media_s,
         sum(tokens_in) tok_in, sum(tokens_out) tok_out
  FROM {T_BRONZE} WHERE prompt = '{PROMPT_FINGERPRINT}' GROUP BY status
"""))

# COMMAND ----------

# MAGIC %md ## 8. Aplanado → tablas `silver`
# MAGIC Una fila por ficha (persona) y tablas hijas para tatuajes, señas y vestimenta (relación 1:N).
# MAGIC `id_ficha` = SHA-256 del folio (o del hash de la imagen si no hay folio): seudónimo estable para unir tablas.

# COMMAND ----------

def id_ficha(ficha: FichaExtraida, sha_img: str) -> str:
    clave = (ficha.folio_unico or "").strip().upper() or f"IMG:{sha_img}"
    return hashlib.sha256(clave.encode()).hexdigest()[:24]


def aplanar(filas_bronze: list[dict]) -> dict[str, pd.DataFrame]:
    persona, tatuajes, senas, ropa = [], [], [], []
    for fb in filas_bronze:
        ficha = FichaExtraida.model_validate_json(fb["json"])
        if not ficha.es_ficha_busqueda:
            continue
        fid = id_ficha(ficha, fb["sha256"])
        persona.append({
            "id_ficha": fid, "folio_unico": ficha.folio_unico, "sexo": ficha.sexo,
            "edad_desaparicion": ficha.edad_desaparicion, "fecha_hechos": ficha.fecha_hechos,
            "estado_hechos": ficha.estado_hechos, "municipio_hechos": ficha.municipio_hechos,
            "autoridad_emisora": ficha.autoridad_emisora,
            **ficha.media_filiacion.model_dump(),
            "n_tatuajes": len(ficha.tatuajes), "n_senas": len(ficha.senas_particulares),
            "n_prendas": len(ficha.vestimenta),
            "menciona_perfil_genetico": ficha.menciona_perfil_genetico,
            "menciona_huellas": ficha.menciona_huellas, "calidad_foto": ficha.calidad_foto,
            "notas_extraccion": ficha.notas_extraccion,
            "ruta": fb["ruta"], "pagina": fb["pagina"], "modelo": fb["modelo"], "prompt": fb["prompt"],
        })
        tatuajes += [{"id_ficha": fid, "n": i, **t.model_dump()} for i, t in enumerate(ficha.tatuajes, 1)]
        senas += [{"id_ficha": fid, "n": i, **s.model_dump()} for i, s in enumerate(ficha.senas_particulares, 1)]
        ropa += [{"id_ficha": fid, "n": i, **p.model_dump()} for i, p in enumerate(ficha.vestimenta, 1)]
    return {"persona": pd.DataFrame(persona), "tatuajes": pd.DataFrame(tatuajes),
            "senas": pd.DataFrame(senas), "vestimenta": pd.DataFrame(ropa)}

# COMMAND ----------

# DBTITLE 1,8. Aplanado → silver
filas_ok = [r.asDict() for r in spark.sql(f"""
  SELECT * FROM {T_BRONZE}
  WHERE status = 'ok' AND prompt = '{PROMPT_FINGERPRINT}' AND modelo = '{ENDPOINT}'
  QUALIFY row_number() OVER (PARTITION BY sha256 ORDER BY ts DESC) = 1
""").collect()]

tablas = aplanar(filas_ok)
# La misma ficha puede venir en varias compilaciones: nos quedamos con una por id_ficha.
tablas["persona"] = tablas["persona"].drop_duplicates("id_ficha", keep="last")
ids = set(tablas["persona"]["id_ficha"]) if len(tablas["persona"]) else set()
for k in ("tatuajes", "senas", "vestimenta"):
    if len(tablas[k]):
        tablas[k] = tablas[k].drop_duplicates(["id_ficha", "n"], keep="last")

for nombre, tabla in [("persona", T_PERSONA), ("tatuajes", T_TATUAJES),
                      ("senas", T_SENAS), ("vestimenta", T_VESTIMENTA)]:
    pdf = tablas[nombre]
    if len(pdf):
        # edad_desaparicion es Optional[int]; None → NaN(float64) → Spark infiere DOUBLE, no INT
        if nombre == "persona" and "edad_desaparicion" in pdf.columns:
            pdf["edad_desaparicion"] = pdf["edad_desaparicion"].astype("Int32")
        spark.createDataFrame(pdf).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(tabla)
    print(f"{tabla}: {len(pdf)} filas")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Texto canónico por faceta (insumo para embeddings)
# MAGIC Se construye de forma **determinista** desde los campos estructurados (no es texto libre del LLM),
# MAGIC omitiendo `SIN_DATO` y cualquier identificador. Se separa por faceta porque tienen distinta estabilidad:
# MAGIC rasgos y tatuajes casi no cambian; la ropa solo sirve para hallazgos cercanos a la fecha de desaparición.
# MAGIC El mismo constructor se aplicará al libro de necropsias para que ambos lados queden en el mismo "idioma".

# COMMAND ----------

# DBTITLE 1,9. Texto canónico por faceta
def _v(x):
    return None if x in (None, "", SIN_DATO, "sin_dato") else str(x).lower().replace("_", " ")


def texto_rasgos(p: dict) -> str:
    partes = [
        f"sexo {_v(p['sexo'])}" if _v(p["sexo"]) else None,
        f"complexión {_v(p['complexion'])}" if _v(p["complexion"]) else None,
        f"piel {_v(p['color_piel'])}" if _v(p["color_piel"]) else None,
        f"cara {_v(p['forma_cara'])}" if _v(p["forma_cara"]) else None,
        "cabello " + " ".join(filter(None, [_v(p["cabello_color"]), _v(p["cabello_largo"]), _v(p["cabello_tipo"])]))
        if any(_v(p[c]) for c in ("cabello_color", "cabello_largo", "cabello_tipo")) else None,
        "ojos " + " ".join(filter(None, [_v(p["ojos_color"]), _v(p["ojos_tamano"])]))
        if any(_v(p[c]) for c in ("ojos_color", "ojos_tamano")) else None,
        f"nariz {_v(p['nariz'])}" if _v(p["nariz"]) else None,
        f"boca {_v(p['boca'])}" if _v(p["boca"]) else None,
        f"labios {_v(p['labios'])}" if _v(p["labios"]) else None,
        f"vello facial {_v(p['vello_facial'])}" if _v(p["vello_facial"]) else None,
        f"estatura {int(p['estatura_cm'])} cm" if pd.notna(p.get("estatura_cm")) else None,
        f"peso {int(p['peso_kg'])} kg" if pd.notna(p.get("peso_kg")) else None,
    ]
    return "; ".join(x for x in partes if x)


def texto_tatuajes(rows: list[dict]) -> str:
    out = []
    for t in rows:
        s = f"tatuaje en {_v(t['region']) or 'región no especificada'}"
        if _v(t["lado"]):
            s += f" {_v(t['lado'])}"
        s += f": {t['motivo'].lower()}"
        if t.get("texto_legible"):
            s += f", texto '{t['texto_legible']}'"
        if _v(t["color_tinta"]):
            s += f", tinta {_v(t['color_tinta'])}"
        out.append(s)
    return "; ".join(out)


def texto_senas(rows: list[dict]) -> str:
    return "; ".join(
        f"{_v(s['tipo'])} en {_v(s['region']) or 'región no especificada'}"
        + (f" {_v(s['lado'])}" if _v(s["lado"]) else "") + f": {s['descripcion'].lower()}"
        for s in rows)


def texto_vestimenta(rows: list[dict]) -> str:
    return "; ".join(
        " ".join(filter(None, [_v(r["tipo"]), _v(r["color"]), (r.get("descripcion") or "").lower() or None,
                               f"marca {r['marca']}" if r.get("marca") else None]))
        for r in rows)


def construir_textos(t: dict[str, pd.DataFrame]) -> pd.DataFrame:
    def agrupar(df):
        return {} if not len(df) else {k: g.sort_values("n").to_dict("records") for k, g in df.groupby("id_ficha")}
    tat, sen, rop = agrupar(t["tatuajes"]), agrupar(t["senas"]), agrupar(t["vestimenta"])
    filas = []
    for p in t["persona"].to_dict("records"):
        fid = p["id_ficha"]
        filas.append({
            "id_ficha": fid,
            # columnas de blocking (filtros duros en Vector Search)
            "sexo": p["sexo"], "edad_desaparicion": p["edad_desaparicion"],
            "fecha_hechos": p["fecha_hechos"], "estado_hechos": p["estado_hechos"],
            "municipio_hechos": p["municipio_hechos"],
            # facetas de texto
            "texto_rasgos": texto_rasgos(p),
            "texto_tatuajes": texto_tatuajes(tat.get(fid, [])),
            "texto_senas": texto_senas(sen.get(fid, [])),
            "texto_vestimenta": texto_vestimenta(rop.get(fid, [])),
        })
    df = pd.DataFrame(filas)
    if len(df):
        df["texto_completo"] = df[["texto_rasgos", "texto_tatuajes", "texto_senas", "texto_vestimenta"]].apply(
            lambda r: " | ".join(x for x in r if x), axis=1)
    return df


df_texto = construir_textos(tablas)
if len(df_texto):
    # edad_desaparicion es Optional[int]; None → NaN(float64) → Spark infiere DOUBLE, no INT
    if "edad_desaparicion" in df_texto.columns:
        df_texto["edad_desaparicion"] = df_texto["edad_desaparicion"].astype("Int32")
    spark.createDataFrame(df_texto).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(T_TEXTO)
    spark.sql(f"ALTER TABLE {T_TEXTO} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")  # requerido por Vector Search
display(spark.table(T_TEXTO).select("id_ficha", "sexo", "edad_desaparicion", "texto_completo").limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Señales 4 y 5: perfil genético y huellas (fuera del LLM)
# MAGIC
# MAGIC Estos datos **no se extraen con IA**: vienen de laboratorio (genética) y de AFIS (dactiloscopia) y se unen por
# MAGIC `id_ficha` / número de registro. Se definen en formato **largo** (una fila por marcador) para no fijar columnas
# MAGIC antes de conocer la tabla real que mostró el perito (≈5–6 características numéricas/alfanuméricas).
# MAGIC Ajustar nombres de marcadores cuando se tenga el formato oficial.
# MAGIC
# MAGIC En el pipeline funcionan como **confirmación** del candidato (identificador primario), no como insumo de embeddings.

# COMMAND ----------

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {T_GENETICO} (
  id_registro STRING COMMENT 'id_ficha (ante-mortem: familiar de referencia) o número de necropsia (post-mortem)',
  lado STRING COMMENT 'ANTE_MORTEM | POST_MORTEM',
  tipo_muestra STRING COMMENT 'p. ej. familiar_directo, hueso, sangre',
  marcador STRING COMMENT 'nombre del locus / característica',
  valor_1 STRING, valor_2 STRING,
  laboratorio STRING, fecha_resultado DATE
) COMMENT 'Perfil genético en formato largo. ALTAMENTE SENSIBLE. No se procesa con LLM.'
""")

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {T_HUELLAS} (
  id_registro STRING, lado STRING,
  disponible BOOLEAN, sistema STRING COMMENT 'AFIS u otro', referencia_externa STRING,
  calidad STRING, fecha_registro DATE
) COMMENT 'Solo referencia a la existencia de huellas; el cotejo lo hace el sistema dactiloscópico.'
""")
print("Tablas de identificadores primarios listas:", T_GENETICO, T_HUELLAS)

# COMMAND ----------

# MAGIC %md ## 11. Controles de calidad rápidos (alimentan el EDA)

# COMMAND ----------

from pyspark.sql import functions as F

per = spark.table(T_PERSONA)
cols_cat = ["sexo", "complexion", "color_piel", "forma_cara", "cabello_color", "cabello_largo", "cabello_tipo",
            "ojos_color", "ojos_tamano", "nariz", "boca", "labios", "vello_facial"]
n = per.count()
faltantes = per.select(
    *[F.round(100 * F.avg((F.col(c) == SIN_DATO).cast("int")), 1).alias(c) for c in cols_cat],
    *[F.round(100 * F.avg(F.col(c).isNull().cast("int")), 1).alias(c)
      for c in ["edad_desaparicion", "estatura_cm", "peso_kg", "fecha_hechos", "municipio_hechos"]],
)
print(f"Fichas: {n}  |  % SIN_DATO / nulos por campo:")
display(faltantes)

display(per.groupBy("calidad_foto").count())
display(per.select(
    F.round(F.avg((F.col("n_tatuajes") > 0).cast("int")) * 100, 1).alias("pct_con_tatuajes"),
    F.round(F.avg((F.col("n_senas") > 0).cast("int")) * 100, 1).alias("pct_con_senas"),
    F.round(F.avg((F.col("n_prendas") > 0).cast("int")) * 100, 1).alias("pct_con_vestimenta"),
))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Siguientes pasos
# MAGIC 1. **Validación manual de la extracción:** revisar ~30 fichas contra la imagen y medir exactitud por campo
# MAGIC    (es la métrica de calidad del paso 1 en CRISP-ML; también permite comparar endpoints/prompts).
# MAGIC 2. **EDA de fichas** sobre las tablas `silver`: faltantes por autoridad emisora, cardinalidad de `motivo`/color,
# MAGIC    distribución de edad y sexo, tendencia temporal por `fecha_hechos`.
# MAGIC 3. **Aplicar el mismo esquema al libro de necropsias** (campos de vestimenta, cicatrices, tatuajes y señas) para
# MAGIC    homologar ambos lados antes de vectorizar.
# MAGIC 4. **Notebook 02:** embeddings por faceta + índice de Databricks Vector Search con filtros de blocking.