# elder_brain_v8_5_pro_prod.py
# (baseado no seu elder_brain_v8_5_pro.py, com segurança de PROD)
"""
Elder Brain Analytics — v9.1 Corporate (PROD)
- Keys lidas via st.secrets (sem exibir para o usuário)
- Modo Admin protegido por senha (ADMIN_PASSWORD em secrets)
- Tokens & Custos visíveis apenas ao Admin
- Mantém todas as funcionalidades (Extração, Análise, Gráficos, PDF, Chat, Treinamento)
"""

import os, io, re, json, time, tempfile
from datetime import datetime
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
from fpdf import FPDF
from pdfminer.high_level import extract_text
import streamlit as st
import os

# Remove variáveis de ambiente de proxy que podem causar conflitos
for proxy_var in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'ALL_PROXY', 'all_proxy', 'GROQ_PROXY']:
    os.environ.pop(proxy_var, None)
# ========= LLM Clients =========
try:
    from groq import Groq
except Exception:
    Groq = None
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# ========= Token helpers (para estimar quando a API não retornar usage) =========
try:
    import tiktoken
except Exception:
    tiktoken = None

# ========= Diretórios / Constantes =========
TRAINING_DIR = "training_data"
PROCESSED_DIR = "relatorios_processados"
os.makedirs(TRAINING_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

MODELOS_SUGERIDOS_GROQ = [
    "llama-3.1-8b-instant",
    "llama-3.1-70b-versatile",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
]
MODELOS_SUGERIDOS_OPENAI = ["gpt-4o-mini", "gpt-4o"]

MAX_TOKENS_FIXED = 4096
TEMP_FIXED = 0.3
# preços apenas referência (mantidos do seu código)
GPT_PRICE_INPUT_PER_1K = 0.005
GPT_PRICE_OUTPUT_PER_1K = 0.015

# ========= Tema / CSS =========
DARK_CSS = """
<style>
:root{
  --bg:#20152b; --panel:#2a1f39; --panel-2:#332447; --accent:#9b6bff;
  --text:#EAE6F5; --muted:#B9A8D9; --success:#2ECC71; --warn:#F39C12; --danger:#E74C3C;
}
html, body, .stApp { background: var(--bg); color: var(--text) !important; }
section[data-testid="stSidebar"] { background: #1b1c25; border-right: 1px solid #3b3d4b; }
header[data-testid="stHeader"] { display:none !important; }
.kpi-card{background:var(--panel); border:1px solid #3f4151; border-radius:14px; padding:14px; box-shadow:0 8px 24px rgba(0,0,0,.22)}
.small{color:var(--muted);font-size:.9rem}
.badge{display:inline-block;background:#2a2b36;color:var(--muted);padding:.25rem .55rem;border-radius:999px;border:1px solid #3f4151;margin-right:.35rem}
.stButton>button,.stDownloadButton>button{background:linear-gradient(135deg,var(--accent),#7c69d4); color:white; border:0; padding:.55rem 1rem; border-radius:12px; font-weight:700; box-shadow:0 10px 22px rgba(96,81,155,.25)}
.stButton>button:hover,.stDownloadButton>button:hover{filter:brightness(1.06)}
</style>
"""

# ========= Fonts (Montserrat opcional para PDF) =========
def _download_font(dst: str, url: str) -> bool:
    try:
        import requests
        r = requests.get(url, timeout=15)
        if r.ok:
            with open(dst, "wb") as f:
                f.write(r.content)
            return True
    except Exception:
        pass
    return False

def _register_montserrat(pdf: FPDF) -> bool:
    os.makedirs("fonts", exist_ok=True)
    font_map = {
        "Montserrat-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/montserrat/Montserrat-Regular.ttf",
        "Montserrat-Bold.ttf": "https://github.com/google/fonts/raw/main/ofl/montserrat/Montserrat-Bold.ttf",
        "Montserrat-Italic.ttf": "https://github.com/google/fonts/raw/main/ofl/montserrat/Montserrat-Italic.ttf",
    }
    ok = True
    for fname, url in font_map.items():
        path = os.path.join("fonts", fname)
        if not os.path.exists(path):
            if not _download_font(path, url):
                ok = False
    if not ok:
        return False
    try:
        pdf.add_font("Montserrat", "", os.path.join("fonts", "Montserrat-Regular.ttf"), uni=True)
        pdf.add_font("Montserrat", "B", os.path.join("fonts", "Montserrat-Bold.ttf"), uni=True)
        pdf.add_font("Montserrat", "I", os.path.join("fonts", "Montserrat-Italic.ttf"), uni=True)
        return True
    except Exception:
        return False

# ========= Token Accounting =========
@dataclass
class TokenStep:
    prompt: int = 0
    completion: int = 0
    @property
    def total(self): return self.prompt + self.completion

@dataclass
class TokenTracker:
    steps: Dict[str, TokenStep] = field(default_factory=lambda: {
        "extracao": TokenStep(),
        "analise": TokenStep(),
        "chat": TokenStep(),
        "pdf": TokenStep()  # lógico, sem custo
    })
    model: str = ""
    provider: str = ""

    def add(self, step: str, prompt_tokens: int, completion_tokens: int):
        if step not in self.steps:
            self.steps[step] = TokenStep()
        self.steps[step].prompt += int(prompt_tokens or 0)
        self.steps[step].completion += int(completion_tokens or 0)

    def dict(self):
        return {k: {"prompt": v.prompt, "completion": v.completion, "total": v.total} for k, v in self.steps.items()}

    @property
    def total_prompt(self): return sum(s.prompt for s in self.steps.values())
    @property
    def total_completion(self): return sum(s.completion for s in self.steps.values())
    @property
    def total_tokens(self): return self.total_prompt + self.total_completion

    def cost_usd_gpt(self) -> float:
        return (self.total_prompt/1000.0)*GPT_PRICE_INPUT_PER_1K + (self.total_completion/1000.0)*GPT_PRICE_OUTPUT_PER_1K

def _estimate_tokens(text: str) -> int:
    if not text: return 0
    try:
        if tiktoken:
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
    except Exception:
        pass
    return max(1, int(len(text) / 4))  # heurística

# ========= Prompts =========
EXTRACTION_PROMPT = """Você é um especialista em análise de relatórios BFA (Big Five Analysis) para seleção de talentos.

Sua tarefa: extrair dados do relatório abaixo e retornar APENAS um JSON válido, sem texto adicional.

ESTRUTURA OBRIGATÓRIA:
{
  "candidato": {
    "nome": "string ou null",
    "cargo_avaliado": "string ou null"
  },
  "traits_bfa": {
    "Abertura": número 0-10 ou null,
    "Conscienciosidade": número 0-10 ou null,
    "Extroversao": número 0-10 ou null,
    "Amabilidade": número 0-10 ou null,
    "Neuroticismo": número 0-10 ou null
  },
  "competencias_ms": [
    {"nome": "string", "nota": número, "classificacao": "string"}
  ],
  "facetas_relevantes": [
    {"nome": "string", "percentil": número, "interpretacao": "string resumida"}
  ],
  "indicadores_saude_emocional": {
    "ansiedade": número 0-100 ou null,
    "irritabilidade": número 0-100 ou null,
    "estado_animo": número 0-100 ou null,
    "impulsividade": número 0-100 ou null
  },
  "potencial_lideranca": "BAIXO" | "MÉDIO" | "ALTO" ou null,
  "integridade_fgi": número 0-100 ou null,
  "resumo_qualitativo": "texto do resumo presente no relatório",
  "pontos_fortes": ["lista de 3-5 pontos"],
  "pontos_atencao": ["lista de 2-4 pontos"],
  "fit_geral_cargo": número 0-100
}

REGRAS:
1) Normalize percentis; 2) Big Five: percentil 60 -> 6.0/10; 3) Extraia TODAS as competências;
4) Use null quando não houver evidência; 5) resumo_qualitativo = texto original;
6) pontos_fortes (3-5) e pontos_atencao (2-4); 7) fit_geral_cargo 0-100 baseado no cargo: {cargo}.

RELATÓRIO:
\"\"\"{text}\"\"\"

MATERIAIS (opcional):
\"\"\"{training_context}\"\"\"

Retorne apenas o JSON puro.
"""

ANALYSIS_PROMPT = """Você é um consultor sênior de RH especializado em análise comportamental.

Cargo avaliado: {cargo}

DADOS (JSON extraído):
{json_data}

PERFIL IDEAL DO CARGO:
{perfil_cargo}

Responda em JSON:
{
  "compatibilidade_geral": 0-100,
  "decisao": "RECOMENDADO" | "RECOMENDADO COM RESSALVAS" | "NÃO RECOMENDADO",
  "justificativa_decisao": "texto",
  "analise_tracos": {
    "Abertura": "texto",
    "Conscienciosidade": "texto",
    "Extroversao": "texto",
    "Amabilidade": "texto",
    "Neuroticismo": "texto"
  },
  "competencias_criticas": [
    {"competencia": "nome", "avaliacao": "texto", "status": "ATENDE" | "PARCIAL" | "NÃO ATENDE"}
  ],
  "saude_emocional_contexto": "texto",
  "recomendacoes_desenvolvimento": ["a","b","c"],
  "cargos_alternativos": [{"cargo":"nome","justificativa":"texto"}],
  "resumo_executivo": "100-150 palavras"
}"""

# ========= Helpers I/O =========
@st.cache_resource(show_spinner=False)
def get_llm_client_cached(provider: str, api_key: str):
    """Cria cliente LLM compatível com múltiplas versões, sem proxies."""
    if not api_key:
        raise RuntimeError("Chave da API não configurada. Defina nos Secrets do Streamlit.")
    pv = (provider or "Groq").lower()
    try:
        if pv == "groq":
            if Groq is None:
                raise RuntimeError("Biblioteca 'groq' não instalada. Execute: pip install groq")
            try:
                # tentativa padrão
                return Groq(api_key=api_key)
            except Exception:
                # fallback para versões que rejeitam proxies
                import groq
                client = groq.Groq()
                if hasattr(client, "api_key"):
                    client.api_key = api_key
                elif hasattr(client, "configuration"):
                    client.configuration.api_key = api_key
                return client

        elif pv == "openai":
            if OpenAI is None:
                raise RuntimeError("Biblioteca 'openai' não instalada. Execute: pip install openai>=1.0.0")
            try:
                return OpenAI(api_key=api_key)
            except Exception:
                import openai
                client = openai.OpenAI()
                if hasattr(client, "api_key"):
                    client.api_key = api_key
                return client
        else:
            raise RuntimeError(f"Provedor não suportado: {provider}")
    except Exception as e:
        raise RuntimeError(f"[Erro cliente] {e}")
    
def gerar_perfil_cargo_dinamico(cargo: str) -> Dict:
    return {
        "traits_ideais": {"Abertura": (5,8), "Conscienciosidade": (6,9), "Extroversão": (4,8), "Amabilidade": (5,8), "Neuroticismo": (0,5)},
        "competencias_criticas": ["Adaptabilidade","Comunicação","Trabalho em Equipe","Resolução de Problemas"],
        "descricao": f"Perfil para {cargo}"
    }

# ========= Wrappers de completion (captura usage quando disponível) =========
def _chat_completion_json(provider, client, model, messages, force_json=True):
    usage = None
    if (provider or "").lower() == "groq":
        kwargs = dict(model=model, messages=messages, max_tokens=MAX_TOKENS_FIXED, temperature=TEMP_FIXED)
        if force_json: kwargs["response_format"] = {"type":"json_object"}
        resp = client.chat.completions.create(**kwargs)
        content = resp.choices[0].message.content.strip()
        usage = getattr(resp, "usage", None)
        if usage:
            usage = {"prompt_tokens": usage.prompt_tokens, "completion_tokens": usage.completion_tokens, "total_tokens": usage.total_tokens}
        return content, usage
    else:
        resp = client.chat.completions.create(
            model=model,
            messages=messages if not force_json else ([{"role":"system","content":"Responda apenas com JSON válido."}] + messages),
            temperature=TEMP_FIXED,
            max_tokens=MAX_TOKENS_FIXED,
            response_format={"type":"json_object"} if force_json else None
        )
        content = resp.choices[0].message.content.strip()
        usage = getattr(resp, "usage", None)
        if usage:
            usage = {"prompt_tokens": usage.prompt_tokens, "completion_tokens": usage.completion_tokens, "total_tokens": usage.total_tokens}
        return content, usage

def _add_usage_or_estimate(tracker: TokenTracker, step: str, messages: List[Dict], content: str, usage: Optional[Dict]):
    if usage:
        tracker.add(step, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))
        return
    prompt_text = "\n".join([m.get("content","") for m in messages])
    tracker.add(step, _estimate_tokens(prompt_text), _estimate_tokens(content))

# ========= Core IA =========
def extract_bfa_data(text, cargo, training_context, provider, model_id, token, tracker: TokenTracker):
    try:
        client = get_llm_client_cached(provider, token)
    except Exception as e:
        return None, f"[Erro cliente] {e}"
    prompt = (EXTRACTION_PROMPT
              .replace("{text}", text[:10000])
              .replace("{training_context}", training_context[:3000])
              .replace("{cargo}", cargo))
    try:
        content, usage = _chat_completion_json(provider, client, model_id.strip(), [{"role":"user","content":prompt}], True)
        _add_usage_or_estimate(tracker, "extracao", [{"role":"user","content":prompt}], content, usage)
        try:
            return json.loads(content), content
        except Exception:
            m = re.search(r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}', content, re.DOTALL)
            if m:
                return json.loads(m.group(0)), content
            return None, f"Nenhum JSON válido: {content[:800]}..."
    except Exception as e:
        msg = f"[Erro LLM] {e}"
        if hasattr(e, "response") and getattr(e.response, "text", None):
            msg += f" - Resposta: {e.response.text}"
        return None, msg

def analyze_bfa_data(bfa_data, cargo, perfil_cargo, provider, model_id, token, tracker: TokenTracker):
    try:
        client = get_llm_client_cached(provider, token)
    except Exception as e:
        return None, f"[Erro cliente] {e}"
    prompt = (ANALYSIS_PROMPT
              .replace("{cargo}", cargo)
              .replace("{json_data}", json.dumps(bfa_data, ensure_ascii=False, indent=2))
              .replace("{perfil_cargo}", json.dumps(perfil_cargo, ensure_ascii=False, indent=2)))
    try:
        content, usage = _chat_completion_json(provider, client, model_id.strip(),
                                               [{"role":"system","content":"Responda estritamente em JSON."},
                                                {"role":"user","content":prompt}], True)
        _add_usage_or_estimate(tracker, "analise",
                               [{"role":"system","content":"Responda estritamente em JSON."},{"role":"user","content":prompt}],
                               content, usage)
        try:
            return json.loads(content), content
        except Exception:
            fix, usage2 = _chat_completion_json(provider, client, model_id.strip(),
                                        [{"role":"system","content":"Retorne apenas o JSON válido."},
                                         {"role":"user","content":f"Converta em JSON:\n{content}"}],
                                        True)
            _add_usage_or_estimate(tracker, "analise",
                                   [{"role":"system","content":"Retorne apenas o JSON válido."},{"role":"user","content":f"Converta em JSON:\n{content}"}],
                                   fix, usage2)
            return json.loads(fix), fix
    except Exception as e:
        return None, f"[Erro durante análise] {e}"

def chat_with_elder_brain(question, bfa_data, analysis, cargo, provider, model_id, token, tracker: TokenTracker):
    try:
        client = get_llm_client_cached(provider, token)
    except Exception as e:
        return f"Erro ao conectar com a IA: {e}"
    contexto = f"""
Você é um consultor executivo de RH analisando um relatório BFA.

DADOS (JSON): {json.dumps(bfa_data, ensure_ascii=False)}
ANÁLISE (JSON): {json.dumps(analysis, ensure_ascii=False)}
CARGO: {cargo}

PERGUNTA: {question}
Responda de forma objetiva e profissional.
""".strip()
    try:
        content, usage = _chat_completion_json(provider, client, model_id.strip(),
                                               [{"role":"user","content":contexto}], False)
        _add_usage_or_estimate(tracker, "chat", [{"role":"user","content":contexto}], content, usage)
        return content
    except Exception as e:
        msg = f"Erro na resposta da IA: {e}"
        if hasattr(e, "response") and getattr(e.response, "text", None):
            msg += f" - Detalhes: {e.response.text}"
        return msg

# ========= Gráficos =========
COLOR_CANDIDATO = "#60519b"
COLOR_IDEAL_MAX = "rgba(46, 213, 115, 0.35)"
COLOR_IDEAL_MIN = "rgba(46, 213, 115, 0.15)"
COLOR_WARN      = "#F39C12"
COLOR_GOOD      = "#2ECC71"
COLOR_BAD       = "#E74C3C"

def criar_radar_bfa(traits: Dict[str, Optional[float]], traits_ideais: Dict = None) -> go.Figure:
    labels = ["Abertura", "Conscienciosidade", "Extroversão", "Amabilidade", "Neuroticismo"]
    vals = []
    for k in labels:
        v = traits.get(k, None)
        if v is None:
            norm = k.replace("ã","a").replace("ç","c").replace("õ","o").replace("é","e").replace("ó","o")
            v = traits.get(norm, 0)
        vals.append(float(v or 0))
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=vals, theta=labels, fill='toself', name='Candidato',
                                  line=dict(color=COLOR_CANDIDATO)))
    if traits_ideais:
        vmin = [traits_ideais.get(k, (0,10))[0] for k in labels]
        vmax = [traits_ideais.get(k, (0,10))[1] for k in labels]
        fig.add_trace(go.Scatterpolar(r=vmax, theta=labels, fill='toself', name='Faixa Ideal (Máx)',
                                      line=dict(color=COLOR_GOOD), fillcolor=COLOR_IDEAL_MAX))
        fig.add_trace(go.Scatterpolar(r=vmin, theta=labels, fill='tonext', name='Faixa Ideal (Mín)',
                                      line=dict(color=COLOR_GOOD), fillcolor=COLOR_IDEAL_MIN))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
                      showlegend=True, title="Big Five x Perfil Ideal", height=500)
    return fig

def fig_to_png_path(fig: "go.Figure", width=1280, height=800, scale=2) -> Optional[str]:
    try:
        import plotly.io as pio
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            pio.write_image(fig, tmp.name, format="png", width=width, height=height, scale=scale)
            return tmp.name
    except Exception:
        return None

def criar_grafico_competencias(competencias: List[Dict]) -> Optional[go.Figure]:
    if not competencias: return None
    df = pd.DataFrame(competencias).copy()
    if df.empty or "nota" not in df.columns: return None
    df = df.sort_values('nota', ascending=True).tail(15)
    cores = [COLOR_BAD if n < 45 else COLOR_WARN if n < 55 else COLOR_GOOD for n in df['nota']]
    fig = go.Figure(go.Bar(x=df['nota'], y=df['nome'], orientation='h',
                           marker=dict(color=cores),
                           text=df['nota'].round(0).astype(int), textposition='outside'))
    fig.update_layout(title="Competências MS (Top 15)", xaxis_title="Nota", yaxis_title="",
                      height=600, showlegend=False)
    fig.add_vline(x=45, line_dash="dash", line_color=COLOR_WARN)
    fig.add_vline(x=55, line_dash="dash", line_color=COLOR_GOOD)
    return fig

def criar_gauge_fit(fit_score: float) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta", value=float(fit_score or 0),
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Fit para o Cargo", 'font': {'size': 24}},
        delta={'reference': 70},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': COLOR_CANDIDATO},
            'steps': [{'range': [0, 40], 'color': COLOR_BAD},
                      {'range': [40, 70], 'color': COLOR_WARN},
                      {'range': [70, 100], 'color': COLOR_GOOD}],
            'threshold': {'line': {'color': "#ff0040", 'width': 4}, 'thickness': 0.75, 'value': 70}
        }
    ))
    fig.update_layout(height=400)
    return fig

# ========= PDF (Deluxe) =========
class PDFReport(FPDF):
    def __init__(self,*a,**k):
        super().__init__(*a,**k)
        self.set_auto_page_break(auto=True, margin=15)
        self.set_margins(15, 15, 15)
        self._family = "Helvetica"
        self._unicode = False
    def set_main_family(self,fam,uni): self._family, self._unicode = fam, uni
    def _safe(self, s: str) -> str:
        rep = {"\u2014": "-", "\u2013": "-", "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u2026": "...", "\u00a0": " "}
        for k,v in rep.items(): s = (s or "").replace(k,v)
        try: return s if self._unicode else s.encode("latin-1","ignore").decode("latin-1")
        except Exception: return s
    def cover(self, titulo, subtitulo, autor, versao, logo_path=None):
        self.add_page()
        if logo_path and os.path.exists(logo_path):
            try: self.image(logo_path, x=15, y=18, w=28)
            except Exception: pass
        self.set_font(self._family,'B',22); self.ln(18); self.cell(0,12,self._safe(titulo),align='C',ln=1)
        self.set_font(self._family,'',12); self.cell(0,8,self._safe(subtitulo),align='C',ln=1); self.ln(6)
        self.set_font(self._family,'',11)
        self.multi_cell(0,7,self._safe(f"Desenvolvedor Responsável: {autor}\nVersão: {versao}\nData: {datetime.now().strftime('%d/%m/%Y')}"), align='C'); self.ln(4)
    def header(self):
        if self.page_no()==1: return
        self.set_font(self._family,'B',12); self.cell(0,8,self._safe('Elder Brain Analytics — Relatório Corporativo'), align='C', ln=1); self.ln(1)
    def footer(self):
        if self.page_no()==1: return
        self.set_y(-15); self.set_font(self._family,'',8); self.cell(0,10,self._safe(f'Página {self.page_no()}'), align='C')
    def heading(self, title):
        self.set_font(self._family,'B',12)
        self.set_fill_color(96, 81, 155)
        self.set_text_color(255,255,255)
        self.cell(0,10,self._safe(title),align='L',ln=1,fill=True)
        self.set_text_color(0,0,0); self.ln(1)
    def paragraph(self, body, size=10):
        self.set_font(self._family,'',size); self.multi_cell(0,5,self._safe(body or "")); self.ln(1)

def gerar_pdf_corporativo(bfa_data: Dict, analysis: Dict, cargo: str, save_path: str = None, logo_path: Optional[str] = None) -> io.BytesIO:
    try:
        pdf = PDFReport(orientation="P", unit="mm", format="A4")
        if _register_montserrat(pdf): pdf.set_main_family("Montserrat", True)
        else: pdf.set_main_family("Helvetica", False)

        # CAPA
        pdf.cover("Elder Brain Analytics PRO — Versão Deluxe",
                  "Relatório de Análise Comportamental (BFA) com IA",
                  "André de Lima","V9.1 CORPORATE",logo_path)

        # 1. INFOS
        pdf.heading('1. INFORMAÇÕES DO CANDIDATO')
        candidato = bfa_data.get('candidato', {}) or {}
        info_text = f"""Nome: {candidato.get('nome', 'Não informado')}
Cargo Avaliado: {cargo}
Data da Análise: {datetime.now().strftime('%d/%m/%Y %H:%M')}"""
        pdf.paragraph(info_text, size=10)

        # 2. DECISÃO
        pdf.heading('2. DECISÃO E COMPATIBILIDADE')
        decisao = (analysis or {}).get('decisao','N/A')
        compat = float((analysis or {}).get('compatibilidade_geral',0) or 0)
        pdf.set_fill_color(230,230,230); pdf.set_font(pdf._family,'B',12)
        pdf.cell(0,8,pdf._safe(f"DECISÃO: {decisao} | COMPATIBILIDADE: {compat:.0f}%"), align='C', ln=1, fill=True)
        justificativa = (analysis or {}).get('justificativa_decisao','')
        if justificativa: pdf.paragraph(justificativa, size=10)

        # 3. RESUMO
        pdf.heading('3. RESUMO EXECUTIVO')
        resumo = (analysis or {}).get('resumo_executivo', justificativa)
        if resumo: pdf.paragraph(resumo, size=10)

        # 4. BIG FIVE
        pdf.heading('4. TRAÇOS DE PERSONALIDADE (BIG FIVE)')
        traits = (bfa_data or {}).get('traits_bfa', {}) or {}
        for trait_name, valor in traits.items():
            if valor is None: continue
            pdf.set_font(pdf._family,'B',10); pdf.cell(70,6,pdf._safe(f'{trait_name}:'))
            pdf.set_font(pdf._family,'',10)
            try: txt_val = f'{float(valor):.1f}/10'
            except Exception: txt_val = f'{valor}/10'
            pdf.cell(0,6,pdf._safe(txt_val), ln=1)
        analise_tracos = (analysis or {}).get('analise_tracos', {}) or {}
        for trait, analise in analise_tracos.items():
            if analise: pdf.paragraph(f'{trait}: {analise}', size=9)

        # 5. GRÁFICOS
        pdf.add_page()
        pdf.heading('5. VISUALIZAÇÕES (GRÁFICOS)')
        perfil = gerar_perfil_cargo_dinamico(cargo)
        radar_fig = criar_radar_bfa(traits, perfil.get('traits_ideais', {}))
        comp_fig = criar_grafico_competencias((bfa_data or {}).get('competencias_ms', []) or [])
        gauge_fig = criar_gauge_fit(float((analysis or {}).get('compatibilidade_geral', 0) or 0))

        def _embed(fig, w):
            path = fig_to_png_path(fig, width=1200, height=900, scale=2)
            if path:
                try: pdf.image(path, w=w)
                except Exception: pass
                try: os.remove(path)
                except Exception: pass
                return True
            return False

        if not _embed(radar_fig, 180):
            pdf.paragraph("⚠️ Instale 'kaleido' para embutir gráficos no PDF.", size=9)
        if comp_fig:
            if not _embed(comp_fig, 180):
                pdf.paragraph("⚠️ Falha ao embutir gráfico de Competências.", size=9)
        else:
            pdf.paragraph("Sem competências para exibir.", size=9)
        if not _embed(gauge_fig, 150):
            pdf.paragraph("⚠️ Falha ao embutir gráfico de Fit.", size=9)

        # 6. SAÚDE
        pdf.heading('6. SAÚDE EMOCIONAL E RESILIÊNCIA')
        saude = (analysis or {}).get('saude_emocional_contexto','')
        if saude: pdf.paragraph(saude, size=10)
        indicadores = (bfa_data or {}).get('indicadores_saude_emocional', {}) or {}
        for k,v in indicadores.items():
            if v is None: continue
            pdf.set_font(pdf._family,'',9)
            pdf.cell(70,5,pdf._safe(f'{k.replace("_"," ").capitalize()}: '))
            pdf.cell(0,5,pdf._safe(f'{float(v):.0f}/100'), ln=1)

        # 7/8. PONTOS
        pf = (bfa_data or {}).get('pontos_fortes', []) or []
        if pf:
            pdf.heading('7. PONTOS FORTES')
            for item in pf:
                if item: pdf.paragraph(f'+ {item}', size=10)
        pa = (bfa_data or {}).get('pontos_atencao', []) or []
        if pa:
            pdf.heading('8. PONTOS DE ATENÇÃO')
            for item in pa:
                if item: pdf.paragraph(f'! {item}', size=10)

        # 9/10. RECOMENDAÇÕES / CARGOS
        pdf.add_page()
        pdf.heading('9. RECOMENDAÇÕES DE DESENVOLVIMENTO')
        recs = (analysis or {}).get('recomendacoes_desenvolvimento', []) or []
        for i, rec in enumerate(recs, 1):
            if rec:
                pdf.set_font(pdf._family,'B',10); pdf.cell(10,6,pdf._safe(f'{i}.'))
                pdf.set_font(pdf._family,'',10); pdf.multi_cell(0,6,pdf._safe(rec))
        cargos_alt = (analysis or {}).get('cargos_alternativos', []) or []
        if cargos_alt:
            pdf.heading('10. CARGOS ALTERNATIVOS SUGERIDOS')
            for cargo_info in cargos_alt:
                nome = cargo_info.get('cargo',''); just = cargo_info.get('justificativa','')
                if not nome: continue
                pdf.set_font(pdf._family,'B',10); pdf.multi_cell(0,6,pdf._safe(f"- {nome}"))
                if just: pdf.set_font(pdf._family,'',9); pdf.multi_cell(0,5,pdf._safe(f"   {just}"))

        pdf.ln(2); pdf.set_font(pdf._family,'I',8)
        pdf.multi_cell(0,4,pdf._safe("Este relatório auxilia a decisão e não substitui avaliação profissional. Uso interno — Elder Brain Analytics PRO (Versão Deluxe)."))

        try:
            out_bytes = pdf.output(dest='S')
            if isinstance(out_bytes, str): out_bytes = out_bytes.encode('latin-1','replace')
        except Exception:
            fb = PDFReport(); fb.set_main_family("Helvetica", False); fb.add_page()
            fb.set_font(fb._family,'B',14); fb.cell(0,10,fb._safe('RELATÓRIO DE ANÁLISE COMPORTAMENTAL'), ln=1, align='C')
            fb.set_font(fb._family,'',11); fb.multi_cell(0,8,fb._safe(f"Relatório gerado para: {cargo}\nData: {datetime.now().strftime('%d/%m/%Y %H:%M')}"))
            out_bytes = fb.output(dest='S')
            if isinstance(out_bytes,str): out_bytes = out_bytes.encode('latin-1','replace')

        buf = io.BytesIO(out_bytes); buf.seek(0)
        if save_path:
            try:
                with open(save_path,'wb') as f: f.write(buf.getbuffer())
            except Exception as e:
                st.error(f"Erro ao salvar PDF: {e}")
        return buf
    except Exception as e:
        st.error(f"Erro crítico na geração do PDF: {e}")
        return io.BytesIO(b'%PDF-1.4\n%EOF\n')

# ========= UI helpers =========
def kpi_card(title, value, sub=None):
    st.markdown(
        f'<div class="kpi-card"><div style="font-weight:700;font-size:1.02rem">{title}</div>'
        f'<div style="font-size:1.9rem;margin:.2rem 0 .25rem 0">{value}</div>'
        f'<div class="small">{sub or ""}</div></div>', unsafe_allow_html=True
    )

# ========= APP =========
def main():
    st.set_page_config(page_title="EBA — Corporate PROD", page_icon="🧠", layout="wide")
    st.markdown(DARK_CSS, unsafe_allow_html=True)

    ss = st.session_state
    ss.setdefault('provider', "Groq")
    ss.setdefault('modelo', "llama-3.1-8b-instant")
    ss.setdefault('cargo', "")
    ss.setdefault('analysis_complete', False)
    ss.setdefault('bfa_data', None)
    ss.setdefault('analysis', None)
    ss.setdefault('pdf_generated', None)
    ss.setdefault('tracker', TokenTracker())
    ss.setdefault('admin_mode', False)   # 🔐 sempre inicia como usuário comum

    # ===== Topo
    st.markdown("## 🧠 Elder Brain Analytics — Corporate (PROD)")
    st.markdown('<span class="badge">PDF Deluxe</span> <span class="badge">Seguro</span> <span class="badge">Streamlit Cloud</span>', unsafe_allow_html=True)

    # ===== Sidebar (Config + Admin)
    with st.sidebar:
        st.header("⚙️ Configuração")
        provider = st.radio("Provedor", ["Groq","OpenAI"], index=0, key="provider")
        modelo = st.text_input("Modelo", value=ss['modelo'],
                               help=("Sugestões: " + ", ".join(MODELOS_SUGERIDOS_GROQ if provider=="Groq" else MODELOS_SUGERIDOS_OPENAI)))
        ss['modelo'] = modelo

        # 🔐 SEM CAMPO DE API KEY PARA O USUÁRIO
        # keys são lidas de st.secrets conforme o provedor escolhido
        token = st.secrets.get("GROQ_API_KEY","") if provider=="Groq" else st.secrets.get("OPENAI_API_KEY","")

        st.caption("Temperatura fixa: 0.3 · Máx tokens: 4096")
        ss['cargo'] = st.text_input("Cargo para análise", value=ss['cargo'])
        if ss['cargo']:
            with st.expander("Perfil gerado (dinâmico)"):
                st.json(gerar_perfil_cargo_dinamico(ss['cargo']))

        st.markdown("---")
        st.subheader("🔒 Painel Administrativo")
        admin_pwd = st.text_input("Senha do Admin", type="password", placeholder="somente administradores")
        if admin_pwd:
            if admin_pwd == st.secrets.get("ADMIN_PASSWORD",""):
                ss['admin_mode'] = True
                st.success("Acesso administrativo concedido")
            else:
                ss['admin_mode'] = False
                st.error("Senha incorreta")
        else:
            ss['admin_mode'] = False

        # 📈 Token Log — visível SOMENTE para admin
        if ss['admin_mode']:
            st.markdown("---")
            st.header("📈 Token Log")
            td = ss['tracker'].dict()
            for step in ["extracao","analise","chat","pdf"]:
                d = td.get(step, {"prompt":0,"completion":0,"total":0})
                st.write(f"- **{step.capitalize()}**: {d['total']}  (prompt {d['prompt']} / output {d['completion']})")
            st.write(f"**Total:** {ss['tracker'].total_tokens} tokens")
            st.write(f"**Custo (estimado):** ${ss['tracker'].cost_usd_gpt():.4f}")
        else:
            st.caption("modo usuário — sem métricas financeiras visíveis")

    # ===== KPIs (cliente NUNCA vê custo/tokens)
    c1,c2,c3,c4 = st.columns(4)
    with c1: kpi_card("Status", "Pronto", "Aguardando PDF")
    if ss['admin_mode']:
        with c2: kpi_card("Tokens (Total)", f"{ss['tracker'].total_tokens}", "desde o início")
        with c3: kpi_card("Prompt/Output", f"{ss['tracker'].total_prompt}/{ss['tracker'].total_completion}", "tokens")
        with c4: kpi_card("Custo Estimado", f"${ss['tracker'].cost_usd_gpt():.4f}", "apenas admin")
    else:
        with c2: kpi_card("Relatórios", "—", "em sessão")
        with c3: kpi_card("Andamento", "—", "")
        with c4: kpi_card("Disponibilidade", "Online", "")

    # ===== Upload
    st.markdown("### 📄 Upload do Relatório BFA")
    uploaded_file = st.file_uploader("Carregue o PDF do relatório BFA", type=["pdf"])

    with st.expander("📚 Materiais de Treinamento (Opcional)"):
        training_files = st.file_uploader("Arraste PDFs/TXTs", accept_multiple_files=True, key="training")
        if training_files:
            for f in training_files:
                save_path = os.path.join(TRAINING_DIR, f"{int(time.time())}_{f.name}")
                with open(save_path, "wb") as out:
                    out.write(f.getbuffer())
            st.success(f"{len(training_files)} arquivo(s) salvos")

    # ===== Processamento
    if uploaded_file:
        if not ss['cargo']: st.error("Informe o cargo na sidebar"); st.stop()
        if not token:
            st.error("Chave da API não configurada nos Secrets do Streamlit. Defina GROQ_API_KEY/OPENAI_API_KEY.")
            st.stop()
        if not (ss['modelo'] and ss['modelo'].strip()): st.error("Informe o modelo"); st.stop()

        with st.spinner("Extraindo texto do PDF..."):
            raw_text = extract_pdf_text_bytes(uploaded_file)
        if raw_text.startswith("[ERRO"): st.error(raw_text); st.stop()
        st.success("✓ Texto extraído")
        st.text_area("Prévia do texto (início)", raw_text[:1500], height=180)

        if st.button("🔬 ANALISAR RELATÓRIO", type="primary", use_container_width=True):
            training_context = load_all_training_texts()
            tracker: TokenTracker = ss['tracker']
            tracker.model, tracker.provider = ss['modelo'], ss['provider']

            with st.spinner("Etapa 1/2: Extraindo dados estruturados..."):
                bfa_data, raw1 = extract_bfa_data(raw_text, ss['cargo'], training_context, ss['provider'], ss['modelo'], token, tracker)
            if not bfa_data:
                st.error("Falha na extração"); 
                with st.expander("Resposta bruta da IA"):
                    st.code(raw1)
                st.stop()

            perfil = gerar_perfil_cargo_dinamico(ss['cargo'])
            with st.spinner("Etapa 2/2: Analisando compatibilidade..."):
                analysis, raw2 = analyze_bfa_data(bfa_data, ss['cargo'], perfil, ss['provider'], ss['modelo'], token, tracker)
            if not analysis:
                st.error("Falha na análise");
                with st.expander("Resposta bruta da IA"):
                    st.code(raw2)
                st.stop()

            ss['bfa_data'], ss['analysis'], ss['analysis_complete'] = bfa_data, analysis, True
            st.success("✓ Análise concluída!"); st.rerun()

    # ===== Resultados
    if ss.get('analysis_complete') and ss.get('bfa_data') and ss.get('analysis'):
        st.markdown("## 📊 Resultados")
        decisao = ss['analysis'].get('decisao','N/A')
        compat = float(ss['analysis'].get('compatibilidade_geral',0) or 0)

        c1,c2,c3 = st.columns([2,1,1])
        with c1: st.markdown(f"### 🏷️ Decisão: **{decisao}**")
        with c2: st.metric("Compatibilidade", f"{compat:.0f}%")
        with c3: st.metric("Liderança", ss['bfa_data'].get('potencial_lideranca','N/A'))

        with st.expander("📋 Resumo Executivo", expanded=True):
            st.write(ss['analysis'].get('resumo_executivo',''))
        st.info(ss['analysis'].get('justificativa_decisao',''))

        tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎯 Big Five","💼 Competências","🧘 Saúde Emocional","📈 Desenvolvimento","📄 Dados Brutos"])
        with tab1:
            traits = ss['bfa_data'].get('traits_bfa',{})
            fig_radar = criar_radar_bfa(traits, gerar_perfil_cargo_dinamico(ss['cargo']).get('traits_ideais',{}))
            st.plotly_chart(fig_radar, use_container_width=True)
        with tab2:
            comps = ss['bfa_data'].get('competencias_ms',[])
            figc = criar_grafico_competencias(comps)
            if figc: st.plotly_chart(figc, use_container_width=True)
            for comp in ss['analysis'].get('competencias_criticas',[]):
                status = comp.get('status'); compn = comp.get('competencia'); txt = comp.get('avaliacao','')
                if status == 'ATENDE': st.success(f"✓ {compn} — {status}"); st.caption(txt)
                elif status == 'PARCIAL': st.warning(f"⚠ {compn} — {status}"); st.caption(txt)
                else: st.error(f"✗ {compn} — {status}"); st.caption(txt)
        with tab3:
            st.write(ss['analysis'].get('saude_emocional_contexto',''))
            indicadores = ss['bfa_data'].get('indicadores_saude_emocional',{})
            cols = st.columns(2)
            for i,(k,v) in enumerate(indicadores.items()):
                if v is None: continue
                with cols[i%2]: st.metric(k.replace('_',' ').title(), f"{float(v):.0f}")
        with tab4:
            recs = ss['analysis'].get('recomendacoes_desenvolvimento',[])
            for i,r in enumerate(recs,1): st.markdown(f"**{i}.** {r}")
            alt = ss['analysis'].get('cargos_alternativos',[])
            if alt:
                st.markdown("#### Cargos Alternativos")
                for c in alt: st.markdown(f"- **{c.get('cargo','')}** — {c.get('justificativa','')}")
        with tab5:
            c1,c2 = st.columns(2)
            with c1: st.json(ss['bfa_data'])
            with c2: st.json(ss['analysis'])

        st.markdown("### 🎯 Compatibilidade")
        st.plotly_chart(criar_gauge_fit(compat), use_container_width=True)

        st.markdown("### 📄 Gerar PDF")
        logo_path = st.text_input("Caminho para logo (opcional)", value="")
        if st.button("🔨 Gerar PDF", key="gen_pdf"):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            nome = ((ss['bfa_data'].get('candidato',{}) or {}).get('nome') or 'candidato')
            nome = re.sub(r'[^\w\s-]', '', str(nome)).strip().replace(' ', '_')
            fname = f"relatorio_{nome}_{ts}.pdf"
            path = os.path.join(PROCESSED_DIR, fname)
            buf = gerar_pdf_corporativo(ss['bfa_data'], ss['analysis'], ss['cargo'], save_path=path, logo_path=logo_path if logo_path else None)
            ss['tracker'].add("pdf", 0, 0)  # registra passo lógico (sem custo)
            if buf.getbuffer().nbytes > 100:
                ss['pdf_generated'] = {'buffer': buf, 'filename': fname}
                st.success(f"✓ PDF gerado: {fname}")
            else:
                st.error("Arquivo PDF vazio (erro na geração).")
        if ss.get('pdf_generated'):
            st.download_button("⬇️ Download do PDF", data=ss['pdf_generated']['buffer'].getvalue(),
                               file_name=ss['pdf_generated']['filename'], mime="application/pdf", use_container_width=True)

        st.markdown("### 💬 Chat com o Elder Brain")
        q = st.text_input("Pergunte sobre este relatório", placeholder="Ex.: Principais riscos para este cargo?")
        if q and st.button("Enviar", key="ask"):
            with st.spinner("Pensando..."):
                ans = chat_with_elder_brain(q, ss['bfa_data'], ss['analysis'], ss['cargo'],
                                            ss['provider'], ss['modelo'], token, ss['tracker'])
            st.markdown(f"**Você:** {q}")
            st.markdown(f"**Elder Brain:** {ans}")

    st.caption(f"📁 Relatórios salvos em: `{PROCESSED_DIR}`")

if __name__ == "__main__":
    main()
