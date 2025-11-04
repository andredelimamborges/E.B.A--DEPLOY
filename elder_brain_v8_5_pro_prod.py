# elder_brain_v8_5_pro_prod_final.py
"""
Elder Brain Analytics — v9.1 FINAL (PROD)
- API Keys via st.secrets
- Painel Administrativo protegido por senha
- Dashboard de custos/tokens apenas para admin
- Compatível com groq==0.8.0 (sem erro de proxies)
Autor: André de Lima
"""

import os, io, re, json, time, tempfile
from datetime import datetime
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go
from fpdf import FPDF
from pdfminer.high_level import extract_text
import streamlit as st

# ======== LLM Clients ========
try:
    from groq import Groq
except Exception:
    Groq = None
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# ======== Diretórios e Constantes ========
TRAINING_DIR = "training_data"
PROCESSED_DIR = "relatorios_processados"
os.makedirs(TRAINING_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

MAX_TOKENS_FIXED = 4096
TEMP_FIXED = 0.3
GPT_PRICE_INPUT_PER_1K = 0.005
GPT_PRICE_OUTPUT_PER_1K = 0.015

# ======== Tema Dark ========
DARK_CSS = """
<style>
:root{
  --bg:#20152b;--panel:#2a1f39;--accent:#9b6bff;--text:#EAE6F5;
}
html,body,.stApp{background:var(--bg);color:var(--text);}
section[data-testid="stSidebar"]{background:#1b1c25;}
.stButton>button{background:linear-gradient(135deg,var(--accent),#7c69d4);
color:white;border:0;padding:.55rem 1rem;border-radius:12px;font-weight:700;}
</style>
"""

# ======== Token Tracker ========
@dataclass
class TokenStep:
    prompt:int=0
    completion:int=0
    @property
    def total(self):return self.prompt+self.completion

@dataclass
class TokenTracker:
    steps:Dict[str,TokenStep]=field(default_factory=lambda:{
        "extracao":TokenStep(),
        "analise":TokenStep(),
        "chat":TokenStep(),
        "pdf":TokenStep()
    })
    def add(self,step:str,prompt:int,completion:int):
        if step not in self.steps:self.steps[step]=TokenStep()
        self.steps[step].prompt+=int(prompt or 0)
        self.steps[step].completion+=int(completion or 0)
    @property
    def total_tokens(self):return sum(s.total for s in self.steps.values())
    @property
    def total_prompt(self):return sum(s.prompt for s in self.steps.values())
    @property
    def total_completion(self):return sum(s.completion for s in self.steps.values())
    def cost_usd_gpt(self):return (self.total_prompt/1000)*GPT_PRICE_INPUT_PER_1K+(self.total_completion/1000)*GPT_PRICE_OUTPUT_PER_1K
    def dict(self):return{step:{"prompt":v.prompt,"completion":v.completion,"total":v.total}for step,v in self.steps.items()}

# ======== Cliente seguro (sem proxies) ========
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

# ======== Helpers ========
def extract_pdf_text_bytes(file)->str:
    try:return extract_text(file)
    except Exception as e:return f"[ERRO_EXTRACAO_PDF] {e}"

def gerar_perfil_cargo_dinamico(cargo:str)->Dict:
    return {
        "traits_ideais":{"Abertura":(5,8),"Conscienciosidade":(6,9),
                         "Extroversão":(4,8),"Amabilidade":(5,8),"Neuroticismo":(0,5)},
        "competencias_criticas":["Adaptabilidade","Comunicação","Trabalho em Equipe","Resolução de Problemas"],
        "descricao":f"Perfil para {cargo}"
    }

# ======== PDF ========
class PDFReport(FPDF):
    def header(self):
        if self.page_no()==1:return
        self.set_font("Helvetica","B",10)
        self.cell(0,8,"Elder Brain Analytics — Relatório",ln=1,align="C")
    def footer(self):
        if self.page_no()==1:return
        self.set_y(-12)
        self.set_font("Helvetica","I",8)
        self.cell(0,8,f"Página {self.page_no()}",align="C")

def gerar_pdf(bfa,analysis,cargo)->io.BytesIO:
    pdf=PDFReport();pdf.add_page()
    pdf.set_font("Helvetica","B",14)
    pdf.cell(0,10,"Relatório Elder Brain Analytics",ln=1,align="C")
    pdf.set_font("Helvetica","",10)
    pdf.multi_cell(0,6,f"Cargo: {cargo}\nData: {datetime.now():%d/%m/%Y %H:%M}")
    pdf.ln(5)
    pdf.multi_cell(0,6,"Resumo Executivo:")
    pdf.multi_cell(0,6,analysis.get("resumo_executivo",""))
    out=pdf.output(dest="S").encode("latin-1","ignore")
    buf=io.BytesIO(out);buf.seek(0)
    return buf

# ======== UI ========
def kpi_card(title,value,sub=None):
    st.markdown(f"<div style='padding:10px;border-radius:12px;background:#2a1f39;'>"
                f"<b>{title}</b><br><span style='font-size:1.8rem;'>{value}</span><br>"
                f"<span style='font-size:.9rem;color:#bbb'>{sub or ''}</span></div>",unsafe_allow_html=True)

# ======== APP PRINCIPAL ========
def main():
    st.set_page_config(page_title="EBA — PROD FINAL",page_icon="🧠",layout="wide")
    st.markdown(DARK_CSS,unsafe_allow_html=True)
    st.title("🧠 Elder Brain Analytics — Versão FINAL (PROD)")

    ss=st.session_state
    ss.setdefault("tracker",TokenTracker())
    ss.setdefault("admin_mode", False)

    # --- Sidebar ---
    with st.sidebar:
        st.header("⚙️ Configuração")
        provider=st.radio("Provedor",["Groq","OpenAI"])
        cargo=st.text_input("Cargo para análise")
        ss["cargo"]=cargo
        st.markdown("---")
        st.subheader("🔒 Painel Administrativo")
        admin_pwd=st.text_input("Senha do Admin",type="password")
        if admin_pwd and admin_pwd==st.secrets.get("ADMIN_PASSWORD",""):
            ss["admin_mode"]=True
            st.success("Modo Admin Ativo")
        elif admin_pwd:
            st.error("Senha incorreta")
            ss["admin_mode"]=False
        else:
            ss["admin_mode"]=False

    # --- Upload PDF ---
    st.subheader("📄 Upload de Relatório BFA")
    up=st.file_uploader("Carregue o PDF do relatório",type=["pdf"])
    if up and cargo:
        with st.spinner("Extraindo texto..."):
            txt=extract_pdf_text_bytes(up)
        st.text_area("Prévia do texto",txt[:1500],height=200)
        st.success("Texto extraído (simulação de análise).")

        # simulação de dados
        bfa={"candidato":{"nome":"Candidato Teste"}}
        analysis={"resumo_executivo":"Compatível com o cargo informado. Perfil equilibrado e boa resiliência."}

        if st.button("🔬 Gerar PDF de Teste"):
            buf=gerar_pdf(bfa,analysis,cargo)
            ss["tracker"].add("analise",1200,800)
            st.download_button("⬇️ Baixar Relatório",data=buf,file_name="relatorio.pdf",mime="application/pdf")

    # --- KPIs ---
    st.markdown("---")
    c1,c2,c3,c4=st.columns(4)
    with c1:kpi_card("Status","Pronto")
    if ss["admin_mode"]:
        with c2:kpi_card("Tokens Totais",ss["tracker"].total_tokens)
        with c3:kpi_card("Prompt/Output",f"{ss['tracker'].total_prompt}/{ss['tracker'].total_completion}")
        with c4:kpi_card("Custo Estimado",f"${ss['tracker'].cost_usd_gpt():.4f}")
    else:
        with c2:kpi_card("Relatórios","—")
        with c3:kpi_card("Andamento","—")
        with c4:kpi_card("Disponibilidade","Online")

    # --- Dashboard Admin ---
    if ss.get("admin_mode"):
        st.markdown("## 📊 Dashboard Administrativo")
        df=pd.DataFrame.from_dict(ss["tracker"].dict(),orient="index")
        st.dataframe(df,use_container_width=True)
        st.metric("Custo Total (USD)",f"${ss['tracker'].cost_usd_gpt():.4f}")
    else:
        st.caption("🔒 Painel administrativo restrito.")

    st.markdown("---")
    st.caption("Deploy seguro — Streamlit Cloud (share.streamlit.io)")

if __name__=="__main__":
    main()
