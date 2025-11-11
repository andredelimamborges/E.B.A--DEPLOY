# elder_brain_v8_5_pro_prod.py
"""
Elder Brain Analytics — v9.1 PROD (Streamlit Cloud)
Deploy seguro com:
 - API Keys protegidas via st.secrets
 - Painel Administrativo protegido por senha
 - Dashboard de tokens/custos apenas para admin
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

# ======== Importações LLM ========
try:
    from groq import Groq
except Exception:
    Groq = None
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# ======== Constantes ========
TRAINING_DIR = "training_data"
PROCESSED_DIR = "relatorios_processados"
os.makedirs(TRAINING_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

GPT_PRICE_INPUT_PER_1K = 0.005
GPT_PRICE_OUTPUT_PER_1K = 0.015
MAX_TOKENS_FIXED = 4096
TEMP_FIXED = 0.3

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

# ======== Funções Auxiliares ========
def get_llm_client(provider:str):
    """Cria cliente seguro usando secrets"""
    if provider.lower()=="groq":
        key=st.secrets.get("GROQ_API_KEY","")
        if not key:raise RuntimeError("Groq API Key não configurada em secrets.")
        return Groq(api_key=key)
    elif provider.lower()=="openai":
        key=st.secrets.get("OPENAI_API_KEY","")
        if not key:raise RuntimeError("OpenAI API Key não configurada em secrets.")
        return OpenAI(api_key=key)
    else:
        raise RuntimeError(f"Provedor inválido: {provider}")

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

# ======== Funções Gráficas ========
def criar_gauge_fit(fit:float)->go.Figure:
    fig=go.Figure(go.Indicator(
        mode="gauge+number",value=float(fit or 0),
        gauge={'axis':{'range':[None,100]},
               'bar':{'color':"#9b6bff"},
               'steps':[{'range':[0,40],'color':'#E74C3C'},
                        {'range':[40,70],'color':'#F39C12'},
                        {'range':[70,100],'color':'#2ECC71'}]},
        title={'text':"Fit para o Cargo"}))
    fig.update_layout(height=350)
    return fig

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
    st.set_page_config(page_title="EBA — PROD",page_icon="🧠",layout="wide")
    st.markdown(DARK_CSS,unsafe_allow_html=True)
    st.title("🧠 Elder Brain Analytics — Versão Testes")
    ss=st.session_state
    ss.setdefault("tracker",TokenTracker())

    # --- Login Admin ---
    with st.sidebar:
        st.header("⚙️ Configuração")
        provider=st.radio("Provedor",["Groq","OpenAI"])
        cargo=st.text_input("Cargo para análise")
        st.session_state["cargo"]=cargo
        st.markdown("---")
        st.subheader("🔒 Painel Administrativo")
        admin_pwd=st.text_input("Senha do Admin",type="password")
        if admin_pwd==st.secrets.get("ADMIN_PASSWORD",""):
            ss["admin_mode"]=True;st.success("Modo Admin Ativo")
        else:ss["admin_mode"]=False

    # --- Upload PDF ---
    st.subheader("📄 Upload de Relatório BFA/BOL")
    up=st.file_uploader("Carregue o PDF do relatório",type=["pdf"])
    if up and cargo:
        with st.spinner("Extraindo texto..."):
            txt=extract_pdf_text_bytes(up)
        st.text_area("Prévia do texto",txt[:1500],height=200)
        st.success("Texto extraído. (Simulação de análise)")

        # Simulação: geração de dados fictícios
        bfa={"candidato":{"nome":"Candidato Teste"}}
        analysis={"resumo_executivo":"Compatível com o cargo informado. Perfil equilibrado e boa resiliência."}

        if st.button("🔬 Gerar PDF de Teste"):
            buf=gerar_pdf(bfa,analysis,cargo)
            ss["tracker"].add("analise",1200,800)
            st.download_button("⬇️ Baixar Relatório",data=buf,file_name="relatorio.pdf",mime="application/pdf")

    # --- KPIs gerais ---
    st.markdown("---")
    c1,c2,c3,c4=st.columns(4)
    with c1:kpi_card("Status","Pronto")
    with c2:kpi_card("Tokens Totais",ss["tracker"].total_tokens)
    with c3:kpi_card("Prompt/Output",f"{ss['tracker'].total_prompt}/{ss['tracker'].total_completion}")
    with c4:kpi_card("Custo Estimado",f"${ss['tracker'].cost_usd_gpt():.4f}")

    # --- Dashboard Admin ---
    if ss.get("admin_mode"):
        st.markdown("## 📊 Dashboard Administrativo")
        td=ss["tracker"].dict()
        df=pd.DataFrame.from_dict(td,orient="index")
        st.dataframe(df,use_container_width=True)
        st.metric("Custo Total (USD)",f"${ss['tracker'].cost_usd_gpt():.4f}")
    else:
        st.caption("🔒 Painel administrativo restrito.")

    st.markdown("---")
    st.caption("Deploy seguro — Streamlit Cloud (share.streamlit.io)")

if __name__=="__main__":
    main()
