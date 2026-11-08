# ============================================================
# app.py — Recomendador de cesta (Market Basket Analysis)
# ------------------------------------------------------------
# Serve os três modelos treinados no notebook:
#   • Regras de associação (Apriori)
#   • Autoencoder de cesta (embeddings de item)
#   • Popularidade (baseline)
# Não retreina nada: apenas lê modelo_cesta/bundle.pkl.
# Rodar com:  streamlit run app.py
# ============================================================
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import streamlit as st

DIR = Path(__file__).parent / "../modelos"
CAMINHO = DIR / "bundle.pkl"

st.set_page_config(page_title="Recomendador de Cesta",
                   page_icon="🛒", layout="wide")


# ------------------------------------------------------------
# Carregamento
# ------------------------------------------------------------
@st.cache_resource(show_spinner="Carregando modelo...")
def carregar():
    with open(CAMINHO, "rb") as f:
        b = pickle.load(f)

    # índice item -> posição, para não varrer a lista a cada clique
    b["idx"] = {it: i for i, it in enumerate(b["itens"])}

    # regras: converte antecedentes em frozenset para filtro O(1)
    for r in b["regras"]:
        r["ant_set"] = frozenset(r["antecedents"])
        r["template"] = (f"{', '.join(r['antecedents'])} → "
                         f"{', '.join(r['consequents'])}")

    # presets de demonstração: antecedentes das regras mais fortes
    vistas, presets = set(), []
    for r in sorted(b["regras"], key=lambda x: -x["lift"]):
        chave = tuple(r["antecedents"])
        if chave not in vistas and len(r["antecedents"]) <= 3:
            vistas.add(chave)
            presets.append(r["antecedents"])
        if len(presets) == 3:
            break
    b["presets"] = presets
    return b


if not CAMINHO.exists():
    st.error(
        "**Modelo não encontrado.** Rode a célula de exportação no notebook "
        "e copie a pasta `modelo_cesta/` para o diretório deste app."
    )
    st.code("modelo_cesta/bundle.pkl", language="text")
    st.stop()

M = carregar()


# ------------------------------------------------------------
# Motores de recomendação
# ------------------------------------------------------------
def rec_regras(cesta, k, min_lift=1.0, min_sup=0.0):
    """Filtra as regras cujo antecedente está contido na cesta."""
    conj = set(cesta)
    out = []
    for r in M["regras"]:
        if not r["ant_set"] <= conj:
            continue
        if r["lift"] < min_lift or r["support"] < min_sup:
            continue
        for it in r["consequents"]:
            if it in conj:
                continue
            out.append({
                "item": it,
                "score": r["lift"] * r["confidence"],
                "lift": r["lift"],
                "confidence": r["confidence"],
                "support": r["support"],
                "explicacao": r["template"],
            })
    out.sort(key=lambda d: -d["score"])

    vistos, final = set(), []
    for d in out:
        if d["item"] not in vistos:
            vistos.add(d["item"])
            final.append(d)
    return final[:k]


def rec_neural(cesta, k):
    """Média dos embeddings dos itens da cesta, projetada no catálogo."""
    pos = [M["idx"][i] for i in cesta if i in M["idx"]]
    if not pos:
        return []
    v = M["item_emb"][pos].mean(axis=0)
    score = v @ M["item_emb"].T

    mascara = np.ones(len(score), dtype=bool)
    mascara[pos] = False
    if mascara.sum() == 0:
        return []

    s = score[mascara]
    lo, hi = s.min(), s.max()
    s = (s - lo) / (hi - lo) if hi > lo else np.zeros_like(s)
    candidatos = np.flatnonzero(mascara)[np.argsort(-s)[:k]]

    return [{"item": M["itens"][i],
             "score": float(score[i]),
             "explicacao": "similaridade latente aprendida pelo autoencoder"}
            for i in candidatos]


def rec_popular(cesta, k):
    conj = set(cesta)
    ordem = np.argsort(-M["pop"])
    out = []
    for i in ordem:
        it = M["itens"][i]
        if it not in conj:
            out.append({"item": it, "score": float(M["pop"][i]),
                        "explicacao": "item com maior volume de cestas"})
        if len(out) == k:
            break
    return out


# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------
cfg = M["config"]

with st.sidebar:
    st.header("🛒 Cesta atual")
    cesta = st.multiselect(
        "Selecione os itens da cesta",
        options=M["itens"],
        default=[],
        placeholder="Digite para buscar...",
    )

    if M["presets"]:
        st.caption("Cestas de exemplo")
        cols = st.columns(len(M["presets"]))
        for c, preset in zip(cols, M["presets"]):
            rotulo = preset[0][:12] + ("…" if len(preset[0]) > 12 else "")
            if c.button(rotulo, use_container_width=True,
                        help=" + ".join(preset)):
                st.session_state["cesta_pre"] = preset

    # aplica preset vindo do botão (no próximo rerun)
    if "cesta_pre" in st.session_state:
        cesta = list(st.session_state.pop("cesta_pre"))

    st.divider()
    st.header("⚙️ Parâmetros")
    k = st.slider("Recomendações por modelo", 1, 10, 5)
    min_lift = st.slider("Lift mínimo (regras)", 1.0, 4.0, 1.2, 0.1)
    min_sup = st.slider("Suporte mínimo (regras)", 0.0,
                        float(cfg["min_support"]) * 3,
                        float(cfg["min_support"]), 0.001,
                        format="%.3f")

    st.divider()
    st.caption(
        f"**Modelo** · {M['fonte']}\n\n"
        f"`min_support` {cfg['min_support']:.2%} · `max_len` {cfg['max_len']}\n\n"
        f"{len(M['itens'])} itens · {len(M['regras'])} regras\n\n"
        f"Atualizado em {M['data']}"
    )


# ------------------------------------------------------------
# Cabeçalho
# ------------------------------------------------------------
st.title("🛒 Recomendador de cesta")
st.caption(
    f"Comparação lado a lado entre **regras de associação**, "
    f"**autoencoder de cesta** e **popularidade** — "
    f"{M['fonte']}, {cfg['n_trans']:,} cestas.".replace(",", ".")
)

if not cesta:
    st.info("Selecione itens na barra lateral — ou use uma cesta de exemplo.")
    st.stop()

st.markdown("**Cesta atual:** " + " · ".join(f"`{i}`" for i in cesta))
st.divider()


# ------------------------------------------------------------
# Abas
# ------------------------------------------------------------
aba_rec, aba_regras, aba_diag, aba_sobre = st.tabs(
    ["🎯 Recomendações", "🔗 Regras", "📊 Diagnóstico", "ℹ️ Sobre"])


# ---------- Recomendações ----------
with aba_rec:
    r_regra = rec_regras(cesta, k, min_lift, min_sup)
    r_neural = rec_neural(cesta, k)
    r_pop = rec_popular(cesta, k)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.subheader("🔗 Regras de associação")
        if not r_regra:
            st.warning("Nenhuma regra cobre esta cesta com os filtros atuais. "
                       "Reduza o lift ou o suporte mínimo.")
        for d in r_regra:
            with st.container(border=True):
                st.markdown(f"**{d['item']}**")
                st.caption(f"lift **{d['lift']:.2f}** · "
                           f"conf. {d['confidence']:.0%} · "
                           f"sup. {d['support']:.2%}")
                with st.expander("por quê?"):
                    st.code(d["explicacao"], language="text")

    with c2:
        st.subheader("🧠 Autoencoder")
        if not r_neural:
            st.warning("Cesta sem itens reconhecidos pelo modelo.")
        for d in r_neural:
            with st.container(border=True):
                st.markdown(f"**{d['item']}**")
                st.caption(f"escore latente {d['score']:.3f}")
                with st.expander("por quê?"):
                    st.write(d["explicacao"])

    with c3:
        st.subheader("📈 Popularidade")
        for d in r_pop:
            with st.container(border=True):
                st.markdown(f"**{d['item']}**")
                st.caption(f"{d['score']:.2%} das cestas")

    st.divider()

    # consenso entre os modelos
    conj_regra = {d["item"] for d in r_regra}
    conj_neural = {d["item"] for d in r_neural}
    conj_pop = {d["item"] for d in r_pop}
    consenso = conj_regra & conj_neural & conj_pop

    if consenso:
        st.success("**Consenso dos três modelos:** " +
                   " · ".join(f"`{i}`" for i in sorted(consenso)))
    else:
        st.caption("Nenhum item aparece nos três rankings — os modelos discordam "
                   "nesta cesta, o que é comum quando o sinal de co-compra é fraco.")

    # tabela consolidada + download
    linhas = []
    for nome, lst in [("Regras", r_regra), ("Autoencoder", r_neural),
                      ("Popularidade", r_pop)]:
        for pos, d in enumerate(lst, 1):
            linhas.append({"modelo": nome, "posicao": pos, "item": d["item"],
                           "score": round(d["score"], 4),
                           "explicacao": d["explicacao"]})
    df_rec = pd.DataFrame(linhas)

    with st.expander("📋 Ver tabela consolidada"):
        st.dataframe(df_rec, use_container_width=True, hide_index=True)
        st.download_button(
            "Baixar CSV das recomendações",
            df_rec.to_csv(index=False).encode("utf-8-sig"),
            file_name="recomendacoes.csv",
            mime="text/csv",
        )


# ---------- Regras ----------
with aba_regras:
    st.subheader("Explorador de regras")

    f1, f2, f3 = st.columns(3)
    filtro_lift = f1.slider("Lift mínimo", 1.0, 6.0, 1.5, 0.1, key="rf_lift")
    filtro_conf = f2.slider("Confiança mínima", 0.0, 1.0, 0.0, 0.05, key="rf_conf")
    filtro_sup = f3.slider("Suporte mínimo", 0.0, float(cfg["min_support"]) * 3,
                           float(cfg["min_support"]), 0.001,
                           format="%.3f", key="rf_sup")

    busca = st.text_input("Filtrar por item (contém)",
                          placeholder="ex.: CHURN, MUG, BAG")
    piso_sup = cfg["min_support"]

    filtradas = [r for r in M["regras"]
                 if r["lift"] >= filtro_lift
                 and r["confidence"] >= filtro_conf
                 and r["support"] >= filtro_sup
                 and (not busca or busca.upper() in r["template"].upper())]

    df_regras = pd.DataFrame([{
        "regra": r["template"],
        "support": round(r["support"], 4),
        "confidence": round(r["confidence"], 4),
        "lift": round(r["lift"], 3),
        "leverage": round(r["leverage"], 5),
        "fragil": "⚠️" if r["support"] < piso_sup else "",
    } for r in filtradas]).sort_values("lift", ascending=False)

    m1, m2, m3 = st.columns(3)
    m1.metric("Regras no filtro", f"{len(df_regras):,}".replace(",", "."))
    m2.metric("Lift mediano",
              f"{df_regras['lift'].median():.2f}" if len(df_regras) else "—")
    m3.metric("Com suporte ≥ piso",
              int((df_regras["support"] >= piso_sup).sum()) if len(df_regras) else 0)

    st.dataframe(df_regras, use_container_width=True, hide_index=True,
                 height=460)
    st.caption("⚠️ = lift alto sustentado por suporte abaixo do piso — "
               "hipótese para teste, não para decisão.")


# ---------- Diagnóstico ----------
with aba_diag:
    st.subheader("Comparação dos modelos")
    st.caption(f"Avaliação leave-one-out com k = {cfg['k_avaliacao']}")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Todas as cestas com 3+ itens**")
        st.dataframe(pd.DataFrame(M["avaliacao"]).T.round(4),
                     use_container_width=True)

    with col_b:
        st.markdown("**Recorte: cestas de 3 a 6 itens**")
        st.dataframe(pd.DataFrame(M["avaliacao_restrita"]).T.round(4),
                     use_container_width=True)

    st.divider()
    st.subheader("Itens mais frequentes")
    meta = M["item_meta"].sort_values("cestas", ascending=False).head(20)
    st.bar_chart(meta.set_index("item")["cestas"], height=340)

    st.divider()
    st.subheader("Base transacional")
    eda = M["eda"]
    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Itens/cesta (média)", f"{eda['media']:.2f}")
    e2.metric("Itens/cesta (mediana)", f"{eda['mediana']:.0f}")
    e3.metric("Itens raros (≤10 cestas)", f"{eda['frac_raros']:.1%}")
    e4.metric("Spearman freq×receita", f"{eda['spearman']:.3f}")

    with st.expander("Matriz de similaridade entre itens da cesta"):
        if len(cesta) >= 2:
            pos = [M["idx"][i] for i in cesta if i in M["idx"]]
            sim = M["item_emb"][pos] @ M["item_emb"][pos].T
            st.dataframe(
                pd.DataFrame(np.round(sim, 3),
                             index=[M["itens"][i] for i in pos],
                             columns=[M["itens"][i] for i in pos]),
                use_container_width=True)
        else:
            st.caption("Selecione ao menos 2 itens.")


# ---------- Sobre ----------
with aba_sobre:
    st.subheader("Como os três modelos funcionam")
    st.markdown(f"""
**🔗 Regras de associação (Apriori).** Filtra as regras cujo antecedente está
contido na cesta e ordena por `lift × confidence`. É o único modelo que
**explica** a recomendação — daí a frase "quem comprou X também levou Y".
Limitação: só recomenda itens que cruzaram o `min_support` de
{cfg['min_support']:.2%}.

**🧠 Autoencoder de cesta.** Comprime a cesta em um espaço latente e projeta a
média dos embeddings dos itens sobre o catálogo inteiro. Cobre os
{M['eda']['n_itens_catalogo']:,} itens do catálogo original, inclusive a cauda
longa que o Apriori nunca vê. Não explica o motivo item a item.

**📈 Popularidade.** Baseline obrigatório. Se este ranking empata com os
outros, o aparato todo não agregou valor — e essa é uma conclusão válida.

> Em produção usa-se as três camadas: a rede ordena, as regras justificam,
> a popularidade desempata em cold start.
""".replace(",", "."))

    st.divider()
    st.caption(
        f"Modelo v{M['versao']} · treinado em {M['data']} · "
        f"embeddings: {M['origem_emb']} · {cfg['n_trans']:,} cestas".replace(",", ".")
    )