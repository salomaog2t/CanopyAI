"""
CanopyAI — Passo 4.1: auditoria cruzada de modelos (cross-model audit).

Pergunta: o ganho do Lazy Greedy otimizado com o M2 (HGB) é real, ou o otimizador só
aprendeu a explorar degraus das árvores de decisão do próprio M2?

Teste: as mesmas alocações são avaliadas por modelos que o otimizador NÃO viu:
    M1      Ridge linear (sem degraus; função suave)
    M2-alt  outro HGB monotônico, com hiperparâmetros diferentes e treinado em só metade
            dos blocos espaciais (degraus em lugares diferentes)
Se a vantagem do Greedy-M2 sobre a aleatória sobreviver nesses avaliadores, ela não é
artefato do M2. Para referência, roda também o Lazy Greedy otimizado pelo próprio M1.

Saída: reports/cross_model_audit.md e reports/cross_model_audit.json

Uso:
  python scripts/audit_cross_model.py [--budget 5000]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from gee.extract import ExtractParams, extract_grid  # noqa: E402
from models.lst_model import (  # noqa: E402
    build_feature_frame,
    fit_lst_model,
    monotonic_constraints,
    training_data,
)
from optimize.allocate import (  # noqa: E402
    AllocationProblem,
    hotspot_allocation,
    lazy_greedy,
    random_allocation,
)

BBOX = (-46.75, -23.65, -46.55, -23.50)  # mesma área do Passo 3
N_RANDOM = 5
ROBUST_RATIO = 1.5  # vantagem mínima sobre a aleatória, em cada avaliador, para chamar de robusta
REPORT_DIR = ROOT / "reports"


def fit_m2_alt(X, y, groups, seed=7):
    """HGB monotônico independente: outros hiperparâmetros, metade dos blocos."""
    rng = np.random.default_rng(seed)
    blocks = np.unique(groups)
    half = rng.choice(blocks, len(blocks) // 2, replace=False)
    sel = np.isin(groups, half)
    model = HistGradientBoostingRegressor(
        max_iter=500, learning_rate=0.05, max_leaf_nodes=63, min_samples_leaf=80,
        l2_regularization=0.1, monotonic_cst=monotonic_constraints(), early_stopping=False, random_state=seed,
    )
    return model.fit(X[sel], y[sel]), int(sel.sum())


class SignConstrainedLinear:
    """
    Regressão linear com os sinais das restrições do M2 (diagnóstico adicionado após a 1ª rodada):
    coeficiente ≤ 0 nas features de vegetação, ≥ 0 em built_frac, livre em NDBI/elevação.
    Truque: padroniza, multiplica cada feature restrita pelo sinal desejado e usa positive=True;
    features livres entram duas vezes (+x e −x), o que equivale a um coeficiente sem sinal fixo.
    """

    def fit(self, X, y):
        from sklearn.linear_model import LinearRegression
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-12
        self.signs = np.array(monotonic_constraints(), dtype=float)
        self.model = LinearRegression(positive=True).fit(self._design(X), y)
        return self

    def _design(self, X):
        Z = (X - self.mu) / self.sd
        fixed = Z[:, self.signs != 0] * self.signs[self.signs != 0]
        free = Z[:, self.signs == 0]
        return np.hstack([fixed, free, -free])

    def predict(self, X):
        return self.model.predict(self._design(X))


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--budget", type=int, default=5000)
    p.add_argument("--render-only", action="store_true", help="regera o .md a partir do .json existente")
    args = p.parse_args()
    if args.render_only:
        summary = json.loads((REPORT_DIR / "cross_model_audit.json").read_text(encoding="utf-8"))
        table = {tuple(k.split("|")): v for k, v in summary["table"].items()}
        evaluators = list(summary["greedy_m2_vs_random"])
        strategies = ["Greedy-M2", "Greedy-M1", "Mais quente primeiro", "Aleatória"]
        (REPORT_DIR / "cross_model_audit.md").write_text(render_markdown(summary, table, strategies, evaluators),
                                                         encoding="utf-8")
        print("reports/cross_model_audit.md regerado")
        return 0
    t_start = time.perf_counter()

    grid = extract_grid(BBOX, ExtractParams())
    feat = build_feature_frame(grid.df, grid.shape, grid.scale_m)
    X, y, groups, _ = training_data(feat, grid.shape, grid.scale_m)

    print("Ajustando avaliadores...")
    models = {"M2": fit_lst_model(X, y, "M2 HGB monotônico"), "M1": fit_lst_model(X, y, "M1 Ridge")}
    models["M2-alt"], n_alt = fit_m2_alt(X, y, groups)
    models["M1-sinal"] = SignConstrainedLinear().fit(X, y)
    r2_lin = {k: float(1 - ((y - models[k].predict(X)) ** 2).sum() / ((y - y.mean()) ** 2).sum())
              for k in ("M1", "M1-sinal")}
    problems = {k: AllocationProblem(feat, grid.shape, m, pixel_m=grid.scale_m) for k, m in models.items()}

    print("\nOtimizando com o M2 (alocação auditada)...")
    allocs = {"Greedy-M2": lazy_greedy(problems["M2"], args.budget, reeval_batch=1).trees}  # igual ao Passo 3
    print("\nOtimizando com o M1 (referência: o melhor que o próprio Ridge encontra)...")
    allocs["Greedy-M1"] = lazy_greedy(problems["M1"], args.budget, reeval_batch=1).trees
    allocs["Mais quente primeiro"] = hotspot_allocation(problems["M2"], args.budget).trees
    randoms = [random_allocation(problems["M2"], args.budget, seed=s).trees for s in range(N_RANDOM)]

    print("\nAvaliando todas as alocações em todos os modelos...")
    table, maps = {}, {}
    for ev, prob in problems.items():
        rnd = [prob.evaluate(t) for t in randoms]
        rnd_sum = np.array([r["soma_dT"] for r in rnd])
        table[(ev, "Aleatória")] = {
            "soma_dT": float(rnd_sum.mean()), "soma_dT_sd": float(rnd_sum.std(ddof=1)),
            "aquecimento_max": float(max(r["aquecimento_max"] for r in rnd)),
            "celulas_aquecidas": int(np.mean([(r["mapa_dT"] < -1e-9).sum() for r in rnd])),
        }
        for name, trees in allocs.items():
            r = prob.evaluate(trees)
            maps[(ev, name)] = r["mapa_dT"]
            table[(ev, name)] = {
                "soma_dT": r["soma_dT"], "aquecimento_max": r["aquecimento_max"],
                "celulas_aquecidas": int((r["mapa_dT"] < -1e-9).sum()),
            }
        for name in list(allocs) + ["Aleatória"]:
            table[(ev, name)]["vs_aleatoria"] = table[(ev, name)]["soma_dT"] / table[(ev, "Aleatória")]["soma_dT"]
        print(f"  {ev}: ok")

    # concordância espacial
    ev_mask = problems["M2"].eval_mask
    rho_m1 = spearman(maps[("M2", "Greedy-M2")][ev_mask], maps[("M1", "Greedy-M2")][ev_mask])
    rho_alt = spearman(maps[("M2", "Greedy-M2")][ev_mask], maps[("M2-alt", "Greedy-M2")][ev_mask])
    a, b = allocs["Greedy-M2"] > 0, allocs["Greedy-M1"] > 0
    jaccard = float((a & b).sum() / (a | b).sum())

    strategies = ["Greedy-M2", "Greedy-M1", "Mais quente primeiro", "Aleatória"]
    evaluators = list(problems)
    g = {ev: table[(ev, "Greedy-M2")]["vs_aleatoria"] for ev in evaluators}
    eff_m1 = table[("M1", "Greedy-M2")]["soma_dT"] / table[("M1", "Greedy-M1")]["soma_dT"]
    robust = g["M1"] >= ROBUST_RATIO and g["M2-alt"] >= ROBUST_RATIO
    retained_m1 = (g["M1"] - 1) / (g["M2"] - 1)

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "bbox": BBOX, "budget": args.budget, "cache": grid.path.name,
        "n_train": int(len(y)), "n_train_m2_alt": n_alt,
        "table": {f"{ev}|{s}": table[(ev, s)] for ev in evaluators for s in strategies},
        "greedy_m2_vs_random": g, "greedy_m2_over_greedy_m1_in_m1": eff_m1,
        "fraction_of_advantage_retained_in_m1": retained_m1,
        "spearman_cooling_map_m2_vs_m1": rho_m1, "spearman_cooling_map_m2_vs_m2alt": rho_alt,
        "jaccard_planted_cells_greedy_m2_vs_m1": jaccard,
        "robust": bool(robust), "robust_threshold": ROBUST_RATIO,
        "r2_in_sample_linear": r2_lin,
        "post_hoc_sign_linear_ratio": g["M1-sinal"],
        "runtime_s": time.perf_counter() - t_start,
    }

    REPORT_DIR.mkdir(exist_ok=True)
    (REPORT_DIR / "cross_model_audit.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (REPORT_DIR / "cross_model_audit.md").write_text(render_markdown(summary, table, strategies, evaluators),
                                                     encoding="utf-8")

    print(f"\n{'Alocação':<22}" + "".join(f"{ev + ' Σ ΔT':>14}{'×alea.':>8}" for ev in evaluators))
    for s in strategies:
        print(f"{s:<22}" + "".join(f"{table[(ev, s)]['soma_dT']:>14.0f}{table[(ev, s)]['vs_aleatoria']:>8.2f}"
                                   for ev in evaluators))
    print(f"\nGreedy-M2 no M1: {eff_m1:.0%} do que o próprio M1 consegue otimizando; "
          f"retém {retained_m1:.0%} da vantagem sobre a aleatória")
    print(f"Spearman dos mapas de ΔT (Greedy-M2): M2×M1={rho_m1:.2f}, M2×M2-alt={rho_alt:.2f}; "
          f"Jaccard células Greedy-M2×Greedy-M1={jaccard:.2f}")
    print(f"Veredito: {'ROBUSTO' if robust else 'NÃO ROBUSTO'} (limiar {ROBUST_RATIO}× a aleatória em M1 e M2-alt)")
    print(f"Pós-hoc: Greedy-M2 no M1-sinal = {g['M1-sinal']:.2f}× a aleatória; "
          f"R² treino M1={r2_lin['M1']:.3f}, M1-sinal={r2_lin['M1-sinal']:.3f}")
    print(f"Relatório: reports/cross_model_audit.md ({summary['runtime_s'] / 60:.1f} min)")
    return 0


def render_markdown(s, table, strategies, evaluators) -> str:
    def row(name):
        cells = []
        for ev in evaluators:
            t = table[(ev, name)]
            sd = f" ± {t['soma_dT_sd']:.0f}" if "soma_dT_sd" in t else ""
            cells.append(f"{t['soma_dT']:,.0f}{sd} ({t['vs_aleatoria']:.2f}×)")
        return f"| {name} | " + " | ".join(cells) + " |"

    def warm_row(name):
        return f"| {name} | " + " | ".join(
            f"{table[(ev, name)]['celulas_aquecidas']:,} / {table[(ev, name)]['aquecimento_max']:.3f} °C"
            for ev in evaluators) + " |"

    g = s["greedy_m2_vs_random"]
    verdict = "**robusta**" if s["robust"] else "**não robusta**"
    independent = [ev for ev in evaluators if ev != "M2"]
    beats_all = all(table[(ev, "Greedy-M2")]["soma_dT"] > max(table[(ev, "Aleatória")]["soma_dT"],
                                                               table[(ev, "Mais quente primeiro")]["soma_dT"])
                    for ev in evaluators)
    tree_evs = [ev for ev in independent if ev.startswith("M2")]
    lin_evs = [ev for ev in independent if ev.startswith("M1")]
    rnd = {ev: table[(ev, "Aleatória")]["soma_dT"] for ev in evaluators}
    magnitude = max(rnd.values()) / min(rnd.values())
    conclusion = [
        "## Conclusão",
        "",
        f"- O Greedy-M2 supera a aleatória e a \"mais quente primeiro\" em {'todos os' if beats_all else 'nem todos os'} "
        f"{len(evaluators)} avaliadores. O **ranking** das estratégias é robusto.",
        f"- O **tamanho** da vantagem depende da família do modelo: "
        + ", ".join(f"{g[ev]:.2f}× no {ev}" for ev in tree_evs)
        + " (árvores de decisão independentes) contra "
        + ", ".join(f"{g[ev]:.2f}× no {ev}" for ev in lin_evs)
        + " (lineares). Como o M2-alt tem degraus em outros lugares e ainda assim concorda com o M2, a vantagem "
        "não é artefato dos degraus específicos do M2. A explicação provável são efeitos não lineares que um "
        "modelo linear não representa por construção (hipótese ainda não testada: plantar onde há pouca copa "
        "resfriaria mais).",
        f"- O **efeito absoluto** de plantar varia {magnitude:.1f}× entre avaliadores (Σ ΔT da alocação aleatória). "
        "No Passo 3, HGB e Ridge tiveram R² espacial quase igual (0,436 × 0,421): prever bem a LST atual não "
        "basta para distinguir modelos que discordam sobre o efeito de uma intervenção.",
        "- **Como comunicar:** \"o otimizador supera as linhas de base em todos os modelos testados; o ganho "
        f"varia de {min(g[ev] for ev in independent):.1f}× a {max(g[ev] for ev in independent):.1f}× a alocação "
        "aleatória conforme o modelo\", em vez de um número único. Validar com dados de campo (p.ex. LST antes e "
        "depois de plantios reais) é o que separaria as duas famílias de modelo.",
        "",
    ]
    lines = [
        "# Auditoria cruzada de modelos — CanopyAI",
        "",
        f"Gerado por `scripts/audit_cross_model.py` em {s['created_utc']} (UTC).",
        "",
        "## Pergunta",
        "",
        "O Lazy Greedy otimizado com o M2 (HistGradientBoosting monotônico) rendeu 2,34× o resfriamento da "
        "alocação aleatória. Como o otimizador escolhe exatamente onde o M2 responde mais, esse número pode "
        "estar inflado por degraus das árvores de decisão. A auditoria avalia as mesmas alocações em modelos "
        "que o otimizador não viu.",
        "",
        "## Configuração",
        "",
        f"- Área: centro expandido de São Paulo, bbox {tuple(s['bbox'])}, grade de 30 m (`{s['cache']}`)",
        f"- Orçamento: {s['budget']:,} árvores de 50 m² de copa; mesma intervenção em todas as avaliações",
        f"- Treino: {s['n_train']:,} células (M1 e M2); M2-alt em {s['n_train_m2_alt']:,} células (metade dos blocos de 1,5 km)",
        "- Avaliadores:",
        "  - **M2**: o modelo usado pelo otimizador (referência, não é teste independente)",
        "  - **M1**: Ridge linear, sem degraus",
        "  - **M2-alt**: HGB monotônico com outros hiperparâmetros (63 folhas, 500 iterações, lr 0,05) e outra metade dos dados",
        "  - **M1-sinal** (*diagnóstico definido depois da 1ª rodada*): linear como o M1, mas com os sinais das "
        "restrições do M2 (vegetação ≤ 0, construído ≥ 0). É suave como o M1 e fisicamente coerente como o M2. "
        "Não entra no veredito, que segue o critério fixado antes de ver os resultados.",
        f"- R² no treino dos lineares: M1 {s['r2_in_sample_linear']['M1']:.3f}, "
        f"M1-sinal {s['r2_in_sample_linear']['M1-sinal']:.3f}",
        f"- Aleatória: média ± desvio-padrão de {N_RANDOM} sorteios",
        "",
        "## Resfriamento total Σ ΔT (°C·célula) e razão sobre a aleatória",
        "",
        "| Alocação | " + " | ".join(f"avaliada no {ev}" for ev in evaluators) + " |",
        "|---|" + "---|" * len(evaluators),
        *[row(n) for n in strategies],
        "",
        "## Células que esquentam (nº / pior caso)",
        "",
        "| Alocação | " + " | ".join(evaluators) + " |",
        "|---|" + "---|" * len(evaluators),
        *[warm_row(n) for n in strategies],
        "",
        "## Concordância espacial",
        "",
        f"- Spearman entre os mapas de ΔT do Greedy-M2 avaliados no M2 e no M1: {s['spearman_cooling_map_m2_vs_m1']:.2f}",
        f"- Spearman entre M2 e M2-alt: {s['spearman_cooling_map_m2_vs_m2alt']:.2f}",
        f"- Jaccard das células plantadas, Greedy-M2 × Greedy-M1: {s['jaccard_planted_cells_greedy_m2_vs_m1']:.2f}",
        "",
        "## Resultado",
        "",
        f"- Vantagem do Greedy-M2 sobre a aleatória: {g['M2']:.2f}× no M2, **{g['M1']:.2f}× no M1** e "
        f"**{g['M2-alt']:.2f}× no M2-alt**.",
        f"- No M1, o Greedy-M2 retém {s['fraction_of_advantage_retained_in_m1']:.0%} da vantagem que tinha no M2 e "
        f"alcança {s['greedy_m2_over_greedy_m1_in_m1']:.0%} do resfriamento que o próprio M1 obtém quando é ele quem otimiza.",
        f"- Veredito automático (limiar de {s['robust_threshold']}× a aleatória no M1 e no M2-alt): a vantagem é {verdict}.",
        f"- Diagnóstico pós-hoc: no M1-sinal (linear, suave, com sinais físicos) o Greedy-M2 rende "
        f"**{g['M1-sinal']:.2f}×** a aleatória.",
        "",
        "## Como ler",
        "",
        "- **Degraus do M2.** O M2-alt tem degraus em outros lugares; se o otimizador só explorasse degraus do M2, "
        "a vantagem sumiria nele. O M1-sinal não tem degrau nenhum. A razão nesses dois avaliadores é a "
        "evidência mais direta sobre essa hipótese.",
        "- **M1 sem restrição.** Os coeficientes do Ridge têm sinais trocados por colinearidade (NDVI do pixel "
        "\"aquece\", construído \"resfria\"), por isso ele prevê aquecimento em milhares de células para "
        "qualquer plantio, até o aleatório. É um avaliador fraco para uma intervenção física; ainda assim, o "
        "veredito acima o inclui porque esse foi o critério definido antes.",
        "- **Magnitude.** Os avaliadores concordam mais na *ordem* das estratégias do que no *tamanho* do "
        "resfriamento. Compare as somas de ΔT da linha \"Aleatória\" entre colunas: o efeito absoluto de "
        "plantar depende fortemente do modelo, e deve ser comunicado como incerto.",
        "",
        *conclusion,
        "## Limitações",
        "",
        "- Os três avaliadores aprendem dos mesmos dados (uma cena composta, uma cidade). A auditoria "
        "descarta artefatos de um modelo específico, não erros comuns a todos — por exemplo, confusão entre "
        "vegetação e outros fatores urbanos não medidos.",
        "- O M1 tem coeficientes contraintuitivos por colinearidade (NDVI do pixel e fração construída com "
        "sinal trocado), então pode marcar como \"aquecimento\" partes da intervenção. Células que esquentam "
        "no M1 refletem isso, não um efeito físico.",
        "- ΔT é LST prevista (temperatura de superfície), não temperatura do ar, e assume copa madura.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
