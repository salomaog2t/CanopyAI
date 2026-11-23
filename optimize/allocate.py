"""
CanopyAI — Passo 3: alocação de árvores para máxima redução de LST.

Modelo de intervenção (por célula de 30 m = 900 m²)
--------------------------------------------------
Plantar n árvores de copa madura com área a acrescenta F = n·a / 900 de fração arbórea.

  Área plantável = fração "outra" (1 − árvore − construído − água: grama, solo, arbusto)
                   + street_share × fração construída (calçadas, canteiros, estacionamentos)
  Capacidade     = ⌊área plantável · 900 / a⌋ árvores
  A copa ocupa primeiro a área "outra" e depois a parcela plantável do construído, que então
  diminui (built_frac −= o que foi ocupado).
  NDVI           = NDVI₀ + F · (P95 − NDVI₀), onde P95 é o percentil 95 do NDVI das células
                   quase puramente arbóreas da cena (tree_frac ≥ 0,9). É uma mistura linear:
                   a fração F da célula passa a ter a assinatura espectral de copa madura.
  O NDBI não é alterado (conservador: na prática ele cairia junto, resfriando mais).

Tudo é calculado a partir do estado original (n total de árvores na célula), então o resultado
não depende da ordem em que as árvores foram adicionadas.

Depois da intervenção, as features de vizinhança são recalculadas por convolução
(features.spatial) e o modelo M2 prevê a nova anomalia de LST. ΔT = previsto antes − depois,
somado sobre as células: °C·célula (1 célula = 0,09 ha).

Otimização: Lazy Greedy (CELF)
------------------------------
Uma ação = adicionar até `step` árvores numa célula. O ganho de uma ação é a soma de ΔT em
todas as células afetadas (o disco de 600 m em volta dela); a fila ordena pelo ganho por árvore. Como as médias de vizinhança são
lineares, o efeito numa célula i é Δ·k(i−j)/den(i) — atualização local exata, sem reconvoluir a
grade. A fila de prioridade guarda o último ganho conhecido de cada ação; ao sair do topo, uma
ação desatualizada é reavaliada e volta para a fila, e só é aceita se continuar no topo.

Observação honesta: com um modelo de árvores (HGB) o ganho não é submodular, então o guloso
preguiçoso é uma heurística — boa, mas sem a garantia (1 − 1/e) do caso submodular.

Para caber no tempo, um substituto linear do próprio M2 (calculado por convolução para todas
as células de uma vez) pré-seleciona um pool de candidatas; só o pool recebe avaliação exata.
"""

from __future__ import annotations

import heapq
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import ndimage

from features.spatial import DEFAULT_RADII_M, DEFAULT_WATER_THRESHOLD, add_neighborhood_features, disk_kernel, feature_name
from models.lst_model import FEATURES, VEGETATION_COLUMNS


@dataclass(frozen=True)
class InterventionParams:
    canopy_area_m2: float = 50.0        # copa madura (~8 m de diâmetro)
    street_share_of_built: float = 0.15  # parte da fração construída onde cabe árvore
    ndvi_tree_quantile: float = 0.95
    pure_tree_threshold: float = 0.9


@dataclass
class AllocationResult:
    strategy: str
    trees: np.ndarray                  # (H·W,) árvores por célula
    runtime_s: float
    predicted_gain: float | None = None  # Σ ΔT acumulado pelo guloso (atualização local)
    n_evaluations: int = 0
    pool_size: int = 0
    fallback_trees: int = 0            # árvores colocadas sem avaliação (ganho exato zerou)


class AllocationProblem:
    """Estado da grade + modelo; avalia intervenções localmente (guloso) ou na grade inteira."""

    def __init__(
        self,
        feat_df: pd.DataFrame,
        shape: tuple[int, int],
        model,
        features=FEATURES,
        params: InterventionParams = InterventionParams(),
        pixel_m: float = 30.0,
        radii_m=DEFAULT_RADII_M,
    ):
        self.shape = shape
        self.H, self.W = shape
        self.model = model
        self.features = list(features)
        self.params = params
        self.pixel_m = pixel_m
        self.radii = tuple(radii_m)
        self.cell_area = pixel_m**2
        self.base_df = feat_df
        self.col = {f: i for i, f in enumerate(self.features)}

        self.tree0 = feat_df["tree_frac"].to_numpy(np.float64)
        self.built0 = feat_df["built_frac"].to_numpy(np.float64)
        self.ndvi0 = feat_df["NDVI"].to_numpy(np.float64)
        water = feat_df["water_frac"].to_numpy(np.float64)
        self.other0 = np.clip(1 - self.tree0 - self.built0 - np.nan_to_num(water), 0, 1)
        self.street0 = params.street_share_of_built * self.built0

        X = feat_df[self.features].to_numpy(np.float64)
        land = np.isfinite(water) & (water < DEFAULT_WATER_THRESHOLD)
        self.eval_mask = land & np.isfinite(X).all(axis=1)
        plantable = np.nan_to_num(self.other0 + self.street0)
        self.capacity = np.where(
            self.eval_mask, np.floor(plantable * self.cell_area / params.canopy_area_m2 + 1e-9), 0
        ).astype(np.int32)

        pure = self.eval_mask & (self.tree0 >= params.pure_tree_threshold)
        ref = self.ndvi0[pure] if pure.sum() >= 50 else self.ndvi0[self.eval_mask]
        self.ndvi_tree = float(np.quantile(ref, params.ndvi_tree_quantile))

        # denominadores das médias focais, exatamente como em features.spatial
        self.den = {}
        self.terms = []  # (coluna, índice da feature, raio)
        for c in VEGETATION_COLUMNS:
            values = feat_df[c].to_numpy(np.float64).reshape(shape)
            mask = (land.reshape(shape) & np.isfinite(values)).astype(np.float64)
            for r in self.radii:
                name = feature_name(c, r)
                if name in self.col:
                    self.den[(c, r)] = ndimage.convolve(mask, disk_kernel(r, pixel_m), mode="constant").ravel()
                    self.terms.append((c, self.col[name], r))

        # deslocamentos do maior disco e, para cada raio, quais deles caem dentro
        n = int(max(self.radii) // pixel_m)
        dy, dx = np.mgrid[-n : n + 1, -n : n + 1]
        d2 = (dy**2 + dx**2).ravel()
        keep = d2 <= (max(self.radii) / pixel_m) ** 2 + 1e-9
        self.dy, self.dx = dy.ravel()[keep], dx.ravel()[keep]
        self.in_r = {r: d2[keep] <= (r / pixel_m) ** 2 + 1e-9 for r in self.radii}
        self.center = int(np.flatnonzero((self.dy == 0) & (self.dx == 0))[0])

        self.X0 = X
        self.base_pred = np.full(len(X), np.nan)
        self.base_pred[self.eval_mask] = model.predict(X[self.eval_mask])
        self.reset()

    # ------------------------------------------------------------------ estado
    def reset(self) -> None:
        self.X = self.X0.copy()
        self.pred = self.base_pred.copy()
        self.trees = np.zeros(len(self.X0), dtype=np.int32)

    def pixel_values(self, j, n_trees):
        """(tree_frac, NDVI, built_frac) da célula j com n_trees árvores no total (vetorizado)."""
        F = np.asarray(n_trees, dtype=np.float64) * self.params.canopy_area_m2 / self.cell_area
        d_built = np.maximum(F - self.other0[j], 0.0)
        tree = self.tree0[j] + F
        built = self.built0[j] - d_built
        ndvi = self.ndvi0[j] + F * np.maximum(self.ndvi_tree - self.ndvi0[j], 0.0)
        return tree, ndvi, built

    # --------------------------------------------------------- avaliação local
    def _proposal(self, j: int, add: int):
        r0, c0 = divmod(int(j), self.W)
        rr, cc = r0 + self.dy, c0 + self.dx
        ok = (rr >= 0) & (rr < self.H) & (cc >= 0) & (cc < self.W)
        offs = np.flatnonzero(ok)
        idx = rr[ok] * self.W + cc[ok]
        ev = self.eval_mask[idx]
        offs, idx = offs[ev], idx[ev]

        Xn = self.X[idx].copy()
        n = self.trees[j]
        t0, v0, _ = self.pixel_values(j, n)
        t1, v1, b1 = self.pixel_values(j, n + add)
        me = int(np.flatnonzero(offs == self.center)[0])
        Xn[me, self.col["tree_frac"]] = t1
        Xn[me, self.col["NDVI"]] = v1
        Xn[me, self.col["built_frac"]] = b1
        delta = {"tree_frac": t1 - t0, "NDVI": v1 - v0}
        for c, q, r in self.terms:
            inside = self.in_r[r][offs]
            Xn[inside, q] += delta[c] / self.den[(c, r)][idx[inside]]
        return idx, Xn

    def local_gains(self, cells, adds):
        """Para cada (célula, nº de árvores): (ganho em °C·célula, idx, X novo, previsão nova)."""
        props = [self._proposal(j, a) for j, a in zip(cells, adds)]
        if not props:
            return []
        pred = self.model.predict(np.vstack([X for _, X in props]))
        out, k = [], 0
        for idx, Xn in props:
            p = pred[k : k + len(idx)]
            k += len(idx)
            out.append((float((self.pred[idx] - p).sum()), idx, Xn, p))
        return out

    def commit(self, j: int, add: int, idx, Xn, pnew) -> None:
        self.X[idx] = Xn
        self.pred[idx] = pnew
        self.trees[j] += add

    # ---------------------------------------------------- pré-seleção linear
    def screening_scores(self, step: int, n_fit: int = 50_000, seed: int = 0):
        """Ganho aproximado da 1ª ação de cada candidata via substituto linear do modelo."""
        ev = np.flatnonzero(self.eval_mask)
        sub = np.random.default_rng(seed).choice(ev, min(n_fit, len(ev)), replace=False)
        A = np.column_stack([np.ones(len(sub)), self.X0[sub]])
        coef = np.linalg.lstsq(A, self.base_pred[sub], rcond=None)[0][1:]

        cand = np.flatnonzero(self.capacity > 0)
        add = np.minimum(step, self.capacity[cand])
        t0, v0, b0 = self.pixel_values(cand, 0)
        t1, v1, b1 = self.pixel_values(cand, add)
        delta = {"tree_frac": t1 - t0, "NDVI": v1 - v0}
        change = (t1 - t0) * coef[self.col["tree_frac"]] + (v1 - v0) * coef[self.col["NDVI"]] \
            + (b1 - b0) * coef[self.col["built_frac"]]
        for c, q, r in self.terms:
            inv = np.where(self.eval_mask & (self.den[(c, r)] > 0), 1.0 / np.maximum(self.den[(c, r)], 1), 0.0)
            spread = ndimage.convolve(inv.reshape(self.shape), disk_kernel(r, self.pixel_m), mode="constant").ravel()
            change += delta[c] * coef[q] * spread[cand]
        return cand, -change  # ganho = resfriamento

    # ----------------------------------------------------- avaliação completa
    def apply(self, trees: np.ndarray) -> pd.DataFrame:
        """Grade com a intervenção aplicada e as features de vizinhança recalculadas por convolução."""
        trees = np.asarray(trees).ravel()
        if (trees > self.capacity).any():
            raise ValueError("alocação excede a área plantável de alguma célula")
        planted = np.flatnonzero(trees)
        tree, ndvi, built = self.tree0.copy(), self.ndvi0.copy(), self.built0.copy()
        tree[planted], ndvi[planted], built[planted] = self.pixel_values(planted, trees[planted])
        df = self.base_df.copy()
        df["tree_frac"], df["NDVI"], df["built_frac"] = tree, ndvi, built
        return add_neighborhood_features(df, self.shape, pixel_m=self.pixel_m, radii_m=self.radii)

    def evaluate(self, trees: np.ndarray) -> dict:
        trees = np.asarray(trees).ravel()
        df = self.apply(trees)
        X = df[self.features].to_numpy(np.float64)
        cooling = np.zeros(len(X))
        cooling[self.eval_mask] = self.base_pred[self.eval_mask] - self.model.predict(X[self.eval_mask])
        ev = cooling[self.eval_mask]
        return {
            "arvores": int(trees.sum()),
            "celulas_plantadas": int((trees > 0).sum()),
            "soma_dT": float(ev.sum()),
            "dT_medio_mC": 1000 * float(ev.mean()),
            "dT_max": float(ev.max()),
            "aquecimento_max": float(max(0.0, -ev.min())),
            "celulas_dT_ge_0.1": int((ev >= 0.1).sum()),
            "mapa_dT": cooling,
        }


# ---------------------------------------------------------------------------
# Estratégias
# ---------------------------------------------------------------------------
def lazy_greedy(problem: AllocationProblem, budget: int, step: int = 5, pool_size: int = 6000,
                explore: int = 6000, batch: int = 64, seed: int = 0, verbose: bool = True) -> AllocationResult:
    """
    A prioridade é o ganho POR ÁRVORE da ação (o orçamento é em árvores, e ações têm tamanhos
    diferentes: células com pouco espaço só recebem 1–2). Comparar o ganho total da ação
    favoreceria células grandes, onde o efeito satura.

    Pool = `pool_size` melhores pela pré-seleção linear + `explore` candidatas sorteadas,
    para que regiões onde o substituto linear erra também tenham chance.
    """
    t_start = time.perf_counter()
    problem.reset()
    cand, score = problem.screening_scores(step)
    first_add = np.minimum(step, problem.capacity[cand])
    order = cand[np.argsort(-score / first_add)]
    # a parte pré-selecionada cobre pelo menos 3× o orçamento em capacidade
    need = int(np.searchsorted(np.cumsum(problem.capacity[order]), 3 * budget)) + 1
    top = order[: max(pool_size, need)]
    rest = order[len(top):]
    sampled = np.random.default_rng(seed).choice(rest, min(explore, len(rest)), replace=False)
    pool = np.concatenate([top, sampled])
    if verbose:
        print(f"  pré-seleção: {len(cand):,} candidatas → pool de {len(pool):,} "
              f"({len(top):,} pelo substituto linear + {len(sampled):,} sorteadas; "
              f"capacidade {int(problem.capacity[pool].sum()):,} árvores)")

    heap, n_eval, last = [], 0, {}
    for k in range(0, len(pool), batch):
        chunk = pool[k : k + batch]
        adds = np.minimum(step, problem.capacity[chunk])
        for j, a, res in zip(chunk, adds, problem.local_gains(chunk, adds)):
            heap.append((-res[0] / a, int(j), 0))
            last[int(j)] = (int(a), *res)
        n_eval += len(chunk)
    heapq.heapify(heap)
    if verbose:
        print(f"  ganhos iniciais exatos: {n_eval:,} avaliações em {time.perf_counter() - t_start:.0f} s")

    placed, n_accept, total = 0, 0, 0.0
    while placed < budget and heap:
        neg, j, stamp = heapq.heappop(heap)
        free = int(problem.capacity[j] - problem.trees[j])
        if free <= 0:
            continue
        add = min(step, free, budget - placed)
        if stamp == n_accept and last.get(j, (None,))[0] == add:
            if -neg <= 0:  # nenhuma ação restante resfria segundo o modelo
                heapq.heappush(heap, (neg, j, stamp))
                break
            _, gain, idx, Xn, p = last.pop(j)
            problem.commit(j, add, idx, Xn, p)
            placed += add
            total += gain
            n_accept += 1
            if problem.trees[j] < problem.capacity[j]:
                heapq.heappush(heap, (neg, j, -1))  # ganho antigo como estimativa; será reavaliado
            if verbose and n_accept % 200 == 0:
                print(f"  {placed:,}/{budget:,} árvores, Σ ΔT={total:.1f} °C·célula, {n_eval:,} avaliações")
        else:
            gain, idx, Xn, p = problem.local_gains([j], [add])[0]
            n_eval += 1
            last[j] = (add, gain, idx, Xn, p)
            heapq.heappush(heap, (-gain / add, j, n_accept))

    # se o ganho exato zerou antes do fim do orçamento, completa pela ordem da pré-seleção
    fallback = 0
    for j in order:
        if placed >= budget:
            break
        free = int(problem.capacity[j] - problem.trees[j])
        if free > 0:
            add = min(free, budget - placed)
            problem.trees[j] += add
            placed += add
            fallback += add

    return AllocationResult("Lazy Greedy", problem.trees.copy(), time.perf_counter() - t_start,
                            predicted_gain=total, n_evaluations=n_eval, pool_size=len(pool),
                            fallback_trees=fallback)


def _fill_in_order(problem: AllocationProblem, cells: np.ndarray, budget: int) -> np.ndarray:
    trees = np.zeros(len(problem.capacity), dtype=np.int32)
    caps = problem.capacity[cells]
    cum = np.cumsum(caps)
    k = int(np.searchsorted(cum, budget))  # células totalmente preenchidas antes da última
    trees[cells[:k]] = caps[:k]
    if k < len(cells):
        trees[cells[k]] = budget - (cum[k - 1] if k else 0)
    return trees


def random_allocation(problem: AllocationProblem, budget: int, seed: int = 0) -> AllocationResult:
    t0 = time.perf_counter()
    cand = np.flatnonzero(problem.capacity > 0)
    cells = np.random.default_rng(seed).permutation(cand)
    return AllocationResult(f"Aleatória (seed {seed})", _fill_in_order(problem, cells, budget),
                            time.perf_counter() - t0)


def hotspot_allocation(problem: AllocationProblem, budget: int) -> AllocationResult:
    """Linha de base ingênua: preenche as células mais quentes (LST observada) até a capacidade."""
    t0 = time.perf_counter()
    cand = np.flatnonzero(problem.capacity > 0)
    lst = problem.base_df["LST"].to_numpy()[cand]
    cells = cand[np.argsort(-np.nan_to_num(lst, nan=-np.inf))]
    return AllocationResult("Mais quente primeiro", _fill_in_order(problem, cells, budget),
                            time.perf_counter() - t0)
