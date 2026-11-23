# Auditoria cruzada de modelos — CanopyAI

Gerado por `scripts/audit_cross_model.py` em 2026-11-23T20:49:07+00:00 (UTC).

## Pergunta

O Lazy Greedy otimizado com o M2 (HistGradientBoosting monotônico) rendeu 2,34× o resfriamento da alocação aleatória. Como o otimizador escolhe exatamente onde o M2 responde mais, esse número pode estar inflado por degraus das árvores de decisão. A auditoria avalia as mesmas alocações em modelos que o otimizador não viu.

## Configuração

- Área: centro expandido de São Paulo, bbox (-46.75, -23.65, -46.55, -23.5), grade de 30 m (`grid_3bd5a66c269a64a7.parquet`)
- Orçamento: 5,000 árvores de 50 m² de copa; mesma intervenção em todas as avaliações
- Treino: 335,561 células (M1 e M2); M2-alt em 165,657 células (metade dos blocos de 1,5 km)
- Avaliadores:
  - **M2**: o modelo usado pelo otimizador (referência, não é teste independente)
  - **M1**: Ridge linear, sem degraus
  - **M2-alt**: HGB monotônico com outros hiperparâmetros (63 folhas, 500 iterações, lr 0,05) e outra metade dos dados
  - **M1-sinal** (*diagnóstico definido depois da 1ª rodada*): linear como o M1, mas com os sinais das restrições do M2 (vegetação ≤ 0, construído ≥ 0). É suave como o M1 e fisicamente coerente como o M2. Não entra no veredito, que segue o critério fixado antes de ver os resultados.
- R² no treino dos lineares: M1 0.435, M1-sinal 0.409
- Aleatória: média ± desvio-padrão de 5 sorteios

## Resfriamento total Σ ΔT (°C·célula) e razão sobre a aleatória

| Alocação | avaliada no M2 | avaliada no M1 | avaliada no M2-alt | avaliada no M1-sinal |
|---|---|---|---|---|
| Greedy-M2 | 16,694 (2.34×) | 896 (1.20×) | 14,334 (2.27×) | 1,165 (1.19×) |
| Greedy-M1 | 2,718 (0.38×) | 1,184 (1.58×) | 2,897 (0.46×) | 1,183 (1.21×) |
| Mais quente primeiro | 6,060 (0.85×) | 841 (1.12×) | 6,255 (0.99×) | 1,072 (1.10×) |
| Aleatória | 7,125 ± 190 (1.00×) | 748 ± 8 (1.00×) | 6,313 ± 107 (1.00×) | 976 ± 10 (1.00×) |

## Células que esquentam (nº / pior caso)

| Alocação | M2 | M1 | M2-alt | M1-sinal |
|---|---|---|---|---|
| Greedy-M2 | 0 / 0.000 °C | 5,925 / 0.893 °C | 0 / 0.000 °C | 0 / 0.000 °C |
| Greedy-M1 | 0 / 0.000 °C | 3,958 / 1.250 °C | 0 / 0.000 °C | 0 / 0.000 °C |
| Mais quente primeiro | 0 / 0.000 °C | 3,706 / 0.724 °C | 0 / 0.000 °C | 0 / 0.000 °C |
| Aleatória | 0 / 0.000 °C | 9,502 / 1.336 °C | 0 / 0.000 °C | 0 / 0.000 °C |

## Concordância espacial

- Spearman entre os mapas de ΔT do Greedy-M2 avaliados no M2 e no M1: 0.47
- Spearman entre M2 e M2-alt: 0.66
- Jaccard das células plantadas, Greedy-M2 × Greedy-M1: 0.00

## Resultado

- Vantagem do Greedy-M2 sobre a aleatória: 2.34× no M2, **1.20× no M1** e **2.27× no M2-alt**.
- No M1, o Greedy-M2 retém 15% da vantagem que tinha no M2 e alcança 76% do resfriamento que o próprio M1 obtém quando é ele quem otimiza.
- Veredito automático (limiar de 1.5× a aleatória no M1 e no M2-alt): a vantagem é **não robusta**.
- Diagnóstico pós-hoc: no M1-sinal (linear, suave, com sinais físicos) o Greedy-M2 rende **1.19×** a aleatória.

## Como ler

- **Degraus do M2.** O M2-alt tem degraus em outros lugares; se o otimizador só explorasse degraus do M2, a vantagem sumiria nele. O M1-sinal não tem degrau nenhum. A razão nesses dois avaliadores é a evidência mais direta sobre essa hipótese.
- **M1 sem restrição.** Os coeficientes do Ridge têm sinais trocados por colinearidade (NDVI do pixel "aquece", construído "resfria"), por isso ele prevê aquecimento em milhares de células para qualquer plantio, até o aleatório. É um avaliador fraco para uma intervenção física; ainda assim, o veredito acima o inclui porque esse foi o critério definido antes.
- **Magnitude.** Os avaliadores concordam mais na *ordem* das estratégias do que no *tamanho* do resfriamento. Compare as somas de ΔT da linha "Aleatória" entre colunas: o efeito absoluto de plantar depende fortemente do modelo, e deve ser comunicado como incerto.

## Conclusão

- O Greedy-M2 supera a aleatória e a "mais quente primeiro" em todos os 4 avaliadores. O **ranking** das estratégias é robusto.
- O **tamanho** da vantagem depende da família do modelo: 2.27× no M2-alt (árvores de decisão independentes) contra 1.20× no M1, 1.19× no M1-sinal (lineares). Como o M2-alt tem degraus em outros lugares e ainda assim concorda com o M2, a vantagem não é artefato dos degraus específicos do M2. A explicação provável são efeitos não lineares que um modelo linear não representa por construção (hipótese ainda não testada: plantar onde há pouca copa resfriaria mais).
- O **efeito absoluto** de plantar varia 9.5× entre avaliadores (Σ ΔT da alocação aleatória). No Passo 3, HGB e Ridge tiveram R² espacial quase igual (0,436 × 0,421): prever bem a LST atual não basta para distinguir modelos que discordam sobre o efeito de uma intervenção.
- **Como comunicar:** "o otimizador supera as linhas de base em todos os modelos testados; o ganho varia de 1.2× a 2.3× a alocação aleatória conforme o modelo", em vez de um número único. Validar com dados de campo (p.ex. LST antes e depois de plantios reais) é o que separaria as duas famílias de modelo.

## Limitações

- Os três avaliadores aprendem dos mesmos dados (uma cena composta, uma cidade). A auditoria descarta artefatos de um modelo específico, não erros comuns a todos — por exemplo, confusão entre vegetação e outros fatores urbanos não medidos.
- O M1 tem coeficientes contraintuitivos por colinearidade (NDVI do pixel e fração construída com sinal trocado), então pode marcar como "aquecimento" partes da intervenção. Células que esquentam no M1 refletem isso, não um efeito físico.
- ΔT é LST prevista (temperatura de superfície), não temperatura do ar, e assume copa madura.
