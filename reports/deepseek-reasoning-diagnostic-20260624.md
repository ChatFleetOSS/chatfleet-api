# Diagnostic raisonnement DeepSeek - 2026-06-24

## Objectif

Verifier si la latence DeepSeek vient du temps de raisonnement et si une variante de prompt peut le reduire.

RAG teste: `client-validation-rag-20260530-105619`

Batterie: `reports/deepseek-client-questions-20260624.json`

Modele: `DeepSeek-R1-Distill-Qwen-32B-Q8_0.gguf`

Seuil teste: `max_tokens=500`

## Changements de mesure

Les rapports incluent maintenant:

- `p30` en plus de min/p50/p95/max.
- `reasoning_len` depuis l'evenement provider `llm.response`.
- `content_len`, `finish_reason`, `completion_tokens`.

Les evenements `llm.response` sont rattaches au `corr_id` de la requete RAG.

## Resultats

| Variante | HTTP OK | Qualite | p30 LLM | p50 LLM | p95 LLM | p50 reasoning_len | p95 reasoning_len | p50 content_len | p95 content_len | p50 completion_tokens |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `concise 500` | 16/16 | 12 OK, 4 limite | 12766.3 ms | 14491.8 ms | 28912.1 ms | 1025.5 | 1544.0 | 101.0 | 631.0 | 251.0 |
| `direct_answer 500` | 16/16 | 14 OK, 2 limite | 12018.0 ms | 15397.4 ms | 25361.3 ms | 1059.0 | 1507.0 | 60.5 | 265.0 | 263.5 |

Rapports sources:

- `reports/deepseek-concise-500-reasoning-corr/rag-latency-20260624T145124z0000.md`
- `reports/deepseek-direct-answer-500-reasoning-corr/rag-latency-20260624T144637z0000.md`

## Diagnostic

- Oui, la lenteur vient largement du raisonnement DeepSeek.
- Le modele produit typiquement autour de 1000 caracteres de raisonnement median avant la reponse finale.
- `direct_answer` reduit fortement la taille visible de la reponse (`content_len` p50 60.5 vs 101.0), mais ne reduit pas vraiment le raisonnement median (`1059.0` vs `1025.5`).
- `direct_answer` ameliore le p95 LLM (`25361.3 ms` vs `28912.1 ms`) et la qualite automatique (`14 OK` vs `12 OK`), mais sa p50 LLM est legerement moins bonne que `concise`.
- Le prompt influence la sortie finale et certains cas lents, mais ne desactive pas le mode thinking de DeepSeek.

## Levier serveur

Le `llama-server` local expose des options de reasoning:

- `--reasoning [on|off|auto]`
- `--reasoning-budget N`
- `--reasoning-format FORMAT`

Le serveur DeepSeek actuel est lance avec `--jinja`, mais sans `--reasoning off` ni `--reasoning-budget`.

Conclusion: pour reduire fortement le raisonnement, le prochain test doit etre cote serveur, idealement sur un port separe:

- DeepSeek avec `--reasoning off`, ou
- DeepSeek avec `--reasoning-budget 0`, ou
- DeepSeek avec un budget faible, par exemple `--reasoning-budget 128`.

## Recommendation

- Garder `concise` comme variante applicative prudente.
- Tester `direct_answer` sur plus de runs si la priorite est la qualite courte.
- Lancer ensuite un serveur DeepSeek de test avec reasoning reduit, puis refaire la meme batterie a `max_tokens=500`.
- Ne pas compter uniquement sur le prompt pour supprimer le raisonnement: le modele continue de raisonner meme quand on demande une reponse directe.
