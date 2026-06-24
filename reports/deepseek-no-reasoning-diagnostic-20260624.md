# Diagnostic DeepSeek no-reasoning - 2026-06-24

## Objectif

Tester si DeepSeek peut etre accelere en reduisant le raisonnement cote `llama-server`, sans changer de modele.

Modele teste: `DeepSeek-R1-Distill-Qwen-32B-Q8_0.gguf`

RAG teste: `client-validation-rag-20260530-105619`

Batterie: `reports/deepseek-client-questions-20260624.json`

Seuil: `max_tokens=500`

## Preflight serveur

- `--reasoning off` ne suffit pas avec ce modele: le serveur retourne encore du `reasoning_content` et peut finir en `length` sans reponse visible.
- `--reasoning-budget 0` fonctionne: `reasoning_len=0` en appel direct et dans les appels RAG.
- `--reasoning-budget 0` ajoute un artefact visible `</think>` au debut des reponses. Il faut le nettoyer cote application avant adoption.

## Resultats RAG

| Variante | Verdict | Qualite | p30 LLM | p50 LLM | p95 LLM | p50 reasoning_len | p95 reasoning_len | p50 content_len | p95 content_len | p50 completion_tokens |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `baseline concise 500` | Non OK | 12 OK, 4 limite | 12881.3 ms | 14491.8 ms | 28969.0 ms | 1025.5 | 1591.2 | 101.0 | 651.5 | 251.0 |
| `baseline direct_answer 500` | Non OK | 14 OK, 2 limite | 12655.9 ms | 15397.4 ms | 25875.8 ms | 1059.0 | 1591.2 | 60.5 | 270.0 | 263.5 |
| `budget0 default 500` | Non OK | 10 OK, 6 limite | 1123.9 ms | 1590.4 ms | 18627.3 ms | 0.0 | 0.0 | 74.5 | 1213.0 | 25.0 |
| `budget0 direct_answer 500` | OK | 13 OK, 3 limite | 1489.5 ms | 1700.9 ms | 4789.5 ms | 0.0 | 0.0 | 79.5 | 267.0 | 27.0 |

Rapports sources:

- `reports/deepseek-reasoning-budget-0-500/rag-latency-20260624T165104z0000.md`
- `reports/deepseek-direct-answer-budget-0-500/rag-latency-20260624T165318z0000.md`
- Baselines: `reports/deepseek-concise-500-reasoning-corr/rag-latency-20260624T145124z0000.md` et `reports/deepseek-direct-answer-500-reasoning-corr/rag-latency-20260624T144637z0000.md`

## Diagnostic

- Le levier serveur est confirme: `--reasoning-budget 0` supprime le raisonnement mesure (`reasoning_len=0`).
- Le gain median est tres fort: p50 LLM autour de `1.7s` avec `budget0 direct_answer`, contre `14.5s` a `15.4s` sur les baselines.
- Le p95 devient acceptable avec `budget0 direct_answer`: `4.8s`, contre `25.9s` a `29.0s` sur les baselines.
- `budget0 default` accelere la mediane mais degrade la qualite et garde un long tail sur `summary`; il ne doit pas etre retenu seul.
- `budget0 direct_answer` est le meilleur compromis teste: verdict OK, qualite proche de la baseline `concise`, sorties courtes, plus de raisonnement.

## Decision

Option recommandee pour la suite:

1. Garder DeepSeek.
2. Lancer DeepSeek avec `--reasoning-budget 0`.
3. Utiliser le prompt applicatif `direct_answer`.
4. Ajouter un nettoyage applicatif de prefixe `</think>` avant exposition utilisateur.
5. Refaire une validation qualite apres nettoyage sur au moins 3 runs pour stabiliser p95 et qualite.

Environnement restaure apres test:

- Config runtime ChatFleet remise sur `http://host.docker.internal:2250/v1`.
- API remise sur le prompt `default`.
- Serveur temporaire `2253` arrete.
