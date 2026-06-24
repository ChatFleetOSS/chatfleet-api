# Comparaison DeepSeek prompt as-is vs concise - 2026-06-24

## Validation du RAG client

RAG teste: `client-validation-rag-20260530-105619`.

Les chunks du RAG client contiennent deux copies du meme document `client-validation-source.txt` avec les faits suivants:

- `CF-UPGRADE-001`: le canal stable du curl installer doit resoudre API `v0.1.21` et Web `v0.1.20`.
- `CF-MONGO-002`: les secrets Mongo existants et les URI Mongo non vides doivent etre preserves pendant les mises a jour.
- `CF-VOLUMES-003`: les volumes Docker pour Mongo data, les index RAG et les documents uploades ne doivent pas etre supprimes pendant une mise a jour.
- `CF-RAG-004`: les index RAG sont stockes sous `/var/lib/chatfleet/faiss` dans le conteneur API.
- `CF-ANSWER-005`: les reponses doivent citer des extraits du document uploade.
- `CF-FALLBACK-006`: si le contexte ne contient pas la reponse, ChatFleet ne doit pas inventer.

La batterie validee est stockee dans `reports/deepseek-client-questions-20260624.json`.

## Questions de test

| ID | Intention |
|---|---|
| `version_stable` | Verifier les versions API/Web attendues. |
| `mongo_preservation` | Verifier la preservation Mongo. |
| `volumes_preservation` | Verifier les volumes/donnees a conserver. |
| `rag_index_location` | Verifier le chemin des index RAG. |
| `answer_policy` | Verifier les regles de reponse/citation/fallback. |
| `summary` | Verifier une synthese multi-faits. |
| `out_of_context_weather` | Verifier le fallback hors contexte. |
| `out_of_context_japan` | Verifier un second fallback hors contexte. |

Le scoring automatique est volontairement strict et base sur des termes attendus dans l'aperçu de reponse. Certains `limite` sont dus a des traductions correctes, par exemple `uploaded documents` rendu en `documents telecharges`.

## Resultats

| Variante | max_tokens | HTTP OK | Qualite | p50 total client | max total client | p50 LLM | max LLM | p50 retrieval | p50 first SSE |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| as-is | 500 | 14/16 | 11 OK, 3 limite, 1 KO, 1 non score | 20254.0 ms | 29721.4 ms | 19743.8 ms | 29162.5 ms | 376.4 ms | 431.8 ms |
| as-is | 800 | 16/16 | 12 OK, 4 limite | 20491.0 ms | 42811.8 ms | 19970.0 ms | 42245.9 ms | 381.7 ms | 396.6 ms |
| concise | 500 | 16/16 | 12 OK, 4 limite | 13652.0 ms | 29572.7 ms | 13095.4 ms | 29172.8 ms | 375.7 ms | 370.8 ms |
| concise | 800 | 16/16 | 12 OK, 4 limite | 15201.0 ms | 32362.9 ms | 14535.1 ms | 31842.7 ms | 385.1 ms | 379.3 ms |

Rapports sources:

- `reports/deepseek-as-is-500/rag-latency-20260624T125803z0000.md`
- `reports/deepseek-as-is-800/rag-latency-20260624T130350z0000.md`
- `reports/deepseek-concise-500/rag-latency-20260624T131035z0000.md`
- `reports/deepseek-concise-800/rag-latency-20260624T131507z0000.md`

## Diagnostic

- Le RAG/retrieval est stable: p50 autour de 376-385 ms selon les runs.
- Le premier evenement SSE reste rapide: p50 autour de 371-432 ms.
- La latence longue reste attribuee au LLM DeepSeek.
- Le prompt as-is a besoin de `800` tokens pour etre stable sur cette batterie.
- Le prompt `concise` rend `500` stable: 16/16 HTTP OK, qualite comparable a as-is `800`.
- Le prompt `concise` ameliore aussi le seuil `800`: p50 LLM 14535.1 ms contre 19970.0 ms en as-is.
- Le seuil `concise 500` est le meilleur compromis actuel: qualite comparable au seuil haut as-is avec p50 LLM reduite d'environ 34%.

## Recommendation

- Retenir la variante `concise` comme candidate pour DeepSeek.
- Retester `concise 500` avec `runs=2` ou `runs=3` avant de la promouvoir en comportement produit.
- Ne pas choisir `as-is 500`: il a encore des erreurs `LLM_REASONING_WITHOUT_ANSWER`/HTTP 503.
- Ne pas choisir `800` comme defaut si l'objectif est de reduire la latence: il reste stable mais trop couteux.
- Ajouter ensuite un timeout serveur strict LLM pour eviter les blocages longs.
