# Diagnostic latence RAG local - 2026-06-24

## Contexte

- Installation testee: ChatFleet local via Docker Desktop.
- API: `http://localhost:8080/api`
- Image API: `ghcr.io/chatfleetoss/chatfleet-api:v0.1.22`
- Commit API expose par `/api/health`: `76b7a5a68607617e88e6cc9a75a54986edebfca3`
- RAG client existant: `client-validation-rag-20260530-105619`
- RAG probe cree: `latency-probe-1782296803`
- Rapport brut: `reports/rag-latency-20260624T102643z0000.md`
- Resultats bruts: `reports/rag-latency-20260624T102643z0000.json`

## Campagne 1 - config locale initiale

Configuration observee avant test:

- `provider`: `openai`
- `chat_model`: `gpt-4o-mini`
- `embed_provider`: `openai`
- `embed_model`: `BAAI/bge-m3`

Resultat:

- 48 requetes executees sur un vrai RAG: 24 `/chat`, 24 `/chat/stream`.
- Toutes les requetes ont echoue cote generation avec `LLM_AUTH_ERROR` ou HTTP `503`.
- Aucun cas a 30s n'a ete reproduit dans cette configuration.
- Latence HTTP mesuree avant erreur LLM:
  - Probe `/chat`: p50 `562.5 ms`, max `625.8 ms`.
  - Probe `/chat/stream`: p50 `563.5 ms`, max `579.5 ms`.
  - Client `/chat`: p50 `558.7 ms`, max `562.9 ms`.
  - Client `/chat/stream`: p50 `565.8 ms`, max `719.4 ms`.

Interpretation:

- Le retrieval/index est suffisamment disponible pour atteindre la phase LLM en environ 0.55-0.72s.
- La configuration locale de production n'est pas valide pour generer une reponse: le provider chat est OpenAI mais l'appel retourne `LLM_AUTH_ERROR`.
- Cette campagne ne permet pas de mesurer une latence LLM normale, car la generation ne demarre pas correctement.

## Campagne 2 - bascule vers Gemma local

Action realisee:

- `provider`: `vllm`
- `base_url`: `http://host.docker.internal:2251/v1`
- `chat_model`: `gemma-4-26B-A4B-it-UD-Q8_K_XL.gguf`
- `embed_provider`: `local`
- `embed_model`: `BAAI/bge-m3`

Resultat:

- Les serveurs llama.cpp locaux etaient actifs sur les ports `2250`, `2251`, `2252`.
- Apres passage en embeddings locaux, l'API a cesse de repondre pendant l'indexation du RAG probe.
- `/api/health` time outait via Caddy et directement depuis le conteneur API.
- Le process `uvicorn` restait actif et consommait du CPU, avec une memoire RSS montee au-dessus de 1.5 GB puis redescendue.
- Le blocage a dure plus de 3 minutes avant interruption du test.
- L'installation a ete restauree en redemarrant uniquement `chatfleet-api-1`, puis en remettant la configuration initiale.

Interpretation:

- Sur l'image `v0.1.22`, le chemin `embed_provider=local` peut bloquer le worker API pendant chargement/indexation.
- Ce comportement est compatible avec les symptomes de latence longue ou de requetes pendantes, meme si le cas exact 30s n'a pas ete reproduit en generation reussie.
- Le blocage se situe avant ou pendant l'etape embedding/indexation, pas dans Caddy ni Mongo.

## Limite des mesures serveur

L'image locale `v0.1.22` ne contient pas encore les logs `chatfleet.metrics {...}` ajoutes sur la branche `codex/rag-real-latency-report`.

Consequence:

- Les mesures HTTP sont disponibles.
- L'attribution fine `embedding_ms`, `retrieval_ms`, `llm_ms`, `total_ms` n'est pas disponible sur cette image.
- Pour obtenir la chaine complete, il faut lancer la meme campagne avec l'API instrumentee de cette branche.

## Diagnostic

Etat actuel: pas OK pour un test RAG fiable.

Raisons:

- La configuration initiale locale echoue systematiquement en `LLM_AUTH_ERROR`.
- La configuration Gemma locale avec embeddings locaux bloque le worker API pendant l'indexation.
- Les logs de production locale `v0.1.22` ne permettent pas d'attribuer finement les temps serveur.

Hypothese la plus probable pour les latences de 30s:

- Soit les requetes attendent un provider LLM lent ou mal configure.
- Soit le chemin embedding local bloque le worker pendant chargement/indexation.
- Le retrieval pur ne semble pas etre le suspect principal dans la campagne initiale, car les erreurs LLM reviennent en moins de 1s.

## Suite recommandee

1. Demarrer l'API instrumentee de la branche `codex/rag-real-latency-report` sur la meme stack locale.
2. Garder Mongo et les volumes RAG existants.
3. Configurer un couple coherent:
   - chat provider local Gemma;
   - embeddings qui ne bloquent pas le worker, ou prechargement explicite hors requete.
4. Relancer `scripts/rag_latency_report.py --runs 3`.
5. Comparer `embedding_ms`, `retrieval_ms`, `llm_ms`, `first_sse_event_ms` et `total_ms`.

