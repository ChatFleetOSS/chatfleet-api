# Diagnostic latence RAG local - suivi 2026-06-24

## Contexte

- Branche testee: `codex/rag-real-latency-report`
- Image temporaire: `chatfleet-api:rag-latency-instrumented`
- Build temporaire expose: `rag-latency-instrumented / 4797d11-embedfix`
- Stack restauree apres test: `ghcr.io/chatfleetoss/chatfleet-api:v0.1.22`
- API locale: `http://localhost:8080/api`
- RAG probe: `latency-probe-1782296803`
- RAG client: `client-validation-rag-20260530-105619`

## Corrections appliquees avant mesure

- L'API ne force plus `embed_provider=local` quand `provider=vllm`.
- Le provider embeddings `openai` utilise le client embeddings OpenAI au lieu du `base_url` du chat local.
- `scripts/rag_latency_report.py` accepte `--max-questions-per-suite` pour produire une campagne courte quand le LLM est lent.

## Campagne instrumentee - config initiale OpenAI

- Provider chat: `openai`
- Modele chat: `gpt-4o-mini`
- Provider embeddings: `openai`
- Modele embeddings: `BAAI/bge-m3`
- Rapport: `reports/rag-latency-20260624T113131z0000.md`
- Resultats: `reports/rag-latency-20260624T113131z0000.json`

Resultats:

- 24 requetes executees.
- Toutes les generations echouent vite en auth/config LLM.
- Retrieval p50: `259.0 ms`
- Retrieval max: `1019.6 ms`
- Embedding domine le retrieval: p50 autour de `255 ms`, avec un outlier a `1015.4 ms`.
- Semantic, lexical et fusion sont negligeables: generalement `<2 ms`.

Interpretation:

- Sur cette config, le RAG atteint la phase LLM rapidement.
- La lenteur n'est pas FAISS/BM25.
- Le temps retrieval mesure est essentiellement du temps d'embedding.
- La config LLM n'est pas valide pour une generation reussie.

## Campagne instrumentee - Gemma local

- Provider chat: `vllm`
- Endpoint: `http://host.docker.internal:2251/v1`
- Modele chat: `gemma-4-26B-A4B-it-UD-Q8_K_XL.gguf`
- Provider embeddings: `openai`
- Limite: `--max-questions-per-suite 1`
- Rapport: `reports/rag-latency-20260624T114403z0000.md`
- Resultats: `reports/rag-latency-20260624T114403z0000.json`

Resultats:

| Suite | Mode | HTTP ms | Retrieval ms | Embedding ms | LLM ms | Resultat |
|---|---:|---:|---:|---:|---:|---|
| probe | chat | `19027.3` | `685.4` | `683.8` | `18334.5` | OK |
| probe | stream | `18029.1` | `360.3` | `359.0` | `17534.9` | OK |
| client | chat | `180002.4` | `402.8` | `400.7` | `181368.3` | timeout/erreur provider |
| client | stream | `180727.3` | `713.7` | `712.5` | non termine | timeout client |

Interpretation:

- Le symptome de latence longue est reproduit.
- La cause principale est le LLM/provider local, pas le retrieval.
- Le streaming emet `ready` rapidement apres retrieval:
  - probe stream: first SSE `368.2 ms`
  - client stream: first SSE `725.1 ms`
- Le client RAG declenche une generation Gemma qui depasse 180s.

## Diagnostic final

Etat: non OK pour production avec cette config locale Gemma.

Cause prioritaire:

- LLM/provider local Gemma trop lent ou bloquant sur certains prompts RAG.

Causes secondaires:

- Embeddings distants/configures via `openai` ajoutent `~360-715 ms` par requete.
- La config actuelle utilise une cle non valide pour l'API OpenAI embeddings pendant les tests Gemma; l'API retombe en fallback deterministe apres erreur auth.

Ce qui n'est pas la cause principale:

- FAISS semantic search: generalement `<2 ms`.
- BM25 lexical search: generalement `<2 ms`.
- Fusion RRF: generalement `<0.1 ms`.
- Caddy/proxy: `/health` reste responsive pendant les lenteurs.

## Suite recommandee

1. Ne pas deploiement Gemma `gemma-4-26B-A4B-it-UD-Q8_K_XL.gguf` tel quel pour les RAG client.
2. Ajouter une limite de sortie plus basse pour les tests de latence RAG, par exemple `max_tokens` entre `256` et `500`.
3. Ajouter un timeout serveur explicite autour du job LLM pour eviter les attentes >180s.
4. Separer les secrets chat provider et embedding provider si on veut `vllm` pour le chat et OpenAI-compatible pour les embeddings.
5. Garder les logs `chatfleet.metrics` en production pour attribution fine.
