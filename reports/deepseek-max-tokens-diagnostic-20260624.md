# Diagnostic DeepSeek max_tokens - 2026-06-24

## Contexte

- API testee: `http://localhost:8080/api`
- Image API temporaire: `chatfleet-api:rag-latency-instrumented`
- LLM provider: `vllm`
- Modele: `DeepSeek-R1-Distill-Qwen-32B-Q8_0.gguf`
- Base URL: `http://host.docker.internal:2250/v1`
- Embed provider: `openai`
- RAG probe: `latency-probe-1782296803`
- RAG client: `client-validation-rag-20260530-105619`
- Batterie courte: 1 question probe + 1 question client, en `/chat` et `/chat/stream`.

## Preflight DeepSeek direct

| Test | Resultat |
|---|---:|
| `/v1/models` | 146.7 ms |
| completion `max_tokens=32` | 2807.2 ms, 32 tokens, contenu visible vide |
| completion `max_tokens=128` | 7503.5 ms, 128 tokens, contenu visible partiel |

Interpretation: DeepSeek repond correctement, mais les petits budgets peuvent etre consommes par le raisonnement avant de produire une reponse visible.

## Resultats RAG par seuil

| max_tokens | OK / total | LLM probe chat | LLM probe stream | LLM client chat | LLM client stream | Diagnostic |
|---:|---:|---:|---:|---:|---:|---|
| 128 | 0 / 4 | 7459 ms | 7409 ms | 7434 ms | 7424 ms | Rapide mais inutilisable: pas de reponse visible. |
| 256 | 1 / 4 | 16566 ms | 14786 ms | 15914 ms | 14803 ms | Instable: raisonnement tronque, erreurs `finish_reason=length`. |
| 500 | 3 / 4 | 14275 ms | 20401 ms | 29146 ms | 28999 ms | Meilleur, mais stream client echoue encore et latence client proche de 30s. |
| 800 | 4 / 4 | 16211 ms | 21072 ms | 36190 ms | 37023 ms | Stable, qualite OK, mais trop lent pour une UX synchrone confortable. |

Rapports sources:

- `reports/rag-latency-20260624T123351z0000.md` (`128`)
- `reports/rag-latency-20260624T123225z0000.md` (`256`)
- `reports/rag-latency-20260624T123430z0000.md` (`500`)
- `reports/rag-latency-20260624T123625z0000.md` (`800`)

## Controle qualite cible a 800 tokens

Requete client: `Resume les informations principales disponibles dans ce RAG.`

- HTTP: 200
- Temps total client: 38044.4 ms
- Tokens in: 60
- Tokens out: 642
- Citations: 2
- Taille reponse: 1150 caracteres

Extrait de reponse: la reponse resume correctement les faits du RAG client: versions stable API/Web, preservation Mongo, volumes Docker, emplacement des index RAG, obligation de citer les documents, et fallback anti-hallucination.

## Diagnostic

- Le retrieval est stable et rapide: environ 360-640 ms sur les campagnes.
- Le premier evenement stream arrive vite: environ 370-645 ms.
- La latence longue est attribuee au LLM DeepSeek, pas au RAG.
- DeepSeek est plus stable que Gemma sur cette machine, mais son comportement de modele reasoning consomme beaucoup de tokens avant de produire une reponse visible.
- `128` et `256` ne sont pas des seuils acceptables pour DeepSeek dans ce workflow.
- `500` est le meilleur compromis latence, mais pas assez fiable en stream client.
- `800` est le premier seuil fiable, avec une qualite correcte, mais il produit des reponses client autour de 36-38s.

## Recommandation

- Garder DeepSeek comme modele de reference pour la suite des tests.
- Ne pas retenir `128` ni `256` comme defaut produit.
- Utiliser `800` comme seuil de validation qualite DeepSeek.
- Tester ensuite une optimisation de prompt pour reduire le raisonnement requis, puis retester `500`.
- Ajouter un timeout serveur strict avant tout usage produit, car meme le seuil stable reste trop lent pour certaines UX.
