# La Ghigliottina — un benchmark di associazione semantica per LLM

Uno studio pilota che sottopone 30 item nel formato del gioco televisivo italiano "La Ghigliottina" (5 parole indizio in inglese → un'unica parola che le collega tutte) a 127 modelli linguistici — 111 a temperatura controllata (locali via Ollama + open-weight via OpenRouter) e 16 commerciali di punta (Claude, GPT, Gemini, Grok, Kimi, tra gli altri) — per confrontarne la capacità di riconoscimento di pattern semantici contro il giudizio di un singolo autore umano (N=1, dichiarato esplicitamente come limite, non nascosto).

Il saggio completo, con la discussione estesa dei risultati e le sezioni di riflessione filosofica, è in `documentation/` (italiano e inglese) e verrà pubblicato anche su LessWrong e LinkedIn — link aggiunti qui non appena disponibili.

## Domande di ricerca

1. La soluzione dell'autore coincide con quella trovata dai modelli?
2. Quanto distano, quantitativamente, le risposte dei modelli dalla soluzione di riferimento?
3. Nell'associazione delle parole, la risposta di un modello può risultare più coerente di quella dell'autore (valutazione linguistica soggettiva)?
4. Quanto la risposta di un modello cambia lungo lo spettro di temperatura (T=0.0/0.3/0.7), a parità di top_k/top_p? — *non* un test di riproducibilità a parametri fissi, quello richiederebbe un run aggiuntivo a T identica, non incluso in questo disegno.
5. La dimensione del modello (numero di parametri) correla con l'accuratezza dell'associazione?
6. Quale modello fornisce la risposta migliore, secondo il giudizio dell'autore?

## Alcuni risultati (dettagli completi in `documentation/`)

- Il modello locale/open-weight più affidabile è `stepfun/step-3.5-flash` (71.1% exact match, IC 95% Wilson [61.0%, 79.5%], su 90 osservazioni) — non il nominalmente più alto in classifica, che ha solo 4 osservazioni valide su 90.
- Tra i commerciali, `x-ai/grok-4.6` guida con il 76.7% (IC 95% [59.1%, 88.2%], N=30). Tra i primi 8 modelli di ciascun gruppo, **nessuna coppia resta significativamente diversa** dopo correzione FDR (McNemar) — la posizione esatta in classifica ai vertici non va sovra-interpretata.
- La temperatura sposta l'accuratezza assoluta di pochi punti per un sottoinsieme di modelli (9 su quelli testabili, senza correzione per confronti multipli), ma **non ribalta il ranking** tra modelli (Spearman tra ranking a T diverse: 0.886–0.930).
- La dimensione del modello correla con l'accuratezza (Spearman rho=0.756 su parametri totali verificati, rho=0.690 su parametri attivi per token, n=45) — ma non spiega tutto: `gpt-oss:20b` (3.61B parametri attivi per token, architettura MoE) è tra i migliori del roster con uno dei budget di calcolo più piccoli.
- Le tre misure indipendenti di difficoltà per item (etichetta dichiarata dall'autore, cluster empirico sull'accuratezza, cluster empirico sul tempo di generazione dei modelli reasoning) si accordano solo in parte tra loro (Kappa pesato: etichetta↔accuratezza 0.122, etichetta↔tempo 0.145, accuratezza↔tempo 0.325) — le due misure derivate dal comportamento dei modelli si accordano tra loro più di quanto ciascuna si accordi col giudizio umano.
- Comportamenti qualitativi documentati nel dettaglio: un modello (`starcoder2:15b-instruct`) che in metà delle risposte continua a generare codice Python estraneo al compito dopo aver risposto; casi di modelli reasoning che trovano la risposta corretta nel proprio ragionamento e poi la abbandonano (incluso un caso che oscilla più di dieci volte tra due risposte prima di sbagliare).

## Struttura della repository

```
guillotine/
├── source/                  # dati di input (i risultati grezzi NON sono inclusi, vedi sotto)
│   ├── guillotine.xlsx          # dataset dei 30 item (fonte di verità, editabile a mano)
│   └── guillotine_prompts.json  # stesso dataset, convertito in JSON per gli script
├── python/                  # pipeline di raccolta dati
├── notebook/                # analisi statistica (Jupyter)
│   ├── ghigliottina_analisi_locali_13[_eng].ipynb
│   ├── ghigliottina_analisi_commerciali_01[_eng].ipynb
│   └── ghigliottina_tempo_token_ragionamento_02[_eng].ipynb
├── documentation/           # saggio completo (italiano + inglese)
├── requirements.txt         # dipendenze della pipeline di raccolta
└── .env.example             # variabili di configurazione (copiare in .env)
```

### Dati inclusi ed esclusi

Per tenere la repository leggera e non ridistribuire migliaia di risposte grezze di modelli commerciali (i cui termini di servizio su riuso/ridistribuzione non sono sempre chiari), **sono inclusi solo i due file di input**:

- `source/guillotine.xlsx` — i 30 item originali (indizi, soluzione attesa, nota di difficoltà)
- `source/guillotine_prompts.json` — stessi dati, formato JSON, usato direttamente dagli script e dai notebook

**Non sono inclusi**: i workbook con i risultati grezzi (`complete_ghigliottina_results.xlsx` e le sue versioni intermedie), i log di raccolta (`*.log`), i CSV prodotti dai notebook (`analisi_leaderboard.csv`, `analisi_per_item.csv`, `analisi_temperatura.csv` — rigenerabili rieseguendo i notebook), e i log grezzi JSONL delle chiamate OpenRouter (`raw_logs/`, necessari solo per rigenerare da zero il notebook `ghigliottina_tempo_token_ragionamento_02`, che usa i risultati già calcolati e salvati nelle celle del notebook stesso).

**Conseguenza pratica**: i tre notebook si aprono e si leggono perfettamente così come sono (le celle contengono già i risultati e i grafici dell'esecuzione originale) — ma per *rilanciarli* da zero contro dati nuovi serve prima rigenerare `source/complete_ghigliottina_results.xlsx` con la pipeline sotto, oppure richiedere il dataset completo all'autore.

### Pipeline di raccolta (`python/`)

| Script | Cosa fa |
|---|---|
| `config.py` | Configurazione centrale: path, lista modelli locali (`LOCAL_MODELS`) e OpenRouter (`OPENROUTER_MODELS`, `OPENROUTER_COMMERCIAL`), parametri di sampling, schema colonne output. Nessuna credenziale hardcoded — le chiavi API vivono solo in `.env` |
| `excel_to_json.py` | Converte `guillotine.xlsx` (validato a mano) in `guillotine_prompts.json`, con un controllo di sanità che segnala se la soluzione compare per errore come sottostringa in una clue |
| `generate_template.py` | Crea il workbook risultati vuoto/precompilato (4 fogli: `T_0.0`/`T_0.3`/`T_0.7` per i modelli a temperatura controllata, `Commercial` precompilato per la raccolta manuale) |
| `oldgenerate_template.py` | Versione precedente di `generate_template.py`, superata — mantenuta solo per riferimento storico, non usarla |
| `ollama_client.py` | Client HTTP per `/api/generate` di Ollama (modelli locali) |
| `openrouter_client.py` | Client HTTP per l'endpoint chat di OpenRouter (modelli open-weight su hardware remoto + modelli commerciali); gestisce rate limiting (backoff 5/10/20s su 429), retry, e un backup grezzo JSONL di ogni chiamata prima del parsing |
| `batch_runner.py` | Orchestratore CLI per i modelli locali/OpenRouter a temperatura controllata: per ogni modello, itera 3 temperature × 30 prompt, salva incrementalmente, scarica il modello dalla VRAM. Resumable: al riavvio salta le combinazioni (modello, prompt) già completate |
| `commercial_runner.py` | Stesso principio per i modelli commerciali: un solo run per modello/prompt, nessun parametro di sampling impostato (il provider usa il proprio default) |
| `response_parser.py` | Estrae la parola finale dalla risposta grezza (gestisce testo tra virgolette, tracce di reasoning `<think>`, risposta-poi-spiegazione e viceversa) |
| `scoring.py` | Le tre metriche automatiche: `exact_match`, `same_root` (stemming), `is_similar_lev` (Levenshtein normalizzato) |
| `excel_writer.py` | Scrittura incrementale su Excel con upsert anti-duplicati; il workbook stesso è il checkpoint di resume |
| `merge_log_timings.py` | Utility per ricongiungere i tempi di generazione dai log quando mancanti dal workbook principale |
| `rescore.py` | Ricalcola le metriche automatiche su un workbook già raccolto, senza rilanciare i modelli (usato dopo correzioni manuali a `predicted_word`) |

### Notebook di analisi (`notebook/`)

Ogni notebook esiste in due versioni identiche nei risultati: quella originale in italiano e una tradotta in inglese (`_eng`, stessi dati, stesso codice, titoli/etichette dei grafici e commenti in inglese) — usata anche per generare le figure del saggio in inglese.

| Notebook | Copre |
|---|---|
| `ghigliottina_analisi_locali_13` | RQ1–RQ6 sui 111 modelli a temperatura controllata: caricamento/deduplica dati, controllo di copertura, classifica con IC di Wilson, effetto della temperatura (Cochran's Q), correlazione dimensione/accuratezza (RQ5, con e senza parametri verificati), calibrazione e classificazione empirica della difficoltà degli item (k-means + silhouette), validità convergente (Kappa pesato), analisi Zipf della rarità lessicale |
| `ghigliottina_analisi_commerciali_01` | Stessa impalcatura statistica sui 16 modelli commerciali (run singolo, N=30) — senza RQ4 (nessun asse di temperatura) né RQ5 (i modelli proprietari di punta non dichiarano i parametri) |
| `ghigliottina_tempo_token_ragionamento_02` | Analisi dei log grezzi OpenRouter (non l'Excel): token di ragionamento nativi per provider, velocità di inferenza, rapporto caratteri/token come controllo di trasparenza sui riassunti del ragionamento mostrati da alcuni provider |

## Come riprodurre

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env       # personalizzare GHIGLIOTTINA_DATA_DIR/OLLAMA_HOST se serve
```

Per i modelli locali serve [Ollama](https://ollama.com) in esecuzione (`ollama serve`); per i modelli OpenRouter/commerciali serve una chiave API in `.env` (`OPENROUTER_API_KEY=...`, non incluso in `.env.example` per non suggerire un formato con credenziali).

```bash
# 1. Dataset -> JSON
python python/excel_to_json.py source/guillotine.xlsx source/guillotine_prompts.json

# 2. Crea il workbook vuoto
python python/generate_template.py source/guillotine_prompts.json source/ghigliottina_results.xlsx

# 3. Raccolta — modelli locali/OpenRouter a temperatura controllata (resumable)
python python/batch_runner.py
python python/batch_runner.py --models mistral-small:22b --limit-prompts 3   # smoke test

# 4. Raccolta — modelli commerciali (un run, nessuna temperatura)
python python/commercial_runner.py
```

Per i notebook (non coperti da `requirements.txt`, che riguarda solo la pipeline di raccolta):

```bash
pip install jupyter pandas numpy scipy matplotlib scikit-learn statsmodels nltk rapidfuzz wordfreq
```

Il prompt usato, identico per tutti i modelli:

```
You are playing a word-association game. You will be given 5 clue
words. Find the single word that connects all five (through common
phrases, idioms, or shared concepts). Respond with ONLY the final
word, in uppercase, with no explanation. Clues: {word1}, {word2},
{word3}, {word4}, {word5}.
```

## Limiti principali

- Soluzione, difficoltà attesa e giudizio di coerenza linguistica sono stabiliti da un singolo autore (N=1), dichiarato come tale, non un limite nascosto.
- Il confronto a coppie tra i modelli migliori (McNemar + correzione FDR) non trova differenze statisticamente significative ai vertici della classifica, in entrambi i gruppi — la posizione esatta in classifica dei modelli di punta non va sovra-interpretata.
- I modelli commerciali hanno un solo run per item, senza controllo dei parametri di sampling (costo, non impossibilità tecnica) — confrontabili con i modelli a temperatura controllata solo su RQ1/2/3/5/6, non su RQ4.
- L'estrazione automatica della risposta finale dal testo grezzo è euristica (`format_compliant` traccia i casi a rischio senza eliminarli); la revisione manuale delle risposte non esatte (`synonym_manual`, `coherence_manual`) è comunque completa al 100%.
- 3 dei 30 item originali sono stati sostituiti dopo una revisione metodologica iniziale (misuravano un pattern ortografico o un'etichetta di categoria, non un'associazione semantica) — la versione del dataset in questa repo è quella finale.

## Licenza

Il codice in questa repository (`python/`, `source/*.py`, notebook inclusi) è distribuito con licenza **MIT** — vedi [LICENSE](LICENSE).

Il materiale in `documentation/` (il saggio e le sue analisi) è distribuito con licenza **[Creative Commons Attribuzione 4.0 (CC-BY 4.0)](https://creativecommons.org/licenses/by/4.0/deed.it)**: chiunque può usarlo, modificarlo e redistribuirlo, anche commercialmente, a condizione di citare l'autore e/o linkare questa repository.

## Come citare

Se questo lavoro ti è utile, cita l'autore e/o linka questa repository. Un riferimento più formale (con link al saggio pubblicato) verrà aggiunto qui non appena disponibile.
