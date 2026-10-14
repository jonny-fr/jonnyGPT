# Mini-SmolLM auf einer RTX 4070

Eigenständiges Lernexperiment: zufällig initialisiertes Llama-artiges Modell mit
ungefähr 33 Millionen Parametern, kein Fine-Tuning vortrainierter Modellgewichte.
Nur der Tokenizer wird von HuggingFaceTB/SmolLM2-135M übernommen.

## Installation

Empfohlen: Python 3.11, Linux oder Windows/WSL2, NVIDIA-Treiber und eine
CUDA-fähige PyTorch-Installation. Pakete in einer separaten Umgebung installieren:

    python -m venv .venv
    # Linux / WSL2:
    source .venv/bin/activate
    # Windows PowerShell stattdessen:
    # .venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

Falls CUDA False meldet, zuerst die zum Betriebssystem und Treiber passende
CUDA-Ausgabe von PyTorch installieren. Ein CPU-only-Wheel reicht nicht.
Die Versionsbereiche sind Kompatibilitätsvorgaben, kein getestetes Lockfile.
Das Skript wurde in der Erstellungsumgebung syntaktisch geprüft, nicht auf einer
4070 oder mit den entfernten Datensätzen ausgeführt.

## Schnelltest vor einem großen Download

    python mini_smollm.py prepare --data smoke_data --train-tokens 2000000 --val-tokens 8192
    python mini_smollm.py train --data smoke_data --out smoke_model --hours 0.25 --max-steps 20 --eval-batches 2 --log-every 1
    python mini_smollm.py sample --out smoke_model --prompt "The solar system consists of"

Der Schnelltest prüft die Pipeline, nicht die Textqualität. Es ist normal, dass
so wenig trainierte Modelle unbrauchbare Ausgaben erzeugen.

## Hauptlauf

    python mini_smollm.py prepare --data mini_data --train-tokens 60000000
    python mini_smollm.py train --data mini_data --out mini_model --hours 3
    python mini_smollm.py sample --out mini_model --prompt "The solar system consists of"

Der Download und die CPU-Tokenisierung zählen NICHT zu den drei Stunden.
Das Zeitbudget beginnt in train vor dem Benchmark; Modellladen ist davor.
Benchmark, Evaluation und Training liegen im Budget. Der abschließende Export
kommt gegebenenfalls noch dazu. Das ist ein weiches Budget, kein harter
Prozessabbruch: laufende CUDA-Operationen werden nicht abgebrochen.

prepare lädt FineWeb-Edu/sample-10BT und FineMath/finemath-4plus per Streaming.
Standardmischung: 90% Web, 10% Mathematik. Der Ausschnitt ist vorab lokal
gepackt: 60 Mio. Tokens benötigen bei uint16 etwa 120 MB, plus Validierung und
Cache. Der Netzwerktraffic kann erheblich höher sein; Parquet-Lesevorgänge,
Shuffle-Puffer und die Validierungsauswahl brauchen weitere Daten.
Bereits vorbereitete Verzeichnisse werden nicht überschrieben.

Die erste Nutzung des Hubs braucht Internet und ggf. eine Anmeldung bzw.
Zustimmung zu Bedingungen der jeweiligen Quelle. Es werden KEINE Modelle
hochgeladen und keine kostenpflichtigen APIs aufgerufen. Lizenzen der Quellen
beachten; das Skript übernimmt nicht automatisch sämtliche Attributionen in
abgeleitete Daten oder Modellkarten.

## Optional: Python-Daten ergänzen

Stack-Edu enthält Referenzen auf Software-Heritage-Dateien, nicht die eigentlichen
Code-Inhalte. Für das schnelle Experiment werden diese Downloads nicht
automatisiert. Stelle rechtmäßig nutzbaren Code lokal als JSONL bereit, z.B.:

    {"text": "def add(a, b):\n    \"Return the sum of two values.\"\n    return a + b\n"}

Auch content und code werden als Felder akzeptiert. Eine Zeile pro Dokument,
keine mehrzeiligen JSON-Objekte. Texte unter 80 Zeichen werden übersprungen.
Für eine echte Stack-Edu-Mischung müssen die Texte zuvor aus dessen Referenzen
heruntergeladen werden; beliebiger lokaler Code ist kein Stack-Edu-Ersatz mit
gleichen Eigenschaften. Die Beschaffung samt Lizenzprüfung liegt beim Nutzer.

    python mini_smollm.py prepare --data mixed_data --train-tokens 60000000 --code-jsonl python_data.jsonl
    python mini_smollm.py train --data mixed_data --out mixed_model --hours 3

Damit beträgt die angestrebte Tokenmischung 80% Web / 10% Code / 10% Mathematik.
Bei lokalen Daten müssen genug unterschiedliche Dokumente für beide Splits
vorhanden sein. Die Dokument-Hashes bestimmen den Split (2% Validierung), bevor
gepackt wird. Exakte Duplikate werden über alle Quellen hinweg entfernt;
semantische Duplikate und Benchmark-Kontamination werden nicht geprüft.
Ist eine lokale Quelle vor den Zielmengen erschöpft, werden die tatsächlich
verfügbaren Tokenzahlen im Manifest vermerkt. Zu kleine Splits führen zu Fehlern.
Maximal 40.000 Zeichen und anschließend maximal 8.192 Tokens pro Dokument
werden verwendet. Diese Begrenzungen sind weitere bewusste Vereinfachungen.

## Modell und Trainingsablauf

- 8 Layer, Hidden Size 384, 6 Attention Heads, FFN 1024, SwiGLU, RoPE.
- Geteilte Ein-/Ausgabe-Embeddings, ungefähr 33 Mio. Parameter.
- Kontext 512; Dokumente werden mit EOS getrennt und zu Sequenzen gepackt.
- Attention kann dabei über Dokumentgrenzen hinweg reichen.
- Microbatch 4, Akkumulation 8: 16.384 Trainingstokens pro Optimizer-Schritt.
- Vollständiges Training aller Parameter, AdamW, Weight Decay 0,1.
- FP32-Modellparameter, Autocast BF16 oder FP16, bei FP16 GradScaler.
- SDPA, kein separates flash-attn-Paket erforderlich.
- LR 5e-4, 5% Warmup / 75% konstant / 20% linearer Decay.
- Drei Benchmark-Schritte mit Lernrate null; danach Optimizer-/Scaler-/RNG-Reset.
- Die letzten beiden Benchmark-Schritte bestimmen das geplante Schrittbudget.
- 20% Reserve für Evaluation, Schwankungen und Export.
- Kontrollierter Abbruch zwischen Optimizer-Schritten bei knappem Zeitbudget.

Der Benchmark zählt keine Trainingstokens im eigentlichen Ergebnis. Die
geplante Tokenzahl ist eine Durchsatzschätzung, keine garantierte Qualität oder
4070-Benchmarkangabe. Ist --max-steps angegeben, wird der Lernratenplan an das
kleinere Schrittbudget angepasst.

train zieht zufällige Fenster MIT ZURÜCKLEGEN aus dem vorbereiteten Pool.
Die Datenmischung wird pro Microbatch gewürfelt; langfristig entspricht sie der
Tokenmischung, einzelne Schritte können abweichen. Trainingstokens sind daher
nicht unbedingt einzigartige Tokens. Wenn das Budget den Datenpool weit
übersteigt, mehr Daten vorbereiten oder --max-steps begrenzen. Diese Umsetzung
ist keine exakt einmalige Epoche über einen fortlaufenden Stream.

Die Validierung verwendet bei jeder Auswertung identische zufällige Fenster
je Quelle, damit die Werte vergleichbar sind. Diese Fenster können überlappen;
sie stellen eine kleine Stichprobe und keinen umfassenden Benchmark dar.
Für robustere Messungen --eval-batches erhöhen. Ein zusätzlicher unangetasteter
Testsatz wird nicht automatisch angelegt. Die Daten sind überwiegend Englisch.

## Speicherprobleme

    python mini_smollm.py train --data mini_data --out small_batch_model --hours 3 --batch-size 2 --accumulation 16

Bei Bedarf zusätzlich --gradient-checkpointing verwenden. Nach einem Fehler
für den nächsten Lauf ein neues --out angeben. Auch Sequenzlänge 256 kann
helfen, sofern das Modell später ebenfalls mit dieser Kontextgrenze genügt.
Kein CPU-Offloading, keine QLoRA und keine 4-Bit-Quantisierung.

## Ergebnisse

Im gewählten Modellverzeichnis werden gespeichert:
- Hugging-Face-kompatibles Modell und Tokenizer,
- run.json mit Konfiguration, Quelle und Benchmark,
- result.json mit tatsächlicher Schrittzahl, Tokens, Laufzeit, Abbruchgrund,
- metrics.csv mit Trainings- und Validierungsverlusten.

Strg+C exportiert den aktuellen Modellstand, sobald der Handler greifen kann.
Bei nicht abfangbarem Prozessabbruch/Stromausfall gibt es KEINE Wiederherstellung.
Es gibt keine periodischen Checkpoints und keinen Resume-Modus; der Optimizer
wird nicht gespeichert. Für einen Drei-Stunden-Lernlauf hält das das Skript
überschaubar. Nach endgültigem Export kann das Modell mit sample geladen werden.
Es ist ein Textfortsetzungsmodell, kein instruction-getunter Chatbot.

## Quellen

SmolLM2-Paper: arXiv 2502.02737 (vom Nutzer bereitgestellt).
Datensatzkarten: HuggingFaceFW/fineweb-edu, HuggingFaceTB/finemath,
HuggingFaceTB/stack-edu. Implementierung: Hugging Face Transformers Llama,
Hugging Face Datasets Streaming. Die reduzierte Konfiguration, Mischung und das
Stundenbudget sind ein eigener Experimentvorschlag, keine Paper-Reproduktion.
