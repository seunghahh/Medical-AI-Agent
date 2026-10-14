# Medical AI Agent

A Python agent for clinical conversations. It gathers patient history, requests focused physical examination findings, and produces a SOAP note with a provisional diagnosis and follow-up plan. Optional local medical references support diagnosis review.

![Clinical conversation interface demo](docs/nova-demo.gif)

## Run locally

Use Python 3.12 or newer and [Ollama](https://ollama.com).

Start the local model server:

```bash
ollama serve
```

In another terminal, download the model and start a sample conversation:

```bash
ollama pull gpt-oss:20b
python develop.py --case samples/C01
```

The conversation viewer opens at `http://127.0.0.1:8768`. Records are saved in `results/`.

To run without the viewer:

```bash
python develop.py --case samples/C01 --no-browser
```
