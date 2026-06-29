import requests                              # HTTP requests
import json
import hashlib                               # SHA256 for exact deduplication
import wikipediaapi                          # fetch Wikipedia articles as plain text
import pypdf                                 # extract text from PDFs
import csv
import shutil                       
from pathlib import Path                     # OS-independent paths
from bs4 import BeautifulSoup                # clean web-page HTML
from langdetect import detect                # language filter
from transformers import AutoTokenizer       # Llama tokenizer for chunking


OA = "https://api.openalex.org/works"        # OpenAlex endpoint
OUTPUT_DIR = Path(__file__).parent / "data_corpus"
CHUNK_SIZE = 2048


WIKIPEDIA_ARTICLES = [
    "AUTOSAR",
    # Bus systems, in-vehicle networks
    "CAN bus", "CAN FD", "FlexRay", "Local Interconnect Network", "Vehicle bus",
    "CANopen", "DeviceNet", "MOST Bus", "Time-Triggered Protocol",
    "Time-Sensitive Networking", "Audio Video Bridging", "SAE J1939",
    # ECU architectures
    "Electronic control unit", "Engine control unit", "Body control module",
    "Transmission control unit", "Powertrain control module",
    # Functional safety 
    "ISO 26262", "Automotive Safety Integrity Level", "Functional safety",
    # Diagnostics
    "Unified Diagnostic Services", "On-board diagnostics", "OBD-II", "OBD-II PIDs",
    "ISO 15765-2", "Keyword Protocol 2000", "XCP (protocol)",
    # ADAS & autonomous driving
    "Advanced driver-assistance system", "Self-driving car",
    "Adaptive cruise control", "Lane departure warning system",
    "Sensor fusion", "Lidar", "Radar",     # Sensor fusion bleibt drin → not_relevant-Demo
    # Safety & control systems
    "Anti-lock braking system", "Electronic stability control", "Airbag",
    "Brake-by-wire", "Drive by wire", "Steer-by-wire", "Tire-pressure monitoring system",
    # Connected, telematics & security
    "Vehicle-to-everything", "Vehicular communication systems",
    "Dedicated short-range communications", "Telematics", "Connected car",
    "Vehicle-to-grid", "ECall", "Over-the-air update",
    "Hardware security module", "In-car entertainment"
]

WEB_URLS = [
    "https://www.siemens.com/en-us/technology/autosar/",
    # Bus systems & protocols
    "https://www.csselectronics.com/pages/can-bus-simple-intro-tutorial",
    "https://www.csselectronics.com/pages/can-fd-flexible-data-rate-intro",
    "https://www.csselectronics.com/pages/can-bus-errors-intro-tutorial",
    "https://www.csselectronics.com/pages/lin-bus-protocol-intro-basics",
    "https://www.csselectronics.com/pages/can-dbc-file-database-intro",
    "https://www.csselectronics.com/pages/j1939-explained-simple-intro-tutorial",
    "https://www.kvaser.com/can-protocol-tutorial/",
    "https://www.kvaser.com/about-can/can-standards/",
    "https://www.kvaser.com/about-can/higher-layer-protocols/j1939-introduction/",
    "https://www.ni.com/en/shop/seamlessly-connect-to-third-party-devices-and-supervisory-system/controller-area-network--can--overview.html",
    "https://www.ni.com/en/shop/seamlessly-connect-to-third-party-devices-and-supervisory-system/flexray-automotive-communication-bus-overview.html",
    # Diagnostics
    "https://www.csselectronics.com/pages/obd2-explained-simple-intro",
    "https://www.csselectronics.com/pages/obd2-pid-table-on-board-diagnostics-j1979",
    "https://www.csselectronics.com/pages/uds-protocol-tutorial-unified-diagnostic-services",   
    # German source
    "https://de.wikipedia.org/wiki/Controller_Area_Network",
    # PDFs
    "https://mediatum.ub.tum.de/doc/1638880/mth1hqzs56h0qnkny6syzdham.Disseration_Johannes_Eder_Bib.pdf",
]

SEED_DOIS = [
      "10.5121/csit.2016.60121",         # Bock 2016 — automotive software engineering taxonomy
      "10.1109/meco58584.2023.10154913", # Cuomo 2023 — RISC-V open platform for automotive ECUs
      "10.1016/j.jss.2024.112220",       # Mauser 2024 — centralization of automotive E/E architectures
      "10.48550/arxiv.1709.02435",       # Salay 2017 — ISO 26262 & functional safety for ML in automotive
      "10.48550/arxiv.1703.08557",       # Ulbrich 2017 — functional system architecture for automated vehicles
  ]

EE_KEYWORDS = [
    # General domain
    "automotive", "e/e",
    # AUTOSAR & middleware
    "autosar", "some/ip",
    # Bus systems & in-vehicle networks
    "can bus", "can fd", "controller area network", "canopen", "flexray",
    "lin bus", "j1939", "iso 11898", "time-triggered", "time-sensitive networking",
    "automotive ethernet", "audio video bridging", "in-vehicle",
    # ECU architectures
    "ecu", "electronic control unit", "powertrain",
    # Functional safety
    "iso 26262", "asil", "functional safety",
    # Diagnostics
    "on-board diagnostic", "obd",
    # ADAS & autonomous driving
    "adas", "advanced driver", "driver-assistance",
    # Connected, telematics & security
    "v2x", "vehicle-to", "ecall", "tpms", "tire-pressure",
]

STOPWORDS = {"the","a","an","of","and","to","in","is","for",
             "with","on","that","as","are","by","be","this","or"}


def fetch_wikipedia(articles, output_dir): # Fetch each article as plain text and save it as a .txt file
    output_dir.mkdir(parents=True, exist_ok=True)
    wiki = wikipediaapi.Wikipedia(language="en", user_agent="ee-corpus")
    for article in articles:
        page = wiki.page(article)
        text = page.text
        if not text:
            print(f"Skipping '{article}': not found")
            continue
        (output_dir / f"{page.title}.txt").write_text(text, encoding="utf-8")


def fetch_webpages(urls, output_dir): # Download each page; HTML → text, direct PDF → .pdf
    output_dir.mkdir(parents=True, exist_ok=True)
    for url in urls:
        try:
            response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
            slug = url.split("//")[-1].replace("/", "_").replace(".", "_")[:60]
            if response.content[:5] == b"%PDF-":                 # direct PDF link → let pypdf handle it
                (output_dir / (slug + ".pdf")).write_bytes(response.content)
                continue
            soup = BeautifulSoup(response.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            text = soup.get_text(separator="\n")
            lines = [l.strip() for l in text.splitlines()]
            text = "\n".join(l for l in lines if l)
            with open(output_dir / (slug + ".txt"), "w", encoding="utf-8") as f:
                f.write(text)
        except Exception as e:
            print(f"Skipping {url}: {e}")


def fetch_seed_pdfs(seed_dois, output_dir): # download the seed papers themselves into raw/
    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    for doi in seed_dois:
        w = requests.get(f"{OA}/https://doi.org/{doi}",
                         params={"select": "locations"}, timeout=20).json()
        url = next((loc["pdf_url"] for loc in (w.get("locations") or []) if loc.get("pdf_url")), None)
        if not url:
            print(f"seed {doi}: no OA PDF"); continue
        try:
            r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
            if r.status_code == 200 and r.content[:5] == b"%PDF-":
                (output_dir / f"seed_{downloaded}.pdf").write_bytes(r.content); downloaded += 1
            else:
                print(f"seed {doi}: not downloadable ({r.status_code})")
        except Exception as e:
            print(f"seed {doi}: {e}")
    print(f"downloaded {downloaded} seed PDFs")


def fetch_papers_from_seeds(seed_dois, output_dir, max_download=40): # DEMO: citation-driven
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. resolve each seed by DOI, collect its reference list (clean OpenAlex IDs)
    refs = set()
    for doi in seed_dois:
        seed = requests.get(f"{OA}/https://doi.org/{doi}",
                            params={"select": "referenced_works"}, timeout=20).json()
        refs |= set(seed.get("referenced_works", []))
    print(f"collected {len(refs)} unique references across {len(seed_dois)} seeds")

    # 2. for the open-access ones, take a PDF url from ANY location (not just the best)
    urls = []
    refs = list(refs)
    for i in range(0, len(refs), 50):
        ids = "|".join(r.split("/")[-1] for r in refs[i:i+50])
        works = requests.get(OA, params={
            "filter": f"openalex_id:{ids},open_access.is_oa:true",
            "select": "locations", "per_page": 200}, timeout=30
        ).json().get("results", [])
        for w in works:
            for loc in (w.get("locations") or []):
                if loc.get("pdf_url"):
                    urls.append(loc["pdf_url"]); break
    print(f"{len(urls)} open-access PDF links")

    # 3. download those that return a real PDF (bot-blocked publishers are skipped)
    downloaded = 0
    for url in urls:
        if downloaded >= max_download:
            break
        try:
            r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
            if r.status_code == 200 and r.content[:5] == b"%PDF-":
                (output_dir / f"ref_{downloaded}.pdf").write_bytes(r.content)
                downloaded += 1
        except Exception:
            continue
    print(f"downloaded {downloaded} PDFs")


def extract_texts(output_dir): # return [(path, text), ...] for every .pdf/.txt in raw/
    items = []
    for datei in output_dir.iterdir():
        if datei.suffix == ".pdf":
            try:
                text = "".join(p.extract_text() for p in pypdf.PdfReader(datei).pages)
                items.append((datei, text))
            except Exception as e:
                print(f"Skipping {datei.name}: {e}")
        elif datei.suffix == ".txt":
            items.append((datei, open(datei, encoding="utf-8").read()))
    return items




# Filters

def doc_metrics(text): # the numbers every filter looks at, for one document
    words = text.split()
    n = len(words)
    non_ascii = sum(1 for c in text if ord(c) > 127) / len(text) if text else 0
    stop_ratio = sum(1 for w in words if w.lower() in STOPWORDS) / n if n else 0
    ee = sum(1 for kw in EE_KEYWORDS if kw in text.lower())
    return {"words": n, "stop_ratio": round(stop_ratio, 4),
            "non_ascii": round(non_ascii, 4), "ee_keywords": ee}


def classify(text, m): # rejection reason (or None) derived from the metrics
    if not is_english(text):   return "not_english"
    if m["words"] < 50:        return "too_short"
    if m["words"] > 100000:    return "too_long"
    if m["non_ascii"] > 0.1:   return "non_ascii"
    if m["stop_ratio"] < 0.10: return "low_prose"
    if m["ee_keywords"] < 3:   return "not_relevant"
    return None


def _move_aside(source, filtered_dir): # move a collected raw/ file out of the corpus (seeds stay put)
    if source.parent.name == "raw":
        dest = filtered_dir / source.name
        if dest.exists(): dest.unlink()
        shutil.move(str(source), str(dest))


def filter_docs(items, filtered_dir, stats): # items = [(source, text)]; records stats, moves rejects
    filtered_dir.mkdir(parents=True, exist_ok=True)
    passed = []
    for source, text in items:
        m = doc_metrics(text)
        reason = classify(text, m)
        stats.append({"source": source.name, **m, "status": reason or "passed"})
        if reason:
            _move_aside(source, filtered_dir)
        else:
            passed.append((source, text))
    counts = {}
    for row in stats:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    print(f"Total: {len(stats)} |", " | ".join(f"{k}: {v}" for k, v in counts.items()))
    return passed


def is_english(text): # Language filter: True if langdetect detects English
    try:
        return detect(text) == "en"
    except Exception:
        return False


def deduplicate(items, filtered_dir, stats): # exact dedup; marks + moves duplicates aside
    seen, unique = set(), []
    for source, text in items:
        h = hashlib.sha256(text.encode()).hexdigest()
        if h in seen:
            for row in stats:
                if row["source"] == source.name and row["status"] == "passed":
                    row["status"] = "duplicate"; break
            _move_aside(source, filtered_dir)
        else:
            seen.add(h); unique.append((source, text))
    return unique


def chunk_docs(items): # tokenize each document text and slice into 2048-token chunks
    tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-1B")
    chunks = []
    for _, text in items:
        tokens = tokenizer.encode(text)
        for i in range(0, len(tokens), CHUNK_SIZE):
            chunks.append(tokens[i : i + CHUNK_SIZE])
    return chunks


def save_stats(stats, output_dir): # per-document table for later inspection
    with open(output_dir / "corpus_stats.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["source","words","stop_ratio","non_ascii","ee_keywords","status"])
        writer.writeheader(); writer.writerows(stats)


def save_splits(chunks, output_dir): # 90/10 train/val split
    split = int(len(chunks) * 0.9)
    with open(output_dir / "train.json", "w") as f:
        json.dump(chunks[:split], f)
    with open(output_dir / "val.json", "w") as f:
        json.dump(chunks[split:], f)




# Pipeline:
def main():
    raw_dir      = OUTPUT_DIR / "raw"
    filtered_dir = OUTPUT_DIR / "filtered"

    print("Fetching Wikipedia...");  fetch_wikipedia(WIKIPEDIA_ARTICLES, raw_dir)
    print("Fetching web pages...");  fetch_webpages(WEB_URLS, raw_dir)
    print("Fetching seed papers..."); fetch_seed_pdfs(SEED_DOIS, raw_dir)
    print("Citation driven strategy (demo)..."); fetch_papers_from_seeds(SEED_DOIS, raw_dir)

    items = extract_texts(raw_dir)

    stats = []
    items = filter_docs(items, filtered_dir, stats)
    items = deduplicate(items, filtered_dir, stats)
    print(f"After dedup: {len(items)} docs")

    chunks = chunk_docs(items)
    total_tokens = sum(len(c) for c in chunks)
    print(f"Tokens: {total_tokens} | Chunks Total: {len(chunks)} | "
          f"Chunks Train: {int(len(chunks)*0.9)} | Chunks Val: {len(chunks) - int(len(chunks)*0.9)}")

    save_splits(chunks, OUTPUT_DIR)
    save_stats(stats, OUTPUT_DIR)
    print(f"Per-document stats -> {OUTPUT_DIR/'corpus_stats.csv'}")

if __name__ == "__main__":
    main()