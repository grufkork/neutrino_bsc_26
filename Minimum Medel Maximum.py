#ChatGPT har använts för att utveckla denna kod.
"""
Tre separata träningskörningar med bästa seed från tidigare sweep:
- target = min
- target = mean = (min + max) / 2
- target = max

Upplägg:
- Hela omega-intervallet används (ingen omega < q-filtrering)
- q = 250 MeV används som valideringskurva
- Alla andra q används i träning, inklusive q = 75 MeV
- Samma bästa seed från tidigare sweep återanvänds
- Samma modelltyp/hyperparametrar som bästa modellen återanvänds
- Early stopping på q = 250, precis som tidigare
- 15 figurer sparas: en per responskurva och target-mode
  (5 responskurvor x 3 target-modes)
  Varje figur visar prediction vs true på q = 250 MeV
- Dessutom sparas en checkpoint per target så att modellerna kan laddas senare
"""

from __future__ import annotations
import csv
import hashlib
import json
import math
import os
import random
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn


# ============================================================
# 0. Device + seeds
# ============================================================
# Grundseed som används innan bästa tidigare seed har lästs in.
# Senare i main() återanvänds BEST_SEED från tidigare sweep om den hittas.
BASE_SEED = 20260413


# Sätter slumpfrön i Python, NumPy och PyTorch så att körningen blir reproducerbar.
def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Sätter grundseed direkt vid programstart.
set_global_seed(BASE_SEED)

# Väljer vilken beräkningsenhet som ska användas.
# CUDA prioriteras, därefter Apple MPS och sist CPU.
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

# Skriver ut vald device så användaren vet var träningen körs.
print(f"Using device: {DEVICE}")


# ============================================================
# 1. Global config
# ============================================================
# Mappen där responsfilerna .dat förväntas ligga.
DATA_ROOT = Path(".")

# q-värdet som alltid används som valideringskurva.
VAL_Q = 250

# Huvudmapp för alla resultat från denna körning.
OUTPUT_DIR = Path("output_bestseed_min_mean_max_val250_fullomega")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Sökvägar till sammanfattningar, loggar, plottar och checkpoints.
RUNS_CSV_PATH = OUTPUT_DIR / "run_results.csv"
SUMMARY_JSON_PATH = OUTPUT_DIR / "summary.json"
MANIFEST_OUT_PATH = OUTPUT_DIR / "manifest.json"
LOG_PATH = OUTPUT_DIR / "run_log.txt"
PLOTS_DIR = OUTPUT_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINTS_DIR = OUTPUT_DIR / "checkpoints"
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

# De fem responskurvorna som modellen ska prediktera.
# Ordningen används konsekvent i data, prediktioner, metrik och plottar.
OUTPUT_CURVES = ["R00", "Rt", "Rxy", "Rzz", "R0z"]
NUM_OUTPUTS = len(OUTPUT_CURVES)

# De tre target-lägen som ska tränas separat.
TARGET_MODES = ["min", "mean", "max"]

# Svenska titlar för figurerna.
TARGET_TITLES = {
    "min": "Minimivärde",
    "mean": "Medel",
    "max": "Maximivärde",
}

# Svenska beskrivningar som används i figurernas längre rubriker.
TARGET_DESCRIPTIONS = {
    "min": "minimivärde",
    "mean": "medel",
    "max": "maximivärde",
}

# Regex som identifierar responsfiler och fångar q-värde samt kurvnamn.
FILE_RE = re.compile(r"^CR_q(\d+)_(R00|Rt|Rxy|Rzz|R0z)_.+\.dat$", re.IGNORECASE)

# Mappar olika små/stora bokstavsvarianter till standardiserade kurvnamn.
CANONICAL_CURVE_NAMES = {
    "r00": "R00",
    "rt": "Rt",
    "rxy": "Rxy",
    "rzz": "Rzz",
    "r0z": "R0z",
}

# Fallbacks om metadata/manifest inte hittas automatiskt
# Om tidigare sweep-resultat inte finns används dessa standardvärden.
FALLBACK_BEST_SEED = 20270445
FALLBACK_MODEL_CONFIG = {
    "template_name": "top1_128x6_bestseed",
    "architecture": [128, 128, 128, 128, 128, 128],
    "activation": "gelu",
    "optimizer": "adamw",
    "lr_policy": "fixed",
    "loss_name": "mae",
    "feature_set": "base+logs",
    "normalize": True,
    "unit_system": "MeV",
}

# Möjliga featureuppsättningar som kan byggas från q och omega.
# Den faktiska uppsättningen väljs från tidigare modellkonfiguration eller fallback.
FEATURE_SETS = {
    "base": ["q", "omega"],
    "base+dist": ["q", "omega", "q_minus_omega", "omega_over_q"],
    "base+logs": ["q", "omega", "log1p_q", "log1p_omega"],
    "base+dist+logs": [
        "q",
        "omega",
        "q_minus_omega",
        "omega_over_q",
        "log1p_q",
        "log1p_omega",
    ],
}

# Fasta träningsinställningar
# Dessa styr maximal träningstid, early stopping, learning rate och regularisering.
MAX_EPOCHS = 3000
EARLY_STOP_PATIENCE = 80
MIN_DELTA = 1e-6
BASE_LR = 1e-3
WEIGHT_DECAY = 1e-4

# Parametrar för viktad MAE, om loss_name sätts till weighted_mae.
WEIGHTED_MAE_ALPHA = 4.0
WEIGHTED_MAE_POWER = 1.0


# ============================================================
# 2. Utilities
# ============================================================
# Loggar ett meddelande både till terminalen och till en loggfil med tidsstämpel.
def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# Skriver text till fil på ett atomärt sätt.
# Först skrivs en temporär fil, sedan ersätts målfilen.
def atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


# Skriver CSV-data atomärt med angivna kolumnnamn.
def atomic_write_csv(path: Path, fieldnames: List[str], rows: List[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    os.replace(tmp, path)


# Sparar PyTorch-objekt atomärt, till exempel modellcheckpoints.
def atomic_torch_save(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


# Gör en stabil SHA1-hash av en dictionary.
# Den används för att skapa ett kort run_id från en körningskonfiguration.
def sha1_dict(d: dict) -> str:
    payload = json.dumps(d, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


# Gör ett läsbart namn för arkitekturen, exempelvis 128-128-128-128-128-128.
def architecture_name(layers: List[int]) -> str:
    return "-".join(str(x) for x in layers)


# Räknar antal träningsbara parametrar i modellen.
def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Returnerar första filvägen i candidates som faktiskt finns.
# Om ingen finns returneras None.
def resolve_first_existing(candidates: List[Path]) -> Optional[Path]:
    for path in candidates:
        if path.exists():
            return path
    return None


# Läser JSON från fil om sökvägen finns, annars returneras en tom dictionary.
def load_json_if_exists(path: Optional[Path]) -> dict:
    if path is None:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# Formaterar ett flyttal i vetenskaplig notation med svensk decimal-komma.
def format_sci_sv(x: float, sig_digits: int = 2) -> str:
    decimals = max(sig_digits - 1, 0)
    s = f"{x:.{decimals}e}"
    return s.replace(".", ",")


# ============================================================
# 3. Läs tidigare bästa seed + config
# ============================================================
# Försöker läsa metadata och manifest från en tidigare sweep.
# Om filerna saknas används fallback-värdena ovan.
def load_previous_best_setup() -> dict:
    previous_output_dir = Path("output_top1_128x6_50seeds_trainall_except_val250_fullomega")

    # Försöker hitta metadatafilen antingen i tidigare outputmapp eller i aktuell mapp.
    metadata_path = resolve_first_existing(
        [
            previous_output_dir / "best_model_metadata.json",
            Path("best_model_metadata.json"),
        ]
    )

    # Försöker hitta manifestfilen antingen i tidigare outputmapp eller i aktuell mapp.
    manifest_path = resolve_first_existing(
        [
            previous_output_dir / "manifest.json",
            Path("manifest.json"),
        ]
    )

    # Läser filerna om de hittas.
    metadata = load_json_if_exists(metadata_path)
    manifest = load_json_if_exists(manifest_path)

    # Hämtar tidigare vald modellkonfiguration från manifestet om den finns.
    selected_cfg = manifest.get("selected_model_config", {})

    # Hämtar bästa seed från metadata eller använder fallback.
    best_seed = int(metadata.get("best_seed", FALLBACK_BEST_SEED))

    # Bygger den modellkonfiguration som ska återanvändas.
    # Prioritetsordning: metadata -> manifest -> fallback.
    model_cfg = {
        "template_name": "bestseed_min_mean_max",
        "architecture": metadata.get("architecture", selected_cfg.get("architecture", FALLBACK_MODEL_CONFIG["architecture"])),
        "activation": metadata.get("activation", selected_cfg.get("activation", FALLBACK_MODEL_CONFIG["activation"])),
        "optimizer": selected_cfg.get("optimizer", FALLBACK_MODEL_CONFIG["optimizer"]),
        "lr_policy": selected_cfg.get("lr_policy", FALLBACK_MODEL_CONFIG["lr_policy"]),
        "loss_name": selected_cfg.get("loss_name", FALLBACK_MODEL_CONFIG["loss_name"]),
        "feature_set": metadata.get("feature_set", selected_cfg.get("feature_set", FALLBACK_MODEL_CONFIG["feature_set"])),
        "normalize": bool(metadata.get("normalize", selected_cfg.get("normalize", FALLBACK_MODEL_CONFIG["normalize"]))),
        "unit_system": metadata.get("unit_system", selected_cfg.get("unit_system", FALLBACK_MODEL_CONFIG["unit_system"])),
    }

    # Hämtar learning rate och early stopping-patience från manifest om möjligt.
    base_lr = float(manifest.get("base_lr", BASE_LR))
    early_stop_patience = int(manifest.get("early_stop_patience", EARLY_STOP_PATIENCE))

    # Returnerar både själva konfigurationen och information om vilka filer som användes.
    return {
        "best_seed": best_seed,
        "model_cfg": model_cfg,
        "metadata_path": None if metadata_path is None else str(metadata_path.resolve()),
        "manifest_path": None if manifest_path is None else str(manifest_path.resolve()),
        "base_lr": base_lr,
        "early_stop_patience": early_stop_patience,
    }


# ============================================================
# 4. File loading + curve construction
# ============================================================
# Kontrollerar om en fil matchar det förväntade responsfilformatet.
def is_response_file(path: Path) -> bool:
    return FILE_RE.match(path.name) is not None


# Tolkar filnamnet och returnerar q-värde samt standardiserat kurvnamn.
def parse_filename(path: Path) -> Tuple[int, str]:
    m = FILE_RE.match(path.name)
    if m is None:
        raise ValueError(f"Ogiltigt filnamn: {path.name}")
    q = int(m.group(1))
    curve = CANONICAL_CURVE_NAMES[m.group(2).lower()]
    return q, curve


# Läser en responsfil och returnerar omega, min-kolumnen och max-kolumnen.
def load_single_response_file(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.loadtxt(path)

    # Om filen bara innehåller en rad görs den om till tvådimensionell form.
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    # Hanterar fall där filen verkar vara transponerad.
    if arr.shape[0] == 3 and arr.shape[1] != 3:
        arr = arr.T

    # Minst tre kolumner krävs: omega, min och max.
    if arr.shape[1] < 3:
        raise ValueError(f"Fil {path.name} måste ha minst 3 kolumner, fick shape={arr.shape}")

    # Kolumn 0 är omega, kolumn 1 är minimum och kolumn 2 är maximum.
    omega = arr[:, 0].astype(np.float64)
    y_min = arr[:, 1].astype(np.float64)
    y_max = arr[:, 2].astype(np.float64)
    return omega, y_min, y_max


# Väljer target-vektor från min och max beroende på target_mode.
def select_target_from_minmax(y_min: np.ndarray, y_max: np.ndarray, mode: str) -> np.ndarray:
    if mode == "min":
        return y_min.astype(np.float64)
    if mode == "mean":
        return np.nanmean(np.stack([y_min, y_max], axis=1), axis=1).astype(np.float64)
    if mode == "max":
        return y_max.astype(np.float64)
    raise ValueError(f"Okänd target mode: {mode}")


# Ersätter bara inledande NaN-värden med noll.
# NaN efter första giltiga punkt hanteras senare genom maskning.
def fill_leading_nans_with_zero(y: np.ndarray) -> np.ndarray:
    y = y.copy()
    finite = np.isfinite(y)
    if np.any(finite):
        first_finite = int(np.argmax(finite))
        if first_finite > 0:
            y[:first_finite] = 0.0
    else:
        y[:] = 0.0
    return y


# Uppskattar omega-gridens stegstorlek från befintliga omega-värden.
def infer_zero_padding_step(omega: np.ndarray) -> float:
    diffs = np.diff(np.sort(np.unique(omega)))
    diffs = diffs[np.isfinite(diffs) & (diffs > 1e-12)]
    if len(diffs) == 0:
        return max(float(np.min(omega)), 1.0)
    return float(np.median(diffs))


# Dataklass som samlar all data för en given q-kurva.
@dataclass
class QCurveData:
    q_mev: int
    omega_mev: np.ndarray
    y: np.ndarray
    weights: np.ndarray
    peaks: np.ndarray
    inferred_step_mev: float


# Beräknar vikter relativt varje responskurvas peak.
# Punkter med större relativ respons får högre vikt.
def compute_relative_curve_weights(y: np.ndarray, alpha: float, power: float) -> Tuple[np.ndarray, np.ndarray]:
    peaks = np.max(np.abs(y), axis=0)
    peaks = np.where(peaks < 1e-12, 1.0, peaks)
    rel = np.abs(y) / peaks[None, :]
    weights = 1.0 + alpha * np.power(rel, power)
    return weights.astype(np.float64), peaks.astype(np.float64)


# Läser alla responsfiler och bygger q_data för ett specifikt target-läge.
def build_q_curve_data(data_root: Path, target_mode: str) -> Dict[int, QCurveData]:
    # Hittar alla .dat-filer som matchar responsfilnamnsmönstret.
    files = sorted([p for p in data_root.glob("*.dat") if is_response_file(p)])
    if not files:
        raise FileNotFoundError(
            f"Hittade inga responsfiler i {data_root.resolve()}. "
            f"Förväntade namn som CR_q75_R00_NNLO_GO_450.dat"
        )

    # Läser varje fil och grupperar data efter q-värde och responskurva.
    grouped: Dict[int, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {}
    for path in files:
        q, curve = parse_filename(path)
        omega, y_min, y_max = load_single_response_file(path)
        response = select_target_from_minmax(y_min, y_max, target_mode)
        grouped.setdefault(q, {})[curve] = (omega, response)

    # Kontroll att validerings-q finns i materialet.
    if VAL_Q not in grouped:
        raise ValueError(f"Hittade inga filer för valideringskurvan q={VAL_Q} MeV")

    q_data: Dict[int, QCurveData] = {}

    # Bygger en komplett femkurvematrix för varje q.
    for q in sorted(grouped.keys()):
        curves = grouped[q]
        missing_curves = [c for c in OUTPUT_CURVES if c not in curves]
        if missing_curves:
            raise ValueError(f"q={q} saknar kurvor: {missing_curves}")

        omega_ref = None
        y_cols = []

        # Läser kurvor i den fasta ordningen OUTPUT_CURVES.
        for curve_name in OUTPUT_CURVES:
            omega, y = curves[curve_name]
            y = fill_leading_nans_with_zero(y)

            # Första kurvans omega-grid blir referens för detta q.
            if omega_ref is None:
                omega_ref = omega.copy()
            else:
                # Alla fem responskurvor för samma q måste ha samma omega-grid.
                if len(omega) != len(omega_ref) or not np.allclose(omega, omega_ref, rtol=0.0, atol=1e-9):
                    raise ValueError(
                        f"Omega-grid skiljer sig mellan kurvor för q={q}. "
                        "Skriptet antar samma omega-grid för alla 5 kurvor."
                    )

            y_cols.append(y)

        # Stackar fem responskurvor till en matris med shape: antal omega-punkter x 5.
        omega_ref = np.asarray(omega_ref, dtype=np.float64)
        y_mat = np.stack(y_cols, axis=1)

        # Tar bort rader där omega eller någon respons inte är ändlig.
        mask = np.isfinite(omega_ref) & np.all(np.isfinite(y_mat), axis=1)
        omega_clean = omega_ref[mask]
        y_clean = y_mat[mask]

        if len(omega_clean) == 0:
            raise ValueError(f"Inga giltiga datapunkter kvar för q={q}")

        # Skapar zero-padding från omega=0 upp till första befintliga omega-värde.
        step = infer_zero_padding_step(omega_clean)
        omega_min = float(np.min(omega_clean))

        if omega_min > 1e-12:
            omega_zeros = np.arange(0.0, omega_min, step, dtype=np.float64)
            omega_zeros = omega_zeros[omega_zeros < omega_min - 1e-12]
        else:
            omega_zeros = np.empty((0,), dtype=np.float64)

        # Responsvärdena i paddingområdet sätts till noll för alla fem kurvor.
        y_zeros = np.zeros((len(omega_zeros), NUM_OUTPUTS), dtype=np.float64)

        # Hela intervallet: ingen omega<q-filtering här
        # Kombinerar zero-padding och faktisk data utan att filtrera bort omega >= q.
        omega_aug = np.concatenate([omega_zeros, omega_clean], axis=0)
        y_aug = np.concatenate([y_zeros, y_clean], axis=0)

        # Beräknar vikter och peak-värden för den utökade kurvan.
        weights, peaks = compute_relative_curve_weights(
            y_aug,
            alpha=WEIGHTED_MAE_ALPHA,
            power=WEIGHTED_MAE_POWER,
        )

        # Sparar allt för detta q-värde.
        q_data[q] = QCurveData(
            q_mev=q,
            omega_mev=omega_aug,
            y=y_aug,
            weights=weights,
            peaks=peaks,
            inferred_step_mev=step,
        )

    return q_data


# ============================================================
# 5. Split helper
# ============================================================
# Skapar tränings- och valideringssplitten.
# q=250 används för validering och alla andra q används för träning.
def build_single_split(q_data: Dict[int, QCurveData]) -> dict:
    available_qs = sorted(q_data.keys())

    if VAL_Q not in available_qs:
        raise ValueError(f"Validerings-q={VAL_Q} saknas")

    train_qs = [q for q in available_qs if q != VAL_Q]
    if not train_qs:
        raise ValueError("Inga tränings-q återstår efter att valideringskurvan tagits bort")

    return {
        "train_qs": train_qs,
        "val_q": VAL_Q,
    }


# ============================================================
# 6. Features + data manager
# ============================================================
# Konverterar energi från MeV till valt enhetssystem.
def convert_energy(x_mev: float, unit_system: str) -> float:
    if unit_system == "MeV":
        return float(x_mev)
    if unit_system == "GeV":
        return float(x_mev) / 1000.0
    raise ValueError(f"Okänt enhetssystem: {unit_system}")


# Bygger en featurevektor från q och omega.
# Vilka features som används styrs av feature_names.
def build_feature_vector(q_mev: float, omega_mev: float, feature_names: List[str], unit_system: str) -> List[float]:
    q = convert_energy(q_mev, unit_system)
    omega = convert_energy(omega_mev, unit_system)
    eps = 1e-12

    # Beräknar alla möjliga features och väljer sedan ut dem som efterfrågas.
    values = {
        "q": q,
        "omega": omega,
        "q_minus_omega": q - omega,
        "omega_over_q": 0.0 if abs(q) < eps else omega / q,
        "log1p_q": math.log1p(max(q, 0.0)),
        "log1p_omega": math.log1p(max(omega, 0.0)),
    }
    return [float(values[name]) for name in feature_names]


# Klass som ansvarar för att omvandla q_data till PyTorch-tensorer,
# normalisera data och skapa dataset för träning och validering.
class SplitDataManager:
    def __init__(
        self,
        q_data: Dict[int, QCurveData],
        feature_set_name: str,
        normalize: bool,
        unit_system: str,
        device: torch.device,
    ):
        # Sparar indata och inställningar för featurekonstruktion.
        self.q_data = q_data
        self.feature_set_name = feature_set_name
        self.feature_names = FEATURE_SETS[feature_set_name]
        self.normalize = bool(normalize)
        self.unit_system = unit_system
        self.device = device

        # Normaliseringsstatistik skapas i configure().
        self.x_mean = None
        self.x_std = None
        self.y_mean = None
        self.y_std = None

    # Samlar features, targets, vikter och q-id för en lista q-värden.
    def _collect_for_qs(self, q_list: List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        xs, ys, ws, q_ids = [], [], [], []
        for q in q_list:
            pack = self.q_data[q]
            for i in range(len(pack.omega_mev)):
                x = build_feature_vector(q, float(pack.omega_mev[i]), self.feature_names, self.unit_system)
                xs.append(x)
                ys.append(pack.y[i].tolist())
                ws.append(pack.weights[i].tolist())
                q_ids.append(q)

        # Konverterar listor till NumPy-arrayer med tydliga datatyper.
        X = np.asarray(xs, dtype=np.float32)
        Y = np.asarray(ys, dtype=np.float32)
        W = np.asarray(ws, dtype=np.float32)
        QID = np.asarray(q_ids, dtype=np.int32)
        return X, Y, W, QID

    # Förbereder tränings- och valideringsdata samt beräknar normalisering.
    def configure(self, train_qs: List[int], val_q: int) -> None:
        # Samlar rådata för träning och validering.
        X_train_raw, Y_train_raw, W_train, Q_train = self._collect_for_qs(train_qs)
        X_val_raw, Y_val_raw, W_val, Q_val = self._collect_for_qs([val_q])

        # Flyttar allt till vald PyTorch-device.
        X_train_raw = torch.tensor(X_train_raw, dtype=torch.float32, device=self.device)
        Y_train_raw = torch.tensor(Y_train_raw, dtype=torch.float32, device=self.device)
        W_train = torch.tensor(W_train, dtype=torch.float32, device=self.device)

        X_val_raw = torch.tensor(X_val_raw, dtype=torch.float32, device=self.device)
        Y_val_raw = torch.tensor(Y_val_raw, dtype=torch.float32, device=self.device)
        W_val = torch.tensor(W_val, dtype=torch.float32, device=self.device)

        # Normaliseringsparametrar beräknas endast från träningsdata.
        if self.normalize:
            self.x_mean = X_train_raw.mean(dim=0, keepdim=True)
            self.x_std = X_train_raw.std(dim=0, keepdim=True, unbiased=False)
            self.y_mean = Y_train_raw.mean(dim=0, keepdim=True)
            self.y_std = Y_train_raw.std(dim=0, keepdim=True, unbiased=False)

            # Skyddar mot division med noll eller nästan noll.
            self.x_std = torch.where(self.x_std < 1e-12, torch.ones_like(self.x_std), self.x_std)
            self.y_std = torch.where(self.y_std < 1e-12, torch.ones_like(self.y_std), self.y_std)
        else:
            # Om normalisering är avstängd används identitetstransform.
            self.x_mean = torch.zeros((1, X_train_raw.shape[1]), dtype=torch.float32, device=self.device)
            self.x_std = torch.ones((1, X_train_raw.shape[1]), dtype=torch.float32, device=self.device)
            self.y_mean = torch.zeros((1, Y_train_raw.shape[1]), dtype=torch.float32, device=self.device)
            self.y_std = torch.ones((1, Y_train_raw.shape[1]), dtype=torch.float32, device=self.device)

        # Sparar träningsdata i modellens inputskala och targets i rå skala.
        self.X_train = self.x_to_model_space(X_train_raw)
        self.Y_train_raw = Y_train_raw
        self.W_train = W_train
        self.Q_train = Q_train

        # Sparar valideringsdata på samma sätt.
        self.X_val = self.x_to_model_space(X_val_raw)
        self.Y_val_raw = Y_val_raw
        self.W_val = W_val
        self.Q_val = Q_val

    # Normaliserar features till modellens skala.
    def x_to_model_space(self, X_raw: torch.Tensor) -> torch.Tensor:
        return (X_raw - self.x_mean) / self.x_std

    # Omvandlar modellens output tillbaka till rå target-skala.
    def y_from_model_space(self, Y_model: torch.Tensor) -> torch.Tensor:
        return Y_model * self.y_std + self.y_mean

    # Skapar tensorer för ett enskilt q-värde, exempelvis validerings-q.
    def dataset_for_single_q(self, q: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
        pack = self.q_data[q]
        X = np.asarray(
            [build_feature_vector(q, float(w), self.feature_names, self.unit_system) for w in pack.omega_mev],
            dtype=np.float32,
        )
        Y = pack.y.astype(np.float32)
        W = pack.weights.astype(np.float32)
        omega = pack.omega_mev.astype(np.float64)

        # Konverterar till tensorer och normaliserar X.
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        Y_t = torch.tensor(Y, dtype=torch.float32, device=self.device)
        W_t = torch.tensor(W, dtype=torch.float32, device=self.device)
        X_t = self.x_to_model_space(X_t)
        return X_t, Y_t, W_t, omega


# ============================================================
# 7. Model + loss + metrics
# ============================================================
# Skapar aktiveringsfunktion utifrån namn i konfigurationen.
def make_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU()
    if name == "selu":
        return nn.SELU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Okänd activation: {name}")


# Ett fullt kopplat neuralt nätverk som predikterar fem responskurvor samtidigt.
class MultiOutputMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_layers: List[int], output_dim: int, activation: str):
        super().__init__()
        layers: List[nn.Module] = []
        prev = input_dim

        # Bygger alla dolda lager med linjärt lager följt av aktivering.
        for hidden in hidden_layers:
            layers.append(nn.Linear(prev, hidden))
            layers.append(make_activation(activation))
            prev = hidden

        # Sista lagret mappar till antalet responskurvor.
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    # Forward-pass genom nätverket.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# Mean absolute error i rå target-skala.
def mae_loss_raw(y_pred_raw: torch.Tensor, y_true_raw: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(y_pred_raw - y_true_raw))


# Mean squared error i rå target-skala.
def mse_loss_raw(y_pred_raw: torch.Tensor, y_true_raw: torch.Tensor) -> torch.Tensor:
    return torch.mean((y_pred_raw - y_true_raw) ** 2)


# Viktad MAE där varje punkt vägs med de vikter som byggdes från responskurvorna.
def weighted_mae_loss_raw(y_pred_raw: torch.Tensor, y_true_raw: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    err = torch.abs(y_pred_raw - y_true_raw)
    per_curve = (weights * err).sum(dim=0) / (weights.sum(dim=0) + 1e-12)
    return per_curve.mean()


# Väljer objektivfunktion utifrån loss_name.
def objective_value(loss_name: str, y_pred_raw: torch.Tensor, y_true_raw: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    if loss_name == "mae":
        return mae_loss_raw(y_pred_raw, y_true_raw)
    if loss_name == "mse":
        return mse_loss_raw(y_pred_raw, y_true_raw)
    if loss_name == "weighted_mae":
        return weighted_mae_loss_raw(y_pred_raw, y_true_raw, weights)
    raise ValueError(f"Okänd loss_name: {loss_name}")


# Beräknar globala och kurvvisa valideringsmått.
def evaluate_tensor(y_true: torch.Tensor, y_pred: torch.Tensor, weights: torch.Tensor) -> dict:
    err = y_pred - y_true
    abs_err = torch.abs(err)
    sq_err = err ** 2

    # Globala mått över alla punkter och alla responskurvor.
    mae = torch.mean(abs_err).item()
    mse = torch.mean(sq_err).item()
    per_curve_wmae = (weights * abs_err).sum(dim=0) / (weights.sum(dim=0) + 1e-12)
    wmae = per_curve_wmae.mean().item()

    # Mått separat för varje responskurva.
    per_curve_mae = torch.mean(abs_err, dim=0).detach().cpu().numpy()
    per_curve_mse = torch.mean(sq_err, dim=0).detach().cpu().numpy()
    per_curve_wmae_np = per_curve_wmae.detach().cpu().numpy()

    return {
        "mae": float(mae),
        "mse": float(mse),
        "weighted_mae": float(wmae),
        "per_curve_mae": per_curve_mae,
        "per_curve_mse": per_curve_mse,
        "per_curve_weighted_mae": per_curve_wmae_np,
        "n_points": int(y_true.shape[0]),
    }


# Kör modellen i eval-läge utan gradienter och skalar tillbaka output till rå target-skala.
def predict_on_dataset(model: nn.Module, dm: SplitDataManager, X: torch.Tensor) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        pred_model = model(X)
        pred_raw = dm.y_from_model_space(pred_model)
    return pred_raw


# Predikterar på ett dataset och returnerar metrik mot sann target.
def evaluate_model_on_dataset(model: nn.Module, dm: SplitDataManager, X: torch.Tensor, Y_raw: torch.Tensor, W: torch.Tensor) -> dict:
    pred_raw = predict_on_dataset(model, dm, X)
    return evaluate_tensor(Y_raw, pred_raw, W)


# ============================================================
# 8. Optimizer
# ============================================================
# Skapar optimizer för träning.
# I denna kod stöds AdamW med global weight decay.
def build_optimizer(model: nn.Module, optimizer_name: str, lr: float) -> torch.optim.Optimizer:
    optimizer_name = optimizer_name.lower()
    if optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    raise ValueError(f"Okänd optimizer: {optimizer_name}")


# ============================================================
# 9. Training
# ============================================================
# Dataklass som beskriver en enskild träningskörning.
@dataclass
class RunConfig:
    target_mode: str
    seed: int
    train_qs: List[int]
    val_q: int
    architecture: List[int]
    activation: str
    optimizer: str
    lr_policy: str
    base_lr: float
    early_stop_patience: int
    loss_name: str
    feature_set: str
    normalize: bool
    unit_system: str

    # Skapar ett kort ID från hela körningskonfigurationen.
    def run_id(self) -> str:
        return sha1_dict(asdict(self))[:16]


# Tränar en modell för ett target_mode och returnerar bästa modell samt träningshistorik.
def train_one_run(dm: SplitDataManager, cfg: RunConfig) -> dict:
    input_dim = len(FEATURE_SETS[cfg.feature_set])
    model = MultiOutputMLP(
        input_dim=input_dim,
        hidden_layers=cfg.architecture,
        output_dim=NUM_OUTPUTS,
        activation=cfg.activation,
    ).to(DEVICE)

    optimizer = build_optimizer(model, cfg.optimizer, cfg.base_lr)

    # Variabler för early stopping och bästa checkpoint.
    n_train = dm.X_train.shape[0]
    best_state = None
    best_metrics = None
    best_epoch = -1
    best_objective = float("inf")
    epochs_without_improvement = 0
    history = []

    t0 = time.time()

    # Huvudträningsloop.
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()

        # Slumpar träningspunkternas ordning varje epok.
        perm = torch.randperm(n_train, device=DEVICE)
        xb = dm.X_train[perm]
        yb = dm.Y_train_raw[perm]
        wb = dm.W_train[perm]

        # Ett vanligt träningssteg: forward, loss, backward och optimizer step.
        optimizer.zero_grad(set_to_none=True)
        pred_model = model(xb)
        pred_raw = dm.y_from_model_space(pred_model)
        loss = objective_value(cfg.loss_name, pred_raw, yb, wb)
        loss.backward()
        optimizer.step()

        # Utvärderar modellen på valideringskurvan efter varje epok.
        val_metrics = evaluate_model_on_dataset(model, dm, dm.X_val, dm.Y_val_raw, dm.W_val)
        current_objective = float(val_metrics[cfg.loss_name])

        # Sparar träningshistorik för CSV och analys.
        history.append(
            {
                "epoch": epoch,
                "train_objective": float(loss.item()),
                "val_mae": val_metrics["mae"],
                "val_mse": val_metrics["mse"],
                "val_weighted_mae": val_metrics["weighted_mae"],
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
        )

        # Avgör om valideringsmåttet förbättrats tillräckligt mycket.
        improved = (best_objective - current_objective) > MIN_DELTA
        if np.isfinite(current_objective) and improved:
            best_objective = current_objective
            best_epoch = epoch
            best_metrics = val_metrics
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # Early stopping om ingen förbättring har skett på tillräckligt många epoker.
        if epochs_without_improvement >= cfg.early_stop_patience:
            break

    # Om ingen giltig bästa modell hittades avbryts körningen tydligt.
    if best_state is None:
        raise RuntimeError(f"Ingen giltig checkpoint hittades för target_mode={cfg.target_mode}")

    # Laddar tillbaka bästa vikter innan modellen returneras.
    model.load_state_dict(best_state)
    runtime_sec = time.time() - t0

    # Samlar resultatet från träningen.
    result = {
        "model": model,
        "best_epoch": int(best_epoch),
        "epochs_ran": int(len(history)),
        "best_objective": float(best_objective),
        "best_metrics": best_metrics,
        "runtime_sec": float(runtime_sec),
        "history": history,
        "num_params": int(count_parameters(model)),
    }
    return result


# ============================================================
# 10. Checkpoint save
# ============================================================
# Sparar en modellcheckpoint med modellvikter, normaliseringsstatistik och metadata.
def save_model_checkpoint(
    path: Path,
    model: nn.Module,
    dm: SplitDataManager,
    cfg: RunConfig,
    result: dict,
    val_metrics: dict,
) -> None:
    checkpoint = {
        "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "architecture": list(cfg.architecture),
        "activation": cfg.activation,
        "feature_set": cfg.feature_set,
        "feature_names": list(dm.feature_names),
        "normalize": bool(cfg.normalize),
        "unit_system": cfg.unit_system,
        "input_dim": len(FEATURE_SETS[cfg.feature_set]),
        "output_dim": NUM_OUTPUTS,
        "x_mean": dm.x_mean.detach().cpu().clone(),
        "x_std": dm.x_std.detach().cpu().clone(),
        "y_mean": dm.y_mean.detach().cpu().clone(),
        "y_std": dm.y_std.detach().cpu().clone(),
        "seed": int(cfg.seed),
        "target_mode": cfg.target_mode,
        "best_epoch": int(result["best_epoch"]),
        "val_mae": float(val_metrics["mae"]),
        "val_mse": float(val_metrics["mse"]),
        "val_weighted_mae": float(val_metrics["weighted_mae"]),
        "train_qs": list(cfg.train_qs),
        "val_q": int(cfg.val_q),
    }
    atomic_torch_save(path, checkpoint)


# ============================================================
# 11. Plotting
# ============================================================
# Skapar en figur för en responskurva och ett target-läge.
# Figuren jämför sann valideringskurva med modellens prediktion.
def plot_curve_single_target(
    curve_name: str,
    curve_idx: int,
    target_mode: str,
    store: Dict[str, dict],
    val_q: int,
    out_path: Path,
) -> None:
    # Linjebredder för sann kurva och prediktion.
    true_lw = 3.2
    pred_lw = 3.2

    # Fontstorlekar för olika delar av figuren.
    suptitle_fs = 18
    title_fs = 15
    label_fs = 14
    legend_fs = 12
    tick_fs = 12

    # Skapar en figur med en enda axel.
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Hämtar omega, sann respons, predikterad respons och metrik för valt target-läge.
    omega = store[target_mode]["val_omega"]
    y_true = store[target_mode]["val_true"][:, curve_idx]
    y_pred = store[target_mode]["val_pred"][:, curve_idx]
    metrics = store[target_mode]["val_metrics"]
    mae_text = format_sci_sv(metrics["per_curve_mae"][curve_idx], sig_digits=2)

    # Ritar sann kurva och prediktion i samma figur.
    ax.plot(omega, y_true, linewidth=true_lw, label="Sann")
    ax.plot(omega, y_pred, linewidth=pred_lw, label="Förutsägelse")

    # Sätter titel, axelrubriker, tickstorlek, grid och legend.
    ax.set_title(
        f"{curve_name} | {TARGET_TITLES[target_mode]} | q={val_q} MeV\n"
        f"MAE={mae_text}",
        fontsize=title_fs,
    )
    ax.set_xlabel(r"$\omega$ [MeV]", fontsize=label_fs)
    ax.set_ylabel(r"Respons [GeV$^{-1}$]", fontsize=label_fs)
    ax.tick_params(axis="both", labelsize=tick_fs)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=legend_fs)

    # Övergripande rubrik som förklarar exakt vad figuren visar.
    fig.suptitle(
        f"{curve_name}: Förutsägelse jämfört med sann kurva på valideringskurvan q={val_q} MeV "
        f"för {TARGET_DESCRIPTIONS[target_mode]}",
        fontsize=suptitle_fs,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 12. Main
# ============================================================
# Huvudfunktionen som styr hela körningen från tidigare setup till sparade resultat.
def main() -> None:
    # Läser bästa tidigare seed och modellkonfiguration.
    prev = load_previous_best_setup()

    # Plockar ut effektiv seed, modellkonfiguration, learning rate och patience.
    BEST_SEED = int(prev["best_seed"])
    MODEL_CFG = dict(prev["model_cfg"])
    effective_base_lr = float(prev["base_lr"])
    effective_patience = int(prev["early_stop_patience"])

    # Manifestet dokumenterar alla centrala inställningar för körningen.
    manifest = {
        "data_root": str(DATA_ROOT.resolve()),
        "val_q": VAL_Q,
        "best_seed_reused": BEST_SEED,
        "previous_metadata_path": prev["metadata_path"],
        "previous_manifest_path": prev["manifest_path"],
        "model_cfg": MODEL_CFG,
        "target_modes": TARGET_MODES,
        "max_epochs": MAX_EPOCHS,
        "early_stop_patience": effective_patience,
        "base_lr": effective_base_lr,
        "min_delta": MIN_DELTA,
        "weight_decay": WEIGHT_DECAY,
        "full_interval": True,
        "omega_constraint": None,
    }
    atomic_write_text(MANIFEST_OUT_PATH, json.dumps(manifest, indent=2, ensure_ascii=False))

    # Loggar vilken seed och modell som används.
    log(f"Återanvänder bästa seed från tidigare körning: {BEST_SEED}")
    log(
        f"Modell: arch={architecture_name(MODEL_CFG['architecture'])} | "
        f"activation={MODEL_CFG['activation']} | "
        f"feature_set={MODEL_CFG['feature_set']} | "
        f"optimizer={MODEL_CFG['optimizer']}"
    )

    # run_rows blir tabellrader till CSV, prediction_store sparar data som behövs för plottar.
    run_rows = []
    prediction_store: Dict[str, dict] = {}

    t_global = time.time()

    # Kör tre separata träningskörningar: min, mean och max.
    for target_mode in TARGET_MODES:
        log("=" * 80)
        log(f"Startar target_mode={target_mode}")

        # Sätter samma bästa seed inför varje target-mode för jämförbarhet.
        set_global_seed(BEST_SEED)

        # Bygger data för valt target_mode och skapar train/val-split.
        q_data = build_q_curve_data(DATA_ROOT, target_mode=target_mode)
        split = build_single_split(q_data)

        train_qs = split["train_qs"]
        val_q = split["val_q"]

        # Loggar vilka q-värden som hittades och hur splitten ser ut.
        log("Laddade data för q-värden: " + ", ".join(str(q) for q in sorted(q_data.keys())))
        log(f"Train qs: {train_qs}")
        log(f"Validation q: {val_q}")

        # Skapar data manager med återanvänd modellkonfiguration.
        dm = SplitDataManager(
            q_data=q_data,
            feature_set_name=MODEL_CFG["feature_set"],
            normalize=MODEL_CFG["normalize"],
            unit_system=MODEL_CFG["unit_system"],
            device=DEVICE,
        )
        dm.configure(train_qs=train_qs, val_q=val_q)

        # Hämtar valideringsdatasetet för q=250.
        X_val, Y_val, W_val, omega_val = dm.dataset_for_single_q(val_q)

        # Skapar körningskonfiguration för just detta target_mode.
        cfg = RunConfig(
            target_mode=target_mode,
            seed=BEST_SEED,
            train_qs=train_qs,
            val_q=val_q,
            architecture=list(MODEL_CFG["architecture"]),
            activation=MODEL_CFG["activation"],
            optimizer=MODEL_CFG["optimizer"],
            lr_policy=MODEL_CFG["lr_policy"],
            base_lr=effective_base_lr,
            early_stop_patience=effective_patience,
            loss_name=MODEL_CFG["loss_name"],
            feature_set=MODEL_CFG["feature_set"],
            normalize=bool(MODEL_CFG["normalize"]),
            unit_system=MODEL_CFG["unit_system"],
        )

        # Tränar modellen och hämtar den bästa modellen.
        result = train_one_run(dm, cfg)
        model = result["model"]

        # Utvärderar bästa modellen på valideringskurvan och sparar prediktionerna.
        val_metrics = evaluate_model_on_dataset(model, dm, X_val, Y_val, W_val)
        val_pred = predict_on_dataset(model, dm, X_val).detach().cpu().numpy()
        val_true = Y_val.detach().cpu().numpy()

        # Sparar checkpoint för detta target_mode.
        ckpt_path = CHECKPOINTS_DIR / f"bestseed_target_{target_mode}.pt"
        save_model_checkpoint(
            path=ckpt_path,
            model=model,
            dm=dm,
            cfg=cfg,
            result=result,
            val_metrics=val_metrics,
        )

        # Lagrar allt som behövs senare för att skapa figurer.
        prediction_store[target_mode] = {
            "val_pred": val_pred,
            "val_true": val_true,
            "val_omega": omega_val,
            "val_metrics": val_metrics,
            "checkpoint_path": str(ckpt_path.resolve()),
        }

        # Bygger en rad med sammanfattning av körningen till run_results.csv.
        row = {
            "run_id": cfg.run_id(),
            "target_mode": target_mode,
            "seed": BEST_SEED,
            "architecture": architecture_name(cfg.architecture),
            "activation": cfg.activation,
            "optimizer": cfg.optimizer,
            "lr_policy": cfg.lr_policy,
            "base_lr": cfg.base_lr,
            "early_stop_patience": cfg.early_stop_patience,
            "loss_name": cfg.loss_name,
            "feature_set": cfg.feature_set,
            "normalize": cfg.normalize,
            "unit_system": cfg.unit_system,
            "train_qs": ",".join(str(q) for q in train_qs),
            "val_q": val_q,
            "num_params": result["num_params"],
            "best_epoch": result["best_epoch"],
            "epochs_ran": result["epochs_ran"],
            "runtime_sec": result["runtime_sec"],
            "val_mae": val_metrics["mae"],
            "val_mse": val_metrics["mse"],
            "val_weighted_mae": val_metrics["weighted_mae"],
            "checkpoint_path": str(ckpt_path.resolve()),
        }
        run_rows.append(row)

        # Skriver CSV efter varje target_mode så delresultat finns även om senare körning avbryts.
        atomic_write_csv(RUNS_CSV_PATH, list(run_rows[0].keys()), run_rows)

        # Loggar färdig körning för detta target_mode.
        log(
            f"Klart target_mode={target_mode} | seed={BEST_SEED} | "
            f"best_epoch={result['best_epoch']} | "
            f"val_MAE={val_metrics['mae']:.6e} | "
            f"checkpoint={ckpt_path.name}"
        )

    # Spara 15 figurer, en per responskurva och target-mode
    # För varje target-mode skapas en undermapp och fem figurer.
    for target_mode in TARGET_MODES:
        target_plot_dir = PLOTS_DIR / target_mode
        target_plot_dir.mkdir(parents=True, exist_ok=True)

        for curve_idx, curve_name in enumerate(OUTPUT_CURVES):
            out_path = target_plot_dir / f"{curve_name}_q{VAL_Q}_{target_mode}_prediction_vs_true.png"
            plot_curve_single_target(
                curve_name=curve_name,
                curve_idx=curve_idx,
                target_mode=target_mode,
                store=prediction_store,
                val_q=VAL_Q,
                out_path=out_path,
            )
            log(f"Sparade figur: {out_path}")

    # Sammanfattning som sparas i JSON-format.
    summary = {
        "best_seed_reused": BEST_SEED,
        "val_q": VAL_Q,
        "target_modes": TARGET_MODES,
        "results": run_rows,
        "plots_dir": str(PLOTS_DIR.resolve()),
    }
    atomic_write_text(SUMMARY_JSON_PATH, json.dumps(summary, indent=2, ensure_ascii=False))

    # Loggar slutstatus och viktiga sökvägar.
    total_elapsed = time.time() - t_global
    log(f"KLART | total_elapsed_sec={total_elapsed:.2f}")
    log(f"CSV: {RUNS_CSV_PATH}")
    log(f"Summary: {SUMMARY_JSON_PATH}")
    log(f"Plots: {PLOTS_DIR}")
    log(f"Checkpoints: {CHECKPOINTS_DIR}")

    # Skriver en kompakt sammanfattning i terminalen.
    print("\nSammanfattning:")
    for row in run_rows:
        print(
            f"target={row['target_mode']:>4s} | "
            f"seed={row['seed']} | "
            f"best_epoch={row['best_epoch']} | "
            f"val_MAE={row['val_mae']:.6e} | "
            f"checkpoint={row['checkpoint_path']}"
        )
    print("Plots saved in:", PLOTS_DIR.resolve())


# Kör main() bara om filen körs direkt och inte när den importeras.
if __name__ == "__main__":
    main()
