#ChatGPT har använts för att utveckla denna kod.
"""
Tre träningskörningar med exakt samma NN som i den givna koden,
men med tre olika kombinationer för responserna:

1) other4=min,  R0z=min
2) other4=max,  R0z=max
3) other4=mean, R0z=mean

Definition:
- "other4" = R00, Rt, Rxy, Rzz
- R0z behandlas separat

Datans format per fil:
- kolumn 0: omega [MeV]
- kolumn 1: minimum response value
- kolumn 2: maximum response value

Mean definieras som:
- mean = 0.5 * (col2 + col3)

Precis som i din nuvarande kod:
- validering sker på hela q=250-kurvan
- träning sker på alla övriga q
- samma zero-padding från omega=0 upp till lägsta omega i kurvan
- samma nätverksarkitektur och träningsinställningar
- seed sätts till 20270445

Efter varje träning:
- modellen sparas
- valideringsmått sparas
- differentiella tvärsnittet beräknas för valt E_nu och theta
- kurvor sparas och plottas

Generell kinematik för neutrino scattering med massiv utgående lepton:

Låt
    ε  = E_nu
    ω  = energy transfer
    ε' = ε - ω
    m_l = massan för den utgående leptonen
    k  = ε                  (inkommande neutrino antas masslös)
    k' = sqrt(ε'^2 - m_l^2)

Tre-momentum transfer:
    q(ω) = sqrt(k^2 + k'^2 - 2 k k' cos(theta))

Space-like four-momentum transfer:
    Q^2 = q^2 - ω^2

Leptonic factors:
    v00 = 2 ε ε' [1 + (k'/ε') cos(theta)]

    vzz = (ω^2/q^2)(m_l^2 + v00)
          + (m_l^2/q^2)[m_l^2 + 2ω(ε + ε') + q^2]

    v0z = (ω/q)(m_l^2 + v00)
          + m_l^2(ε + ε')/q

    vxx = Q^2 + [Q^2/(2q^2)](m_l^2 + v00)
          - (m_l^2/q^2)[m_l^2/2 + ω(ε + ε')]

    vxy = Q^2(ε + ε')/q - m_l^2 ω/q

Differentiellt tvärsnitt för neutrino:
    dσ/(dΩ dε') = G_F^2 / (8π^2) * (k'/ε) *
                  [ v00 R00 + vzz Rzz - v0z R0z + vxx Rt + vxy Rxy ]


Det ges error om vald (E_nu, theta)-kinematik ligger utanför
tränad q-range, om inte ALLOW_Q_EXTRAPOLATION = True.
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
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn


# ============================================================
# 0. Device + fixed seed
# ============================================================
# En fast seed gör att slumpmoment som viktinitiering och permutationer
# blir reproducerbara så långt hårdvaran och backend tillåter.
FIXED_SEED = 20270445


# Sätter samma seed i Python random, NumPy och PyTorch.
# Om CUDA används sätts även seed för alla CUDA-enheter.
def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Seed sätts direkt vid start så att allt efter detta blir så reproducerbart som möjligt.
set_global_seed(FIXED_SEED)

# Väljer beräkningsenhet i prioriteringsordning:
# 1. CUDA-GPU om den finns,
# 2. Apple Silicon MPS om den finns,
# 3. CPU som fallback.
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

# Skriver ut vilken enhet och seed som används, så körningen blir lättare att spåra.
print(f"Using device: {DEVICE}")
print(f"Using fixed seed: {FIXED_SEED}")


# ============================================================
# 1. User settings
# ============================================================
# Fysikaliska inställningar som användaren kan ändra:
# inkommande neutrinoenergi och spridningsvinkel.
E_NU_MEV = 620.0
THETA_DEG = 36.0

# Styr om formeln ska tolkas som neutrino eller antineutrino.
# Detta påverkar tecknet framför vxy*Rxy-termen senare.
IS_NEUTRINO = True

# Antal punkter i den täta omega-grid som används för tvärsnittsberäkning och plottning.
N_PLOT_POINTS = 1200

# Om False stoppas körningen när vald kinematik kräver q-värden utanför träningsintervallet.
# Om True tillåts modellen extrapolera i q.
ALLOW_Q_EXTRAPOLATION = False

# Styr om en gemensam overlay-plot ska skapas utöver panel- och bandplottar.
MAKE_OVERLAY_PLOT = True

# Styr vilken enhet som används i plottarna för tvärsnittet.
PLOT_IN_1E38_CM2_PER_SR_PER_GEV = True


# ============================================================
# 2. Global config
# ============================================================
# DATA_ROOT pekar på katalogen där .dat-filerna för responskurvorna förväntas ligga.
DATA_ROOT = Path(".")

# OUTPUT_DIR är huvudkatalogen där alla resultat från körningen sparas.
OUTPUT_DIR = Path("output_diffxs_3comb_seed20270445")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Sökvägar till övergripande metadata, loggar, sammanfattningar och figurer.
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"
LOG_PATH = OUTPUT_DIR / "run_log.txt"
SUMMARY_JSON_PATH = OUTPUT_DIR / "summary.json"
COMBINED_CSV_PATH = OUTPUT_DIR / "all_differential_cross_sections.csv"
PANEL_PLOT_PATH = OUTPUT_DIR / "differential_cross_sections_panel.png"
OVERLAY_PLOT_PATH = OUTPUT_DIR / "differential_cross_sections_overlay.png"
BAND_PLOT_PATH = OUTPUT_DIR / "differential_cross_sections_band.png"

# Namn och ordning för modellens fem utgångskurvor.
# Ordningen är viktig eftersom indexeringen används senare i tvärsnittsformeln.
OUTPUT_CURVES = ["R00", "Rt", "Rxy", "Rzz", "R0z"]
NUM_OUTPUTS = len(OUTPUT_CURVES)
CURVE_TO_INDEX = {name: i for i, name in enumerate(OUTPUT_CURVES)}

# q-värdet som reserveras som valideringskurva.
VAL_Q = 250

# Reguljärt uttryck som identifierar responsfiler och extraherar q samt kurvnamn.
FILE_RE = re.compile(r"^CR_q(\d+)_(R00|Rt|Rxy|Rzz|R0z)_.+\.dat$", re.IGNORECASE)

# Konfiguration för den neurala nätverksmodellen.
# Detta samlar arkitektur, aktivering, optimizer, loss och featureval på ett ställe.
SELECTED_MODEL_CONFIG = {
    "template_name": "top1_128x6_fixedseed_3comb",
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
# Den valda uppsättningen i SELECTED_MODEL_CONFIG är "base+logs".
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

# Träningshyperparametrar.
# Early stopping bryter träningen om valideringsmåttet inte förbättras tillräckligt länge.
MAX_EPOCHS = 3000
EARLY_STOP_PATIENCE = 80
MIN_DELTA = 1e-6
BASE_LR = 1e-3
WEIGHT_DECAY = 1e-4

# Parametrar för viktad MAE om den loss-funktionen används.
# Vikterna gör att punkter nära stora responsvärden kan få större betydelse.
WEIGHTED_MAE_ALPHA = 4.0
WEIGHTED_MAE_POWER = 1.0

# De tre kombinationerna som ska köras.
# other4 gäller R00, Rt, Rxy och Rzz, medan R0z kan styras separat.
COMBINATIONS = {
    "other4_min__R0z_min": {"other4": "min", "R0z": "min"},
    "other4_max__R0z_max": {"other4": "max", "R0z": "max"},
    "other4_mean__R0z_mean": {"other4": "mean", "R0z": "mean"},
}

# Etiketter som används i figurerna för att ge kortare och tydligare namn.
COMBINATION_PLOT_LABELS = {
    "other4_min__R0z_min": "Min",
    "other4_max__R0z_max": "Max",
    "other4_mean__R0z_mean": "Medel",
}


# ============================================================
# 3. Physics constants
# ============================================================
# Fysikaliska konstanter som används i tvärsnittsberäkningen och enhetskonverteringen.
G_F = 1.1663787e-11
HBARC_MEV_FM = 197.3269804
FM2_TO_CM2 = 1e-26

# Massan för utgående lepton sätts här till muonmassan.
MUON_MASS_MEV = 105.6583755
FINAL_LEPTON_MASS_MEV = MUON_MASS_MEV


# ============================================================
# 4. Utilities
# ============================================================
# Enkel loggningsfunktion som både skriver till terminalen och till loggfilen.
def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# Skriver text atomärt: först till en temporär fil, sedan ersätts målfilen.
# Det minskar risken för korrupta filer om programmet avbryts mitt i skrivningen.
def atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


# Skriver CSV atomärt med angivna kolumnnamn och rader.
def atomic_write_csv(path: Path, fieldnames: List[str], rows: List[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    os.replace(tmp, path)


# Sparar PyTorch-objekt atomärt, exempelvis modellcheckpoints.
def atomic_torch_save(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


# Skapar en kort SHA1-hash från en dictionary.
# Används för att ge körningar reproducerbara ID:n baserat på deras konfiguration.
def sha1_dict(d: dict) -> str:
    payload = json.dumps(d, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


# Räknar antalet träningsbara parametrar i modellen.
def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================
# 5. File loading + curve construction
# ============================================================
# Kontrollerar om en fil har rätt namnformat för att räknas som responsfil.
def is_response_file(path: Path) -> bool:
    return FILE_RE.match(path.name) is not None


# Tolkar filnamnet och plockar ut q-värdet och responskurvans namn.
def parse_filename(path: Path) -> Tuple[int, str]:
    m = FILE_RE.match(path.name)
    if m is None:
        raise ValueError(f"Ogiltigt filnamn: {path.name}")
    q = int(m.group(1))
    curve = m.group(2)
    return q, curve


# Läser en enskild responsfil och väljer min, max eller medel beroende på target_kind.
def load_single_response_file(path: Path, target_kind: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    target_kind:
        "min"  -> kolumn 1
        "max"  -> kolumn 2
        "mean" -> 0.5*(kolumn 1 + kolumn 2)
    """
    # np.loadtxt läser in filens numeriska innehåll som en NumPy-array.
    arr = np.loadtxt(path)

    # Om filen bara gav en rad görs den om till en tvådimensionell array.
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    # Hanterar fallet där data verkar vara transponerad med tre rader i stället för tre kolumner.
    if arr.shape[0] == 3 and arr.shape[1] != 3:
        arr = arr.T

    # Varje fil måste minst innehålla omega, min-respons och max-respons.
    if arr.shape[1] < 3:
        raise ValueError(f"Fil {path.name} måste ha minst 3 kolumner, fick shape={arr.shape}")

    # Plockar ut omega-grid och de två responskolumnerna.
    omega = arr[:, 0].astype(np.float64)
    col_min = arr[:, 1].astype(np.float64)
    col_max = arr[:, 2].astype(np.float64)

    # Väljer respons enligt kombinationen som körs.
    if target_kind == "min":
        response = col_min
    elif target_kind == "max":
        response = col_max
    elif target_kind == "mean":
        response = 0.5 * (col_min + col_max)
    else:
        raise ValueError(f"Ogiltigt target_kind: {target_kind}")

    return omega, response


# Ersätter inledande NaN-värden med noll fram till första giltiga datapunkt.
# Det används för att få en rimlig start på kurvor som saknar tidiga omega-värden.
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


# Uppskattar gridsteget i omega från befintliga omega-värden.
# Medianen av positiva differenser används för att vara robust mot små avvikelser.
def infer_zero_padding_step(omega: np.ndarray) -> float:
    diffs = np.diff(np.sort(np.unique(omega)))
    diffs = diffs[np.isfinite(diffs) & (diffs > 1e-12)]
    if len(diffs) == 0:
        return max(float(np.min(omega)), 1.0)
    return float(np.median(diffs))


# Dataklass som samlar all kurvdata för ett q-värde.
# y har en kolumn per responskurva och weights har motsvarande träningsvikter.
@dataclass
class QCurveData:
    q_mev: int
    omega_mev: np.ndarray
    y: np.ndarray
    weights: np.ndarray
    peaks: np.ndarray
    inferred_step_mev: float


# Beräknar relativa vikter för varje punkt och responskurva.
# Större absoluta responsvärden får större vikt jämfört med kurvans peak.
def compute_relative_curve_weights(y: np.ndarray, alpha: float, power: float) -> Tuple[np.ndarray, np.ndarray]:
    peaks = np.max(np.abs(y), axis=0)
    peaks = np.where(peaks < 1e-12, 1.0, peaks)
    rel = np.abs(y) / peaks[None, :]
    weights = 1.0 + alpha * np.power(rel, power)
    return weights.astype(np.float64), peaks.astype(np.float64)


# Bestämmer om en specifik kurva ska använda R0z-valet eller other4-valet.
def response_choice_for_curve(curve_name: str, combo_spec: dict) -> str:
    if curve_name == "R0z":
        return combo_spec["R0z"]
    return combo_spec["other4"]


# Bygger hela datasetet grupperat per q för en vald min/max/mean-kombination.
def build_q_curve_data(data_root: Path, combo_spec: dict) -> Dict[int, QCurveData]:
    # Hittar alla .dat-filer i datakatalogen som matchar förväntat filnamnsmönster.
    files = sorted([p for p in data_root.glob("*.dat") if is_response_file(p)])
    if not files:
        raise FileNotFoundError(
            f"Hittade inga responsfiler i {data_root.resolve()}. "
            f"Förväntade namn som CR_q75_R00_NNLO_GO_450.dat"
        )

    # Grupperar filer efter q-värde och kurvnamn.
    grouped: Dict[int, Dict[str, Path]] = {}
    for path in files:
        q, curve = parse_filename(path)
        grouped.setdefault(q, {})[curve] = path

    # Säkerställer att valideringskurvan finns i data.
    if VAL_Q not in grouped:
        raise ValueError(f"Hittade inga filer för valideringskurvan q={VAL_Q} MeV")

    q_data: Dict[int, QCurveData] = {}

    # För varje q läses alla fem responskurvor in och sätts samman till en matris.
    for q in sorted(grouped.keys()):
        curves = grouped[q]
        missing_curves = [c for c in OUTPUT_CURVES if c not in curves]
        if missing_curves:
            raise ValueError(f"q={q} saknar kurvor: {missing_curves}")

        omega_ref = None
        y_cols = []

        # Läser in kurvorna i den fasta ordningen OUTPUT_CURVES.
        for curve_name in OUTPUT_CURVES:
            target_kind = response_choice_for_curve(curve_name, combo_spec)
            omega, y = load_single_response_file(curves[curve_name], target_kind=target_kind)
            y = fill_leading_nans_with_zero(y)

            # Första kurvans omega-grid blir referensgrid för detta q.
            if omega_ref is None:
                omega_ref = omega.copy()
            else:
                # Alla fem kurvor för samma q måste ligga på samma omega-grid.
                if len(omega) != len(omega_ref) or not np.allclose(omega, omega_ref, rtol=0.0, atol=1e-9):
                    raise ValueError(
                        f"Omega-grid skiljer sig mellan kurvor för q={q}. "
                        "Skriptet antar samma omega-grid för alla 5 kurvor."
                    )

            y_cols.append(y)

        # Stackar responskurvorna till shape: antal omega-punkter x antal responskurvor.
        omega_ref = np.asarray(omega_ref, dtype=np.float64)
        y_mat = np.stack(y_cols, axis=1)

        # Tar bort rader där omega eller någon respons inte är ändlig.
        mask = np.isfinite(omega_ref) & np.all(np.isfinite(y_mat), axis=1)
        omega_clean = omega_ref[mask]
        y_clean = y_mat[mask]

        if len(omega_clean) == 0:
            raise ValueError(f"Inga giltiga datapunkter kvar för q={q}")

        # Beräknar omega-steg och lägger till nollpadding från omega=0 upp till första datapunkt.
        step = infer_zero_padding_step(omega_clean)
        omega_min = float(np.min(omega_clean))

        if omega_min > 1e-12:
            omega_zeros = np.arange(0.0, omega_min, step, dtype=np.float64)
            omega_zeros = omega_zeros[omega_zeros < omega_min - 1e-12]
        else:
            omega_zeros = np.empty((0,), dtype=np.float64)

        # Responsen sätts till noll i det artificiella paddingområdet.
        y_zeros = np.zeros((len(omega_zeros), NUM_OUTPUTS), dtype=np.float64)

        # Kombinerar padding och originaldata.
        omega_aug = np.concatenate([omega_zeros, omega_clean], axis=0)
        y_aug = np.concatenate([y_zeros, y_clean], axis=0)

        # Beräknar träningsvikter för den utökade kurvan.
        weights, peaks = compute_relative_curve_weights(
            y_aug,
            alpha=WEIGHTED_MAE_ALPHA,
            power=WEIGHTED_MAE_POWER,
        )

        # Sparar allt för detta q i q_data.
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
# 6. Split helper
# ============================================================
# Skapar tränings-/valideringssplitten.
# q=VAL_Q används som validering och alla övriga q används för träning.
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
# 7. Features + data manager
# ============================================================
# Konverterar energier mellan MeV och GeV beroende på valt enhetssystem.
def convert_energy(x_mev: float, unit_system: str) -> float:
    if unit_system == "MeV":
        return float(x_mev)
    if unit_system == "GeV":
        return float(x_mev) / 1000.0
    raise ValueError(f"Okänt enhetssystem: {unit_system}")


# Bygger en featurevektor från q och omega.
# Vilka features som faktiskt returneras styrs av feature_names.
def build_feature_vector(q_mev: float, omega_mev: float, feature_names: List[str], unit_system: str) -> List[float]:
    q = convert_energy(q_mev, unit_system)
    omega = convert_energy(omega_mev, unit_system)
    eps = 1e-12

    # Alla möjliga featurevärden beräknas här, men bara de valda namnen returneras.
    values = {
        "q": q,
        "omega": omega,
        "q_minus_omega": q - omega,
        "omega_over_q": 0.0 if abs(q) < eps else omega / q,
        "log1p_q": math.log1p(max(q, 0.0)),
        "log1p_omega": math.log1p(max(omega, 0.0)),
    }
    return [float(values[name]) for name in feature_names]


# SplitDataManager ansvarar för att bygga tensorer till PyTorch,
# normalisera features och targets samt skapa dataset för träning, validering och prediktion.
class SplitDataManager:
    def __init__(
        self,
        q_data: Dict[int, QCurveData],
        feature_set_name: str,
        normalize: bool,
        unit_system: str,
        device: torch.device,
    ):
        # Sparar grunddata och inställningar för featurebyggande.
        self.q_data = q_data
        self.feature_set_name = feature_set_name
        self.feature_names = FEATURE_SETS[feature_set_name]
        self.normalize = bool(normalize)
        self.unit_system = unit_system
        self.device = device

        # Normaliseringsstatistik sätts senare i configure().
        self.x_mean = None
        self.x_std = None
        self.y_mean = None
        self.y_std = None

    # Samlar X, Y, vikter och q-id för en lista av q-värden.
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

    # Förbereder tränings- och valideringsdata, inklusive normalisering.
    def configure(self, train_qs: List[int], val_q: int) -> None:
        # Hämtar rådata för träning och validering.
        X_train_raw, Y_train_raw, W_train, Q_train = self._collect_for_qs(train_qs)
        X_val_raw, Y_val_raw, W_val, Q_val = self._collect_for_qs([val_q])

        # Flyttar data till vald PyTorch-enhet.
        X_train_raw = torch.tensor(X_train_raw, dtype=torch.float32, device=self.device)
        Y_train_raw = torch.tensor(Y_train_raw, dtype=torch.float32, device=self.device)
        W_train = torch.tensor(W_train, dtype=torch.float32, device=self.device)

        X_val_raw = torch.tensor(X_val_raw, dtype=torch.float32, device=self.device)
        Y_val_raw = torch.tensor(Y_val_raw, dtype=torch.float32, device=self.device)
        W_val = torch.tensor(W_val, dtype=torch.float32, device=self.device)

        # Normaliseringsstatistik beräknas endast från träningsdata
        # för att undvika att valideringsdata läcker in i träningen.
        if self.normalize:
            self.x_mean = X_train_raw.mean(dim=0, keepdim=True)
            self.x_std = X_train_raw.std(dim=0, keepdim=True)
            self.y_mean = Y_train_raw.mean(dim=0, keepdim=True)
            self.y_std = Y_train_raw.std(dim=0, keepdim=True)

            # Skyddar mot division med nästan noll standardavvikelse.
            self.x_std = torch.where(self.x_std < 1e-12, torch.ones_like(self.x_std), self.x_std)
            self.y_std = torch.where(self.y_std < 1e-12, torch.ones_like(self.y_std), self.y_std)
        else:
            # Om normalisering är avstängd används identitetstransform.
            self.x_mean = torch.zeros((1, X_train_raw.shape[1]), dtype=torch.float32, device=self.device)
            self.x_std = torch.ones((1, X_train_raw.shape[1]), dtype=torch.float32, device=self.device)
            self.y_mean = torch.zeros((1, Y_train_raw.shape[1]), dtype=torch.float32, device=self.device)
            self.y_std = torch.ones((1, Y_train_raw.shape[1]), dtype=torch.float32, device=self.device)

        # Sparar normaliserad X och rå Y för träning.
        self.X_train = self.x_to_model_space(X_train_raw)
        self.Y_train_raw = Y_train_raw
        self.W_train = W_train
        self.Q_train = Q_train

        # Sparar normaliserad X och rå Y för validering.
        self.X_val = self.x_to_model_space(X_val_raw)
        self.Y_val_raw = Y_val_raw
        self.W_val = W_val
        self.Q_val = Q_val

    # Transformerar features från rå skala till modellens normaliserade skala.
    def x_to_model_space(self, X_raw: torch.Tensor) -> torch.Tensor:
        return (X_raw - self.x_mean) / self.x_std

    # Transformerar modellens prediktioner tillbaka till rå respons-skala.
    def y_from_model_space(self, Y_model: torch.Tensor) -> torch.Tensor:
        return Y_model * self.y_std + self.y_mean

    # Bygger ett dataset för ett enda q-värde, exempelvis valideringskurvan.
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

    # Bygger modellinput för en kinematisk kurva där q och omega varierar punkt för punkt.
    def dataset_for_kinematic_curve(self, q_mev_arr: np.ndarray, omega_mev_arr: np.ndarray) -> torch.Tensor:
        q_mev_arr = np.asarray(q_mev_arr, dtype=np.float64)
        omega_mev_arr = np.asarray(omega_mev_arr, dtype=np.float64)

        # q-arrayen och omega-arrayen måste ha samma form eftersom varje punkt kräver ett q och ett omega.
        if q_mev_arr.shape != omega_mev_arr.shape:
            raise ValueError(
                f"q_mev_arr och omega_mev_arr måste ha samma shape, fick "
                f"{q_mev_arr.shape} och {omega_mev_arr.shape}"
            )

        n_features = len(self.feature_names)

        # Hanterar tom input utan att krascha.
        if q_mev_arr.size == 0:
            X = np.empty((0, n_features), dtype=np.float32)
        else:
            X = np.asarray(
                [
                    build_feature_vector(float(qv), float(ov), self.feature_names, self.unit_system)
                    for qv, ov in zip(q_mev_arr, omega_mev_arr)
                ],
                dtype=np.float32,
            )

        # Konverterar till tensor och normaliserar till modellens skala.
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        X_t = self.x_to_model_space(X_t)
        return X_t


# ============================================================
# 8. Model + loss + metrics
# ============================================================
# Skapar aktiveringsfunktionen som används mellan de linjära lagren.
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


# Ett fullt kopplat multi-output MLP.
# Modellen tar featurevektorer som input och predikterar alla fem responskurvor samtidigt.
class MultiOutputMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_layers: List[int], output_dim: int, activation: str):
        super().__init__()
        layers: List[nn.Module] = []
        prev = input_dim
        # Bygger varje dolt lager följt av vald aktiveringsfunktion.
        for hidden in hidden_layers:
            layers.append(nn.Linear(prev, hidden))
            layers.append(make_activation(activation))
            prev = hidden
        # Sista lagret mappar från sista dolda dimensionen till fem outputvärden.
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    # Forward-pass: skickar input genom hela nätverket.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# Vanlig mean absolute error i rå respons-skala.
def mae_loss_raw(y_pred_raw: torch.Tensor, y_true_raw: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(y_pred_raw - y_true_raw))


# Vanlig mean squared error i rå respons-skala.
def mse_loss_raw(y_pred_raw: torch.Tensor, y_true_raw: torch.Tensor) -> torch.Tensor:
    return torch.mean((y_pred_raw - y_true_raw) ** 2)


# Viktad MAE där fel multipliceras med punkt- och kurvspecifika vikter.
def weighted_mae_loss_raw(y_pred_raw: torch.Tensor, y_true_raw: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    err = torch.abs(y_pred_raw - y_true_raw)
    per_curve = (weights * err).sum(dim=0) / (weights.sum(dim=0) + 1e-12)
    return per_curve.mean()


# Väljer vilken loss/objektivfunktion som ska användas utifrån konfigurationsnamnet.
def objective_value(loss_name: str, y_pred_raw: torch.Tensor, y_true_raw: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    if loss_name == "mae":
        return mae_loss_raw(y_pred_raw, y_true_raw)
    if loss_name == "mse":
        return mse_loss_raw(y_pred_raw, y_true_raw)
    if loss_name == "weighted_mae":
        return weighted_mae_loss_raw(y_pred_raw, y_true_raw, weights)
    raise ValueError(f"Okänd loss_name: {loss_name}")


# Beräknar flera mått mellan sann respons och predikterad respons.
def evaluate_tensor(y_true: torch.Tensor, y_pred: torch.Tensor, weights: torch.Tensor) -> dict:
    err = y_pred - y_true
    abs_err = torch.abs(err)
    sq_err = err ** 2

    # Globala mått över alla punkter och kurvor.
    mae = torch.mean(abs_err).item()
    mse = torch.mean(sq_err).item()
    per_curve_wmae = (weights * abs_err).sum(dim=0) / (weights.sum(dim=0) + 1e-12)
    wmae = per_curve_wmae.mean().item()

    # Kurvvisa mått, ett värde per responskurva.
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


# Gör prediktion på ett dataset utan gradientberäkning och konverterar tillbaka till rå skala.
def predict_on_dataset(model: nn.Module, dm: SplitDataManager, X: torch.Tensor) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        pred_model = model(X)
        pred_raw = dm.y_from_model_space(pred_model)
    return pred_raw


# Predikterar och utvärderar modellen på angiven X/Y/W.
def evaluate_model_on_dataset(model: nn.Module, dm: SplitDataManager, X: torch.Tensor, Y_raw: torch.Tensor, W: torch.Tensor) -> dict:
    pred_raw = predict_on_dataset(model, dm, X)
    return evaluate_tensor(Y_raw, pred_raw, W)


# ============================================================
# 9. Optimizer
# ============================================================
# Skapar optimizer för modellträningen.
# I denna kod stöds AdamW, inklusive global weight decay.
def build_optimizer(model: nn.Module, optimizer_name: str, lr: float) -> torch.optim.Optimizer:
    optimizer_name = optimizer_name.lower()
    if optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    raise ValueError(f"Okänd optimizer: {optimizer_name}")


# ============================================================
# 10. Training
# ============================================================
# Dataklass som beskriver en specifik träningskörning.
# Den innehåller både modellval och data-split, så körningen kan identifieras entydigt.
@dataclass
class RunConfig:
    template_name: str
    combo_name: str
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
    seed: int

    # Skapar ett kort run_id från hela konfigurationen.
    def run_id(self) -> str:
        return sha1_dict(asdict(self))[:16]


# Tränar en modell för en kombination och returnerar modell, historik och bästa valideringsresultat.
def train_one_run(dm: SplitDataManager, cfg: RunConfig) -> dict:
    input_dim = len(FEATURE_SETS[cfg.feature_set])
    model = MultiOutputMLP(
        input_dim=input_dim,
        hidden_layers=cfg.architecture,
        output_dim=NUM_OUTPUTS,
        activation=cfg.activation,
    ).to(DEVICE)

    optimizer = build_optimizer(model, cfg.optimizer, cfg.base_lr)

    # Variabler för early stopping och lagring av bästa modell.
    n_train = dm.X_train.shape[0]
    best_state = None
    best_metrics = None
    best_epoch = -1
    best_objective = float("inf")
    epochs_without_improvement = 0
    history = []

    t0 = time.time()

    # Träningsloopen kör upp till MAX_EPOCHS men kan brytas tidigare av early stopping.
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()

        # Slumpar ordningen på träningspunkterna varje epok.
        perm = torch.randperm(n_train, device=DEVICE)
        xb = dm.X_train[perm]
        yb = dm.Y_train_raw[perm]
        wb = dm.W_train[perm]

        # Standard PyTorch-träningssteg: nollställ gradienter, forward, loss, backward, optimizer step.
        optimizer.zero_grad(set_to_none=True)
        pred_model = model(xb)
        pred_raw = dm.y_from_model_space(pred_model)
        loss = objective_value(cfg.loss_name, pred_raw, yb, wb)
        loss.backward()
        optimizer.step()

        # Utvärderar på hela valideringskurvan efter varje epok.
        val_metrics = evaluate_model_on_dataset(model, dm, dm.X_val, dm.Y_val_raw, dm.W_val)
        current_objective = float(val_metrics[cfg.loss_name])

        # Sparar träningshistorik för senare analys och CSV-export.
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

        # Kontrollerar om valideringsmåttet förbättrades mer än MIN_DELTA.
        improved = (best_objective - current_objective) > MIN_DELTA
        if np.isfinite(current_objective) and improved:
            best_objective = current_objective
            best_epoch = epoch
            best_metrics = val_metrics
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # Stoppar träningen om förbättring uteblivit för länge.
        if epochs_without_improvement >= cfg.early_stop_patience:
            break

    # Återställer modellen till bästa sparade vikter, inte nödvändigtvis sista epoken.
    if best_state is not None:
        model.load_state_dict(best_state)

    runtime_sec = time.time() - t0

    # Samlar allt som behövs efter träningen.
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
# 11. Save helpers
# ============================================================
# Sparar träningshistoriken till CSV om historiken inte är tom.
def save_history_csv(history: List[dict], path: Path) -> None:
    if not history:
        return
    fieldnames = list(history[0].keys())
    atomic_write_csv(path, fieldnames, history)


# Sparar modellens vikter, normaliseringsstatistik och metadata för en kombination.
def save_model_checkpoint(
    combo_dir: Path,
    model: nn.Module,
    dm: SplitDataManager,
    cfg: RunConfig,
    result: dict,
    val_metrics: dict,
    combo_spec: dict,
) -> None:
    checkpoint_path = combo_dir / "best_model_state.pt"
    meta_path = combo_dir / "best_model_metadata.json"

    # Checkpointen innehåller allt som behövs för att återskapa modellen och prediktera senare.
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
        "output_curves": list(OUTPUT_CURVES),
        "x_mean": dm.x_mean.detach().cpu().clone(),
        "x_std": dm.x_std.detach().cpu().clone(),
        "y_mean": dm.y_mean.detach().cpu().clone(),
        "y_std": dm.y_std.detach().cpu().clone(),
        "seed": int(cfg.seed),
        "best_epoch": int(result["best_epoch"]),
        "best_val_mae": float(val_metrics["mae"]),
        "best_val_mse": float(val_metrics["mse"]),
        "best_val_weighted_mae": float(val_metrics["weighted_mae"]),
        "train_qs": list(cfg.train_qs),
        "val_q": int(cfg.val_q),
        "combination": combo_spec,
    }

    atomic_torch_save(checkpoint_path, checkpoint)

    # Metadata sparas separat som JSON för att enkelt kunna läsas utan PyTorch.
    metadata = {
        "seed": int(cfg.seed),
        "best_epoch": int(result["best_epoch"]),
        "best_val_mae": float(val_metrics["mae"]),
        "best_val_mse": float(val_metrics["mse"]),
        "best_val_weighted_mae": float(val_metrics["weighted_mae"]),
        "architecture": list(cfg.architecture),
        "activation": cfg.activation,
        "feature_set": cfg.feature_set,
        "normalize": bool(cfg.normalize),
        "unit_system": cfg.unit_system,
        "train_qs": list(cfg.train_qs),
        "val_q": int(cfg.val_q),
        "combination": combo_spec,
        "checkpoint_path": str(checkpoint_path.resolve()),
    }
    atomic_write_text(meta_path, json.dumps(metadata, indent=2, ensure_ascii=False))


# ============================================================
# 12. Kinematics + differential cross section
# ============================================================
# Beräknar kinematiska storheter och leptoniska faktorer för en omega-grid.
def neutrino_kinematics_general(
    E_nu_mev: float,
    theta_deg: float,
    omega_mev: np.ndarray,
    lepton_mass_mev: float,
) -> dict:
    """
    Generell kinematik med massiv utgående lepton.

    Matematiska formler som används:

        ε  = E_nu
        ω  = energy transfer
        ε' = ε - ω
        m_l = lepton_mass_mev

        k  = ε
        k' = sqrt(ε'^2 - m_l^2)

        q = sqrt(k^2 + k'^2 - 2 k k' cos(theta))

        Q^2 = q^2 - ω^2

        v00 = 2 ε ε' [1 + (k'/ε') cos(theta)]

        vzz = (ω^2/q^2)(m_l^2 + v00)
              + (m_l^2/q^2)[m_l^2 + 2ω(ε + ε') + q^2]

        v0z = (ω/q)(m_l^2 + v00)
              + m_l^2(ε + ε')/q

        vxx = Q^2 + [Q^2/(2q^2)](m_l^2 + v00)
              - (m_l^2/q^2)[m_l^2/2 + ω(ε + ε')]

        vxy = Q^2(ε + ε')/q - m_l^2 ω/q

    Tvärsnitt för neutrino:
        dσ/(dΩ dε') = G_F^2 / (8π^2) * (k'/ε) *
                      [ v00 R00 + vzz Rzz - v0z R0z + vxx Rt + vxy Rxy ]
    """
    # Säkerställer att omega är en float64-array och konverterar vinkeln till radianer.
    omega = np.asarray(omega_mev, dtype=np.float64)
    theta_rad = np.deg2rad(theta_deg)

    # Sätter inkommande energi och leptonmassa som skalärer.
    eps_in = float(E_nu_mev)
    m_l = float(lepton_mass_mev)

    # Utgående leptonenergi är inkommande energi minus överförd energi.
    eps_out = eps_in - omega
    k_in = eps_in

    # k' är utgående leptonens rörelsemängd. Negativa värden under roten markeras som NaN.
    kprime_sq = eps_out**2 - m_l**2
    kprime_sq = np.where(kprime_sq >= 0.0, kprime_sq, np.nan)
    kprime = np.sqrt(kprime_sq)

    cos_theta = np.cos(theta_rad)

    # Beräknar tre-rörelsemängdsöverföringen q.
    q_sq = k_in**2 + kprime**2 - 2.0 * k_in * kprime * cos_theta
    q_sq = np.where(q_sq > 0.0, q_sq, np.nan)
    q = np.sqrt(q_sq)

    # Q2 definieras här som q^2 - omega^2.
    Q2 = q_sq - omega**2

    # Beräknar leptoniska faktorer. errstate gör att divisioner med ogiltiga värden ger NaN/inf utan varningsspam.
    with np.errstate(divide="ignore", invalid="ignore"):
        v00 = 2.0 * eps_in * eps_out * (1.0 + (kprime / eps_out) * cos_theta)

        vzz = (
            (omega**2 / q_sq) * (m_l**2 + v00)
            + (m_l**2 / q_sq) * (m_l**2 + 2.0 * omega * (eps_in + eps_out) + q_sq)
        )

        v0z = (
            (omega / q) * (m_l**2 + v00)
            + m_l**2 * (eps_in + eps_out) / q
        )

        vxx = (
            Q2
            + (Q2 / (2.0 * q_sq)) * (m_l**2 + v00)
            - (m_l**2 / q_sq) * (0.5 * m_l**2 + omega * (eps_in + eps_out))
        )

        vxy = (
            Q2 * (eps_in + eps_out) / q
            - m_l**2 * omega / q
        )

    # Mask som anger vilka omega-punkter som är fysiskt giltiga och numeriskt ändliga.
    physical_mask = (
        np.isfinite(omega)
        & np.isfinite(eps_out)
        & np.isfinite(kprime)
        & np.isfinite(q)
        & np.isfinite(Q2)
        & (eps_out > m_l)
        & (kprime >= 0.0)
        & (q > 0.0)
    )

    # Returnerar alla beräknade storheter i en dictionary så senare steg kan använda dem.
    return {
        "omega_mev": omega,
        "theta_deg": theta_deg,
        "theta_rad": theta_rad,
        "eps_in_mev": eps_in,
        "eps_out_mev": eps_out,
        "lepton_mass_mev": m_l,
        "k_in_mev": k_in,
        "kprime_mev": kprime,
        "q_mev": q,
        "q_sq_mev2": q_sq,
        "Q2_mev2": Q2,
        "v00": v00,
        "vzz": vzz,
        "v0z": v0z,
        "vxx": vxx,
        "vxy": vxy,
        "physical_mask": physical_mask,
    }


# Predikterar de fem responsfunktionerna längs den kinematiska kurvan q(omega).
def predict_responses_on_kinematic_curve(
    model: nn.Module,
    dm: SplitDataManager,
    q_mev_arr: np.ndarray,
    omega_mev_arr: np.ndarray,
) -> np.ndarray:
    X = dm.dataset_for_kinematic_curve(q_mev_arr=q_mev_arr, omega_mev_arr=omega_mev_arr)
    pred_raw = predict_on_dataset(model, dm, X)
    return pred_raw.detach().cpu().numpy()


# Beräknar differentiellt tvärsnitt från predikterade responser och kinematik.
def differential_cross_section_general(
    responses_pred: np.ndarray,
    kin: dict,
    is_neutrino: bool = True,
) -> np.ndarray:
    # Plockar ut varje respons från rätt kolumn enligt CURVE_TO_INDEX.
    R00 = responses_pred[:, CURVE_TO_INDEX["R00"]]
    Rt = responses_pred[:, CURVE_TO_INDEX["Rt"]]
    Rxy = responses_pred[:, CURVE_TO_INDEX["Rxy"]]
    Rzz = responses_pred[:, CURVE_TO_INDEX["Rzz"]]
    R0z = responses_pred[:, CURVE_TO_INDEX["R0z"]]

    # Tecknet för Rxy-termen ändras mellan neutrino och antineutrino.
    sign_xy = +1.0 if is_neutrino else -1.0

    # Hakparentesen i tvärsnittsformeln: leptoniska faktorer gånger responsfunktioner.
    bracket = (
        kin["v00"] * R00
        + kin["vzz"] * Rzz
        - kin["v0z"] * R0z
        + kin["vxx"] * Rt
        + sign_xy * kin["vxy"] * Rxy
    )

    # Prefaktor inklusive Fermi-konstanten och k'/epsilon.
    prefactor = (G_F**2 / (8.0 * np.pi**2)) * (kin["kprime_mev"] / kin["eps_in_mev"])
    xs_nat = prefactor * bracket

    # Ogiltiga kinematiska punkter sätts till NaN.
    xs_nat = np.where(kin["physical_mask"], xs_nat, np.nan)
    return xs_nat


# Konverterar tvärsnittet från naturliga MeV-enheter till 10^-38 cm^2/(sr GeV).
def convert_xs_nat_to_1e38_cm2_per_sr_per_GeV(xs_nat: np.ndarray) -> np.ndarray:
    xs_cm2_per_sr_per_mev = xs_nat * (HBARC_MEV_FM**2) * FM2_TO_CM2
    xs_cm2_per_sr_per_gev = xs_cm2_per_sr_per_mev * 1.0e3
    xs_in_1e38_units = xs_cm2_per_sr_per_gev * 1.0e38
    return xs_in_1e38_units


# ============================================================
# 13. Plot helpers
# ============================================================
# Skapar en panelplot med en subplot per kombination.
def plot_panel(all_results: Dict[str, dict], path: Path, plot_in_converted_units: bool) -> None:
    n = len(all_results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), sharex=True)
    if n == 1:
        axes = [axes]

    # Loopar över kombinationerna och ritar respektive tvärsnittskurva i sin panel.
    for ax, (combo_name, result) in zip(axes, all_results.items()):
        omega = result["omega_plot_mev"]

        # Väljer om y-axeln ska vara i konverterade enheter eller naturliga MeV-enheter.
        if plot_in_converted_units:
            y = result["xs_plot_1e38_cm2_per_sr_per_gev"]
            ylabel = r"$d\sigma/(d\Omega\,d\epsilon')\ [10^{-38}\ \mathrm{cm}^2/(\mathrm{sr}\,\mathrm{GeV})]$"
        else:
            y = result["xs_plot_nat"]
            ylabel = r"$d\sigma/(d\Omega\,d\epsilon')\ [\mathrm{MeV}^{-3}]$"

        ax.plot(omega, y, lw=2)
        ax.set_title(COMBINATION_PLOT_LABELS.get(combo_name, combo_name))
        ax.set_xlabel(r"$\omega$ [MeV]")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

        # Lägger in en liten textruta med viktig körningsinformation.
        txt = (
            f"Eν = {E_NU_MEV:.1f} MeV\n"
            f"θ = {THETA_DEG:.1f}°\n"
            f"q-träning: [{result['q_min_train']:.1f}, {result['q_max_train']:.1f}] MeV"
        )
        ax.text(
            0.02, 0.98, txt,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# Skapar en overlay-plot där alla kombinationer ritas i samma figur.
def plot_overlay(all_results: Dict[str, dict], path: Path, plot_in_converted_units: bool) -> None:
    fig, ax = plt.subplots(figsize=(11, 7))

    for combo_name, result in all_results.items():
        omega = result["omega_plot_mev"]
        if plot_in_converted_units:
            y = result["xs_plot_1e38_cm2_per_sr_per_gev"]
            ylabel = r"$d\sigma/(d\Omega\,d\epsilon')\ [10^{-38}\ \mathrm{cm}^2/(\mathrm{sr}\,\mathrm{GeV})]$"
        else:
            y = result["xs_plot_nat"]
            ylabel = r"$d\sigma/(d\Omega\,d\epsilon')\ [\mathrm{MeV}^{-3}]$"

        ax.plot(omega, y, lw=2, label=COMBINATION_PLOT_LABELS.get(combo_name, combo_name))

    ax.set_title(f"Differentiellt tvärsnitt, Eν={E_NU_MEV:.1f} MeV, θ={THETA_DEG:.1f}°")
    ax.set_xlabel(r"$\omega$ [MeV]")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# Skapar en bandplot mellan min- och maxkombinationerna.
def plot_band(all_results: Dict[str, dict], path: Path, plot_in_converted_units: bool) -> None:
    min_name = "other4_min__R0z_min"
    max_name = "other4_max__R0z_max"

    omega_min = all_results[min_name]["omega_plot_mev"]
    omega_max = all_results[max_name]["omega_plot_mev"]

    # Min- och maxkurvor måste ligga på samma omega-grid för att fill_between ska vara direkt giltig.
    if not np.allclose(omega_min, omega_max, rtol=0.0, atol=1e-9):
        raise ValueError("Omega-griden för min- och max-kurvorna skiljer sig och kan inte användas direkt för bandplott.")

    omega = omega_min

    # Väljer y-data och y-label beroende på enhetsval.
    if plot_in_converted_units:
        y_min = all_results[min_name]["xs_plot_1e38_cm2_per_sr_per_gev"]
        y_max = all_results[max_name]["xs_plot_1e38_cm2_per_sr_per_gev"]
        ylabel = r"$d\sigma/(d\Omega\,d\epsilon')\ [10^{-38}\ \mathrm{cm}^2/(\mathrm{sr}\,\mathrm{GeV})]$"
    else:
        y_min = all_results[min_name]["xs_plot_nat"]
        y_max = all_results[max_name]["xs_plot_nat"]
        ylabel = r"$d\sigma/(d\Omega\,d\epsilon')\ [\mathrm{MeV}^{-3}]$"

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.fill_between(
        omega,
        y_min,
        y_max,
        color="tab:orange",
        alpha=0.45,
        label="Min–max-band",
    )

    ax.set_title(f"Differentiellt tvärsnitt som min–max-band, Eν={E_NU_MEV:.1f} MeV, θ={THETA_DEG:.1f}°")
    ax.set_xlabel(r"$\omega$ [MeV]")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# Sparar alla kombinationers tvärsnitt i en gemensam CSV-fil.
def save_combined_cross_sections_csv(all_results: Dict[str, dict], path: Path) -> None:
    combo_names = list(all_results.keys())

    # Första kombinationens omega- och q-grid används som referenskolumner.
    omega_ref = all_results[combo_names[0]]["omega_plot_mev"]
    q_ref = all_results[combo_names[0]]["q_plot_mev"]

    cols = [omega_ref, q_ref]
    header = ["omega_mev", "q_mev"]

    # Lägger till både naturliga och konverterade tvärsnitt för varje kombination.
    for combo_name in combo_names:
        cols.append(all_results[combo_name]["xs_plot_nat"])
        header.append(f"{combo_name}__xs_nat_mev_minus3")

        cols.append(all_results[combo_name]["xs_plot_1e38_cm2_per_sr_per_gev"])
        header.append(f"{combo_name}__xs_1e38_cm2_per_sr_per_gev")

    arr = np.column_stack(cols)
    np.savetxt(path, arr, delimiter=",", header=",".join(header), comments="")


# ============================================================
# 14. One full combination run
# ============================================================
# Kör hela arbetsflödet för en enda kombination:
# data -> split -> träning -> validering -> prediktion -> tvärsnitt -> filer och plottar.
def run_single_combination(combo_name: str, combo_spec: dict) -> dict:
    combo_dir = OUTPUT_DIR / combo_name
    combo_dir.mkdir(parents=True, exist_ok=True)

    # Loggar tydlig start för denna kombination.
    log("=" * 80)
    log(f"Startar kombination: {combo_name}")
    log(f"Specifikation: {combo_spec}")
    log("=" * 80)

    # Sätter om seed inför varje kombination så att körningarna är jämförbara.
    set_global_seed(FIXED_SEED)

    # Bygger q-data enligt vald min/max/mean-specifikation och skapar träningssplit.
    q_data = build_q_curve_data(DATA_ROOT, combo_spec=combo_spec)
    split = build_single_split(q_data)
    train_qs = split["train_qs"]
    val_q = split["val_q"]

    log(f"[{combo_name}] Train qs: {train_qs}")
    log(f"[{combo_name}] Validation q: {val_q}")

    template = SELECTED_MODEL_CONFIG

    # Initierar datahanteraren och bygger normaliserade tensorer.
    dm = SplitDataManager(
        q_data=q_data,
        feature_set_name=template["feature_set"],
        normalize=template["normalize"],
        unit_system=template["unit_system"],
        device=DEVICE,
    )
    dm.configure(train_qs=train_qs, val_q=val_q)

    # Hämtar valideringsdatasetet för ett enda q separat för slututvärderingen.
    X_val_single, Y_val_single, W_val_single, _ = dm.dataset_for_single_q(val_q)

    # Skapar en komplett run-konfiguration för denna kombination.
    cfg = RunConfig(
        template_name=template["template_name"],
        combo_name=combo_name,
        train_qs=train_qs,
        val_q=val_q,
        architecture=list(template["architecture"]),
        activation=template["activation"],
        optimizer=template["optimizer"],
        lr_policy=template["lr_policy"],
        base_lr=BASE_LR,
        early_stop_patience=EARLY_STOP_PATIENCE,
        loss_name=template["loss_name"],
        feature_set=template["feature_set"],
        normalize=bool(template["normalize"]),
        unit_system=template["unit_system"],
        seed=FIXED_SEED,
    )

    # Tränar modellen och hämtar den bästa modellen från resultatet.
    result = train_one_run(dm, cfg)
    model = result["model"]

    # Utvärderar bästa modellen på valideringskurvan.
    val_metrics = evaluate_model_on_dataset(model, dm, X_val_single, Y_val_single, W_val_single)

    log(
        f"[{combo_name}] klart | best_epoch={result['best_epoch']} | "
        f"epochs_ran={result['epochs_ran']} | val_MAE={val_metrics['mae']:.6e} | "
        f"val_MSE={val_metrics['mse']:.6e} | val_wMAE={val_metrics['weighted_mae']:.6e}"
    )

    # Sparar träningshistorik och modellcheckpoint för denna kombination.
    save_history_csv(result["history"], combo_dir / "training_history.csv")
    save_model_checkpoint(
        combo_dir=combo_dir,
        model=model,
        dm=dm,
        cfg=cfg,
        result=result,
        val_metrics=val_metrics,
        combo_spec=combo_spec,
    )

    # Bestämmer maximal omega som finns i tränings-/responsdatan samt tränat q-intervall.
    max_omega_seen = max(float(np.max(pack.omega_mev)) for pack in q_data.values())
    min_q_train = float(min(train_qs))
    max_q_train = float(max(train_qs))

    # Fysikaliskt övre omega-värde sätts av att utgående lepton måste kunna ha minst sin massa.
    omega_upper_physical = E_NU_MEV - FINAL_LEPTON_MASS_MEV - 1e-6
    if omega_upper_physical <= 0.0:
        raise ValueError(
            f"E_nu={E_NU_MEV} MeV är för låg för att producera en utgående muon "
            f"med massa {FINAL_LEPTON_MASS_MEV} MeV."
        )

    # Den faktiska omega-gränsen blir minsta av datans maxomega och den fysikaliska gränsen.
    omega_upper = min(max_omega_seen, omega_upper_physical)

    # Skapar en tät omega-grid för tvärsnittsberäkningen.
    omega_dense = np.linspace(0.0, omega_upper, N_PLOT_POINTS)

    # Beräknar kinematiska storheter för hela omega-griden.
    kin_all = neutrino_kinematics_general(
        E_nu_mev=E_NU_MEV,
        theta_deg=THETA_DEG,
        omega_mev=omega_dense,
        lepton_mass_mev=FINAL_LEPTON_MASS_MEV,
    )

    # Grundmask: bara fysiska och ändliga kinematiska punkter behålls.
    base_mask = kin_all["physical_mask"] & np.isfinite(kin_all["q_mev"])

    # Beräknar vilket q-intervall vald kinematik faktiskt kräver.
    q_phys = kin_all["q_mev"][base_mask]
    if q_phys.size == 0:
        raise ValueError(
            f"Inga fysiska kinematiska punkter alls för E_nu={E_NU_MEV} MeV och theta={THETA_DEG}°."
        )

    q_phys_min = float(np.nanmin(q_phys))
    q_phys_max = float(np.nanmax(q_phys))

    log(
        f"[{combo_name}] physical q-range from chosen kinematics: "
        f"[{q_phys_min:.3f}, {q_phys_max:.3f}] MeV | "
        f"trained q-range: [{min_q_train:.3f}, {max_q_train:.3f}] MeV"
    )

    # Om extrapolering inte tillåts filtreras punkter utanför tränat q-intervall bort.
    if ALLOW_Q_EXTRAPOLATION:
        mask = base_mask
    else:
        mask = (
            base_mask
            & (kin_all["q_mev"] >= min_q_train)
            & (kin_all["q_mev"] <= max_q_train)
        )

    # Om alla punkter filtreras bort ges ett tydligt felmeddelande med möjliga lösningar.
    if not np.any(mask):
        raise ValueError(
            "\n"
            f"Inga punkter kvar efter q-filter.\n"
            f"Vald kinematik: E_nu={E_NU_MEV} MeV, theta={THETA_DEG} deg\n"
            f"Fysikalisk q-range från denna kinematik: [{q_phys_min:.3f}, {q_phys_max:.3f}] MeV\n"
            f"Tränad q-range: [{min_q_train:.3f}, {max_q_train:.3f}] MeV\n"
            f"Antingen:\n"
            f"  1) välj annan E_nu/theta,\n"
            f"  2) utöka träningsdatan till högre q,\n"
            f"  3) sätt ALLOW_Q_EXTRAPOLATION = True.\n"
        )

    # Skapar den slutliga omega- och q-kurvan som modellen ska prediktera längs.
    omega_plot = omega_dense[mask]
    q_plot = kin_all["q_mev"][mask]

    if len(omega_plot) == 0:
        raise ValueError(
            f"omega_plot är tom för kombination {combo_name}. "
            f"Kontrollera E_nu, theta och q-range."
        )

    # Filtrerar alla arraybaserade kinematiska storheter med samma mask.
    kin_plot = {
        key: (value[mask] if isinstance(value, np.ndarray) else value)
        for key, value in kin_all.items()
    }

    # Predikterar responsfunktionerna längs den valda kinematiska kurvan.
    responses_pred = predict_responses_on_kinematic_curve(
        model=model,
        dm=dm,
        q_mev_arr=q_plot,
        omega_mev_arr=omega_plot,
    )

    # Beräknar tvärsnitt i naturliga enheter och sedan i plottvänliga konverterade enheter.
    xs_nat = differential_cross_section_general(
        responses_pred=responses_pred,
        kin=kin_plot,
        is_neutrino=IS_NEUTRINO,
    )

    xs_conv = convert_xs_nat_to_1e38_cm2_per_sr_per_GeV(xs_nat)

    # Samlar alla kolumner som ska sparas för den beräknade tvärsnittskurvan.
    curve_cols = [
        omega_plot,
        q_plot,
        kin_plot["eps_out_mev"],
        kin_plot["Q2_mev2"],
        kin_plot["v00"],
        kin_plot["vzz"],
        kin_plot["v0z"],
        kin_plot["vxx"],
        kin_plot["vxy"],
        responses_pred[:, CURVE_TO_INDEX["R00"]],
        responses_pred[:, CURVE_TO_INDEX["Rt"]],
        responses_pred[:, CURVE_TO_INDEX["Rxy"]],
        responses_pred[:, CURVE_TO_INDEX["Rzz"]],
        responses_pred[:, CURVE_TO_INDEX["R0z"]],
        xs_nat,
        xs_conv,
    ]
    curve_header = [
        "omega_mev",
        "q_mev",
        "eps_out_mev",
        "Q2_mev2",
        "v00",
        "vzz",
        "v0z",
        "vxx",
        "vxy",
        "R00_pred",
        "Rt_pred",
        "Rxy_pred",
        "Rzz_pred",
        "R0z_pred",
        "xs_nat_mev_minus3",
        "xs_1e38_cm2_per_sr_per_gev",
    ]
    curve_matrix = np.column_stack(curve_cols)
    np.savetxt(
        combo_dir / "differential_cross_section_curve.csv",
        curve_matrix,
        delimiter=",",
        header=",".join(curve_header),
        comments="",
    )

    # Skapar en separat tvärsnittsplot för denna kombination.
    fig, ax = plt.subplots(figsize=(9, 6))
    if PLOT_IN_1E38_CM2_PER_SR_PER_GEV:
        y = xs_conv
        ylabel = r"$d\sigma/(d\Omega\,d\epsilon')\ [10^{-38}\ \mathrm{cm}^2/(\mathrm{sr}\,\mathrm{GeV})]$"
    else:
        y = xs_nat
        ylabel = r"$d\sigma/(d\Omega\,d\epsilon')\ [\mathrm{MeV}^{-3}]$"

    ax.plot(omega_plot, y, lw=2, color="tab:orange")
    ax.set_title(
        f"Differentiellt tvärsnitt, {COMBINATION_PLOT_LABELS.get(combo_name, combo_name)}, "
        f"Eν={E_NU_MEV:.1f} MeV, θ={THETA_DEG:.1f}°"
    )
    ax.set_xlabel(r"$\omega$ [MeV]")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(combo_dir / "differential_cross_section.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Sparar en JSON-sammanfattning för just denna kombination.
    out_meta = {
        "combo_name": combo_name,
        "combination": combo_spec,
        "seed": FIXED_SEED,
        "train_qs": train_qs,
        "val_q": val_q,
        "best_epoch": result["best_epoch"],
        "epochs_ran": result["epochs_ran"],
        "runtime_sec": result["runtime_sec"],
        "val_mae": val_metrics["mae"],
        "val_mse": val_metrics["mse"],
        "val_weighted_mae": val_metrics["weighted_mae"],
        "num_params": result["num_params"],
        "E_nu_mev": E_NU_MEV,
        "theta_deg": THETA_DEG,
        "final_lepton_mass_mev": FINAL_LEPTON_MASS_MEV,
        "q_min_train": min_q_train,
        "q_max_train": max_q_train,
        "allow_q_extrapolation": ALLOW_Q_EXTRAPOLATION,
        "physical_q_min_mev": q_phys_min,
        "physical_q_max_mev": q_phys_max,
        "omega_plot_min_mev": float(np.min(omega_plot)) if len(omega_plot) else None,
        "omega_plot_max_mev": float(np.max(omega_plot)) if len(omega_plot) else None,
        "n_plot_points_kept": int(len(omega_plot)),
    }
    atomic_write_text(combo_dir / "run_summary.json", json.dumps(out_meta, indent=2, ensure_ascii=False))

    # Returnerar både resultat för sammanfattning/plottar och objekt som modell/datamanager.
    return {
        "combo_name": combo_name,
        "combination": combo_spec,
        "model": model,
        "dm": dm,
        "cfg": cfg,
        "result": result,
        "val_metrics": val_metrics,
        "omega_plot_mev": omega_plot,
        "q_plot_mev": q_plot,
        "responses_pred": responses_pred,
        "xs_plot_nat": xs_nat,
        "xs_plot_1e38_cm2_per_sr_per_gev": xs_conv,
        "q_min_train": min_q_train,
        "q_max_train": max_q_train,
    }


# ============================================================
# 15. Main
# ============================================================
# Huvudfunktionen som kör alla tre kombinationer och skapar gemensamma resultatfiler.
def main() -> None:
    # Manifestet beskriver hela körningens inställningar och sparas före själva träningen.
    manifest = {
        "data_root": str(DATA_ROOT.resolve()),
        "output_dir": str(OUTPUT_DIR.resolve()),
        "fixed_seed": FIXED_SEED,
        "selected_model_config": SELECTED_MODEL_CONFIG,
        "max_epochs": MAX_EPOCHS,
        "early_stop_patience": EARLY_STOP_PATIENCE,
        "min_delta": MIN_DELTA,
        "base_lr": BASE_LR,
        "weight_decay": WEIGHT_DECAY,
        "val_q": VAL_Q,
        "full_interval": True,
        "omega_constraint": None,
        "E_nu_mev": E_NU_MEV,
        "theta_deg": THETA_DEG,
        "final_lepton_mass_mev": FINAL_LEPTON_MASS_MEV,
        "is_neutrino": IS_NEUTRINO,
        "allow_q_extrapolation": ALLOW_Q_EXTRAPOLATION,
        "plot_in_1e38_cm2_per_sr_per_gev": PLOT_IN_1E38_CM2_PER_SR_PER_GEV,
        "combinations": COMBINATIONS,
    }
    atomic_write_text(MANIFEST_PATH, json.dumps(manifest, indent=2, ensure_ascii=False))

    # Loggar de viktigaste globala inställningarna.
    log("Startar hela körningen.")
    log(f"E_nu = {E_NU_MEV} MeV")
    log(f"theta = {THETA_DEG} deg")
    log(f"Slutlig leptonmassa = {FINAL_LEPTON_MASS_MEV} MeV")
    log(f"Validerings-q = {VAL_Q}")
    log(f"Seed = {FIXED_SEED}")
    log(f"ALLOW_Q_EXTRAPOLATION = {ALLOW_Q_EXTRAPOLATION}")
    log(f"Kombinationer = {list(COMBINATIONS.keys())}")

    # all_results används för gemensamma plottar, summary för JSON-sammanfattningen.
    all_results: Dict[str, dict] = {}
    summary = {}

    t0 = time.time()

    # Kör hela pipeline för varje kombination.
    for combo_name, combo_spec in COMBINATIONS.items():
        combo_result = run_single_combination(combo_name=combo_name, combo_spec=combo_spec)
        all_results[combo_name] = combo_result

        # Sparar kort sammanfattning per kombination.
        summary[combo_name] = {
            "combination": combo_spec,
            "seed": FIXED_SEED,
            "best_epoch": int(combo_result["result"]["best_epoch"]),
            "epochs_ran": int(combo_result["result"]["epochs_ran"]),
            "runtime_sec": float(combo_result["result"]["runtime_sec"]),
            "val_mae": float(combo_result["val_metrics"]["mae"]),
            "val_mse": float(combo_result["val_metrics"]["mse"]),
            "val_weighted_mae": float(combo_result["val_metrics"]["weighted_mae"]),
            "num_params": int(combo_result["result"]["num_params"]),
            "q_min_train": float(combo_result["q_min_train"]),
            "q_max_train": float(combo_result["q_max_train"]),
            "n_plot_points": int(len(combo_result["omega_plot_mev"])),
        }

    # Skapar panelplot med en figurpanel per kombination.
    plot_panel(
        all_results=all_results,
        path=PANEL_PLOT_PATH,
        plot_in_converted_units=PLOT_IN_1E38_CM2_PER_SR_PER_GEV,
    )

    # Skapar overlay-plot om inställningen är aktiverad.
    if MAKE_OVERLAY_PLOT:
        plot_overlay(
            all_results=all_results,
            path=OVERLAY_PLOT_PATH,
            plot_in_converted_units=PLOT_IN_1E38_CM2_PER_SR_PER_GEV,
        )

    # Skapar min–max-bandplot.
    plot_band(
        all_results=all_results,
        path=BAND_PLOT_PATH,
        plot_in_converted_units=PLOT_IN_1E38_CM2_PER_SR_PER_GEV,
    )

    # Sparar alla tvärsnittskurvor i en gemensam CSV.
    save_combined_cross_sections_csv(
        all_results=all_results,
        path=COMBINED_CSV_PATH,
    )

    total_elapsed = time.time() - t0

    # Lägger till global sammanfattning med körningens gemensamma outputfiler.
    summary["_global"] = {
        "E_nu_mev": E_NU_MEV,
        "theta_deg": THETA_DEG,
        "final_lepton_mass_mev": FINAL_LEPTON_MASS_MEV,
        "seed": FIXED_SEED,
        "val_q": VAL_Q,
        "is_neutrino": IS_NEUTRINO,
        "allow_q_extrapolation": ALLOW_Q_EXTRAPOLATION,
        "total_elapsed_sec": total_elapsed,
        "panel_plot_path": str(PANEL_PLOT_PATH.resolve()),
        "overlay_plot_path": str(OVERLAY_PLOT_PATH.resolve()) if MAKE_OVERLAY_PLOT else None,
        "band_plot_path": str(BAND_PLOT_PATH.resolve()),
        "combined_csv_path": str(COMBINED_CSV_PATH.resolve()),
    }

    # Sparar hela sammanfattningen som JSON.
    atomic_write_text(SUMMARY_JSON_PATH, json.dumps(summary, indent=2, ensure_ascii=False))

    # Skriver en tydlig slutrapport till terminalen med alla viktiga filer.
    print("\nKLART.\n")
    print(f"Output directory: {OUTPUT_DIR.resolve()}")
    print(f"Panel plot:       {PANEL_PLOT_PATH.resolve()}")
    if MAKE_OVERLAY_PLOT:
        print(f"Overlay plot:     {OVERLAY_PLOT_PATH.resolve()}")
    print(f"Band plot:        {BAND_PLOT_PATH.resolve()}")
    print(f"Combined CSV:     {COMBINED_CSV_PATH.resolve()}")
    print(f"Summary JSON:     {SUMMARY_JSON_PATH.resolve()}")

    # Skriver ut filvägar för varje enskild kombination.
    for combo_name in COMBINATIONS:
        combo_dir = OUTPUT_DIR / combo_name
        print(f"\n[{combo_name}]")
        print(f"  model:    {combo_dir / 'best_model_state.pt'}")
        print(f"  metadata: {combo_dir / 'best_model_metadata.json'}")
        print(f"  history:  {combo_dir / 'training_history.csv'}")
        print(f"  curve:    {combo_dir / 'differential_cross_section_curve.csv'}")
        print(f"  plot:     {combo_dir / 'differential_cross_section.png'}")


# Kör main() endast när filen körs direkt, inte när den importeras som modul.
if __name__ == "__main__":
    main()
