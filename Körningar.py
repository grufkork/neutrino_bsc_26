#ChatGPT har använts för att utveckla denna kod.
"""
Top-3-modeller, full omega-range, fast val/test split, 15 plots.

Upplägg:
- Hela omega-intervallet används (ingen omega < q-filtrering)
- Samma target som tidigare: mean(col2, col3)
- Samma zero-padding från omega=0 upp till lägsta omega i kurvan
- Endast de 3 toppmodellerna från första sweepen används
- Validering: q = 250 MeV
- Test: q = 75 MeV
- Träning: alla övriga q som hittas i datan, utom 75 och 250
- 10 oberoende träningskörningar per modell
- Early stopping på valideringskurvan q=250, precis som i första koden
- I slutet sparas:
    * run_results.csv
    * summary_by_model.csv
    * 15 figurer (3 modeller x 5 responskurvor)
      där varje figur har två paneler ovanpå varandra:
        - överst: q=250 (val)
        - nederst: q=75 (test)
      och visar mean ± 1 std över 10 körningar för EN modell åt gången
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
# 0. Device + seeds
# ============================================================
# Grundseed som används för att göra körningarna reproducerbara.
# Senare skapas separata run_seed-värden för varje modell/repetition.
BASE_SEED = 20260413


# Sätter slumpfrö i Python, NumPy och PyTorch.
# Om CUDA används sätts även seed för alla CUDA-enheter.
def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Sätter initial seed innan device väljs och innan någon modell skapas.
set_global_seed(BASE_SEED)

# Väljer beräkningsenhet i prioritetsordning:
# 1. CUDA-GPU om tillgänglig,
# 2. Apple MPS om tillgänglig,
# 3. CPU annars.
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

# Skriver ut vald device så att körningsmiljön blir tydlig i terminalen.
print(f"Using device: {DEVICE}")


# ============================================================
# 1. Global config
# ============================================================
# Mappen där .dat-filerna med responsdata förväntas ligga.
DATA_ROOT = Path(".")

# Huvudmapp där alla resultat från denna körning sparas.
OUTPUT_DIR = Path("output_top3_holdout_q250_test_q75_fullomega_separateplots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Sökvägar till resultatfiler, metadata, logg och plottmapp.
RUNS_CSV_PATH = OUTPUT_DIR / "run_results.csv"
SUMMARY_CSV_PATH = OUTPUT_DIR / "summary_by_model.csv"
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"
LOG_PATH = OUTPUT_DIR / "run_log.txt"
PLOTS_DIR = OUTPUT_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# De fem responskurvor som modellen ska prediktera.
# Ordningen är viktig eftersom samma index används i data, metrik och plottar.
OUTPUT_CURVES = ["R00", "Rt", "Rxy", "Rzz", "R0z"]
NUM_OUTPUTS = len(OUTPUT_CURVES)

# Fast validerings-q, test-q och antal oberoende repetitioner per modell.
VAL_Q = 250
TEST_Q = 75
N_REPEATS = 10

# Reguljärt uttryck som identifierar responsfiler och extraherar q-värde samt kurvnamn.
FILE_RE = re.compile(r"^CR_q(\d+)_(R00|Rt|Rxy|Rzz|R0z)_.+\.dat$", re.IGNORECASE)

# De tre toppmodellerna från första sweepen
# Varje dictionary beskriver en modellarkitektur och dess träningsinställningar.
TOP_MODEL_CONFIGS = [
    {
        "template_name": "top1",
        "architecture": [128, 128, 128, 128, 128, 128],
        "activation": "gelu",
        "optimizer": "adamw",
        "lr_policy": "fixed",
        "loss_name": "mae",
        "feature_set": "base+logs",
        "normalize": True,
        "unit_system": "MeV",
    },
    {
        "template_name": "top2",
        "architecture": [256, 256, 128, 128, 64],
        "activation": "gelu",
        "optimizer": "adamw",
        "lr_policy": "fixed",
        "loss_name": "mae",
        "feature_set": "base+logs",
        "normalize": True,
        "unit_system": "MeV",
    },
    {
        "template_name": "top3",
        "architecture": [256, 256, 128, 128, 64],
        "activation": "gelu",
        "optimizer": "adamw",
        "lr_policy": "fixed",
        "loss_name": "mae",
        "feature_set": "base+dist+logs",
        "normalize": True,
        "unit_system": "MeV",
    },
]

# Möjliga featureuppsättningar som kan byggas från q och omega.
# Varje toppmodell väljer en av dessa via feature_set.
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
# MAX_EPOCHS begränsar träningen, medan EARLY_STOP_PATIENCE stoppar tidigare
# om valideringsmåttet inte förbättras.
MAX_EPOCHS = 3000
EARLY_STOP_PATIENCE = 80
MIN_DELTA = 1e-6
BASE_LR = 1e-3
WEIGHT_DECAY = 1e-4

# Parametrar som används om weighted_mae väljs som loss.
# De styr hur mycket större responsvärden ska viktas.
WEIGHTED_MAE_ALPHA = 4.0
WEIGHTED_MAE_POWER = 1.0


# ============================================================
# 2. Utilities
# ============================================================
# Loggar ett meddelande både till terminalen och till loggfilen.
def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# Skriver text atomärt: först till en temporär fil och sedan ersätts målfilen.
# Det minskar risken för halvskrivna filer om körningen avbryts.
def atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


# Skriver CSV atomärt med givna kolumnnamn och rader.
def atomic_write_csv(path: Path, fieldnames: List[str], rows: List[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    os.replace(tmp, path)


# Skapar en stabil SHA1-hash från en dictionary.
# Detta används för att ge varje träningskörning ett kort run_id.
def sha1_dict(d: dict) -> str:
    payload = json.dumps(d, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


# Gör arkitekturen läsbar som text, till exempel "128-128-128-128-128-128".
def architecture_name(layers: List[int]) -> str:
    return "-".join(str(x) for x in layers)


# Räknar antalet träningsbara parametrar i en PyTorch-modell.
def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================
# 3. File loading + curve construction
# ============================================================
# Kontrollerar om en fil matchar förväntat responsfilnamn.
def is_response_file(path: Path) -> bool:
    return FILE_RE.match(path.name) is not None


# Tolkar filnamnet och returnerar q-värdet samt responskurvans namn.
def parse_filename(path: Path) -> Tuple[int, str]:
    m = FILE_RE.match(path.name)
    if m is None:
        raise ValueError(f"Ogiltigt filnamn: {path.name}")
    q = int(m.group(1))
    curve = m.group(2)
    return q, curve


# Läser en enskild responsfil.
# Filen antas innehålla omega i kolumn 0 och två responskolumner i kolumn 1 och 2.
def load_single_response_file(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.loadtxt(path)

    # Om bara en rad läses in görs arrayen om till 2D-format.
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    # Om data verkar vara transponerad med tre rader i stället för tre kolumner transponeras den.
    if arr.shape[0] == 3 and arr.shape[1] != 3:
        arr = arr.T

    # Varje fil måste minst ha omega, min och max.
    if arr.shape[1] < 3:
        raise ValueError(f"Fil {path.name} måste ha minst 3 kolumner, fick shape={arr.shape}")

    # Omega hämtas från första kolumnen.
    omega = arr[:, 0].astype(np.float64)

    # Target är medelvärdet av kolumn 1 och 2, med NaN-hantering via nanmean.
    response = np.nanmean(arr[:, 1:3], axis=1).astype(np.float64)
    return omega, response


# Ersätter inledande NaN-värden med noll fram till första ändliga värde.
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


# Uppskattar omega-gridens steg genom medianen av positiva differenser.
def infer_zero_padding_step(omega: np.ndarray) -> float:
    diffs = np.diff(np.sort(np.unique(omega)))
    diffs = diffs[np.isfinite(diffs) & (diffs > 1e-12)]
    if len(diffs) == 0:
        return max(float(np.min(omega)), 1.0)
    return float(np.median(diffs))


# Dataklass som samlar all kurvdata för ett q-värde.
@dataclass
class QCurveData:
    q_mev: int
    omega_mev: np.ndarray
    y: np.ndarray
    weights: np.ndarray
    peaks: np.ndarray
    inferred_step_mev: float


# Beräknar relativa vikter per kurva baserat på responsens storlek jämfört med dess peak.
def compute_relative_curve_weights(y: np.ndarray, alpha: float, power: float) -> Tuple[np.ndarray, np.ndarray]:
    peaks = np.max(np.abs(y), axis=0)
    peaks = np.where(peaks < 1e-12, 1.0, peaks)
    rel = np.abs(y) / peaks[None, :]
    weights = 1.0 + alpha * np.power(rel, power)
    return weights.astype(np.float64), peaks.astype(np.float64)


# Läser alla responsfiler och bygger en dictionary med QCurveData per q-värde.
def build_q_curve_data(data_root: Path) -> Dict[int, QCurveData]:
    # Hittar alla .dat-filer i data_root som matchar responsfilformatet.
    files = sorted([p for p in data_root.glob("*.dat") if is_response_file(p)])
    if not files:
        raise FileNotFoundError(
            f"Hittade inga responsfiler i {data_root.resolve()}. "
            f"Förväntade namn som CR_q75_R00_NNLO_GO_450.dat"
        )

    # Grupperar varje fil efter q-värde och kurvnamn.
    grouped: Dict[int, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {}
    for path in files:
        q, curve = parse_filename(path)
        omega, response = load_single_response_file(path)
        grouped.setdefault(q, {})[curve] = (omega, response)

    # Säkerställer att både validerings- och testkurvan finns.
    if VAL_Q not in grouped:
        raise ValueError(f"Hittade inga filer för valideringskurvan q={VAL_Q} MeV")
    if TEST_Q not in grouped:
        raise ValueError(f"Hittade inga filer för testkurvan q={TEST_Q} MeV")

    q_data: Dict[int, QCurveData] = {}

    # Bygger en komplett femkurvematrix för varje q-värde.
    for q in sorted(grouped.keys()):
        curves = grouped[q]
        missing_curves = [c for c in OUTPUT_CURVES if c not in curves]
        if missing_curves:
            raise ValueError(f"q={q} saknar kurvor: {missing_curves}")

        omega_ref = None
        y_cols = []

        # Läser responskurvorna i den fasta ordningen OUTPUT_CURVES.
        for curve_name in OUTPUT_CURVES:
            omega, y = curves[curve_name]
            y = fill_leading_nans_with_zero(y)

            # Den första kurvans omega-grid används som referens.
            if omega_ref is None:
                omega_ref = omega.copy()
            else:
                # Alla fem kurvor för samma q måste ha samma omega-grid.
                if len(omega) != len(omega_ref) or not np.allclose(omega, omega_ref, rtol=0.0, atol=1e-9):
                    raise ValueError(
                        f"Omega-grid skiljer sig mellan kurvor för q={q}. "
                        "Skriptet antar samma omega-grid för alla 5 kurvor."
                    )

            y_cols.append(y)

        # Stackar fem responskurvor till en matris med shape: antal omega-punkter x 5.
        omega_ref = np.asarray(omega_ref, dtype=np.float64)
        y_mat = np.stack(y_cols, axis=1)

        # Tar bort rader med icke-ändliga omega- eller responsvärden.
        mask = np.isfinite(omega_ref) & np.all(np.isfinite(y_mat), axis=1)
        omega_clean = omega_ref[mask]
        y_clean = y_mat[mask]

        if len(omega_clean) == 0:
            raise ValueError(f"Inga giltiga datapunkter kvar för q={q}")

        # Bestämmer stegstorlek och lägsta omega för att kunna skapa zero-padding.
        step = infer_zero_padding_step(omega_clean)
        omega_min = float(np.min(omega_clean))

        # Skapar omega-värden från 0 upp till första faktiska omega-punkt.
        if omega_min > 1e-12:
            omega_zeros = np.arange(0.0, omega_min, step, dtype=np.float64)
            omega_zeros = omega_zeros[omega_zeros < omega_min - 1e-12]
        else:
            omega_zeros = np.empty((0,), dtype=np.float64)

        # Responsen i paddingområdet sätts till noll för alla fem kurvor.
        y_zeros = np.zeros((len(omega_zeros), NUM_OUTPUTS), dtype=np.float64)

        # Hela intervallet: ingen omega<q-filtering här
        # Padding och faktisk data slås ihop utan att klippa bort omega >= q.
        omega_aug = np.concatenate([omega_zeros, omega_clean], axis=0)
        y_aug = np.concatenate([y_zeros, y_clean], axis=0)

        # Beräknar vikter och peakvärden för den utökade kurvan.
        weights, peaks = compute_relative_curve_weights(
            y_aug,
            alpha=WEIGHTED_MAE_ALPHA,
            power=WEIGHTED_MAE_POWER,
        )

        # Sparar färdig data för detta q-värde.
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
# 4. Split helper
# ============================================================
# Skapar den fasta splitten: q=250 för validering, q=75 för test och resten för träning.
def build_single_split(q_data: Dict[int, QCurveData]) -> dict:
    available_qs = sorted(q_data.keys())

    # Kontrollerar att både validerings- och test-q finns i data.
    if VAL_Q not in available_qs:
        raise ValueError(f"Validerings-q={VAL_Q} saknas")
    if TEST_Q not in available_qs:
        raise ValueError(f"Test-q={TEST_Q} saknas")

    # Alla q utom validering och test används för träning.
    train_qs = [q for q in available_qs if q not in (VAL_Q, TEST_Q)]
    if not train_qs:
        raise ValueError("Inga tränings-q återstår efter att val/test tagits bort")

    return {
        "train_qs": train_qs,
        "val_q": VAL_Q,
        "test_q": TEST_Q,
    }


# ============================================================
# 5. Features + data manager
# ============================================================
# Konverterar energi från MeV till valt enhetssystem.
def convert_energy(x_mev: float, unit_system: str) -> float:
    if unit_system == "MeV":
        return float(x_mev)
    if unit_system == "GeV":
        return float(x_mev) / 1000.0
    raise ValueError(f"Okänt enhetssystem: {unit_system}")


# Bygger en featurevektor från q och omega enligt vald featureuppsättning.
def build_feature_vector(q_mev: float, omega_mev: float, feature_names: List[str], unit_system: str) -> List[float]:
    q = convert_energy(q_mev, unit_system)
    omega = convert_energy(omega_mev, unit_system)
    eps = 1e-12

    # Alla möjliga featurevärden beräknas här.
    values = {
        "q": q,
        "omega": omega,
        "q_minus_omega": q - omega,
        "omega_over_q": 0.0 if abs(q) < eps else omega / q,
        "log1p_q": math.log1p(max(q, 0.0)),
        "log1p_omega": math.log1p(max(omega, 0.0)),
    }

    # Returnerar bara de features vars namn finns i feature_names.
    return [float(values[name]) for name in feature_names]


# Hanterar dataomvandling till tensorer, normalisering och dataset för en viss modellkonfiguration.
class SplitDataManager:
    def __init__(
        self,
        q_data: Dict[int, QCurveData],
        feature_set_name: str,
        normalize: bool,
        unit_system: str,
        device: torch.device,
    ):
        # Sparar grunddata och featureinställningar.
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

    # Samlar alla datapunkter för en lista av q-värden.
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

        # Konverterar samlade listor till NumPy-arrayer med rätt datatyper.
        X = np.asarray(xs, dtype=np.float32)
        Y = np.asarray(ys, dtype=np.float32)
        W = np.asarray(ws, dtype=np.float32)
        QID = np.asarray(q_ids, dtype=np.int32)
        return X, Y, W, QID

    # Förbereder tränings- och valideringsdata och beräknar normalisering.
    def configure(self, train_qs: List[int], val_q: int) -> None:
        # Samlar rå train- och valideringsdata.
        X_train_raw, Y_train_raw, W_train, Q_train = self._collect_for_qs(train_qs)
        X_val_raw, Y_val_raw, W_val, Q_val = self._collect_for_qs([val_q])

        # Flyttar arrayer till PyTorch-tensorer på vald device.
        X_train_raw = torch.tensor(X_train_raw, dtype=torch.float32, device=self.device)
        Y_train_raw = torch.tensor(Y_train_raw, dtype=torch.float32, device=self.device)
        W_train = torch.tensor(W_train, dtype=torch.float32, device=self.device)

        X_val_raw = torch.tensor(X_val_raw, dtype=torch.float32, device=self.device)
        Y_val_raw = torch.tensor(Y_val_raw, dtype=torch.float32, device=self.device)
        W_val = torch.tensor(W_val, dtype=torch.float32, device=self.device)

        # Normalisering beräknas endast på träningsdata för att undvika dataläckage.
        if self.normalize:
            self.x_mean = X_train_raw.mean(dim=0, keepdim=True)
            self.x_std = X_train_raw.std(dim=0, keepdim=True)
            self.y_mean = Y_train_raw.mean(dim=0, keepdim=True)
            self.y_std = Y_train_raw.std(dim=0, keepdim=True)

            # Undviker division med mycket små standardavvikelser.
            self.x_std = torch.where(self.x_std < 1e-12, torch.ones_like(self.x_std), self.x_std)
            self.y_std = torch.where(self.y_std < 1e-12, torch.ones_like(self.y_std), self.y_std)
        else:
            # Om normalisering inte används blir transformen identitet.
            self.x_mean = torch.zeros((1, X_train_raw.shape[1]), dtype=torch.float32, device=self.device)
            self.x_std = torch.ones((1, X_train_raw.shape[1]), dtype=torch.float32, device=self.device)
            self.y_mean = torch.zeros((1, Y_train_raw.shape[1]), dtype=torch.float32, device=self.device)
            self.y_std = torch.ones((1, Y_train_raw.shape[1]), dtype=torch.float32, device=self.device)

        # Sparar normaliserade features och råa targets/vikter för träning.
        self.X_train = self.x_to_model_space(X_train_raw)
        self.Y_train_raw = Y_train_raw
        self.W_train = W_train
        self.Q_train = Q_train

        # Sparar normaliserade features och råa targets/vikter för validering.
        self.X_val = self.x_to_model_space(X_val_raw)
        self.Y_val_raw = Y_val_raw
        self.W_val = W_val
        self.Q_val = Q_val

    # Skalar features till modellens normaliserade inputrum.
    def x_to_model_space(self, X_raw: torch.Tensor) -> torch.Tensor:
        return (X_raw - self.x_mean) / self.x_std

    # Skalar modellens output tillbaka till rå respons-skala.
    def y_from_model_space(self, Y_model: torch.Tensor) -> torch.Tensor:
        return Y_model * self.y_std + self.y_mean

    # Skapar ett dataset för ett enskilt q-värde, exempelvis validering eller test.
    def dataset_for_single_q(self, q: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
        pack = self.q_data[q]
        X = np.asarray(
            [build_feature_vector(q, float(w), self.feature_names, self.unit_system) for w in pack.omega_mev],
            dtype=np.float32,
        )
        Y = pack.y.astype(np.float32)
        W = pack.weights.astype(np.float32)
        omega = pack.omega_mev.astype(np.float64)

        # Konverterar till tensorer på rätt device och normaliserar X.
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        Y_t = torch.tensor(Y, dtype=torch.float32, device=self.device)
        W_t = torch.tensor(W, dtype=torch.float32, device=self.device)
        X_t = self.x_to_model_space(X_t)
        return X_t, Y_t, W_t, omega


# ============================================================
# 6. Model + loss + metrics
# ============================================================
# Returnerar vald aktiveringsfunktion som PyTorch-modul.
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


# Fullt kopplat multi-output-nätverk som predikterar alla fem responskurvor samtidigt.
class MultiOutputMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_layers: List[int], output_dim: int, activation: str):
        super().__init__()
        layers: List[nn.Module] = []
        prev = input_dim

        # Bygger ett linjärt lager följt av aktivering för varje dolt lager.
        for hidden in hidden_layers:
            layers.append(nn.Linear(prev, hidden))
            layers.append(make_activation(activation))
            prev = hidden

        # Sista lagret producerar en output per responskurva.
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    # Forward-pass genom hela nätverket.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# Mean absolute error i rå respons-skala.
def mae_loss_raw(y_pred_raw: torch.Tensor, y_true_raw: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(y_pred_raw - y_true_raw))


# Mean squared error i rå respons-skala.
def mse_loss_raw(y_pred_raw: torch.Tensor, y_true_raw: torch.Tensor) -> torch.Tensor:
    return torch.mean((y_pred_raw - y_true_raw) ** 2)


# Viktad MAE där fel multipliceras med kurv- och punktvikter.
def weighted_mae_loss_raw(y_pred_raw: torch.Tensor, y_true_raw: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    err = torch.abs(y_pred_raw - y_true_raw)
    per_curve = (weights * err).sum(dim=0) / (weights.sum(dim=0) + 1e-12)
    return per_curve.mean()


# Väljer loss-funktion utifrån textnamnet i konfigurationen.
def objective_value(loss_name: str, y_pred_raw: torch.Tensor, y_true_raw: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    if loss_name == "mae":
        return mae_loss_raw(y_pred_raw, y_true_raw)
    if loss_name == "mse":
        return mse_loss_raw(y_pred_raw, y_true_raw)
    if loss_name == "weighted_mae":
        return weighted_mae_loss_raw(y_pred_raw, y_true_raw, weights)
    raise ValueError(f"Okänd loss_name: {loss_name}")


# Beräknar flera utvärderingsmått mellan sann och predikterad respons.
def evaluate_tensor(y_true: torch.Tensor, y_pred: torch.Tensor, weights: torch.Tensor) -> dict:
    err = y_pred - y_true
    abs_err = torch.abs(err)
    sq_err = err ** 2

    # Globala mått över alla punkter och alla kurvor.
    mae = torch.mean(abs_err).item()
    mse = torch.mean(sq_err).item()
    per_curve_wmae = (weights * abs_err).sum(dim=0) / (weights.sum(dim=0) + 1e-12)
    wmae = per_curve_wmae.mean().item()

    # Kurvvisa mått för varje responskurva separat.
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


# Predikterar utan gradientberäkning och skalar tillbaka till rå respons-skala.
def predict_on_dataset(model: nn.Module, dm: SplitDataManager, X: torch.Tensor) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        pred_model = model(X)
        pred_raw = dm.y_from_model_space(pred_model)
    return pred_raw


# Utvärderar en modell på ett dataset genom att först prediktera och sedan beräkna mått.
def evaluate_model_on_dataset(model: nn.Module, dm: SplitDataManager, X: torch.Tensor, Y_raw: torch.Tensor, W: torch.Tensor) -> dict:
    pred_raw = predict_on_dataset(model, dm, X)
    return evaluate_tensor(Y_raw, pred_raw, W)


# ============================================================
# 7. Optimizer
# ============================================================
# Skapar optimizer för modellen.
# Här stöds AdamW med global weight decay.
def build_optimizer(model: nn.Module, optimizer_name: str, lr: float) -> torch.optim.Optimizer:
    optimizer_name = optimizer_name.lower()
    if optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    raise ValueError(f"Okänd optimizer: {optimizer_name}")


# ============================================================
# 8. Training
# ============================================================
# Dataklass som beskriver en specifik träningskörning.
@dataclass
class RunConfig:
    template_name: str
    repeat_idx: int
    train_qs: List[int]
    val_q: int
    test_q: int
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

    # Skapar ett kort ID baserat på hela körningskonfigurationen.
    def run_id(self) -> str:
        return sha1_dict(asdict(self))[:16]


# Tränar en modell för en specifik RunConfig och returnerar bästa modell samt metadata.
def train_one_run(dm: SplitDataManager, cfg: RunConfig) -> dict:
    # Inputdimensionen beror på vilken featureuppsättning modellen använder.
    input_dim = len(FEATURE_SETS[cfg.feature_set])

    # Skapar modellen och flyttar den till vald device.
    model = MultiOutputMLP(
        input_dim=input_dim,
        hidden_layers=cfg.architecture,
        output_dim=NUM_OUTPUTS,
        activation=cfg.activation,
    ).to(DEVICE)

    # Skapar optimizer.
    optimizer = build_optimizer(model, cfg.optimizer, cfg.base_lr)

    # Variabler för early stopping och bästa modellvikter.
    n_train = dm.X_train.shape[0]
    best_state = None
    best_metrics = None
    best_epoch = -1
    best_objective = float("inf")
    epochs_without_improvement = 0
    history = []

    t0 = time.time()

    # Träningsloop med max MAX_EPOCHS epoker.
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()

        # Slumpar ordningen på träningspunkterna varje epok.
        perm = torch.randperm(n_train, device=DEVICE)
        xb = dm.X_train[perm]
        yb = dm.Y_train_raw[perm]
        wb = dm.W_train[perm]

        # Ett vanligt PyTorch-träningssteg.
        optimizer.zero_grad(set_to_none=True)
        pred_model = model(xb)
        pred_raw = dm.y_from_model_space(pred_model)
        loss = objective_value(cfg.loss_name, pred_raw, yb, wb)
        loss.backward()
        optimizer.step()

        # Utvärderar modellen på valideringskurvan q=250 efter varje epok.
        val_metrics = evaluate_model_on_dataset(model, dm, dm.X_val, dm.Y_val_raw, dm.W_val)
        current_objective = float(val_metrics[cfg.loss_name])

        # Sparar träningshistorik.
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

        # Kontrollerar om valideringsmåttet förbättrats mer än MIN_DELTA.
        improved = (best_objective - current_objective) > MIN_DELTA
        if np.isfinite(current_objective) and improved:
            best_objective = current_objective
            best_epoch = epoch
            best_metrics = val_metrics
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # Early stopping om förbättring uteblivit för länge.
        if epochs_without_improvement >= cfg.early_stop_patience:
            break

    # Återställer modellen till bästa sparade vikter om en förbättring hittades.
    if best_state is not None:
        model.load_state_dict(best_state)

    runtime_sec = time.time() - t0

    # Returnerar modell och körningsinformation.
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
# 9. Plotting (separate model plots)
# ============================================================
# Skapar en figur för en modell och en responskurva.
# Figuren har två paneler: validering q=250 överst och test q=75 nederst.
def plot_curve_single_model(
    model_name: str,
    agg: dict,
    curve_name: str,
    curve_idx: int,
    val_q: int,
    test_q: int,
    out_path: Path,
) -> None:
    # Ger top1 ett mer beskrivande namn i figurerna.
    display_model_name = "Bästa valideringsmodell" if model_name == "top1" else model_name

    # Tjockare kurvor för bättre läsbarhet
    true_lw = 3.2
    pred_lw = 3.2

    # Större text för bättre läsbarhet
    suptitle_fs = 18
    title_fs = 15
    label_fs = 14
    legend_fs = 12
    tick_fs = 12

    # Validering överst, test nederst
    fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=False, sharey=False)

    # -----------------------------
    # Överst: validation q=250
    # -----------------------------
    ax = axes[0]
    omega_val = agg["val_omega"]
    y_val_true = agg["val_true"][:, curve_idx]

    # Hämtar alla prediktioner för denna kurva över de 10 körningarna.
    preds_val = agg["val_preds"][:, :, curve_idx]
    mean_val = preds_val.mean(axis=0)
    std_val = preds_val.std(axis=0, ddof=0)

    # Ritar sann kurva, medelprediktion och ±1 standardavvikelse.
    ax.plot(omega_val, y_val_true, linewidth=true_lw, label=f"Sann q={val_q}")
    ax.plot(omega_val, mean_val, linewidth=pred_lw, label=f"{display_model_name} medel")
    ax.fill_between(
        omega_val,
        mean_val - std_val,
        mean_val + std_val,
        color="#f4a261",
        alpha=0.35,
        label=f"{display_model_name} ±1 standardavvikelse",
    )

    # Formaterar valideringspanelen.
    ax.set_title(f"{curve_name} | {display_model_name} | Validering q={val_q} MeV", fontsize=title_fs)
    ax.set_xlabel(r"$\omega$ [MeV]", fontsize=label_fs)
    ax.set_ylabel(r"Respons [GeV$^{-1}$]", fontsize=label_fs)
    ax.tick_params(axis="both", labelsize=tick_fs)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=legend_fs)

    # -----------------------------
    # Nederst: test q=75
    # -----------------------------
    ax = axes[1]
    omega_test = agg["test_omega"]
    y_test_true = agg["test_true"][:, curve_idx]

    # Hämtar alla testprediktioner för denna kurva över de 10 körningarna.
    preds_test = agg["test_preds"][:, :, curve_idx]
    mean_test = preds_test.mean(axis=0)
    std_test = preds_test.std(axis=0, ddof=0)

    # Ritar sann testkurva, medelprediktion och ±1 standardavvikelse.
    ax.plot(omega_test, y_test_true, linewidth=true_lw, label=f"Sann q={test_q}")
    ax.plot(omega_test, mean_test, linewidth=pred_lw, label=f"{display_model_name} medel")
    ax.fill_between(
        omega_test,
        mean_test - std_test,
        mean_test + std_test,
        color="#f4a261",
        alpha=0.35,
        label=f"{display_model_name} ±1 standardavvikelse",
    )

    # Formaterar testpanelen.
    ax.set_title(f"{curve_name} | {display_model_name} | Test q={test_q} MeV", fontsize=title_fs)
    ax.set_xlabel(r"$\omega$ [MeV]", fontsize=label_fs)
    ax.set_ylabel(r"Respons [GeV$^{-1}$]", fontsize=label_fs)
    ax.tick_params(axis="both", labelsize=tick_fs)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=legend_fs)

    # Sätter gemensam titel, justerar layout, sparar figuren och stänger den.
    fig.suptitle(
        f"{curve_name}: sann kurva vs {display_model_name} medel ± 1 standardavvikelse över {N_REPEATS} körningar",
        fontsize=suptitle_fs,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 10. Export helpers
# ============================================================
# Skapar en sammanfattnings-CSV där varje modell aggregeras över sina repetitioner.
def export_summary(rows: List[dict], path: Path) -> None:
    # Om inga körningsrader finns finns inget att exportera.
    if not rows:
        return

    # Grupperar alla körningar efter modellnamn/template_name.
    grouped = {}
    for row in rows:
        key = row["template_name"]
        grouped.setdefault(key, []).append(row)

    out_rows = []
    for key, group in grouped.items():
        # Plockar ut relevanta mått över de 10 repetitionerna.
        val_maes = np.array([r["val_mae"] for r in group], dtype=float)
        val_mses = np.array([r["val_mse"] for r in group], dtype=float)
        test_maes = np.array([r["test_mae"] for r in group], dtype=float)
        test_mses = np.array([r["test_mse"] for r in group], dtype=float)
        best_epochs = np.array([r["best_epoch"] for r in group], dtype=float)
        runtimes = np.array([r["runtime_sec"] for r in group], dtype=float)

        # Använder första raden i gruppen för modellens statiska metadata.
        exemplar = group[0]
        out_rows.append(
            {
                "template_name": key,
                "architecture": exemplar["architecture"],
                "activation": exemplar["activation"],
                "feature_set": exemplar["feature_set"],
                "normalize": exemplar["normalize"],
                "unit_system": exemplar["unit_system"],
                "n_repeats": len(group),
                "mean_val_mae": float(val_maes.mean()),
                "std_val_mae": float(val_maes.std(ddof=0)),
                "mean_val_mse": float(val_mses.mean()),
                "std_val_mse": float(val_mses.std(ddof=0)),
                "mean_test_mae": float(test_maes.mean()),
                "std_test_mae": float(test_maes.std(ddof=0)),
                "mean_test_mse": float(test_mses.mean()),
                "std_test_mse": float(test_mses.std(ddof=0)),
                "mean_best_epoch": float(best_epochs.mean()),
                "std_best_epoch": float(best_epochs.std(ddof=0)),
                "mean_runtime_sec": float(runtimes.mean()),
                "std_runtime_sec": float(runtimes.std(ddof=0)),
            }
        )

    # Sorterar modellerna efter bäst genomsnittlig val-MAE och därefter test-MAE.
    out_rows = sorted(out_rows, key=lambda r: (r["mean_val_mae"], r["mean_test_mae"]))
    fieldnames = list(out_rows[0].keys())
    atomic_write_csv(path, fieldnames, out_rows)


# ============================================================
# 11. Main
# ============================================================
# Huvudfunktionen som kör hela experimentet.
def main() -> None:
    # Läser och förbereder all q-data från responsfilerna.
    q_data = build_q_curve_data(DATA_ROOT)
    split = build_single_split(q_data)

    # Hämtar train/val/test-q från splitten.
    train_qs = split["train_qs"]
    val_q = split["val_q"]
    test_q = split["test_q"]

    # Manifestet dokumenterar exakt vilka inställningar som användes.
    manifest = {
        "data_root": str(DATA_ROOT.resolve()),
        "available_qs": sorted(q_data.keys()),
        "train_qs": train_qs,
        "val_q": val_q,
        "test_q": test_q,
        "n_repeats_per_model": N_REPEATS,
        "top_model_configs": TOP_MODEL_CONFIGS,
        "max_epochs": MAX_EPOCHS,
        "early_stop_patience": EARLY_STOP_PATIENCE,
        "base_lr": BASE_LR,
        "min_delta": MIN_DELTA,
        "weight_decay": WEIGHT_DECAY,
        "full_interval": True,
        "omega_constraint": None,
    }
    atomic_write_text(MANIFEST_PATH, json.dumps(manifest, indent=2, ensure_ascii=False))

    # Loggar vilka q-värden som laddats och information om varje q-kurva.
    log("Laddade data för q-värden: " + ", ".join(str(q) for q in sorted(q_data.keys())))
    for q in sorted(q_data.keys()):
        pack = q_data[q]
        log(
            f"q={q} | n_points={len(pack.omega_mev)} | omega_min={pack.omega_mev.min():.6f} MeV | "
            f"omega_max={pack.omega_mev.max():.6f} MeV | step~{pack.inferred_step_mev:.6f} MeV"
        )

    # Loggar den fasta train/val/test-splitten.
    log(f"Train qs: {train_qs}")
    log(f"Validation q: {val_q}")
    log(f"Test q: {test_q}")

    # Här samlas en rad per träningskörning för senare CSV-export.
    run_rows = []

    # För plottarna: lagra alla 10 prediktioner per modell för val/test
    prediction_store: Dict[str, dict] = {}

    # Räknare och tidtagning för att kunna logga framsteg.
    total_runs = len(TOP_MODEL_CONFIGS) * N_REPEATS
    run_counter = 0
    t_global = time.time()

    # Loopar över de tre toppmodellerna.
    for model_idx, template in enumerate(TOP_MODEL_CONFIGS, start=1):
        template_name = template["template_name"]
        log(
            f"Startar modell {template_name} | arch={architecture_name(template['architecture'])} | "
            f"feats={template['feature_set']}"
        )

        # Skapar en data manager för den aktuella modellens feature_set och normalisering.
        dm = SplitDataManager(
            q_data=q_data,
            feature_set_name=template["feature_set"],
            normalize=template["normalize"],
            unit_system=template["unit_system"],
            device=DEVICE,
        )
        dm.configure(train_qs=train_qs, val_q=val_q)

        # Hämtar separata dataset för validerings-q och test-q.
        X_val_single, Y_val_single, W_val_single, omega_val = dm.dataset_for_single_q(val_q)
        X_test_single, Y_test_single, W_test_single, omega_test = dm.dataset_for_single_q(test_q)

        # Listor som samlar prediktioner från alla repetitioner för denna modell.
        val_preds_all = []
        test_preds_all = []

        # Kör N_REPEATS oberoende träningar för samma modell.
        for repeat_idx in range(1, N_REPEATS + 1):
            run_counter += 1

            # Skapar unik seed för varje modell och repetition.
            run_seed = BASE_SEED + model_idx * 10000 + repeat_idx
            set_global_seed(run_seed)

            # Bygger körningskonfigurationen för denna specifika repetition.
            cfg = RunConfig(
                template_name=template_name,
                repeat_idx=repeat_idx,
                train_qs=train_qs,
                val_q=val_q,
                test_q=test_q,
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
            )

            # Tränar modellen med early stopping på valideringskurvan.
            result = train_one_run(dm, cfg)
            model = result["model"]

            # Utvärderar bästa modellen på både validerings- och testkurvan.
            val_metrics = evaluate_model_on_dataset(model, dm, X_val_single, Y_val_single, W_val_single)
            test_metrics = evaluate_model_on_dataset(model, dm, X_test_single, Y_test_single, W_test_single)

            # Sparar faktiska prediktioner för senare medel/std-plottar.
            val_pred = predict_on_dataset(model, dm, X_val_single).detach().cpu().numpy()
            test_pred = predict_on_dataset(model, dm, X_test_single).detach().cpu().numpy()

            val_preds_all.append(val_pred)
            test_preds_all.append(test_pred)

            # Skapar en resultatrad för denna körning.
            row = {
                "run_id": cfg.run_id(),
                "template_name": template_name,
                "repeat_idx": repeat_idx,
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
                "seed": run_seed,
                "train_qs": ",".join(str(q) for q in train_qs),
                "val_q": val_q,
                "test_q": test_q,
                "num_params": result["num_params"],
                "best_epoch": result["best_epoch"],
                "epochs_ran": result["epochs_ran"],
                "runtime_sec": result["runtime_sec"],
                "val_mae": val_metrics["mae"],
                "val_mse": val_metrics["mse"],
                "val_weighted_mae": val_metrics["weighted_mae"],
                "test_mae": test_metrics["mae"],
                "test_mse": test_metrics["mse"],
                "test_weighted_mae": test_metrics["weighted_mae"],
            }
            run_rows.append(row)

            # Beräknar ungefärlig återstående tid baserat på medeltid per körning hittills.
            elapsed = time.time() - t_global
            avg_time = elapsed / run_counter
            remaining = avg_time * max(total_runs - run_counter, 0)
            hh = int(remaining // 3600)
            mm = int((remaining % 3600) // 60)
            ss = int(remaining % 60)

            # Loggar status för denna repetition.
            log(
                f"[{run_counter}/{total_runs}] completed | model={template_name} | repeat={repeat_idx}/{N_REPEATS} | "
                f"best_epoch={result['best_epoch']} | val_MAE={val_metrics['mae']:.6e} | "
                f"test_MAE={test_metrics['mae']:.6e} | ETA~{hh:02d}:{mm:02d}:{ss:02d}"
            )

        # När alla repetitioner för modellen är klara sparas prediktionerna i prediction_store.
        prediction_store[template_name] = {
            "val_preds": np.stack(val_preds_all, axis=0),
            "test_preds": np.stack(test_preds_all, axis=0),
            "val_true": Y_val_single.detach().cpu().numpy(),
            "test_true": Y_test_single.detach().cpu().numpy(),
            "val_omega": omega_val,
            "test_omega": omega_test,
        }

    # Export CSV
    # Sparar alla individuella körningar och en aggregerad modellöversikt.
    if run_rows:
        fieldnames = list(run_rows[0].keys())
        atomic_write_csv(RUNS_CSV_PATH, fieldnames, run_rows)
        export_summary(run_rows, SUMMARY_CSV_PATH)

    # Plotta 15 figurer totalt: 3 modeller x 5 kurvor
    # Varje modell får en egen undermapp med fem responskurvefigurer.
    for model_name, agg in prediction_store.items():
        model_plot_dir = PLOTS_DIR / model_name
        model_plot_dir.mkdir(parents=True, exist_ok=True)

        for curve_idx, curve_name in enumerate(OUTPUT_CURVES):
            out_path = model_plot_dir / f"{model_name}_curve_{curve_name}_val{val_q}_test{test_q}_mean_pm_std.png"
            plot_curve_single_model(
                model_name=model_name,
                agg=agg,
                curve_name=curve_name,
                curve_idx=curve_idx,
                val_q=val_q,
                test_q=test_q,
                out_path=out_path,
            )
            log(f"Sparade figur: {out_path}")

    # Loggar total körningstid och viktiga outputfiler.
    total_elapsed = time.time() - t_global
    log(
        f"Klart. total_elapsed_sec={total_elapsed:.2f}. "
        f"Filer: {RUNS_CSV_PATH}, {SUMMARY_CSV_PATH}, {PLOTS_DIR}"
    )


# Kör main() endast när filen körs direkt, inte om den importeras som modul.
if __name__ == "__main__":
    main()
