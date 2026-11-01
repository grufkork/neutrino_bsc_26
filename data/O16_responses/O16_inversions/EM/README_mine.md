# 16O electromagnetic response functions R_L, R_T (ħω 12/14/16 comparison)

Inclusive (e,e′) responses of ¹⁶O, with an uncertainty band that covers the inversion, the ħω dependence and the
form-factor correction. 

- Hamiltonian ΔNNLO_GO(450), N_max = 14, CCSD ground state, CCSD Lanczos LIT, one-body currents. The isoscalar and
  isovector channels are computed separately and summed.
- q = 50, 100, 150, 200, 250, 300, 350, 400 MeV. 

## Files

`EM_RL_q<q>.dat` and `EM_RT_q<q>.dat`, with R_L = r00 and R_T = rte + rtm. Each is on a 0.125 MeV grid of E_x from 0 to
~520 MeV (R = 0 below the threshold).

| column | content |
|---|---|
| `E_x` | excitation energy of ¹⁶O (MeV) |
| `omega` | energy transfer ω = E_x + q²/2M_A (MeV), M_A = 16 × 931.494 MeV |
| `R` | response (MeV⁻¹), ħω16 central value: midpoint of the ħω16 band of accepted solutions |
| `R_low`, `R_high` | total band: R ∓ √(max(dR_env, dR_hw)² + dR_FF²), each side separately |
| `dR_env_low`, `dR_env_high` | envelope of all accepted solutions of all ħω, relative to R (inversion + ħω) |
| `dR_hw` | ħω spread of the central values (see below) |
| `dR_FF` | error of the form-factor correction |
| `R_hw12`, `R_hw14`, `R_hw16` | central value (band midpoint) per ħω; `nan` where that ħω was not run or had no accepted solution |

`EM_responses.png`: overview of R_L and R_T at all q with the total band, ω ≤ q


## Cross section

Electron scattering in the one-photon exchange, electron mass neglected:

    d²σ/dΩ dω = σ_Mott · [ (Q²/q²)² R_L(q,ω) + ( Q²/(2q²) + tan²(θ/2) ) R_T(q,ω) ]

    σ_Mott = α² cos²(θ/2) / (4 ε² sin⁴(θ/2)),     α = 1/137.036

- ε is the beam energy, ε′ = ε − ω, and θ is the scattering angle.
- q² = ε² + ε′² − 2εε′ cos θ, and Q² = q² − ω² = 4εε′ sin²(θ/2).
- **Units:** α²/ε² [MeV⁻²] · R [MeV⁻¹] gives MeV⁻³ per sr. Multiply by (ħc)² = 3.8938×10⁴ MeV² fm² to get fm²/(MeV sr)
  (1 fm² = 10 mb = 10⁷ nb).


## Caveats

- **Threshold.** These files use the **experimental** threshold, S_p = 12.13 MeV, for every ħω.
- **Near-threshold resonances.** At q50 (and q100–150 in R_L) the resonance region near threshold, at 12–30 MeV, is
  resolved only at Γ = 10 MeV. Peak heights there are less reliable than their integrals.
- **R_T at q50** has a long flat tail up to several hundred MeV. This is the J = 1 magnetic strength, the same for all ħω.
