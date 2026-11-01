# 16O charged-current (CR) response functions, hw = 14 MeV

Inverted LIT response functions for ν_l + ¹⁶O → l⁻ + X (CR = n → p, i.e. neutrino scattering), with a total
uncertainty band. 
- Hamiltonian ΔNNLO_GO(450), N_max = 14, ħω = 14 MeV, CCSD ground state, CCSD Lanczos LIT
- One-body currents only. Form factors: Kelly (vector), dipole axial with M_A = 1020 MeV, g_A = 1.2724.
  Darwin–Foldy is included in the vector charge.
- Inversion settings: power_exp basis, α = 1.5, N = 6–8, Γ = 10 MeV, threshold S_n(¹⁶O) = 15.66 MeV (experimental).
- q = 50, 75, 100, 150, 200, 250, 300, 350, 400 MeV.

## Files

`CR_<R>_q<q>.dat` with R = R00, Rzz, R0z, Rxx, Rxy. There are 45 files, each on the same grid of 0.125 MeV steps.

| column | content |
|---|---|
| `E_x` | excitation energy of the final nucleus relative to the ¹⁶O ground state (MeV), from 15.66 to 515.66 |
| `omega` | energy transfer ω = E_x + q²/2M_A (MeV), with M_A = 16 × 931.494 MeV |
| `R` | response (MeV⁻¹), best inversion |
| `R_low`, `R_high` | total band: R ∓ √(dR_inv² + dR_FF²), each side separately |
| `dR_inv_low`, `dR_inv_high` | inversion error, ≥ 0 |
| `dR_FF` | form-factor-correction error, ≥ 0, symmetric |


## Cross section

Neutrino (upper sign) or antineutrino (lower sign) on the nucleus, from Eq. (cross_section) of the notes, with the
responses of these files:

    d²σ/dΩ dε′ = G²/(8π²) · (k′/ε) · [ v00 R00 + vzz Rzz − v0z R0z + vxx Rxx ± vxy Rxy ]


- G = G_F V_ud = 1.1663788×10⁻¹¹ MeV⁻² × 0.97373.
- ε is the neutrino energy, ε′ = ε − ω is the lepton energy, k′ = √(ε′² − m_l²), and θ is the lepton scattering angle.
- q² = ε² + k′² − 2εk′ cos θ, and Q² = q² − ω².

The lepton factors (the 1/4-normalised lepton traces, MeV²):

    v00 = 2εε′ (1 + (k′/ε′) cos θ) = 2ε (ε′ + k′ cos θ)
    vzz = (ω²/q²)(m_l² + v00) + (m_l²/q²) [m_l² + 2ω(ε+ε′) + q²]
    v0z = (ω/q)(m_l² + v00) + m_l² (ε+ε′)/q
    vxx = Q² + (Q²/2q²)(m_l² + v00) − (m_l²/q²) [m_l²/2 + ω(ε+ε′)]
    vxy = Q² (ε+ε′)/q − m_l² ω/q

**Units.** G² [MeV⁻⁴] · v [MeV²] · R [MeV⁻¹] gives MeV⁻³ per sr. Multiply by (ħc)² = 3.8938×10⁻²² MeV² cm² to get
cm²/(MeV sr) per ¹⁶O nucleus.

## Caveats

- **ω** includes the nuclear recoil but not the n–p mass difference. The threshold used, 15.66 MeV, is the experimental S_n(¹⁶O) (16O → 16F → 15O + p).
- **Discrete states.** Bound-state poles separated by the post-processing are not in the files. They are all negligible:
  the largest is 6×10⁻⁴ in r0z_im_im at q300, at E = 12.2 MeV.
- **Tail.** Above E_x ≈ 400–500 MeV the responses are an extrapolation of the inversion basis. The LIT was computed for
  σ ≤ 500 MeV.
