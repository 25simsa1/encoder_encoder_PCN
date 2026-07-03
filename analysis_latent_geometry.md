# E4 latent-geometry battery (8k, held-out N=2000, ladder bar >3 hits)

| system | erank img/txt (mu2 i/t) | matched-derangement gap (sigma) | align_cos | unif img/txt | raw | centered | Procrustes | whitened | ridge | probe R2 (null) | verdict |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| PC_armA | 10.6/7.0 (0.95/1.00) | 0.0005 (7.5) | 0.958 | -0.20/-0.01 | 1 | 1 | 1 | 0 | 2 | 0.490 (-0.019) | mean-direction collapse (uncentered erank ~1 / mean carries the energy, no rung rescues); residual caption signal (probe R2=0.49) with no cross-modal map |
| PC_armB | 11.3/15.9 (0.98/1.00) | 0.0001 (5.9) | 0.967 | -0.07/-0.01 | 0 | 3 | 0 | 2 | 4 | 0.727 (-0.052) | mean-direction collapse (uncentered erank ~1 / mean carries the energy, no rung rescues); residual caption signal (probe R2=0.73) with no cross-modal map |
| BPonF | 3.9/3.7 (0.97/0.99) | 0.0001 (1.8) | 0.973 | -0.10/-0.02 | 1 | 0 | 0 | 0 | 0 | 0.363 (-0.000) | mean-direction collapse (uncentered erank ~1 / mean carries the energy, no rung rescues); residual caption signal (probe R2=0.36) with no cross-modal map |
| E1_adam | 75.2/39.9 (0.02/0.00) | 0.0292 (9.5) | 0.027 | -3.81/-3.78 | 5 | 4 | 6 | 5 | 3 | 0.964 (-0.041) | coupled (raw retrieval above the pre-registered bar) |
| E1L_lars | 184.2/44.6 (0.05/0.03) | 0.0316 (14.9) | 0.027 | -3.75/-3.69 | 6 | 5 | 5 | 4 | 4 | 0.937 (-0.049) | coupled (raw retrieval above the pre-registered bar) |
| BPonF_free | 7.4/5.2 (0.86/0.90) | 0.0085 (8.7) | 0.889 | -0.52/-0.36 | 0 | 2 | 2 | 3 | 1 | 0.384 (-0.025) | mean-direction collapse (uncentered erank ~1 / mean carries the energy, no rung rescues); residual caption signal (probe R2=0.38) with no cross-modal map |

PREDICTION HOLDS: PC and BPonF share the failure geometry while the InfoNCE systems differ
