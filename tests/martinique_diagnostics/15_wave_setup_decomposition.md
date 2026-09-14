# Decomposing the residual gap: wave setup vs. tide/surge magnitude

Follow-up to `ngm_to_goco06s_conversion.csv`, using the user-supplied TRI Fort-de-France/Lamentin
source-model breakdown for RP100 (02Moy), all in mNGM (IGN 1987 datum):

| component | value (mNGM) |
|---|---|
| Highest astronomical tide (HAT) | 0.4 |
| Cyclonic storm surge | 0.7 |
| Wave setup | 1.6 |
| **Total** | **2.7** (matches the previously-reported 2.7 m NGM figure exactly) |

## Rule used

The NGM->GOCO06s offset (+0.6106 m, `ngm_to_goco06s_conversion.csv` step 5) converts a physical
water-surface elevation from one datum's zero to another's. It must be applied **once** to
whichever total is being converted - applying it separately to each summed component and
re-adding would triple-count it, since datum offsets are a shift of the reference zero, not a
per-term correction: `(a+k)+(b+k)+(c+k) = a+b+c+3k ≠ (a+b+c)+k`.

Wave setup is a *differential* quantity (height added by wave breaking on top of the still-water
level) - it does not need its own datum conversion; it is the same 1.6 m regardless of which
elevation datum the still-water baseline is expressed in.

## Two comparisons, apples-to-apples

COAST-RP's `storm_tide_rp_*` product (and hence GFM's applied boundary forcing) represents
astronomical tide + storm surge only - GTSM/STORM-class models do not simulate wave setup
(a nearshore, wave-breaking-driven process requiring a separate wave model + nearshore
transformation). So the correct comparison excludes wave setup on the TRI side too:

| | mNGM | GOCO06s |
|---|---|---|
| TRI tide + surge only (0.4 + 0.7) | 1.100 | **1.711** (1.100 + 0.611) |
| GFM/COAST-RP RP100, MDT sign fixed | - | **1.016** (`martinique_hitrate_ceiling_grid.csv`) |
| **Tide/surge-magnitude gap** | | **0.695 m** |

| | GOCO06s |
|---|---|
| TRI full total (2.7 + 0.611) | **3.311** |
| GFM/COAST-RP RP100, MDT sign fixed | **1.016** |
| **Full residual gap** | **2.294 m** |
| ... of which wave setup (unmodeled by GFM, expected) | 1.600 m |
| ... of which tide/surge-magnitude discrepancy | 0.695 m |

`0.695 + 1.600 = 2.295 m`, matching the direct full-total subtraction (2.294 m) to rounding -
internally consistent.

## Reading

The user's instinct - that the residual gap is "roughly" the missing wave-setup term - is
essentially right and is the dominant, single largest, and entirely *expected* piece: GFM (like
Spain's own benchmark comparison earlier this session) does not simulate wave setup/run-up at
all, a known, already-documented limitation, not a bug.

But the two numbers are not identical: doing the comparison in one consistent frame shows an
additional **~0.7 m gap in tide+surge magnitude alone**, even before wave setup is considered -
COAST-RP's RP100 storm tide at Martinique is itself lower than the French source model's own
assumed 0.4 m HAT + 0.7 m cyclonic surge. This could reflect real differences in surge
climatology/methodology (COAST-RP's global GTSM/STORM-based approach vs. a local
French model), a return-period definition mismatch, or something else not yet investigated -
flagged as open, not resolved.
