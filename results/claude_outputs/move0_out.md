# Biomechanical Analysis — Climbing Move 0 (t = 6.1s → 9.3s, 33 frames)

**Chart attached:** `move0_analysis.png` — COM speed with deadpoints, key joint angles, and contact-state timeline, all aligned on the same time axis.

## Key structural finding first
Across **all 33 frames**, `left_hand_contact` = **-1** and both `left_foot_contact`/`right_foot_contact` = **-1**. The only active contact point is the **right hand**, which holds **hold #9** from t=6.1–7.3s, goes airborne, and lands on **hold #47** from t=7.5s onward. This is therefore a **single-point-of-contact (one-arm) dynamic swing/bump**, not a foot-driven dyno — every bit of momentum has to come from arm/shoulder pre-tension plus a free-swinging leg/hip pendulum, since there is nothing to push off.

---

## 1. Dynamism / Explosivity & Accuracy

**Contact-change event (exact):**
- t=7.3s: `right_hand_contact = 9` (still attached)
- **t=7.4s: `right_hand_contact = -1`** ← release frame
- **t=7.5s: `right_hand_contact = 47`** ← catch frame
- Flight time ≈ **0.1 s** — a very short, committed swing.

**Load phase (t=6.1–7.0s):** COM speed is low and oscillating (5.8–110 px/frame, mostly <55), right armpit angle stays near 0–8° (shoulder tucked to torso — efficient tension position) and right elbow sits mid-flexed (67–87°). This is a static rocking/positioning phase on the single hold.

**Drive phase (t=7.0–7.4s):** COM speed rises monotonically and steeply: 28 → 52 → 73 → 96 → **118 px/frame** — roughly a 4× acceleration in 0.4 s. This is *simultaneous* with:
- Right armpit angle jumping 8° → 51° (shoulder driving the throw)
- Right elbow angle collapsing 76° → **17°** (a rapid "cocking" flexion — pulling the elbow in tight right before extension)
- Right elbow velocity spiking to 470 → **580 px/frame** (peak exactly at the release frame, t=7.4s)
- Right wrist velocity peaking at **1045 px/frame at t=7.3s**, the single highest reading in the whole dataset — the literal instant of "the throw"
- Hip/knee velocities rising in parallel (right hip vel 143 px/frame, left hip vel 132 px/frame at t=7.4s) — confirming the **legs/hips are swinging in sync with the arm**, not passively along for the ride.

**Accuracy:** The hand lands on the new hold in a single frame (no multi-frame "fumbling"), and COM y-velocity flips sharply from +69 to -6 px/frame at the moment of catch — an abrupt arrest, evidence of a well-aimed throw. However, residual horizontal drift (com_vel_x still ~90 px/frame at catch) and continued vertical oscillation for 3 more frames (462→457→446→436) show the swing wasn't perfectly "dead" on arrival — some post-catch correction/absorption was needed (mild barn-door).

**A second, larger dynamic burst** occurs post-catch: COM speed climbs again to a dataset-maximum **197.8 px/frame at t=8.6s**, coincident with the right ankle velocity peak (**782 px/frame**, the largest velocity value anywhere in the data) and right knee velocity peaking (466 px/frame). This is a big leg-kick used to drive the body upward *after* the hand-swap, larger than the transition swing itself.

---

## 2. Power Usage & Efficiency

- **Pre-throw isometric loading:** Sustained mid-flexion of the right elbow (67–87°) and near-zero armpit angle for ~0.9s (t=6.1–7.0s) reflects real, sustained shoulder/forearm isometric loading before the dynamic release — this is the highest-fatigue-risk phase, since grip/forearm endurance is being spent while essentially static.
- **Stretch-shorten pattern:** The elbow's rapid flex-then-extend cycle (76°→17°→93°→122° over t=6.9–8.8s) is a classic wind-up/release/lock sequence — using a brief compact "cocked" position before extension is a mechanically efficient way to generate throw velocity without pure brute-force pulling.
- **Leg/hip contribution offloads the arm:** Left hip angle swings across an enormous range (166°→64°→178°) and right knee goes from deeply bent (105°) to nearly straight (173°). This large-amplitude hip/knee motion is generating pendulum momentum that the single working arm would otherwise have to supply alone — good technique, since it reduces the pure grip/arm force needed to complete the reach.
- **Costliest phase for power:** After the catch, the right armpit angle keeps climbing all the way to ~124° (near full overhead shoulder flexion) while the elbow extends to 88–95°, sustained for over a second (t=7.5–8.9s) — this is a long, one-armed, near-locked-out static hang/reach, a significant shoulder/lat/triceps power-endurance demand, arguably costing more cumulative energy than the dynamic swing itself.
- **Efficiency verdict:** The move is mechanically efficient in its momentum generation (legs/hips doing the "engine" work, arm doing a compact cock-and-release), but energy-inefficient in its *duration* — both the pre-throw static hang and the post-catch extended one-arm lock-off are prolonged loaded phases that cost more forearm/shoulder endurance than the actual 0.1s dynamic transition. Shortening these dwell times would meaningfully reduce fatigue cost.

---

## 3. Body Positioning & Balance

- **Clearest deadpoint:** **t=6.8s, COM speed = 5.8 px/frame** (lowest value in the entire sequence), with a secondary near-stall at t=6.6s (9.2 px/frame). This is the natural "quiet point" of the pre-swing pendulum, and the data shows the climber correctly times the explosive pull to begin right after this stall (speed climbs 5.8→12.7→28→52 over the next 3 frames).
- **Second deadpoint cluster: t=8.9–9.0s (19.2 and 13.9 px/frame)** — this occurs right after the big post-catch leg-driven surge (peak at t=8.6s), functioning as a "reset/settle" point where the body re-stabilizes (right armpit ~124°, elbow ~86–89°, near lock-off) before the free (left) arm explosively swings out again at t=9.1s (elbow velocity 700 px/frame, wrist velocity 953 px/frame) to presumably initiate the next move.
- **Net displacement:** COM moves from (363, 401) to (504, 412) — **+141 px horizontally, only +11 px vertically net**. Despite large *intermediate* vertical excursions (com_y sags from 401→462 during the swing, dips again to 467 by t=8.3s, then climbs sharply back to 412 by t=9.3s), the move is fundamentally a **rightward/lateral traverse-style dynamic move**, with the body dropping and re-climbing through a double-sag pendulum path rather than a pure vertical dyno.
- **Balance risk:** With zero foot contact for the entire sequence, all orientation control depends on arm tension and hip positioning. Marked left/right hip-angle asymmetry mid-sequence (e.g., t=8.3s: left hip 81° vs. right hip 136°) indicates the torso is twisted/off-square during the drive — a classic single-point-contact "barn-door" risk. The large right-leg kick at t=8.3–8.6s (knee 101°→173°, ankle velocity peaking at 782 px/frame) appears to function as a **free-swinging counterbalance/flag** to control body rotation, since there's no foothold to push against.

---

## Summary & Suggestions for Improvement

| Aspect | Finding | Suggestion |
|---|---|---|
| Set-up dwell | ~0.9s static one-arm hang before initiating swing (t=6.1–7.0s) | Shorten the load phase — begin the pendulum swing sooner after establishing tension, to conserve grip/forearm endurance |
| Release timing | Peak wrist velocity (1045 px/frame) occurs at t=7.3s, one frame *before* actual release (t=7.4s), while elbow velocity is still climbing to its own peak at the release frame | Release fractionally earlier, at the true velocity peak, to convert maximum swing energy into flight rather than "holding on" through the peak |
| Post-catch stabilization | Residual horizontal drift + 3-frame vertical oscillation after the catch signals imperfect "dead" landing; body also shows hip-angle asymmetry (barn-door risk) throughout the no-feet phase | If any foothold is available, use even a light foot touch/flag during the swing and immediately after the catch to kill excess rotation/swing-through and reduce reliance on the free-leg counterbalance |
| Post-catch lock-off | Sustained near-full shoulder flexion + elbow extension one-armed for ~1.4s (t=7.5–8.9s) is a major power/endurance cost | Reduce the duration of the extended one-arm reach — match hands or find a foothold sooner rather than holding a long static extension |
| Second swing (t=8.0–8.6s) | Larger COM-speed peak (197.8 px/frame) and largest ankle velocity in the dataset (782 px/frame) than the original transition swing | If this second surge is meant only to reposition for the next move, consider a more controlled, lower-amplitude drive to conserve energy for what follows, rather than a second near-maximal pendulum swing |

**Overall:** This is a well-timed, well-coordinated single-point dynamic swing (good use of leg/hip momentum to offload the arm, good synchrony between body and arm at the moment of release, and a clean single-frame catch). The main inefficiencies are **temporal** — a longer-than-necessary static hang before the throw and an extended one-arm lock-off after the catch — both of which unnecessarily tax grip and shoulder endurance relative to the brief, well-executed dynamic transition itself.