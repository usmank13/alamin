# API cost and timing

| Stage | Unit | Estimated API cost | Approximate time |
| --- | --- | --- | --- |
| Agent LLM | Scene authoring | $0.18 | 2 min |
| fal PBR materials | Material set | $0.07 | 1.5 min |
| fal textured mesh | Asset | $0.375 | 2.5 min |
| Offline render | Image | $0 | 25 s |
| Full RGB-D capture | 1 simulated minute | $0 | 11 min |
| State/range/IMU capture | 1 simulated minute | $0 | 1.6 min |

**fal accounts for the majority of API cost.** For example, five new material sets
and six new meshes cost about **$2.60**, versus **$0.18** for the agent LLM:
approximately **$2.78 total**, with **94% spent on fal**. Cached fal assets incur
no additional generation charge.

The LLM estimate assumes 40,000 input and 4,000 output tokens at illustrative rates
of $3 and $15 per million tokens, respectively; it is not recorded billing.
Timings are approximate and requests can overlap. Local compute costs are excluded.
