"""One-off script to render a results summary table as a PNG for social posts."""

import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 1, figsize=(11, 7.5), gridspec_kw={"height_ratios": [3, 1.3]})
fig.suptitle(
    "DPO-VP Squeezing Falsification — Qwen2.5-0.5B-Instruct + GSM8K",
    fontsize=15, fontweight="bold", y=0.98,
)

# ── Table 1: main results ────────────────────────────────────────────────
ax1 = axes[0]
ax1.axis("off")

columns = ["Run", "pass@1", "chosen lp", "rejected lp", "gap", "entropy", "KL-from-SFT", "Squeezing?"]
rows = [
    ["Base model",        "0.420", "—",      "—",      "—",      "—",      "—",      "—"],
    ["DPO-VP Round 1",     "0.473", "-0.203", "-0.262", "+0.059", "0.276",  "0.0042",  "No"],
    ["DPO-VP Round 2",     "0.463", "-0.206", "-0.270", "+0.064", "0.282",  "0.0076",  "No"],
    ["DPO-VP Round 3",     "0.480", "-0.208", "-0.273", "+0.065", "0.281",  "0.0096",  "No"],
    ["GRPO (600 steps)",   "0.493", "—",      "—",      "—",      "0.238",  "0.0146",  "No"],
]

col_widths1 = [0.20, 0.10, 0.13, 0.13, 0.10, 0.11, 0.13, 0.13]
tbl = ax1.table(
    cellText=rows, colLabels=columns, loc="center", cellLoc="center",
    colWidths=col_widths1,
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(11)
tbl.scale(1, 2.1)
for (r, c), cell in tbl.get_celld().items():
    if c == 0:
        cell.set_text_props(ha="left")
        cell.PAD = 0.02

header_color = "#1f4e79"
chosen_color = "#eaf2fb"
for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor("#cccccc")
    if r == 0:
        cell.set_facecolor(header_color)
        cell.set_text_props(color="white", fontweight="bold")
    else:
        cell.set_facecolor(chosen_color if r % 2 == 0 else "white")
        if c == 0:
            cell.set_text_props(fontweight="bold", ha="left")
        if c == len(columns) - 1:
            cell.set_text_props(color="#1a7a1a", fontweight="bold")

ax1.set_title(
    "Healthy DPO signature across all 3 rounds: chosen logprob flat, rejected falls faster,\n"
    "gap grows monotonically. GRPO baseline equally stable. No squeezing observed.",
    fontsize=10.5, style="italic", pad=14,
)

# ── Table 2: 135M capacity-floor footnote ───────────────────────────────
ax2 = axes[1]
ax2.axis("off")

columns2 = ["Run", "base pass@1", "probe pair_rate", "round-1 pair_rate", "squeeze probe"]
rows2 = [["SmolLM2-135M (GSM8K)", "0.030", "0.057 (17/300)", "0.125 (150/1200)", "empty — no usable pairs"]]

col_widths2 = [0.26, 0.16, 0.20, 0.20, 0.26]
tbl2 = ax2.table(
    cellText=rows2, colLabels=columns2, loc="center", cellLoc="center",
    colWidths=col_widths2,
)
tbl2.auto_set_font_size(False)
tbl2.set_fontsize(10.5)
tbl2.scale(1, 2.1)
for (r, c), cell in tbl2.get_celld().items():
    if c == 0:
        cell.set_text_props(ha="left")
        cell.PAD = 0.02

for (r, c), cell in tbl2.get_celld().items():
    cell.set_edgecolor("#cccccc")
    if r == 0:
        cell.set_facecolor("#7a2d2d")
        cell.set_text_props(color="white", fontweight="bold")
    else:
        cell.set_facecolor("#fbeaea")
        if c == 0:
            cell.set_text_props(fontweight="bold", ha="left")

ax2.set_title(
    "135M follow-up: capacity floor reached before squeezing could ever be tested",
    fontsize=10.5, style="italic", pad=10,
)

plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig("results_summary.png", dpi=200, bbox_inches="tight")
print("Saved results_summary.png")
