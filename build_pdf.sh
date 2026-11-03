#!/bin/sh
# REPORT.md -> REPORT.pdf (pandoc + xelatex). GIF swapped for a still frame; ≫ rendered as \gg (missing in DejaVu Serif).
T=$(mktemp --suffix=.md)
sed -e 's|runs/tetris_agents.gif|runs/fig_agents_frame.png|' \
    -e 's|^\*Figure 2. Same piece sequence.|*Figure 2. A still frame (step ~100) of the side-by-side gameplay recording, same piece sequence (animated version in the repository: `runs/tetris_agents.gif`).|' \
    -e 's/≫/$\\gg$/g' REPORT.md > "$T"
pandoc -f markdown+lists_without_preceding_blankline "$T" -o REPORT.pdf --pdf-engine=xelatex --resource-path=. \
  -V mainfont="DejaVu Serif" -V sansfont="DejaVu Sans" -V monofont="DejaVu Sans Mono" -V fontsize=10pt \
  -V geometry:margin=2cm -V colorlinks=true -V linkcolor=blue -V urlcolor=blue \
  -V header-includes='\usepackage{float}\floatplacement{figure}{H}'
rm -f "$T"
