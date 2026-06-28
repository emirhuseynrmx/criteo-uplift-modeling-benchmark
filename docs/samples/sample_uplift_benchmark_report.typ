#set page(width: 210mm, height: 297mm, margin: 14mm)
#set text(font: "Arial", size: 9.4pt)

#let blue = rgb("#2563eb")
#let green = rgb("#059669")
#let muted = rgb("#64748b")
#let border = rgb("#d8e0ea")
#let soft = rgb("#f5f8fb")

#let card(title, value, accent: blue) = block(
  fill: soft,
  radius: 5pt,
  inset: 8pt,
  width: 42mm,
)[
  #text(size: 6.6pt, weight: "bold", fill: muted)[#upper(title)]
  #linebreak()
  #text(size: 15pt, weight: "bold", fill: accent)[#value]
]

#let cell(body, strong: false) = table.cell(inset: 4pt)[
  #text(size: 8.2pt, weight: if strong { "bold" } else { "regular" })[#body]
]

#text(size: 17pt, weight: "bold")[Criteo Uplift Benchmark Report]

#v(2pt)
#text(fill: muted)[Sample report for campaign incrementality modeling. The benchmark asks which users convert because of treatment, not only which users are likely to convert.]

#v(10pt)
#grid(columns: (1fr, 1fr, 1fr, 1fr), gutter: 7pt)[
  #card("Rows", "7M", accent: blue)
][
  #card("Best Tradeoff", "S-Learner", accent: green)
][
  #card("Test AUUC", "3405.48", accent: blue)
][
  #card("Top Decile", "6.23x", accent: green)
]

#v(12pt)
#text(size: 12pt, weight: "bold")[Model Comparison]
#v(4pt)
#table(
  columns: (1.4fr, .8fr, .7fr, .8fr),
  stroke: border,
  [#cell("Model", strong: true)], [#cell("Validation AUUC", strong: true)], [#cell("Runtime", strong: true)], [#cell("Peak RSS", strong: true)],
  [#cell("S-Learner")], [#cell("3495.68")], [#cell("252.3s")], [#cell("5041 MB")],
  [#cell("DR-Learner")], [#cell("3462.59")], [#cell("1646.7s")], [#cell("5727 MB")],
  [#cell("Naive response ranker")], [#cell("3396.22")], [#cell("196.4s")], [#cell("5113 MB")],
  [#cell("X-Learner")], [#cell("3375.79")], [#cell("521.7s")], [#cell("5288 MB")],
  [#cell("Causal Forest")], [#cell("3345.46")], [#cell("3335.2s")], [#cell("14458 MB")],
)

#v(10pt)
#grid(columns: (1fr, 1fr), gutter: 10pt)[
  #text(size: 11pt, weight: "bold")[Accuracy vs Runtime]
  #v(4pt)
  #image("../../assets/tradeoff_accuracy_runtime.png", width: 86mm)
][
  #text(size: 11pt, weight: "bold")[Policy Simulation]
  #v(4pt)
  #image("../../assets/policy_simulation_test.png", width: 86mm)
]

#v(9pt)
#text(size: 11pt, weight: "bold")[Business Read]
#v(3pt)
#block(fill: soft, radius: 5pt, inset: 8pt)[
  #list(
    [Use uplift ranking for campaign targeting, not raw conversion probability.],
    [S-Learner is the practical winner in this run: strong AUUC with much lower runtime than DR-Learner and Causal Forest.],
    [The top decile shows a 6.23x relative uplift signal, which is the first segment to test in a controlled campaign rollout.],
    [The paired bootstrap interval overlaps zero, so this should be treated as a production trade-off decision rather than a universal model claim.]
  )
]

#v(8pt)
#text(size: 11pt, weight: "bold")[Delivered Artifacts]
#v(3pt)
#table(
  columns: (1fr, 2fr),
  stroke: border,
  [#cell("Artifact", strong: true)], [#cell("Purpose", strong: true)],
  [#cell("benchmark_results.csv")], [#cell("AUUC, Qini, runtime, and memory by model")],
  [#cell("policy_segments.csv")], [#cell("Treatment targeting groups and expected incremental response")],
  [#cell("diagnostic_charts/")], [#cell("Qini curve, uplift by decile, SHAP surrogate, policy simulation")],
  [#cell("README assets")], [#cell("Repo-ready visuals for portfolio and stakeholder review")],
)
