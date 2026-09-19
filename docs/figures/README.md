# Thesis figures

These PNGs were extracted directly from [Davyd Ilnytskyi’s bachelor’s thesis](../../Bachelor%20Ilnytskyi%20Davyd.pdf). Plot data, labels, colors, and resolution are unchanged. The images are original PDF image objects, not regenerated results.

| Asset | Thesis reference | Printed page | PDF page (1-based) |
| :--- | :--- | ---: | ---: |
| `allen-cahn-comparison.png` | Figure 5.8: L1-regularized and vanilla Allen–Cahn solutions against FEM at t = 0.5 | 38 | 49 |
| `burgers-jacobian-heatmap.png` | Figure 5.6(c): Jacobian-regularized Burgers’ solution, FEM reference, and absolute difference | 36 | 47 |
| `burgers-training-error.png` | Figure 5.1(f): mean Burgers’ relative L2 error over five runs, EMA α = 0.01 | 29 | 40 |

The comparison and heatmap figures illustrate individual final-evaluation runs. They should not be interpreted as the five-seed mean values in README tables, which come from thesis Tables 5.5–5.7. The training-error figure does average five runs.

To repeat the extraction from the repository root with Poppler’s `pdfimages` utility:

```bash
mkdir -p /tmp/pinns-readme-assets
pdfimages -f 49 -l 49 -png 'Bachelor Ilnytskyi Davyd.pdf' /tmp/pinns-readme-assets/comparison
pdfimages -f 47 -l 47 -png 'Bachelor Ilnytskyi Davyd.pdf' /tmp/pinns-readme-assets/heatmap
pdfimages -f 40 -l 40 -png 'Bachelor Ilnytskyi Davyd.pdf' /tmp/pinns-readme-assets/training
cp /tmp/pinns-readme-assets/comparison-000.png docs/figures/allen-cahn-comparison.png
cp /tmp/pinns-readme-assets/heatmap-004.png docs/figures/burgers-jacobian-heatmap.png
cp /tmp/pinns-readme-assets/training-010.png docs/figures/burgers-training-error.png
```
