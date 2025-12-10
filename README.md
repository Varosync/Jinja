# Jinja

**Deep Learning Pipeline for GPCR Drug Discovery**

Predicting protein activation pathways and allosteric binding sites using deep learning.

## The Problem

GPCRs are targets for **35% of FDA-approved drugs**, but identifying allosteric binding sites requires expensive molecular dynamics simulations that take weeks to months.

## The Solution

Jinja predicts where a protein sits on its activation pathway in **<50ms** — turning weeks of simulation into instant inference.

```mermaid
graph LR
    A[3D Structure] --> B[ESM3 Tokenization]
    B --> C[Committor Model]
    C --> D[Activation State]
    D --> E[Drug Targets]
    style C fill:#10b981
    style E fill:#f59e0b
```

## Key Results

| Metric | Value |
|--------|-------|
| Validation Accuracy | **100%** |
| Inference Time | **<50ms** |
| Training Dataset | 98,800 conformations |
| Free Energy Barrier | 10.60 kJ/mol |

## Validation

| PDB | State | Predicted | p_B |
|-----|-------|-----------|-----|
| 2RH1 | Inactive | ✓ Inactive | 0.008 |
| 3P0G | Active | ✓ Active | 0.998 |
| 3D4S | Transition | ✓ Transition | 0.463 |

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Run demo
python demo.py

# Web interface
cd web_demo && python -m http.server 8080
```

## Technology

- **ESM3** structure tokenization (EvolutionaryScale)
- **Transformer** committor model (3M parameters)
- **AIMMD** free energy analysis (Jung et al. 2023)
- **Claude 3.5** scientific interpretations

## Applications

- Allosteric drug design
- Biased agonist development
- Virtual compound screening
- Lead optimization

## Citation

```bibtex
@software{jinja2025,
  title={Jinja: AI-Powered GPCR Drug Discovery},
  author={Harry Kabodha, Ayman Khaleq},
  year={2025},
  url={https://github.com/varosync/Jinja}
}
```

## License

MIT License
