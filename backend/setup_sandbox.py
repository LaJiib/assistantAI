#!/usr/bin/env python3
"""
setup_sandbox.py — Initialise la structure du sandbox sur AISSD.

À exécuter une seule fois :
    python backend/setup_sandbox.py
"""
from pathlib import Path

SANDBOX_ROOT = Path("/Volumes/AISSD/iris-sandbox")

STRUCTURE = {
    "workspace": "Dossier de travail principal — copies de fichiers à éditer",
    "experiments": "Scripts et prototypes temporaires",
    "notes": "Notes de l'agent sur les tâches en cours",
    "output": "Résultats produits par l'agent (code généré, rapports...)",
}

README = """\
# Iris Coding Sandbox

Environnement sandboxé pour l'agent de coding d'Iris.

## Structure

| Dossier      | Usage                                              |
|--------------|----------------------------------------------------|
| workspace/   | Copies de fichiers du projet à lire / modifier     |
| experiments/ | Scripts temporaires, prototypes, tests             |
| notes/       | Notes de l'agent (todos, contexte de tâche)        |
| output/      | Code produit, rapports, fichiers générés           |

## Règles

- L'agent ne peut pas accéder à des fichiers hors de ce dossier.
- Toute commande shell doit être approuvée manuellement.
- Copier les fichiers à modifier dans workspace/ avant de les éditer.
- Ne jamais stocker de secrets ou clés API ici.
"""

def setup():
    if not Path("/Volumes/AISSD").exists():
        print("⚠️  /Volumes/AISSD introuvable — vérifie que le disque est monté.")
        return

    SANDBOX_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"✓ Sandbox root : {SANDBOX_ROOT}")

    for folder, description in STRUCTURE.items():
        path = SANDBOX_ROOT / folder
        path.mkdir(exist_ok=True)
        (path / ".gitkeep").touch()
        print(f"  ✓ {folder}/  — {description}")

    readme_path = SANDBOX_ROOT / "README.md"
    if not readme_path.exists():
        readme_path.write_text(README)
        print(f"  ✓ README.md créé")
    else:
        print(f"  · README.md existant, non modifié")

    print(f"\n✅ Sandbox prêt : {SANDBOX_ROOT}")


if __name__ == "__main__":
    setup()
