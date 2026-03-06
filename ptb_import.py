"""
ptb_import.py — Utilitaires pour importer et explorer le dataset 
PTB Diagnostic ECG Database (1.0.0)

Fonctionnalités principales :
- Parcours du répertoire des patients/enregistrements
- Lecture des headers (.hea) avec wfdb
- Extraction robuste des diagnostics à partir des commentaires
- Construction d'un DataFrame récapitulatif (pandas)
- Lecture d'un signal ECG (optionnellement subset de leads)

Dépendances: wfdb, pandas, numpy

Auteur: généré par M365 Copilot
"""

from __future__ import annotations
import os
import re
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

try:
    import wfdb  # pip install wfdb
except Exception as e:
    raise ImportError(
        "Le module 'wfdb' est requis. Installe-le avec: pip install wfdb"
    ) from e

# ------------------------------
# Helpers
# ------------------------------

def _is_record_dir(path: str) -> bool:
    """Retourne True si le chemin contient au moins un .hea et .dat."""
    if not os.path.isdir(path):
        return False
    files = os.listdir(path)
    has_hea = any(f.endswith('.hea') for f in files)
    has_dat = any(f.endswith('.dat') for f in files)
    return has_hea and has_dat


def list_patient_dirs(root_dir: str) -> List[str]:
    """Liste les répertoires de patients (patient001, patient002, ...)."""
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"Dossier introuvable: {root_dir}")
    entries = [os.path.join(root_dir, d) for d in os.listdir(root_dir)]
    return sorted([p for p in entries if os.path.isdir(p)])


def list_records(root_dir: str) -> List[Tuple[str, str, str]]:
    """
    Parcourt root_dir et retourne la liste des enregistrements disponibles.

    Retourne: liste de tuples (patient_id, record_stem, record_dir)
    où record_stem est sans extension (ex: 's0010_re').
    """
    records = []
    for patient_dir in list_patient_dirs(root_dir):
        patient_id = os.path.basename(patient_dir)
        # chaque enregistrement a un trio .hea/.dat/.xyz (xyz optionnel)
        for f in sorted(os.listdir(patient_dir)):
            if f.endswith('.hea'):
                stem = f[:-4]
                dat = os.path.join(patient_dir, stem + '.dat')
                hea = os.path.join(patient_dir, stem + '.hea')
                if os.path.exists(dat) and os.path.exists(hea):
                    records.append((patient_id, stem, patient_dir))
    return records


# ------------------------------
# Lecture des headers et diagnostics
# ------------------------------

def read_header(record_dir: str, record_stem: str) -> wfdb.io.record.Record:
    """Lit le header WFDB pour un enregistrement donné (sans charger le signal)."""
    base = os.path.join(record_dir, record_stem)
    header = wfdb.rdheader(base)
    return header


def parse_diagnoses_from_comments(comments: List[str]) -> Dict[str, Any]:
    """
    Extrait diagnostic principal + liste brute à partir des commentaires du header.

    Les fichiers .hea du PTB contiennent souvent des lignes du type:
      - "#Diagnosis: Myocardial infarction" ou
      - "#Diagnoses: Healthy control" ou
      - "#Reason for admission: Myocardial infarction"

    Cette fonction tente de normaliser quelques catégories communes et
    renvoie un dictionnaire:
      {
        'diagnoses_raw': List[str],
        'primary_pathology': str,  # ex: 'myocardial infarction', 'healthy control', 'other'
        'is_normal': int           # 1 normal, 0 anormal
      }
    """
    diags = []
    for c in comments or []:
        # Nettoyage
        c_clean = re.sub(r'^#', '', c).strip()
        diags.append(c_clean)

    text = " \n ".join(d.lower() for d in diags)

    # Heuristiques de détection
    mapping = {
        r"healthy control": "healthy control",
        r"myocardial infarction": "myocardial infarction",
        r"bundle branch block": "bundle branch block",
        r"(dys)?rhythmia": "dysrhythmia",
        r"hypertroph": "hypertrophy",
        r"cardiomyopath": "cardiomyopathy",
        r"myocarditis": "myocarditis",
        r"valvular": "valvular heart disease",
        r"heart failure": "heart failure",
    }

    primary = None
    for pattern, label in mapping.items():
        if re.search(pattern, text):
            primary = label
            break

    if primary is None:
        # Récupérer éventuellement après 'diagnosis' ou 'reason for admission'
        m = re.search(r"(diagnos\w*|reason for admission)\s*[:\-]\s*(.+)", text)
        if m:
            primary = m.group(2).strip().split('\n')[0].strip()
        else:
            primary = 'other'

    is_normal = 1 if primary == 'healthy control' else 0

    return {
        'diagnoses_raw': diags,
        'primary_pathology': primary,
        'is_normal': is_normal,
    }


# ------------------------------
# Construction DataFrame metadata
# ------------------------------

def build_metadata_df(root_dir: str) -> pd.DataFrame:
    """
    Construit un DataFrame avec une ligne par enregistrement.

    Colonnes principales:
      - patient_id
      - record_stem
      - fs (Hz)
      - n_leads
      - siglen (nb d'échantillons)
      - duration_sec
      - sig_names (liste de leads)
      - diagnoses_raw (liste de str)
      - primary_pathology (str)
      - is_normal (int: 1=normal, 0=anormal)
      - record_path (chemin sans extension)
    """
    rows = []
    for patient_id, stem, rec_dir in list_records(root_dir):
        header = read_header(rec_dir, stem)
        fs = header.fs
        n_leads = header.n_sig
        siglen = header.sig_len
        sig_names = header.sig_name
        diags = parse_diagnoses_from_comments(header.comments)
        duration = None
        if fs and siglen:
            try:
                duration = float(siglen) / float(fs)
            except Exception:
                duration = None

        rows.append({
            'patient_id': patient_id,
            'record_stem': stem,
            'fs': fs,
            'n_leads': n_leads,
            'siglen': siglen,
            'duration_sec': duration,
            'sig_names': sig_names,
            'diagnoses_raw': diags['diagnoses_raw'],
            'primary_pathology': diags['primary_pathology'],
            'is_normal': diags['is_normal'],
            'record_path': os.path.join(rec_dir, stem),
        })

    df = pd.DataFrame(rows)
    # Tri pour reproductibilité
    if not df.empty:
        df = df.sort_values(['patient_id', 'record_stem']).reset_index(drop=True)
    return df


# ------------------------------
# Lecture des signaux
# ------------------------------

def read_signal(record_path_no_ext: str,
                leads: Optional[List[int]] = None,
                start: Optional[int] = None,
                stop: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Lit le signal ECG via wfdb.rdrecord.

    Args:
        record_path_no_ext: chemin sans extension vers le record (ex: 
                            '.../patient001/s0010_re')
        leads: indices de leads à garder (ex: [0,1,2]). Si None, tous.
        start, stop: limites d'échantillons (fenêtre). Si None, tout.

    Returns: (signal, meta)
        signal: ndarray shape (T, C)
        meta: dict {'fs', 'sig_name', 'units'} etc.
    """
    rec = wfdb.rdrecord(record_path_no_ext, sampfrom=start or 0, sampto=stop)
    sig = rec.p_signal  # (T, C)
    if leads is not None:
        sig = sig[:, leads]
        sig_names = [rec.sig_name[i] for i in leads]
    else:
        sig_names = rec.sig_name

    meta = {
        'fs': rec.fs,
        'sig_name': sig_names,
        'units': getattr(rec, 'units', None),
        'n_leads': len(sig_names),
        'siglen': sig.shape[0],
    }
    return sig, meta


# ------------------------------
# Patient-level split pour éviter le leakage
# ------------------------------

def train_val_test_split_by_patient(df: pd.DataFrame,
                                    train=0.7, val=0.15, test=0.15,
                                    random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Découpe le DataFrame par patients (groupage par patient_id) pour éviter 
    que des enregistrements du même patient soient dans train et test.
    """
    assert abs((train + val + test) - 1.0) < 1e-6, "Les ratios doivent sommer à 1.0"
    patients = df['patient_id'].unique()
    rng = np.random.default_rng(random_state)
    rng.shuffle(patients)
    n = len(patients)
    n_train = int(round(train * n))
    n_val = int(round(val * n))
    train_pat = set(patients[:n_train])
    val_pat = set(patients[n_train:n_train+n_val])
    test_pat = set(patients[n_train+n_val:])

    df_train = df[df['patient_id'].isin(train_pat)].copy()
    df_val = df[df['patient_id'].isin(val_pat)].copy()
    df_test = df[df['patient_id'].isin(test_pat)].copy()

    return df_train, df_val, df_test


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Construire un CSV metadata pour PTB Diagnostic ECG')
    parser.add_argument('--root', required=True, help='Chemin vers le dossier racine du dataset (qui contient patient001/, patient002/, ...)')
    parser.add_argument('--out', default='ptb_metadata.csv', help='Chemin du CSV de sortie')
    args = parser.parse_args()

    df = build_metadata_df(args.root)
    df.to_csv(args.out, index=False)
    print(f"✅ Metadata sauvegardées dans {args.out} | {len(df)} enregistrements")
