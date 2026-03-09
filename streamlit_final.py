#biblios
import pandas as pd 
import numpy as np
import streamlit as st
import os, sys, json
import matplotlib.pyplot as plt
import seaborn as sns
import re
import scipy.signal as sp

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import librosa

# Le module utilitaire généré
import importlib
if 'ptb_import' not in sys.modules:
    sys.path.append('.')
ptb_import = importlib.import_module('ptb_import')


import pickle
#meta_df = pd.read_csv('ptb_metadata.csv')
meta_df = pd.read_csv('https://raw.githubusercontent.com/Melanie94480/heartbeat/refs/heads/main/ptb_metadata.csv')

with open("all_signals_20s.pkl", "rb") as f:
    all_signals_20s = pickle
    
    
st.sidebar.title("Navigation des deux datasets : PTB & MITBIH")

dataset = st.sidebar.selectbox(
    "Choisir un dataset :",
    ["PTB", "MITBIH"],
    index=0,
    key="dataset_choice"
)

pages = {
    "PTB": ["Exploration du DataFrame", "Sélection Patient", "Modèles ML","TEST"],
    "MITBIH": ["Exploration", "DataVizualisation", "Modélisation" ]
}

# Options adaptées au dataset sélectionné
page_options = pages[dataset]

page = st.sidebar.radio(
    f"Pages disponibles pour {dataset} :",
    page_options,
    index=0,
    key="page_choice"
)


#PAGE 1
if page == "Exploration du DataFrame":
    #titre page :
    st.title('Données brutes du dataset PTB diag-ecg-database-1.0.0')

    # recuperer csv metadatas et records
    st.write('Exploration du dataframe')
    st.write(meta_df.head())
    st.write('Taille du dataset : ' , meta_df.shape)
    
    st.subheader(' Repartition pathologies')
    #visualisation des pathologies principales :
    label_counts = meta_df['primary_pathology'].value_counts().sort_values(ascending=False)
    
    plt.figure(figsize=(10,5))
    label_counts.head(5).plot(kind='bar')
    plt.title('Principales pathologies (primary_pathology)sur les patients du dataset')
    plt.ylabel('Nombre d"enregistrements')
    plt.xlabel('Pathologie')
    plt.tight_layout()
    plt.xticks(rotation=45, ha="right")
    st.pyplot(plt) #affichage sur streamlit 
    st.write(label_counts.head())
    
        
    # Créer deux colonnes
    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Repartition patients')  #(avec: 1 normal, 0 anormal)
        class_counts = meta_df['is_normal'].map({1:'Normal',0:'Anormal'}).value_counts()
        fig, ax = plt.subplots(figsize=(5,5))
        ax.pie(
            class_counts.values,
            labels=class_counts.index,
            autopct='%1.1f%%',
            startangle=90,
            colors=["#EE5A5AA4", "#BCF7A6"],  # Vert / Rouge
            explode=[0.05, 0.05]            # léger zoom pour le style
        )

        ax.set_title("Répartition Normal vs Anormal")
        ax.axis('equal')  # découpe circulaire propre
        st.pyplot(fig)
        #st.write('Rappel : normal=1, anormal=0 ')

    with col2 :
        st.subheader('Repartition sexe ') #(avec: "female": 1, "male": 0)
        class_counts = meta_df['sex'].map({1:'female',0:'male'}).value_counts()
        fig, ax = plt.subplots(figsize=(5,5))
        ax.pie(
            class_counts.values,
            labels=class_counts.index,
            autopct='%1.1f%%',
            startangle=90,
            colors=["#A8C7F3A2", "#EA9ED8D6"],  # Vert / Rouge
            explode=[0.05, 0.05]            # léger zoom pour le style
        )

        ax.set_title("Répartition Normal vs Anormal")
        ax.axis('equal')  # découpe circulaire propre
        st.pyplot(fig)
    
    
    # Plot repartition normal / anormale par sexe
    df = meta_df.copy()
    df['sex'] = df['sex'].map({1:'female', 0 : 'male'})
    df['is_normal'] = df['is_normal'].map({1:'Normal', 0:'Anormale'})

    st.subheader('Répartition Normal / Anormal par sexe')
    fig, ax = plt.subplots(figsize=(8,5))
    sns.countplot(data=df, x="sex", hue="is_normal",
                palette=["#EE5A5A", "#8DEB8D"], ax=ax)

    for bars in ax.containers:
        for bar in bars:
            sex = bar.get_x() + bar.get_width()/2
            sex_name = ax.get_xticklabels()[round(sex)].get_text()
            pct = bar.get_height() / df[df['sex']==sex_name].shape[0] * 100
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height(),
                    f"{pct:.1f}%", ha='center', va='bottom')
    ax.set_title("Répartition Normal / Anormal par sexe")
    ax.set_ylabel("Nombre d'enregistrements")

    st.pyplot(fig)



#PAGE 2
if page == "Sélection Patient":
    st.title('Visualisation pour un patient')
    
    # Sélection du patient
    #meta_df = pd.read_csv('ptb_metadata.csv')
    patients = sorted(meta_df["patient_id"].unique())
    selected_patient = st.selectbox("Choisissez un patient", patients)

    # Filtrage des enregistrements de ce patient
    patient_records = meta_df[meta_df["patient_id"] == selected_patient]

    # Extraire les enregistrements disponibles
    record_stem = patient_records["record_stem"].tolist() 
    selected_record = st.selectbox("Choisissez un enregistrement", record_stem)

    # Afficher l’enregistrement sélectionné
    st.write("Données de l'enregistrement sélectionné :")
    st.write(patient_records[patient_records["record_stem"] == selected_record])


    # AFFICHER METADATA 
    # --- État persistant pour afficher/masquer car description metadata longue
    if "show_diag" not in st.session_state:
        st.session_state.show_diag = False  # initialement masqué

    # afficher metadata du patient associé à l’enregistrement sélectionné :
    if st.button("Afficher/Masquer le diagnostic"):
        st.session_state.show_diag = not st.session_state.show_diag
        
    if st.session_state.show_diag:
        def linebreak(text):
            # Mot commençant par une majuscule et ayant plus de 4 lettres
            pattern = r'\b([A-Z][a-zA-Z]{4,})'
            return re.sub(pattern, r'\n\1', text)
        
        #st.write(patient_records['diagnoses'][patient_records["record_stem"] == selected_record])
        diag = patient_records.loc[patient_records["record_stem"] == selected_record, "diagnoses"].iloc[0]
        diag_clean = linebreak(diag)
        st.text(f"Diagnostic : {diag_clean}")

    #AFFICHER ECG:
    # --- État persistant pour afficher/masquer car description metadata longue
    if "show_ecg" not in st.session_state:
        st.session_state.show_ecg = False  # initialement masqué

    # afficher metadata du patient associé à l’enregistrement sélectionné :
    if st.button("Afficher/Masquer l'ECG"):
        st.session_state.show_ecg = not st.session_state.show_ecg
        
    if st.session_state.show_ecg:
        # Sélection d'un enregistrement
        row = patient_records[patient_records["record_stem"] == selected_record].iloc[0]
        record_path = row['record_path']
        print('Record:', record_path)

        sig, meta = ptb_import.read_signal(record_path)
        fs = meta['fs']
        lead_names = meta['sig_name']

        # Visualiser les 3 premiers leads sur 20 secondes
        sec = 20
        stop = int(sec * fs)
        plt.figure(figsize=(12,6))
        
        for i in range(min(3, sig.shape[1])):
            plt.subplot(3,1,i+1)
            plt.plot(sig[:stop, i])
            plt.title(f"Lead {lead_names[i]} — {sec}s")
            plt.xlabel('échantillon')
            plt.ylabel('mV')
        plt.tight_layout()
        st.pyplot(plt)
        st.write(f"Visualisation de l'enregistrement ({selected_record}) des 3 premières dérivations pour le {selected_patient} , sur 20sec")
        
        
        #DETECTION RR
        st.title(" Détection des intervalles RR pour le ML avec la méthode Pan_Thompkins")

        # Charger le fichier 20s pré-calculé
        with open("all_signals_20s.pkl", "rb") as f:
            all_signals_20s = pickle.load(f)

        # Retrouver l'entrée correspondant au record_path sélectionné
        entry = next(e for e in all_signals_20s if e["record_path"] == record_path)

        # Récupération du segment 20s (NE PAS utiliser meta ici)
        fs = entry["fs"]
        lead_names = [n.lower() for n in entry["lead_names"]]
        idx_ii = lead_names.index("ii")

        # Le vrai signal 20s :
        sig_20s = entry["signal_20s"][:, idx_ii].astype(float)

        # Détection R-peaks sur 20 secondes
        r_peaks, sig_filt = ptb_import.pan_tompkin_precise3(sig_20s, fs)
        n_rr = max(0, len(r_peaks) - 1)

        # Plot
        t = np.arange(len(sig_20s)) / fs
        fig, ax = plt.subplots(figsize=(14, 3))
        ax.plot(t, sig_filt, lw=1)
        ax.scatter(r_peaks/fs, sig_filt[r_peaks], color="red", s=40)
        ax.set_title(f"Détection R-peaks — {n_rr} intervalles RR (20 s)")
        ax.grid(alpha=0.3)
        st.pyplot(fig)

        st.success(f"Nombre d’intervalles RR détectés sur 20s: {n_rr}")
        
        
#PAGE 3
if page == "Modèles ML":
    
    with open("all_signals_20s.pkl", "rb") as f:
        all_signals_20s = pickle.load(f)

    st.title('Mise en place Machine Learning ')
    df_train = pd.read_csv("ptb_train.csv")
    df_val   = pd.read_csv("ptb_val.csv")
    df_test  = pd.read_csv("ptb_test.csv")
    
    st.write('Visualisation de la répartition apès découpage :')
    st.write('df_train : ')
    st.dataframe(df_train.head(2))
    st.write('df_val : ')
    st.dataframe(df_val.head(2))
    st.write('df_test : ')
    st.dataframe(df_test.head(2))

    X_train, y_train = ptb_import.build_rr_dataset(df_train, all_signals_20s, L=30)
    X_val,   y_val   = ptb_import.build_rr_dataset(df_val,   all_signals_20s, L=30)
    X_test,  y_test  = ptb_import.build_rr_dataset(df_test,  all_signals_20s, L=30)

    st.write("Jeu de données Train (X_train, y_train) :", X_train.shape, y_train.shape)
    st.write("Jeu de données Val (X_val,   y_val)  :", X_val.shape,   y_val.shape)
    st.write("Jeu de données Test (X_test,  y_test) :", X_test.shape,  y_test.shape)

    st.write("SELECTION DU MODELE ML ")
    selected_modele = st.selectbox("Choisissez un modèle ", ("RandomForest", "Regression logistique" , "PYTORCH  avec input : RR ", "PYTORCH  avec inputs : RR + métadonnées"))
    
    if selected_modele == 'PYTORCH  avec input : RR ': 
        #creation des dataloaders
        class RRDataset(Dataset):
            def __init__(self, X, y, normalize=True):
                X = np.asarray(X, dtype=np.float32)
                y = np.asarray(y, dtype=np.float32)

                self.X = torch.from_numpy(X)           # (N, 30)
                self.y = torch.from_numpy(y).view(-1, 1)  # (N, 1)

            def __len__(self):
                return len(self.X)

            def __getitem__(self, idx):
                return self.X[idx], self.y[idx]
            
                    
        train_ds = RRDataset(X_train, y_train)
        val_ds   = RRDataset(X_val, y_val)
        test_ds  = RRDataset(X_test, y_test)

        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        val_loader   = DataLoader(val_ds, batch_size=32, shuffle=False)
        test_loader  = DataLoader(test_ds, batch_size=32, shuffle=False)
            
        #creation du modele    
        class RRClassifierCNN(nn.Module):
            def __init__(self):
                super().__init__()

                self.feature_extractor = nn.Sequential(
                    nn.Conv1d(1, 16, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.BatchNorm1d(16),
                    nn.Conv1d(16, 32, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.BatchNorm1d(32),
                    nn.AdaptiveAvgPool1d(1)   # -> (batch, 32, 1)
                )

                self.classifier = nn.Sequential(
                    nn.Dropout(0.3),
                    nn.Linear(32, 1)
                )
            def forward(self, x):
                x = x.unsqueeze(1)      # (B,1,30)
                z = self.feature_extractor(x).squeeze(-1)  # (B,32)
                return self.classifier(z)

            
        #definition loss et optimizer
        device = "cpu"
        model = RRClassifierCNN().to(device)

        pos = float((y_train == 1).sum())
        neg = float((y_train == 0).sum())
        ratio = neg / max(1.0, pos)

        cap = 2.0  # ou 3.0 à tester
        pos_weight = torch.tensor([min(ratio, cap)], dtype=torch.float32, device=device)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)

        #boucle d'entrainement :
        best_val = float('inf')
        for epoch in range(1, 50):
            train_loss = ptb_import.train_one_epoch(model, train_loader, optimizer, criterion)
            val_loss, val_acc = ptb_import.eval_one_epoch(model, val_loader, criterion)

            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.cpu().clone() for k,v in model.state_dict().items()}

            print(f"[Epoch {epoch:02d}] Train: {train_loss:.4f} | Val: {val_loss:.4f} | Acc Val: {val_acc:.3f}")

        # Charger meilleurs poids
        model.load_state_dict(best_state)
        model.to(device)
        
        #test final
        test_loss, test_acc = ptb_import.eval_one_epoch(model, test_loader, criterion)
        st.write(f"Test loss: {test_loss:.4f} | Test Acc: {test_acc:.3f}")
        
        st.write("=== MODÈLE UTILISÉ POUR LE TEST ===")
        st.write(model)
        
        #correlation matrice
        model.eval()
        all_probs = []
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                logits = model(xb)
                probs = torch.sigmoid(logits)

                preds = (probs >= 0.5).float()

                all_probs.append(probs.cpu())
                all_preds.append(preds.cpu())
                all_targets.append(yb.cpu())

        # Concat
        probs_test = torch.vstack(all_probs).numpy().ravel()
        preds_test = torch.vstack(all_preds).numpy().ravel()
        targets_test = torch.vstack(all_targets).numpy().ravel()
        
        cm = confusion_matrix(targets_test, preds_test)

        plt.figure(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Prédit 0 (anormal)", "Prédit 1 (normal)"],
                    yticklabels=["Vrai 0 (anormal)", "Vrai 1 (normal)"])
        plt.title("Matrice de confusion – Test")
        st.pyplot(plt)
    
    
    
    if selected_modele == "PYTORCH  avec inputs : RR + métadonnées" : 
        # 1) Fit sur train
        enc = ptb_import.fit_meta_encoder(df_train)

        # 2) Transform sur train/val/test avec le même encodeur
        X_meta_train = ptb_import.transform_meta(df_train, enc)
        X_meta_val   = ptb_import.transform_meta(df_val,   enc)
        X_meta_test  = ptb_import.transform_meta(df_test,  enc)

        #st.write("meta_dim (via encodage):", enc["meta_dim"])
        #st.write("Train meta shape:", X_meta_train.shape)
        #st.write("Val   meta shape:", X_meta_val.shape)
        #st.write("Test  meta shape:", X_meta_test.shape)

        # Sanity check
        assert X_meta_train.shape[1] == X_meta_val.shape[1] == X_meta_test.shape[1] == enc["meta_dim"]
            
        #maj dataset 
        class RRMetaDataset(Dataset):
            def __init__(self, X_rr, X_meta, y):
                self.X_rr = torch.tensor(X_rr, dtype=torch.float32)
                self.X_meta = torch.tensor(X_meta, dtype=torch.float32)
                self.y = torch.tensor(y, dtype=torch.float32).view(-1,1)

            def __len__(self):
                return len(self.X_rr)

            def __getitem__(self, idx):
                return self.X_rr[idx], self.X_meta[idx], self.y[idx]
            
            
        train_ds = RRMetaDataset(X_train, X_meta_train, y_train)
        val_ds   = RRMetaDataset(X_val,   X_meta_val,   y_val)
        test_ds  = RRMetaDataset(X_test,  X_meta_test,  y_test)

        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        val_loader   = DataLoader(val_ds, batch_size=64)
        test_loader  = DataLoader(test_ds, batch_size=64)
        
        #model
        class RRMetaCNN(nn.Module):
            def __init__(self, meta_dim):
                super().__init__()

                # CNN sur RR
                self.cnn = nn.Sequential(
                    nn.Conv1d(1, 16, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv1d(16, 32, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool1d(1)
                )

                # MLP sur metadata
                self.meta_mlp = nn.Sequential(
                    nn.Linear(meta_dim, 32),
                    nn.ReLU(),
                    nn.Dropout(0.1)
                )

                # Fusion
                self.fusion = nn.Sequential(
                    nn.Linear(32 + 32, 32),
                    nn.ReLU(),
                    nn.Linear(32, 1)
                )

            def forward(self, rr, meta):
                rr = rr.unsqueeze(1)                 # (batch, 1, 30)
                rr_feat = self.cnn(rr).squeeze(-1)   # (batch, 32)

                meta_feat = self.meta_mlp(meta)      # (batch, 32)

                x = torch.cat([rr_feat, meta_feat], dim=1)
                return self.fusion(x)
            
        device ='cpu'

        # Calcul du poids de classe (utile car dataset déséquilibré)
        pos = (y_train == 1).sum()
        neg = (y_train == 0).sum()
        pos_weight = torch.tensor([neg / pos], dtype=torch.float32).to(device)
        
        meta_dim = X_meta_train.shape[1]
        model2 = RRMetaCNN(meta_dim).to(device)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(model2.parameters(), lr=1e-3)
        
        #Boucle d’entraînement complète (train + val)
        best_val = float('inf')
        EPOCHS = 50
        device = 'cpu'

        for epoch in range(1, EPOCHS + 1):
            train_loss = ptb_import.train_one(model2, train_loader, optimizer, criterion, device)
            val_loss, val_acc = ptb_import.eval_one(model2, val_loader, criterion, device)

            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.cpu().clone() for k, v in model2.state_dict().items()}

            print(f"Epoch {epoch:02d} | "
                f"Train Loss = {train_loss:.4f} | "
                f"Val Loss = {val_loss:.4f} | "
                f"Val Acc = {val_acc:.3f}")

        # Charger les meilleurs poids
        model2.load_state_dict(best_state)
        model2.to(device)
        
        # Évaluation finale sur le test set
        test_loss, test_acc = ptb_import.eval_one(model2, test_loader, criterion, device)
        st.write(f"\n[TEST] Loss = {test_loss:.4f} | Accuracy = {test_acc:.3f}")
        
        st.write("=== MODÈLE UTILISÉ POUR LE TEST ===")
        st.write(model2)

        #PRED
        # === Générer prédictions test ===
        model2.eval()
        st.session_state["model2"] = model2
        preds_test = []

        with torch.no_grad():
            for rr, meta, y in test_loader:
                rr, meta = rr.to(device), meta.to(device)
                logits = model2(rr, meta)
                probs = torch.sigmoid(logits).cpu().numpy().flatten()
                preds = (probs >= 0.5).astype(int)
                preds_test.extend(preds)

        # Vraies étiquettes
        targets_test = y_test.astype(int)  
        
        #correlation matrice
        cm1 = confusion_matrix(targets_test, preds_test)
        
        plt.figure(figsize=(5,4))
        sns.heatmap(cm1, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Prédit 0 (anormal)", "Prédit 1 (normal)"],
                    yticklabels=["Vrai 0 (anormal)", "Vrai 1 (normal)"])
        plt.title("Matrice de confusion – Test")
        st.pyplot(plt)
        
        # Stockage dans la session pour la page TEST
        st.session_state["model2"] = model2
        st.session_state["df_test"] = df_test
        st.session_state["X_test"] = X_test
        st.session_state["X_meta_test"] = X_meta_test
        st.session_state["y_test"] = y_test
        
        
    if selected_modele == "RandomForest": 
        rf = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            random_state=42,
            class_weight="balanced"  # très important car classes déséquilibrées
        )

        rf.fit(X_train, y_train)

        # Validation
        y_val_pred = rf.predict(X_val)
        acc_val = accuracy_score(y_val, y_val_pred)

        st.write(f"Accuracy Validation RF : {acc_val:.3f}")

        # Test final
        y_test_pred = rf.predict(X_test)
        acc_test = accuracy_score(y_test, y_test_pred)

        st.write(f"Accuracy Test RF : {acc_test:.3f}")
        st.text(classification_report(y_test, y_test_pred))

        cm = confusion_matrix(y_test, y_test_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        st.pyplot(fig)
        
        
    if selected_modele == "Regression logistique":          
        logreg = LogisticRegression(
            penalty="l2",
            solver="liblinear",
            class_weight="balanced",
            max_iter=2000,
            random_state=42
        )

        logreg.fit(X_train, y_train)

        # Validation
        y_val_pred = logreg.predict(X_val)
        acc_val = accuracy_score(y_val, y_val_pred)
        st.write(f"Accuracy Validation LogReg : {acc_val:.3f}")

        # Test final
        y_test_pred = logreg.predict(X_test)
        acc_test = accuracy_score(y_test, y_test_pred)
        st.write(f"Accuracy Test LogReg : {acc_test:.3f}")

        st.text(classification_report(y_test, y_test_pred))
        cm = confusion_matrix(y_test, y_test_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        st.pyplot(plt)
        
        

##PAge 4
if page == "TEST":
    st.title("🔍 Prédiction patient ")
    st.header("Modele CNN (RR + métadonnées)")
    
#  Récupération du modèle et des données
    #model
    device = "cpu"
    model2 = st.session_state.get("model2", None)
    model2.to(device).eval()
    
    #datasets
    df_test  = pd.read_csv("ptb_test.csv")
    df_train = pd.read_csv("ptb_train.csv")
    
    #signaux
    with open("all_signals_20s.pkl", "rb") as f:
        all_signals_20s = pickle.load(f)
        
    X_test,  y_test  = ptb_import.build_rr_dataset(df_test,  all_signals_20s, L=30)

    #encodage des metadonnées
    enc = ptb_import.fit_meta_encoder(df_train)
    X_meta_test = ptb_import.transform_meta(df_test, enc)

    # 3) Selectbox patient_id
    patient_ids = df_test["patient_id"].unique().tolist()
    selected_patient = st.selectbox("Sélectionne un patient", patient_ids)

    df_patient = df_test[df_test["patient_id"] == selected_patient]

    # 4) Selectbox record_stem
    record_list = df_patient["record_stem"].unique().tolist()
    selected_record = st.selectbox("Sélectionne un enregistrement", record_list)

    # Trouver l'index correspondant dans df_test
    row = df_test[(df_test["patient_id"] == selected_patient) &
                (df_test["record_stem"] == selected_record)].iloc[0]
    idx = row.name   # index global dans df_test
    
    # ============================
    # 5) Bouton de prédiction
    # ============================
    if st.button("🔮 Lancer la prédiction"):

    # RR : remettre absolument en 1D
        rr_np = np.array(X_test[idx]).squeeze()                 # (L,)
        rr_t  = torch.tensor(rr_np, dtype=torch.float32).unsqueeze(0)  # (1,L)

        # Meta : (B, meta_dim)
        meta_np = X_meta_test[idx]
        meta_t  = torch.tensor(meta_np, dtype=torch.float32).unsqueeze(0)           # (1,meta_dim)


        rr_t, meta_t = rr_t.to(device), meta_t.to(device)

        
        # Inférence
        with torch.no_grad():
            logits = model2(rr_t, meta_t)
            prob = torch.sigmoid(logits).item()

        y_pred = int(prob >= 0.5)

        # Résultats
        st.subheader("🩺 Résultat")
        st.write(f"**Patient ID** : {selected_patient}")
        st.write(f"**Record stem** : {selected_record}")
        st.write(f"**Probabilité classe 1 (Normal)** : {prob:.4f}")
        st.write(f"**Prédiction** : {y_pred}")


    

#===========================================================


#url= https://www.dropbox.com/scl/fi/kf22pylagnd5gxuu22cma/mitbih_train.zip?rlkey=d4jyzmffeulto7rhqskc0gk3r&st=z6hrmj1o&dl=1
#url= https://www.dropbox.com/scl/fi/g8fd2f2yfq7p3kmu8s2lq/mitbih_test.zip?rlkey=alghgxzbu2p3qw2l1y6atgg5l&st=194m0cnk&dl=1

import io, zipfile, requests, pandas as pd, streamlit as st

URL_TRAIN = "https://drive.google.com/file/d/1ywt8H4j98_ZUGC7YheS0BtfM6Tn4B62a/"
URL_TEST  = "https://drive.google.com/file/d/1j1t1NfPTRy0UWWkxdxxz-0VifsI5UL4S/ "

@st.cache_data(show_spinner=True, ttl=24*3600)  # cache 24h
def read_csv_from_dropbox_zip(url: str, inner_csv: str, sep=None):
    # ⚠️ verify=False pour ignorer les certificats du proxy d'entreprise
    r = requests.get(url, timeout=180, verify=False)
    r.raise_for_status()

    # Lecture du ZIP en mémoire
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        with z.open(inner_csv) as f:
            return pd.read_csv(f, sep=sep)


df1_train = read_csv_from_dropbox_zip(URL_TRAIN, "mitbih_train.csv")
df1_test  = read_csv_from_dropbox_zip(URL_TEST,  "mitbih_test.csv")
#df1_train.columns = [i+1 for i in range(187)] + ["Classe"]

if page == "Exploration" : 
    st.write("## Introduction")
    st.write("")
    st.write("")
    df1_train.columns = [i+1 for i in range(187)] + ["Classe"]
    st.success(f"Train: {df1_train.shape}, Test: {df1_test.shape}")
    
    first_cols = df1_train.iloc[:, :5]
    last_cols = df1_train.iloc[:, -5:]
    milieu = pd.DataFrame(["..."] * len(df1_train), columns=["..."])

    st.write("<h3 style='text-align: center;'>Aperçu</h3>", unsafe_allow_html=True)
    df_apercu = pd.concat([first_cols, milieu, last_cols], axis=1)
    st.dataframe(df_apercu.head(10), use_container_width=True)
    st.write("")

    st.write("<h3 style='text-align: center;'>Shape</h3>", unsafe_allow_html=True)  
    st.write(df1_train.shape)
    st.write("")

    st.write("<h3 style='text-align: center;'>Describe</h3>", unsafe_allow_html=True)
    milieu_desc = pd.DataFrame(["..."] * len(first_cols.describe()), index=first_cols.describe().index, columns=["..."])
    df_describe = pd.concat([first_cols.describe(), milieu_desc, last_cols.describe()], axis=1)
    st.dataframe(df_describe, use_container_width=True, height=350)

if page =="DataVizualisation" : 
    st.write("## DataVizualisation")
    df1_train.columns = [i+1 for i in range(187)] + ["Classe"]
    
    fig = plt.figure()
    sns.countplot(x=df1_train["Classe"], palette='viridis')
    plt.title("Distribution des classes dans les données d'entraînement")
    plt.xlabel("Classe")
    plt.ylabel("Nombre d'échantillons")
    st.pyplot(fig, use_container_width=True)
    st.write("")

    fig = plt.figure()
    noms_classes = ['Normal', 'Supraventriculaire', 'Ventriculaire', 'Fusion', 'Inconnu']
    echantillons = [df1_train[df1_train["Classe"] == i].iloc[0, :145].values for i in range(5)]
    temps = np.arange(len(echantillons[0]))
    for i, echantillon in enumerate(echantillons):
        plt.plot(temps, echantillon, label=f'Classe {i} - {noms_classes[i]}', linewidth=2)
        plt.title("Signaux ECG par classe (1 exemple par classe)")
        plt.xlabel("Pas de temps")
        plt.ylabel("Amplitude")
        plt.legend(fontsize=6)
        plt.grid(True, alpha=0.3)
    st.pyplot(fig, use_container_width=True)
    st.write("")

    # Valeurs de l'ECG
    fig=plt.figure()
    df_train_145 = df1_train.iloc[:, list(range(145)) + [-1]]
    sns.histplot(
        df_train_145.iloc[:, :-1].values.flatten(),
        bins=100,
        kde=True,
        color ='purple')
    plt.title("Distribution des valeurs du signal ECG")
    plt.xlabel("Amplitude")
    plt.ylabel("Fréquence")
    plt.tight_layout()
    plt.show()
    st.pyplot(fig, use_container_width=True)
    st.write("")

    #Spectrogramme
    df1_train.columns = [i+1 for i in range(187)] + ["Classe"]
    y = df1_train[df1_train["Classe"] == 0].iloc[0, :-1].values[:145].astype(float)
    D = np.abs(librosa.stft(y, n_fft=64, hop_length=8))
    fig, ax = plt.subplots(figsize=(8,4))
    sns.heatmap(D, cmap="inferno_r", ax=ax)
    ax.set_title("Spectrogramme (145 premières valeurs)")
    ax.set_xlabel("Temps (frames)")
    ax.set_ylabel("Fréquence")
    ax.invert_yaxis()
    st.pyplot(fig)

if page == "Modélisation" :
    st.write("## Modélisation")
    st.write("")
    Model=st.radio("Choisir le type de modélisation 👉", options=["Random Forest", "CNN simple", "Spectrogramme et CNN"],)
    
    if Model=="Random Forest":
        from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
        from sklearn.ensemble import RandomForestClassifier

        df1_train.columns = [i+1 for i in range(187)] + ["Classe"]
        df1_test.columns  = [i+1 for i in range(187)] + ["Classe"]
        N = 145
        X_train = df1_train.iloc[:, :N]
        y_train = df1_train["Classe"]
        X_test = df1_test.iloc[:, :N]
        y_test = df1_test["Classe"]

        clf = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        affichage = st.radio("Que souhaitez-vous montrer ?", ["Accuracy", "Confusion matrix", "Classification report"])
        
        if affichage == "Accuracy":
            st.write(accuracy_score(y_test, y_pred))
        elif affichage == "Confusion matrix":
            cm = confusion_matrix(y_test, y_pred)
            st.dataframe(pd.DataFrame(cm))
        else:
            rep = classification_report(y_test, y_pred, output_dict=True)
            st.dataframe(pd.DataFrame(rep).T, use_container_width=True)
        st.write("")
        
        
        st.write("## Test sur 1 échantillon")
        noms_classes = ['Normal', 'Supraventriculaire', 'Ventriculaire', 'Fusion', 'Inconnu']
        echantillons = {}
        classes = sorted(df1_test["Classe"].unique())
        
        for i in classes:
            i = int(i)
            df_classe = df1_test[df1_test["Classe"] == i]
            ligne = df_classe.iloc[0, :N]
            echantillons[i] = ligne.values.astype(float)
            
        classe_origine = st.selectbox("Choisir l’échantillon (classe vraie)", list(echantillons.keys()),format_func=lambda i: f"Classe {i} - {noms_classes[i]}")
        x = echantillons[classe_origine].reshape(1, -1)
        pred = clf.predict(x)[0]
        
        st.write("Classe vraie :", int(classe_origine))

        st.write("Classe prédite :", int(pred), "-", noms_classes[int(pred)])
