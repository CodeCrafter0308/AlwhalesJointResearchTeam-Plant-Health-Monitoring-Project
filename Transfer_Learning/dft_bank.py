import torch

# 吸附能 (eV)，行=VOC id (与 voc_registry.py 的 VOC_KNOWLEDGE_BASE 完全一致)，列=[E_no, E_pos, E_neg]
# id: 0 Acetic acid, 1 Nonanal, 2 Decanoic acid, 3 Heptanal, 4 Hexanal, 5 Decanal, 6 2-Octanone
_ADSORPTION_E = torch.tensor([[-0.8222, -0.7414, -1.2054], [-1.0109, -0.6987, -2.0269], [-1.6983, -0.9402, -2.5079], [-1.4308, -0.8782, -1.9302], [-0.9972, -0.7104, -1.7632], [-0.8796, -0.741, -1.883], [-1.3812, -1.021, -1.7079]], dtype=torch.float32)

def build_adsorption_feature_bank(device=None, dtype=None) -> torch.Tensor:
    """
    返回 DFT 特征银行: [N_voc=7, D_dft=6]
    特征定义:
      0: E_no   (无电场吸附能)
      1: E_pos  (正电场吸附能)
      2: E_neg  (负电场吸附能)
      3: d_neg0 = E_neg - E_no
      4: d_negp = E_neg - E_pos
      5: enh_abs = |E_neg| - |E_no|   (负电场吸附强度提升量, 以绝对值衡量)
    """
    E = _ADSORPTION_E
    if dtype is None:
        dtype = E.dtype
    E = E.to(dtype=dtype)
    E_no = E[:, 0:1]
    E_pos = E[:, 1:2]
    E_neg = E[:, 2:3]
    d_neg0 = E_neg - E_no
    d_negp = E_neg - E_pos
    enh_abs = torch.abs(E_neg) - torch.abs(E_no)
    feat = torch.cat([E_no, E_pos, E_neg, d_neg0, d_negp, enh_abs], dim=1)
    if device is not None:
        feat = feat.to(device)
    return feat



def build_dft_feature_bank(band_dir: str = None, dos_dir: str = None, device=None, dtype=None) -> torch.Tensor:
    """
    Build the full DFT feature bank by concatenating:
      - adsorption features (always available, hard-coded)
      - band-structure features (optional; loaded from band_dir)
      - DOS/PDOS features (optional; loaded from dos_dir)

    Returns:
      Tensor [N_voc=7, D_total]
    """
    feat = build_adsorption_feature_bank(device=device, dtype=dtype)

    # Band-structure features
    if band_dir is not None and str(band_dir).strip() != "":
        try:
            from pathlib import Path
            from dft_band import build_band_feature_bank

            band_path = Path(band_dir)
            if not band_path.is_absolute():
                band_path = Path(__file__).resolve().parent / band_path
            band_dir_resolved = str(band_path)

            band = build_band_feature_bank(band_dir_resolved, device=device, dtype=feat.dtype)
            feat = torch.cat([feat, band], dim=1)
        except Exception as e:
            print(f"[WARN] Failed to load band features from {band_dir}. Skipping band. Error: {e}")

    # DOS/PDOS features
    if dos_dir is not None and str(dos_dir).strip() != "":
        try:
            from pathlib import Path
            from dft_dos import build_dos_feature_bank

            dos_path = Path(dos_dir)
            if not dos_path.is_absolute():
                dos_path = Path(__file__).resolve().parent / dos_path
            dos_dir_resolved = str(dos_path)

            dos = build_dos_feature_bank(dos_dir_resolved, device=device, dtype=feat.dtype)
            feat = torch.cat([feat, dos], dim=1)
        except Exception as e:
            print(f"[WARN] Failed to load DOS/PDOS features from {dos_dir}. Skipping DOS. Error: {e}")

    return feat
