from pathlib import Path
from typing import Tuple, List
import random
from fractions import Fraction
import argparse

def list_basenames(p: Path) -> List[str]:
    # Implementiere nach deinem Bedarf (z.B. alle *.png ohne Suffix)
    return [f.stem for f in sorted(p.glob("*")) if f.is_file()]

def write_list(path: Path, names: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for n in names:
            f.write(n + "\n")

def _ratio_to_units(p_fake: float, max_den: int = 100):
    """0.6 -> (num_fake=3, denom_total=5) ; 0.5 -> (1,2) ; 0.7 -> (7,10)"""
    frac = Fraction(p_fake).limit_denominator(max_den)
    num_fake = frac.numerator
    denom_total = frac.denominator
    return num_fake, denom_total

def create_splits(
    real_dir: str,
    fake_dir: str,
    output_dir: str,
    split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1),  # (train, val, test)
    fake_ratio_in_test: float = 0.5, 
    fake_ratio_in_val: float = 0.5, 
    fake_ratio_in_train: float = 0.6,  # fraction of FAKE inside TRAIN
    val_test_only_fake: bool = False,
    use_all_leftovers_in_train: bool = True,  # add leftovers (real/fake) to TRAIN
    seed: int = 42,
) -> None:
    """
    Verteilt alle Fake-Bilder und erzeugt train/val/test nach:
      - Ziel-Splits (z.B. 80/10/10) so nahe wie möglich
      - Innen-Ratios (z.B. 60/40, 50/50, 50/50) exakt (ganzzahlig)
    Falls val_test_only_fake=True: Val/Test enthalten nur Fake (Innen-Ratio = 100% Fake).
    Wenn use_all_leftovers_in_train=True: übrige Real/Fake landen zusätzlich im Train.
    """
    random.seed(seed)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    s_tr, s_val, s_test = split_ratio
    assert abs((s_tr + s_val + s_test) - 1.0) < 1e-6, "split_ratio must sum to 1.0"
    assert 0.0 <= fake_ratio_in_train <= 1.0
    assert 0.0 <= fake_ratio_in_val   <= 1.0
    assert 0.0 <= fake_ratio_in_test  <= 1.0

    real_dir = Path(real_dir)
    fake_dir = Path(fake_dir)
    real_all = list_basenames(real_dir)
    fake_all = list_basenames(fake_dir)

    # (Optional) Warnen bei basename-Überschneidungen
    overlap = set(real_all) & set(fake_all)
    if overlap:
        print(f"[WARN] {len(overlap)} basenames real∩fake (z.B. {next(iter(overlap))}). "
              "Sie werden anhand des Ordners klassifiziert (kein 'Dedupe' zwischen Ordnern).")

    random.shuffle(real_all)
    random.shuffle(fake_all)

    F_total = len(fake_all)
    R_total = len(real_all)

    # Innen-Ratios in ganzzahlige Einheiten umwandeln
    if val_test_only_fake:
        tr_num, tr_den = _ratio_to_units(fake_ratio_in_train)
        va_num, va_den = 1, 1  # 100% Fake
        te_num, te_den = 1, 1  # 100% Fake
    else:
        tr_num, tr_den = _ratio_to_units(fake_ratio_in_train)
        va_num, va_den = _ratio_to_units(fake_ratio_in_val)
        te_num, te_den = _ratio_to_units(fake_ratio_in_test)

    # Wir suchen ganzzahlige a,b,c:
    # Train: T_tr = tr_den*a, F_tr = tr_num*a, R_tr = (tr_den-tr_num)*a
    # Val:   T_va = va_den*b, F_va = va_num*b, R_va = (va_den-va_num)*b
    # Test:  T_te = te_den*c, F_te = te_num*c, R_te = (te_den-te_num)*c
    # Nebenbedingung: F_tr + F_va + F_te == F_total  -> tr_num*a + va_num*b + te_num*c = F_total
    # Ziel: Minimiere (T_tr/N - s_tr)^2 + (T_va/N - s_val)^2 + (T_te/N - s_test)^2

    best = None
    # sinnvolle Grenzen
    max_a = F_total // max(1, tr_num)
    for a in range(max_a + 1):
        rem1 = F_total - tr_num * a
        if rem1 < 0: break
        max_b = rem1 // max(1, va_num)
        for b in range(max_b + 1):
            c_num = max(1, te_num)
            c = rem1 - va_num * b
            if c < 0 or c % c_num != 0:
                continue
            c //= c_num

            # Totale
            T_tr, F_tr, R_tr = tr_den*a, tr_num*a, (tr_den - tr_num)*a
            T_va, F_va, R_va = va_den*b, va_num*b, (va_den - va_num)*b
            T_te, F_te, R_te = te_den*c, te_num*c, (te_den - te_num)*c
            N = T_tr + T_va + T_te
            if N == 0: 
                continue

            # Reale-Anforderung prüfen
            R_need = R_tr + R_va + R_te
            if R_need > R_total:
                continue  # nicht genug Realbilder → verwerfen

            p_tr, p_va, p_te = T_tr / N, T_va / N, T_te / N
            err = (p_tr - s_tr)**2 + (p_va - s_val)**2 + (p_te - s_test)**2
            score = (err, abs(p_tr - s_tr), N)  # erst Fehler, dann Train-Nähe, dann kleinere N

            if best is None or score < best[0]:
                best = (score, (a,b,c, T_tr,T_va,T_te, F_tr,F_va,F_te, R_tr,R_va,R_te, N, p_tr,p_va,p_te))

    if best is None:
        raise RuntimeError("Keine gültige Integer-Lösung gefunden (zu wenig Real-Bilder?). "
                           "Erhöhe Real-Datensatz oder lockere die Innen-Ratios.")

    (a,b,c, T_tr,T_va,T_te, F_tr,F_va,F_te, R_tr,R_va,R_te, N, p_tr,p_va,p_te) = best[1]

    # --- Stichproben ziehen (ohne Überschneidungen zwischen Splits) ---
    # Train
    train_fake = fake_all[:F_tr]
    fake_left  = fake_all[F_tr:]
    train_real = real_all[:R_tr]
    real_left  = real_all[R_tr:]

    # Val
    val_fake = fake_left[:F_va]
    fake_left = fake_left[F_va:]
    val_real = [] if val_test_only_fake else real_left[:R_va]
    real_left = real_left[R_va:] if not val_test_only_fake else real_left

    # Test
    test_fake = fake_left[:F_te]
    fake_left = fake_left[F_te:]
    test_real = [] if val_test_only_fake else real_left[:R_te]
    real_left = real_left[R_te:] if not val_test_only_fake else real_left

    # Reste optional in Train
    if use_all_leftovers_in_train:
        train_fake = train_fake + fake_left
        train_real = train_real + real_left
        fake_left = []
        real_left = []

    # Finale Namenslisten
    train_names = sorted(set(train_fake + train_real))
    val_names   = sorted(set(val_fake   + val_real))
    test_names  = sorted(set(test_fake  + test_real))

    # Write out
    out = Path(output_dir)
    write_list(out / "train.txt", train_names)
    write_list(out / "val.txt",   val_names)
    write_list(out / "test.txt",  test_names)

    # Stats
    fake_set = set(fake_all)
    real_set = set(real_all)

    def _count_fr(names):
        f = sum(1 for n in names if n in fake_set)
        r = sum(1 for n in names if n in real_set)
        return f, r, len(names)

    tr_f, tr_r, tr_n = _count_fr(train_names)
    va_f, va_r, va_n = _count_fr(val_names)
    te_f, te_r, te_n = _count_fr(test_names)
    total = tr_n + va_n + te_n

    print(f"Done! Splits in {out}")
    print(f"Train: {tr_n} (fake {tr_f} / real {tr_r})  -> {100*tr_n/total:.3f}%")
    print(f"Val:   {va_n} (fake {va_f} / real {va_r})  -> {100*va_n/total:.3f}%")
    print(f"Test:  {te_n} (fake {te_f} / real {te_r})  -> {100*te_n/total:.3f}%")
    if tr_n:
        print(f"[TRAIN ratio] fake={tr_f/tr_n:.3f} real={tr_r/tr_n:.3f}")
    if va_n:
        print(f"[VAL   ratio] fake={va_f/va_n:.3f} real={va_r/va_n:.3f}")
    if te_n:
        print(f"[TEST  ratio] fake={te_f/te_n:.3f} real={te_r/te_n:.3f}")



if __name__ == "__main__":
    create_splits(
        real_dir =  "../dataset/real_images",
        fake_dir = "../dataset/fake_images",
        output_dir = "./",
        split_ratio = (0.8, 0.1, 0.1),
        fake_ratio_in_val = 0.5,
        fake_ratio_in_test = 0.5,
        fake_ratio_in_train= 0.6,
        val_test_only_fake= False,
        use_all_leftovers_in_train = False,
        seed= 42,)

