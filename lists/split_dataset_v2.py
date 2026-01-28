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

def safe_needed_reals(n_fake: int, ratio: float) -> int:
    if ratio <= 0.0:
        return 0  # 0% Fake → undefiniert; hier: keine Reals anfordern (oder raise)
    if ratio >= 1.0:
        return 0  # 100% Fake → 0 Reals
    return int(round(n_fake * (1.0/ratio - 1.0)))

def create_splits(
    real_dir: str,
    fake_dir: str,
    output_dir: str,
    split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1),  # (train, val, test)
    fake_ratio_in_test: float = 0.5, 
    fake_ratio_in_val: float = 1, 
    fake_ratio_in_train: float = 1,  # fraction of FAKE inside TRAIN
    val_test_only_fake: bool = True,
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
    real_dir = Path(real_dir)
    fake_dir = Path(fake_dir)
    real_all = list_basenames(real_dir)
    fake_all = list_basenames(fake_dir)
    all_fake = len(fake_all)

    s_tr, s_val, s_test = split_ratio
    num_fake_train = int(s_tr * all_fake)
    num_fake_val = int(s_val * all_fake)
    num_fake_test = int(s_test * all_fake)
    random.seed(seed)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    assert abs((s_tr + s_val + s_test) - 1.0) < 1e-6, "split_ratio must sum to 1.0"
    assert 0.0 <= fake_ratio_in_train <= 1.0
    assert 0.0 <= fake_ratio_in_val   <= 1.0
    assert 0.0 <= fake_ratio_in_test  <= 1.0


    # Fake aufteilen (wie gehabt)
    train_fake = random.sample(fake_all, num_fake_train)
    left_over_fake = [x for x in fake_all if x not in train_fake]
    test_fake = random.sample(left_over_fake, num_fake_test)
    left_over_fake_2 = [x for x in left_over_fake if x not in test_fake]
    val_fake = random.sample(left_over_fake_2, num_fake_val)

    # benötigte Real-Mengen robust berechnen
    num_real_train = safe_needed_reals(len(train_fake), fake_ratio_in_train)
    num_real_val   = safe_needed_reals(len(val_fake),   fake_ratio_in_val)
    num_real_test  = safe_needed_reals(len(test_fake),  fake_ratio_in_test)

    # Reals ziehen (mit Clamp + Warnung)
    def take_sample(pool, k, tag):
        k_eff = min(k, len(pool))
        if k_eff < k:
            print(f"[WARN] Not enough REAL for {tag}: asked {k}, taking {k_eff}.")
        picks = random.sample(pool, k_eff)
        rest  = [x for x in pool if x not in picks]
        return picks, rest

    train_real, left_over_real = take_sample(real_all, num_real_train, "train")
    test_real,  left_over_real = take_sample(left_over_real, num_real_test, "test")
    val_real,   left_over_real = take_sample(left_over_real, num_real_val, "val")

    # optional: nur Fake in Val/Test
    if val_test_only_fake:
        val_real = []
        test_real = []

    # optional: alle Rest-Reals/Fakes in Train kippen
    if use_all_leftovers_in_train:
        # restliche Fakes, die nicht vergeben wurden
        leftover_fakes = [x for x in left_over_fake_2 if x not in val_fake]
        train_fake += leftover_fakes
        train_real += left_over_real

    # korrekt mischen (shuffle gibt None zurück!)
    train_names = train_fake + train_real
    val_names   = val_fake   + val_real
    test_names  = test_fake  + test_real
    random.shuffle(train_names)
    random.shuffle(val_names)
    random.shuffle(test_names)

    # Write out
    out = Path(output_dir)
    write_list(out / "train.txt", train_names)
    write_list(out / "val.txt",   val_names)
    write_list(out / "test.txt",  test_names)

    tr_n = len(train_fake) + len(train_real)
    te_n = len(test_fake) + len(test_real)
    va_n = len(val_fake) + len(val_real)

    total = tr_n + te_n + va_n

    print(f"Done! Splits in {out}")
    print(f"Train: {tr_n} (fake {len(train_fake)} / real {len(train_real)})  -> {100*tr_n/total:.3f}%")
    print(f"Val:   {va_n} (fake {len(val_fake)} / real {len(val_real)})  -> {100*va_n/total:.3f}%")
    print(f"Test:  {te_n} (fake {len(test_fake)} / real {len(test_real)})  -> {100*te_n/total:.3f}%")

    if tr_n:
        print(f"[TRAIN ratio] fake={len(train_fake)/tr_n:.3f} real={len(train_real)/tr_n:.3f}")
    if va_n:
        print(f"[VAL   ratio] fake={len(val_fake)/va_n:.3f} real={len(val_real)/va_n:.3f}")
    if te_n:
        print(f"[TEST  ratio] fake={len(test_fake)/te_n:.3f} real={len(test_real)/te_n:.3f}")

if __name__ == "__main__":
    create_splits(
        real_dir =  "../dataset/real_images",
        fake_dir = "../dataset/fake_images",
        output_dir = "./",
        split_ratio = (0.8, 0.1, 0.1),
        fake_ratio_in_val = 0.6,
        fake_ratio_in_test = 0.6,
        fake_ratio_in_train= 0.6,
        val_test_only_fake= False,
        use_all_leftovers_in_train = False,
        seed= 42,)

