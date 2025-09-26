import os
import math
import random
from pathlib import Path
import argparse
from typing import List, Tuple

def list_basenames(p: Path) -> List[str]:
    """Return basenames (without extension) for all .png files in a folder."""
    return [f.stem for f in p.glob("*.png")]

def write_list(path: Path, names: List[str]) -> None:
    """Write one basename per line to a text file."""
    with path.open("w", encoding="utf-8") as f:
        for n in names:
            f.write(n + "\n")

def create_splits(
    real_dir: str,
    fake_dir: str,
    output_dir: str,
    split_ratio: Tuple[float, float, float] = (0.7, 0.15, 0.15),  # (train, val, test)
    fake_ratio_in_train: float = 0.7,  # fraction of FAKE inside TRAIN
    val_test_only_fake: bool = False,
    use_all_leftovers_in_train: bool = True,  # << add leftovers (real/fake) to TRAIN
    seed: int = 42,
) -> None:
    """
    Create train/val/test splits. If val_test_only_fake=True, val/test use only fake images.
    If use_all_leftovers_in_train=True, any remaining (unused) images are appended to TRAIN
    so nothing is wasted (train ratio may drift slightly).
    """
    random.seed(seed)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    s_tr, s_val, s_test = split_ratio
    assert abs((s_tr + s_val + s_test) - 1.0) < 1e-6, "split_ratio must sum to 1.0"
    assert 0.0 <= fake_ratio_in_train <= 1.0, "fake_ratio_in_train must be in [0,1]"
    real_ratio_in_train = 1.0 - fake_ratio_in_train

    real_dir = Path(real_dir)
    fake_dir = Path(fake_dir)
    real_all = list_basenames(real_dir)
    fake_all = list_basenames(fake_dir)

    # Optional sanity: overlapping basenames across real/fake
    overlap = set(real_all) & set(fake_all)
    if overlap:
        print(f"[WARN] {len(overlap)} basenames found in BOTH real and fake (e.g., {next(iter(overlap))}). "
              "They will be treated as FAKE for val/test-only-fake and by folder for train composition.")

    # Shuffle once for reproducibility
    random.shuffle(real_all)
    random.shuffle(fake_all)

    if not val_test_only_fake:
        # ---------- Standard mode: mix all splits ----------
        # Max train supported by class pools given desired train composition
        max_train_by_real = math.floor(len(real_all) / real_ratio_in_train) if real_ratio_in_train > 0 else float("inf")
        max_train_by_fake = math.floor(len(fake_all) / fake_ratio_in_train) if fake_ratio_in_train > 0 else float("inf")
        max_train_supported = min(max_train_by_real, max_train_by_fake)
        N = max_train_supported if s_tr == 1.0 else math.floor(max_train_supported / s_tr)
        if N <= 0:
            raise RuntimeError("Not enough images to satisfy requested ratios.")

        n_train = int(round(N * s_tr))
        n_val   = int(round(N * s_val))
        n_test  = N - n_train - n_val

        n_train_fake = int(round(n_train * fake_ratio_in_train))
        n_train_real = n_train - n_train_fake

        train_fake = random.sample(fake_all, min(n_train_fake, len(fake_all)))
        fake_left  = [x for x in fake_all if x not in train_fake]
        train_real = random.sample(real_all, min(n_train_real, len(real_all)))
        real_left  = [x for x in real_all if x not in train_real]

        # Build val/test from remaining total (mix)
        total_left = list(set(fake_left + real_left))
        random.shuffle(total_left)
        val_names  = total_left[:n_val]
        test_names = total_left[n_val:n_val + n_test]

        train_names = sorted(set(train_fake + train_real))

        # Optionally add ALL leftovers to TRAIN (no waste)
        if use_all_leftovers_in_train:
            used = set(train_names) | set(val_names) | set(test_names)
            leftovers = [x for x in (set(real_all) | set(fake_all)) if x not in used]
            if leftovers:
                print(f"[INFO] Appending {len(leftovers)} leftover samples to TRAIN (standard mode).")
                train_names = sorted(set(train_names) | set(leftovers))

        write_list(out / "train.txt", train_names)
        write_list(out / "val.txt",   sorted(val_names))
        write_list(out / "test.txt",  sorted(test_names))

        print(f"Done! Created splits in {out}")
        print(f"Train: {len(train_names)}, Val: {len(val_names)}, Test: {len(test_names)}")
        return

    # ---------- Special mode: VAL/TEST only FAKE ----------
    # Per-N needs:
    fake_need_per_N = (s_val + s_test) + s_tr * fake_ratio_in_train
    real_need_per_N = s_tr * real_ratio_in_train
    if fake_need_per_N <= 0:
        raise RuntimeError("Invalid configuration: fake_need_per_N <= 0.")

    # Choose the largest feasible N so requested composition is possible
    max_N_by_fake = math.floor(len(fake_all) / fake_need_per_N)
    max_N_by_real = math.floor(len(real_all) / real_need_per_N) if real_need_per_N > 0 else float("inf")
    N = min(max_N_by_fake, max_N_by_real)
    if N <= 0:
        raise RuntimeError(
            "Not enough images to satisfy 'val/test only fake' with given split_ratio and train mix.\n"
            f"Available real={len(real_all)}, fake={len(fake_all)}."
        )

    # Exact split sizes (enforce equal val/test if ratios equal)
    n_train = int(round(N * s_tr))
    n_val   = int(round(N * s_val))
    n_test  = N - n_train - n_val
    if abs(s_val - s_test) < 1e-9:
        total_vt = n_val + n_test
        n_val = total_vt // 2
        n_test = total_vt - n_val

    # Reserve FAKE for val/test first (guarantees equality & constraint)
    vt_needed = n_val + n_test
    fake_for_vt = random.sample(fake_all, vt_needed)
    val_names  = fake_for_vt[:n_val]
    test_names = fake_for_vt[n_val:n_val + n_test]

    # Remaining pools for TRAIN
    fake_left = [x for x in fake_all if x not in set(fake_for_vt)]
    n_train_fake = int(round(n_train * fake_ratio_in_train))
    n_train_real = n_train - n_train_fake

    # Sample TRAIN core according to target ratio
    train_fake_core = random.sample(fake_left, min(n_train_fake, len(fake_left))) if n_train_fake > 0 else []
    fake_left_after = [x for x in fake_left if x not in set(train_fake_core)]

    train_real_core = random.sample(real_all, min(n_train_real, len(real_all))) if n_train_real > 0 else []
    real_left_after = [x for x in real_all if x not in set(train_real_core)]

    train_names = sorted(set(train_fake_core + train_real_core))

    # Optionally append ALL leftovers (real & fake) to TRAIN so nothing is wasted
    if use_all_leftovers_in_train:
        used = set(train_names) | set(val_names) | set(test_names)
        # All images we have:
        all_imgs = set(fake_all)
        leftovers = [x for x in all_imgs if x not in used]
        if leftovers:
            print(f"[INFO] Appending {len(leftovers)} leftover samples to TRAIN (val/test-only-fake mode).")
            train_names = sorted(set(train_names) | set(leftovers))

    # Write out
    write_list(out / "train.txt", train_names)
    write_list(out / "val.txt",   sorted(val_names))
    write_list(out / "test.txt",  sorted(test_names))

    # Stats
    print(f"Done! Created splits in {out}")
    print(f"Train: {len(train_names)}, Val: {len(val_names)}, Test: {len(test_names)}")

    # Quick ratio report for TRAIN (for information only)
    real_set = set(real_all)
    fake_set = set(fake_all)
    tr_fake = sum(1 for n in train_names if n in fake_set and n not in real_set)
    tr_real = sum(1 for n in train_names if n in real_set and n not in fake_set)
    total_tr = max(1, tr_fake + tr_real)
    print(f"[TRAIN ratio] fake={tr_fake} ({tr_fake/total_tr:.3f}), real={tr_real} ({tr_real/total_tr:.3f})")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--real_dir", type=str, default="../dataset/real_images")
    p.add_argument("--fake_dir", type=str, default="../dataset/fake_images")
    p.add_argument("--output_dir", type=str, default="./")
    p.add_argument("--split", type=str, default="0.8, 0.1, 0.1", help="train,val,test (must sum to 1.0)")
    p.add_argument("--fake_ratio_in_train", type=float, default=0.7, help="fraction of FAKE inside TRAIN")
    p.add_argument("--val_test_only_fake", action="store_true",default=True, help="val/test contain only fake images")
    p.add_argument("--no_use_all_leftovers_in_train", action="store_true", default=False,
                   help="do NOT append leftovers to TRAIN (default: append)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    s = tuple(float(x) for x in args.split.split(","))
    assert len(s) == 3
    return args, s

if __name__ == "__main__":
    args, split_ratio = parse_args()
    create_splits(
        real_dir=args.real_dir,
        fake_dir=args.fake_dir,
        output_dir=args.output_dir,
        split_ratio=split_ratio,
        fake_ratio_in_train=args.fake_ratio_in_train,
        val_test_only_fake=args.val_test_only_fake,
        use_all_leftovers_in_train=not args.no_use_all_leftovers_in_train,
        seed=args.seed,
    )
