import argparse
import os
import uuid


def build_new_name(old_name):
    # base, ext = os.path.splitext(old_name)
    # if ext.lower() != ".mat":
    #     return None

    # # Remove '-downsample' wherever it appears in the stem.
    # stem = base.replace("-downsample", "")

    # # Ensure '-25Hz' is present exactly once before the extension.
    # if not stem.endswith("-25Hz"):
    #     stem = f"{stem}-25Hz"

    # return f"{stem}.mat"

    if "50.0" not in old_name:
        return None
    return old_name.replace("50.0", "50")


def list_mat_renames(target_dir):
    plan = []
    for name in sorted(os.listdir(target_dir)):
        old_path = os.path.join(target_dir, name)
        if not os.path.isfile(old_path):
            continue
        if not name.lower().endswith(".mat"):
            continue

        new_name = build_new_name(name)
        if not new_name or new_name == name:
            continue

        plan.append((name, new_name))
    return plan


def apply_renames(target_dir, plan, dry_run=False):
    if not plan:
        print("No files need renaming.")
        return

    # Detect conflicts in target names before applying changes.
    targets = [new_name for _, new_name in plan]
    if len(targets) != len(set(targets)):
        raise RuntimeError("Rename plan has duplicate target names. Aborting.")

    existing_names = set(os.listdir(target_dir))
    old_names = {old for old, _ in plan}
    for old_name, new_name in plan:
        if new_name in existing_names and new_name not in old_names:
            raise RuntimeError(
                f"Target file already exists and is not part of rename set: {new_name}"
            )

    print(f"Planned renames: {len(plan)}")
    for old_name, new_name in plan:
        print(f"  {old_name} -> {new_name}")

    if dry_run:
        print("Dry run enabled. No files were renamed.")
        return

    # Two-phase rename prevents collisions on case-insensitive filesystems.
    temp_moves = []
    for old_name, _ in plan:
        old_path = os.path.join(target_dir, old_name)
        temp_name = f".__tmp_rename__{uuid.uuid4().hex}__{old_name}"
        temp_path = os.path.join(target_dir, temp_name)
        os.rename(old_path, temp_path)
        temp_moves.append((temp_name, old_name))

    temp_lookup = {old_name: temp_name for temp_name, old_name in temp_moves}
    for old_name, new_name in plan:
        temp_name = temp_lookup[old_name]
        temp_path = os.path.join(target_dir, temp_name)
        new_path = os.path.join(target_dir, new_name)
        os.rename(temp_path, new_path)

    print("Rename complete.")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Rename files by replacing '50.0' with '50' in filenames."
        )
    )
    parser.add_argument(
        "--dir",
        default="data_clean_segmented",
        help="Directory containing .mat files to rename",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show planned renames without changing files",
    )
    args = parser.parse_args()

    target_dir = args.dir
    if not os.path.isdir(target_dir):
        raise FileNotFoundError(f"Directory not found: {target_dir}")

    plan = list_mat_renames(target_dir)
    apply_renames(target_dir, plan, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
