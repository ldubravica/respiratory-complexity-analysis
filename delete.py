import argparse
import os


def find_matching_files(target_dir, token):
    matches = []
    for name in sorted(os.listdir(target_dir)):
        path = os.path.join(target_dir, name)
        if not os.path.isfile(path):
            continue
        if token in name:
            matches.append(path)
    return matches


def main():
    parser = argparse.ArgumentParser(
        description="Delete files in a directory whose names contain a token."
    )
    parser.add_argument(
        "--dir",
        default="data_clean_segmented",
        help="Target directory",
    )
    parser.add_argument(
        "--token",
        default="50Hz",
        help="Substring to match in filenames",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Actually delete files (without this flag, only prints planned deletions)",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.dir):
        raise FileNotFoundError(f"Directory not found: {args.dir}")

    matches = find_matching_files(args.dir, args.token)

    if not matches:
        print("No matching files found.")
        return

    print(f"Matching files: {len(matches)}")
    for path in matches:
        print(f"  {path}")

    if not args.yes:
        print("Dry run only. Use --yes to delete these files.")
        return

    for path in matches:
        os.remove(path)

    print(f"Deleted {len(matches)} files.")


if __name__ == "__main__":
    main()
