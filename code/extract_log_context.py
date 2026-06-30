

import argparse
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.dirname(SCRIPT_DIR)
LOGHUB2_PATH = os.path.join(os.path.dirname(BASE_PATH), "loghub2DATA")


def extract_log_context(system_name, event_id, line_id, context_lines=5, output_dir=None):

    base_path = LOGHUB2_PATH
    log_file = os.path.join(base_path, system_name, system_name, f"{system_name}_full.log")

    if not os.path.exists(log_file):
        raise FileNotFoundError(f"Log file does not exist: {log_file}")

    target_line = int(line_id)
    start_line = max(1, target_line - context_lines)
    end_line = target_line + context_lines

    print(f"Log file: {log_file}")
    print(f"Target line: {target_line}")
    print(f"Extraction range: {start_line} - {end_line} (total {end_line - start_line + 1} lines)")

    extracted_lines = []
    with open(log_file, 'r', encoding='utf-8', errors='replace') as f:
        for current_line_num, line in enumerate(f, 1):
            if current_line_num < start_line:
                continue
            if current_line_num > end_line:
                break
            marker = " >>> " if current_line_num == target_line else "     "
            extracted_lines.append(f"{current_line_num:>10}{marker}{line.rstrip()}")

    if not extracted_lines:
        raise ValueError(f"Failed to extract any lines, please check if line number {line_id} is valid")

    if output_dir is None:
        output_dir = os.path.join(base_path, system_name)

    os.makedirs(output_dir, exist_ok=True)

    output_filename = f"{system_name}_{event_id}_L{line_id}_C{context_lines}.txt"
    output_path = os.path.join(output_dir, output_filename)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"System: {system_name}\n")
        f.write(f"EventId: {event_id}\n")
        f.write(f"Target LineId: {line_id}\n")
        f.write(f"Context: +/- {context_lines} lines\n")
        f.write(f"Range: {start_line} - {end_line}\n")
        f.write("=" * 100 + "\n\n")
        for line in extracted_lines:
            f.write(line + "\n")

    print(f"\nResult saved to: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Log Context Extraction Tool - Extract specified line and its context",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Extract LineId 187386 and 5 lines of context above and below in BGL system
  python extract_log_context.py --system BGL --event_id E214 --line_id 187386 --context 5

  # Extract LineId 100 and 10 lines of context above and below in Apache system
  python extract_log_context.py --system Apache --event_id E10 --line_id 100 --context 10
        """
    )
    parser.add_argument("--system", type=str, required=True,
                        help="Log system name (e.g., BGL, Apache, Thunderbird, etc.)")
    parser.add_argument("--event_id", type=str, required=True,
                        help="Event ID (used only for output file naming, not for searching)")
    parser.add_argument("--line_id", type=int, required=True,
                        help="Target line number")
    parser.add_argument("--context", type=int, default=5,
                        help="Number of context lines above and below (default: 5)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: loghub2DATA/{system})")

    args = parser.parse_args()

    print("=" * 80)
    print("Log Context Extraction Tool")
    print("=" * 80)
    print(f"System: {args.system}")
    print(f"EventId: {args.event_id}")
    print(f"LineId: {args.line_id}")
    print(f"Context: +/- {args.context} lines")
    print("-" * 80)

    try:
        output_path = extract_log_context(
            system_name=args.system,
            event_id=args.event_id,
            line_id=args.line_id,
            context_lines=args.context,
            output_dir=args.output_dir
        )
        print("\n" + "=" * 80)
        print("Extraction completed!")
        print("=" * 80)

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
