

import csv
import json
import argparse
import os
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.dirname(SCRIPT_DIR)
LOGHUB2_PATH = os.path.join(os.path.dirname(BASE_PATH), "loghub2DATA")


def get_logs_by_event_id(system_name, event_id, output_dir=None):

    base_data_path = LOGHUB2_PATH
    structured_csv = os.path.join(base_data_path, system_name, f"{system_name}_full.log_structured.csv")
    templates_csv = os.path.join(base_data_path, system_name, f"{system_name}_full.log_templates.csv")

    if not os.path.exists(structured_csv):
        raise FileNotFoundError(f"Structured log file does not exist: {structured_csv}")
    if not os.path.exists(templates_csv):
        raise FileNotFoundError(f"Template file does not exist: {templates_csv}")

    template_info = None
    with open(templates_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['EventId'] == event_id:
                template_info = {
                    'EventId': row['EventId'],
                    'EventTemplate': row['EventTemplate'],
                    'Occurrences': int(row['Occurrences'])
                }
                break

    if not template_info:
        raise ValueError(f"EventId not found in {system_name}: {event_id}")

    print(f"Found template: {template_info['EventTemplate']}")
    print(f"Expected log count: {template_info['Occurrences']}")

    matched_logs = []
    with open(structured_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['EventId'] == event_id:
                matched_logs.append({
                    'LineId': row['LineId'],
                    'Content': row['Content'],
                    'EventId': row['EventId'],
                    'EventTemplate': row['EventTemplate']
                })

    print(f"Actual extracted log count: {len(matched_logs)}")

    result = {
        'system_name': system_name,
        'event_id': event_id,
        'template': template_info['EventTemplate'],
        'expected_count': template_info['Occurrences'],
        'actual_count': len(matched_logs),
        'logs': matched_logs
    }

    if output_dir is None:
        output_dir = os.path.join(LOGHUB2_PATH, system_name)

    os.makedirs(output_dir, exist_ok=True)

    output_filename = f"{system_name}_{event_id}_logs.json"
    output_path = os.path.join(output_dir, output_filename)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to: {output_path}")

    txt_output = output_path.replace('.json', '.txt')
    with open(txt_output, 'w', encoding='utf-8') as f:
        f.write(f"System: {system_name}\n")
        f.write(f"EventId: {event_id}\n")
        f.write(f"Template: {template_info['EventTemplate']}\n")
        f.write(f"Log count: {len(matched_logs)}\n")
        f.write("=" * 80 + "\n\n")
        for log in matched_logs:
            f.write(f"[LineId: {log['LineId']}] {log['Content']}\n")

    print(f"Plain text version saved to: {txt_output}")

    csv_output = output_path.replace('.json', '.csv')
    with open(csv_output, 'w', encoding='utf-8', newline='') as f:
        fieldnames = ['LineId', 'Content', 'EventId', 'EventTemplate']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(matched_logs)

    print(f"CSV version saved to: {csv_output}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Extract all logs for a specified template from log dataset")
    parser.add_argument("--system", type=str, required=True,
                       help="Log system name (e.g., Thunderbird, Apache, Linux, etc.)")
    parser.add_argument("--event_id", type=str, required=True,
                       help="Event template ID (e.g., E653, E310, etc.)")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory (default: loghub2DATA/{system})")

    args = parser.parse_args()

    print("=" * 80)
    print(f"Extract Logs")
    print("=" * 80)
    print(f"System name: {args.system}")
    print(f"EventId: {args.event_id}")
    print("-" * 80)

    try:
        result = get_logs_by_event_id(
            system_name=args.system,
            event_id=args.event_id,
            output_dir=args.output_dir
        )

        print("\n" + "=" * 80)
        print("Extraction completed!")
        print("=" * 80)
        print(f"Template: {result['template']}")
        print(f"Extracted log count: {result['actual_count']}")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
