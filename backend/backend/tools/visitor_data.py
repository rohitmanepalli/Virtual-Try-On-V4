"""
Reads and manages the visitor records on this PC (there is deliberately no web route for this).

    venv\\Scripts\\python.exe tools\\visitor_data.py stats
    venv\\Scripts\\python.exe tools\\visitor_data.py export --out visitors.csv
    venv\\Scripts\\python.exe tools\\visitor_data.py delete --phone 9876543210
    venv\\Scripts\\python.exe tools\\visitor_data.py purge --older-than-days 365

export writes names and phone numbers in readable form: keep the file private and delete it when done.
delete handles a visitor's deletion request. purge applies your retention period.
"""
import argparse
import csv
import os
import sys

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("stats")
    export = commands.add_parser("export")
    export.add_argument("--out", required=True)
    delete = commands.add_parser("delete")
    delete.add_argument("--phone", required=True)
    purge = commands.add_parser("purge")
    purge.add_argument("--older-than-days", type=int, required=True)
    args = parser.parse_args()

    out = os.path.abspath(args.out) if args.command == "export" else None
    os.chdir(BACKEND)
    sys.path.insert(0, BACKEND)
    from dotenv import load_dotenv
    load_dotenv()
    from app.services import visitor_store

    if args.command == "stats":
        rows = visitor_store.export_rows()
        print(f"Try-ons: {len(rows)}  Sessions: {len({r['session_id'] for r in rows if r['session_id']})}  "
              f"Identified visitors: {len({r['visitor_id'] for r in rows if r['visitor_id']})}")
    elif args.command == "export":
        rows = visitor_store.export_rows()
        with open(out, "w", newline="", encoding="utf-8") as fh:
            if rows:
                writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        print(f"{len(rows)} rows written to {out}. It holds readable names and phone numbers: delete it when done.")
    elif args.command == "delete":
        print("Details deleted." if visitor_store.delete_visitor_by_phone(args.phone) else "No visitor has that number.")
    else:
        print(f"Removed: {visitor_store.purge_older_than(args.older_than_days)}")


if __name__ == "__main__":
    main()
