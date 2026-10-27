"""
VectorTrade - Offline Telemetry & Trade Details Request CLI
Allows the offline program / user to securely request real-time execution details,
open positions, and health telemetry from the online system.

Supports dual secure channels:
1. Indirect Mailbox (Default): Requests routed through MongoDB collection 'inter_service_requests'.
   Zero open ports or network exposure; works seamlessly across firewalls.
2. Authenticated HTTP: Cryptographically signed REST call with X-Vector-Internal-Token.
   Direct browser access to the URL returns 401 Unauthorized.
"""

import sys
import json
import argparse
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from core.secure_comms import request_online_trade_details


def main():
    parser = argparse.ArgumentParser(description="VectorTrade Secure Offline-to-Online Details Requester")
    parser.add_argument(
        "--action",
        type=str,
        default="GET_TODAY_EXECUTION",
        choices=["GET_TODAY_EXECUTION", "GET_ACTIVE_POSITIONS", "GET_SYSTEM_HEALTH", "GET_ML_TRAINING_FEEDBACK"],
        help="Telemetry action to request from Online Engine."
    )
    parser.add_argument(
        "--channel",
        type=str,
        default="auto",
        choices=["auto", "mailbox", "http"],
        help="Communication channel: 'auto' (preferred), 'mailbox' (indirect DB), or 'http' (secure REST)."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Record limit for list queries (e.g. ML training feedback)."
    )

    args = parser.parse_args()

    print("==================================================================")
    print(f"  VectorTrade Secure Details Request: {args.action}")
    print(f"  Channel: {args.channel} | Security: Pre-Shared Cryptographic Key")
    print("==================================================================")

    res = request_online_trade_details(
        action=args.action,
        params={"limit": args.limit},
        channel=args.channel
    )

    print("\n--- Telemetry Response ---")
    print(json.dumps(res, indent=2))
    print("==================================================================")


if __name__ == "__main__":
    main()
