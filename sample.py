"""Run the baseline against an explicitly published sample with your development key."""
import argparse
import os
from nova_sdk import Client
from run import prepare, consult


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--case', help='Published sample case ID (see --list)')
    group.add_argument('--list', action='store_true', help='List the published sample cases and their IDs')
    args = parser.parse_args()
    client = Client(token=os.environ['NOVA_DEV_KEY'], sample=True)
    if args.list:
        for case in client._request('GET', '/v1/sample/cases')['cases']:
            print(case['case_id'], case['initial']['opening'][:60])
        return
    model = prepare()
    client.create_sample(args.case)
    result = consult(client, model)
    print(result)


if __name__ == '__main__':
    main()
