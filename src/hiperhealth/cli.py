"""
title: Command-line interface for the hiperhealth skill registry.
"""

from __future__ import annotations

import argparse
import json
import sys

from pathlib import Path
from typing import Sequence

from hiperhealth.pipeline.registry import SkillRegistry


def _build_parser() -> argparse.ArgumentParser:
    """
    title: Build the hiperhealth CLI argument parser
    returns:
      type: argparse.ArgumentParser
      description: Configured argument parser.
    """
    parser = argparse.ArgumentParser(prog='hiperhealth')
    parser.add_argument(
        '--registry-dir',
        type=Path,
        default=None,
        help='Artifact directory for installed skills.',
    )
    subparsers = parser.add_subparsers(dest='group', required=True)

    channel_parser = subparsers.add_parser('channel')
    channel_subparsers = channel_parser.add_subparsers(
        dest='channel_command',
        required=True,
    )

    add_parser = channel_subparsers.add_parser('add')
    add_parser.add_argument('source')
    add_parser.add_argument('--name', dest='local_name')
    add_parser.add_argument('--ref')

    channel_subparsers.add_parser('list')

    channel_skills_parser = channel_subparsers.add_parser('skills')
    channel_skills_parser.add_argument('local_name')

    channel_update_parser = channel_subparsers.add_parser('update')
    channel_update_parser.add_argument('local_name')
    channel_update_parser.add_argument('--ref')

    channel_remove_parser = channel_subparsers.add_parser('remove')
    channel_remove_parser.add_argument('local_name')

    channel_install_parser = channel_subparsers.add_parser('install')
    channel_install_parser.add_argument('local_name')
    channel_install_parser.add_argument(
        '--all',
        action='store_true',
        required=True,
        help='Install all skills from the channel.',
    )
    channel_install_parser.add_argument(
        '--include-disabled',
        action='store_true',
        help='Also install disabled skills.',
    )

    skill_parser = subparsers.add_parser('skill')
    skill_subparsers = skill_parser.add_subparsers(
        dest='skill_command',
        required=True,
    )

    skill_list_parser = skill_subparsers.add_parser('list')
    skill_list_parser.add_argument('--channel')
    skill_list_parser.add_argument(
        '--installed-only',
        action='store_true',
        help='Only list installed skills.',
    )

    skill_install_parser = skill_subparsers.add_parser('install')
    skill_install_parser.add_argument('skill_id')

    skill_update_parser = skill_subparsers.add_parser('update')
    skill_update_parser.add_argument('skill_id')
    skill_update_parser.add_argument(
        '--pull',
        action='store_true',
        help='Pull the owning channel before updating the skill.',
    )

    skill_remove_parser = skill_subparsers.add_parser('remove')
    skill_remove_parser.add_argument('skill_id')

    return parser


def _print_json(payload: object) -> None:
    """
    title: Print JSON to stdout
    parameters:
      payload:
        type: object
        description: JSON-serializable payload.
    """
    print(json.dumps(payload, indent=2))


def main(argv: Sequence[str] | None = None) -> int:
    """
    title: Run the hiperhealth CLI
    parameters:
      argv:
        type: Sequence[str] | None
        description: >-
          Optional CLI arguments. When omitted, ``sys.argv`` is used.
    returns:
      type: int
      description: Process exit code.
    """
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    registry = SkillRegistry(registry_dir=args.registry_dir)

    if args.group == 'channel':
        if args.channel_command == 'add':
            print(
                registry.add_channel(
                    args.source,
                    local_name=args.local_name,
                    ref=args.ref,
                )
            )
            return 0
        if args.channel_command == 'list':
            _print_json(
                [
                    record.model_dump(mode='json')
                    for record in registry.list_channels()
                ]
            )
            return 0
        if args.channel_command == 'skills':
            _print_json(
                [
                    record.model_dump(mode='json')
                    for record in registry.list_channel_skills(args.local_name)
                ]
            )
            return 0
        if args.channel_command == 'update':
            _print_json(registry.update_channel(args.local_name, ref=args.ref))
            return 0
        if args.channel_command == 'remove':
            registry.remove_channel(args.local_name)
            print(args.local_name)
            return 0
        if args.channel_command == 'install':
            _print_json(
                registry.install_channel(
                    args.local_name,
                    include_disabled=args.include_disabled,
                )
            )
            return 0

    if args.group == 'skill':
        if args.skill_command == 'list':
            _print_json(
                [
                    record.model_dump(mode='json')
                    for record in registry.list_skills(
                        channel=args.channel,
                        installed_only=args.installed_only,
                    )
                ]
            )
            return 0
        if args.skill_command == 'install':
            print(registry.install_skill(args.skill_id))
            return 0
        if args.skill_command == 'update':
            print(
                registry.update_skill(
                    args.skill_id,
                    pull_channel=args.pull,
                )
            )
            return 0
        if args.skill_command == 'remove':
            registry.remove_skill(args.skill_id)
            print(args.skill_id)
            return 0

    parser.error('unsupported command')
    return 2


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
