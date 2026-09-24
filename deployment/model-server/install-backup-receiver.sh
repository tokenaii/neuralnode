#!/bin/bash
set -euo pipefail

public_key_file=${1:-/tmp/horus_backup_transfer.pub}
test -r "$public_key_file"
read -r key_type key_data _ < "$public_key_file"
[[ "$key_type" == "ssh-ed25519" && -n "$key_data" ]]

mkdir -p "$HOME/.ssh"
touch "$HOME/.ssh/authorized_keys"
chmod 700 "$HOME/.ssh"
chmod 600 "$HOME/.ssh/authorized_keys"

if ! grep -Fq "$key_data" "$HOME/.ssh/authorized_keys"; then
  printf 'restrict,command="/usr/local/bin/receive-tokenai-backup" %s %s tokenai-backup-transfer\n' \
    "$key_type" "$key_data" >> "$HOME/.ssh/authorized_keys"
fi

rm -f "$public_key_file"
