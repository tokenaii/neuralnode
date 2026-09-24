#!/bin/sh
set -eu

backup_dir=${BACKUP_LOCAL_DIR:-/opt/tokenai-early-access/backups}
remote_host=${BACKUP_REMOTE_HOST:-54.91.40.201}
remote_user=${BACKUP_REMOTE_USER:-ec2-user}
remote_key=${BACKUP_REMOTE_SSH_KEY:-/home/ec2-user/.ssh/horus_backup_transfer}
public_key=${BACKUP_ENCRYPTION_PUBLIC_KEY:-/opt/tokenai-early-access/backup_public.pem}
timestamp=$(date -u +%Y%m%dT%H%M%SZ)
name=early-access-$timestamp
work_dir=$(mktemp -d)
umask 077

cleanup() {
  rm -rf "$work_dir"
}
trap cleanup EXIT INT TERM

mkdir -p "$backup_dir"
test -r "$remote_key"
test -r "$public_key"

docker exec tokenai-early-access-postgres-1 sh -lc \
  'pg_dump -U "$POSTGRES_USER" -d "$POSTGRES_DB" --format=custom --no-owner --no-privileges' \
  > "$work_dir/$name.dump"

docker exec -i tokenai-early-access-postgres-1 pg_restore --list \
  < "$work_dir/$name.dump" > /dev/null

openssl rand -base64 48 > "$work_dir/passphrase"
openssl enc -aes-256-cbc -pbkdf2 -iter 200000 -salt \
  -in "$work_dir/$name.dump" \
  -out "$work_dir/database.dump.enc" \
  -pass file:"$work_dir/passphrase"
openssl pkeyutl -encrypt -pubin -inkey "$public_key" \
  -in "$work_dir/passphrase" \
  -out "$work_dir/passphrase.enc" \
  -pkeyopt rsa_padding_mode:oaep \
  -pkeyopt rsa_oaep_md:sha256

dump_sha=$(sha256sum "$work_dir/database.dump.enc" | awk '{print $1}')
key_sha=$(sha256sum "$work_dir/passphrase.enc" | awk '{print $1}')
cat > "$work_dir/manifest.txt" <<EOF
format=tokenai-early-access-backup-v1
created_at=$timestamp
database_cipher_sha256=$dump_sha
wrapped_key_sha256=$key_sha
encryption=aes-256-cbc-pbkdf2-200000+rsa-oaep-sha256
EOF

tar -C "$work_dir" -czf "$backup_dir/$name.tar.gz" database.dump.enc passphrase.enc manifest.txt
archive_sha=$(sha256sum "$backup_dir/$name.tar.gz" | awk '{print $1}')

ssh -o BatchMode=yes -o StrictHostKeyChecking=yes -o UserKnownHostsFile=/home/ec2-user/.ssh/known_hosts -i "$remote_key" \
  "$remote_user@$remote_host" "upload $name.tar.gz $archive_sha" \
  < "$backup_dir/$name.tar.gz"
ssh -o BatchMode=yes -o StrictHostKeyChecking=yes -o UserKnownHostsFile=/home/ec2-user/.ssh/known_hosts -i "$remote_key" \
  "$remote_user@$remote_host" "verify $name.tar.gz"

find "$backup_dir" -type f -name 'early-access-*.tar.gz' -mtime +14 -delete
find "$backup_dir" -type f -name 'early-access-*.dump' -delete
