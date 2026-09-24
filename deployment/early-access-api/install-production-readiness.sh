#!/bin/bash
set -euo pipefail

root=/opt/tokenai-early-access
cd "$root"

for file in \
  model-gateway.js server.js production-readiness-self-test.js production-soak-test.js \
  004-production-readiness.sql 005-manual-distillation-review.sql docker-compose.yml backup-postgres.sh operations-monitor.py \
  tokenai-early-access-backup.service tokenai-early-access-backup.timer \
  tokenai-early-access-monitor.service tokenai-early-access-monitor.timer; do
  sed -i 's/\r$//' "/tmp/$file"
done

install -o ec2-user -g ec2-user -m 644 /tmp/model-gateway.js "$root/api/model-gateway.js"
install -o ec2-user -g ec2-user -m 644 /tmp/server.js "$root/api/server.js"
install -o ec2-user -g ec2-user -m 644 /tmp/production-readiness-self-test.js "$root/api/production-readiness-self-test.js"
install -o ec2-user -g ec2-user -m 644 /tmp/production-soak-test.js "$root/api/production-soak-test.js"
install -o ec2-user -g ec2-user -m 644 /tmp/004-production-readiness.sql "$root/db/004-production-readiness.sql"
install -o ec2-user -g ec2-user -m 644 /tmp/005-manual-distillation-review.sql "$root/db/005-manual-distillation-review.sql"
install -o ec2-user -g ec2-user -m 644 /tmp/docker-compose.yml "$root/docker-compose.yml"
install -o root -g root -m 700 /tmp/backup-postgres.sh "$root/backup-postgres.sh"
install -o root -g root -m 700 /tmp/operations-monitor.py "$root/operations-monitor.py"
install -o root -g root -m 644 /tmp/tokenai-early-access-backup.service /etc/systemd/system/tokenai-early-access-backup.service
install -o root -g root -m 644 /tmp/tokenai-early-access-backup.timer /etc/systemd/system/tokenai-early-access-backup.timer
install -o root -g root -m 644 /tmp/tokenai-early-access-monitor.service /etc/systemd/system/tokenai-early-access-monitor.service
install -o root -g root -m 644 /tmp/tokenai-early-access-monitor.timer /etc/systemd/system/tokenai-early-access-monitor.timer
grep -vE '^(MODEL_UPSTREAM_URLS|MODEL_PRIVATE_IP|GLOBAL_INFERENCE_CAPACITY|GLOBAL_INFERENCE_MAX_QUEUE|GLOBAL_INFERENCE_QUEUE_TIMEOUT_MS|GLOBAL_INFERENCE_LEASE_SECONDS|ALERT_EMAIL|BACKUP_REMOTE_HOST|BACKUP_REMOTE_SSH_KEY)=' .env > .env.next
printf '%s\n' \
  'MODEL_UPSTREAM_URLS=http://172.31.28.151:8011/v1,http://172.31.28.151:8012/v1' \
  'MODEL_PRIVATE_IP=172.31.28.151' \
  'GLOBAL_INFERENCE_CAPACITY=2' \
  'GLOBAL_INFERENCE_MAX_QUEUE=40' \
  'GLOBAL_INFERENCE_QUEUE_TIMEOUT_MS=120000' \
  'GLOBAL_INFERENCE_LEASE_SECONDS=240' \
  'ALERT_EMAIL=info@tokenai.llc' \
  'BACKUP_REMOTE_HOST=54.91.40.201' \
  'BACKUP_REMOTE_SSH_KEY=/home/ec2-user/.ssh/horus_backup_transfer' >> .env.next
chmod 600 .env.next
mv .env.next .env
chown ec2-user:ec2-user .env

docker exec -i tokenai-early-access-postgres-1 sh -lc \
  'psql -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d "$POSTGRES_DB"' \
  < db/004-production-readiness.sql
docker exec -i tokenai-early-access-postgres-1 sh -lc \
  'psql -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d "$POSTGRES_DB"' \
  < db/005-manual-distillation-review.sql
docker compose -f docker-compose.yml config -q
docker build -t tokenai-early-access-api:0.4.0 api

systemctl daemon-reload
curl -fsS http://172.31.28.151:8011/health > /dev/null
curl -fsS http://172.31.28.151:8012/health > /dev/null

docker compose -f docker-compose.yml up -d --remove-orphans
docker exec tokenai-early-access-api-1 node -e \
  "Promise.all([8011,8012].map(p=>fetch('http://172.31.28.151:'+p+'/health').then(r=>{if(!r.ok)throw new Error(String(r.status))}))).catch(()=>process.exit(1))"
systemctl disable --now horus-model-tunnel.service || true
systemctl enable --now tokenai-early-access-backup.timer tokenai-early-access-monitor.timer

rm -f \
  /tmp/model-gateway.js /tmp/server.js /tmp/production-readiness-self-test.js \
  /tmp/production-soak-test.js /tmp/004-production-readiness.sql /tmp/005-manual-distillation-review.sql /tmp/docker-compose.yml \
  /tmp/backup-postgres.sh /tmp/operations-monitor.py /tmp/tokenai-early-access-backup.service \
  /tmp/tokenai-early-access-backup.timer /tmp/tokenai-early-access-monitor.service \
  /tmp/tokenai-early-access-monitor.timer

echo DEPLOYED_0_4_0
