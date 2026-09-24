# Direct VPC cutover

The API instance security group is `sg-0aebc5df9085fcfab`. The model instance
security group is `sg-0ba95845fccc1b60a`. Direct private traffic is currently
blocked by the model security group.

With AWS credentials that can modify EC2 security groups, allow only the API
security group to reach the two model worker ports:

```bash
aws ec2 authorize-security-group-ingress \
  --group-id sg-0ba95845fccc1b60a \
  --ip-permissions 'IpProtocol=tcp,FromPort=8011,ToPort=8012,UserIdGroupPairs=[{GroupId=sg-0aebc5df9085fcfab,Description="TokenAI Early Access API workers"}]' \
  --region us-east-1
```

After `172.31.80.233` can connect to `172.31.28.151:8011` and `:8012`, set:

```text
MODEL_UPSTREAM_URLS=http://172.31.28.151:8011/v1,http://172.31.28.151:8012/v1
```

Restart the API container, verify both workers, then disable and remove
`horus-model-tunnel.service`. Do not add public CIDR rules for these ports.
