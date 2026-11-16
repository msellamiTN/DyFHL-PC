#!/usr/bin/env bash
# Fetch the public datasets used by the paper into data/raw/.
# Xu-Stoller ABAC case studies (GPLv3) and the Amazon employee access data, via a public mirror.
set -e
mkdir -p data/raw
tmp=$(mktemp -d)
git clone --depth 1 https://github.com/saptarshidas13/grace "$tmp/grace"
cp -r "$tmp/grace/datasets/xu-stoller/ABAC-Mining/ABACMining/case-studies" data/raw/xu-stoller
cp "$tmp/grace/datasets/amazon-employee-access/amazon_employee_access.csv" data/raw/
rm -rf "$tmp"
echo "data ready in data/raw"
