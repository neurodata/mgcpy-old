#!/usr/bin/env bash
set -ex

pip install -r docs/source/requirements.txt
cd docs
make html
cd ..

set +ex
