#!/bin/bash

basedir=$(git rev-parse --show-toplevel)
repohookspath="$basedir/.git/hooks/"
localhook="$basedir/git-hooks/pre-commit"

repohook="$repohookspath/pre-commit"

echo "Creating symlink $repohook to $localhook"
ln -si $localhook $repohook
