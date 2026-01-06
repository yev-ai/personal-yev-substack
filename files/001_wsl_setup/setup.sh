#!/bin/bash

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
files=(
  ".zshrc"
  ".p10k.zsh"
  ".zshrc-pre.sh"
  ".zshrc-post.sh"
)

backup_existing() {
  for file in "${files[@]}"; do
    local target="$HOME/$file"

    if [[ -f "$target" ]]; then
      local backup_name="$HOME/.backup$file"
      cp "$target" "$backup_name"
      echo "✅ Backed up $file to $backup_name"
    else
      echo "⚠️  Skipped $file (not found in ~)"
    fi
  done
}

install_files() {
  for file in "${files[@]}"; do
    local source_file="$script_dir/$file"
    local target="$HOME/$file"

    if [[ -f "$source_file" ]]; then
      cp "$source_file" "$target"
      echo "🚀 Installed $file to ~/"
    else
      echo "❌ Error: Could not find $file in script directory ($script_dir)"
    fi
  done
}

backup_existing
install_files