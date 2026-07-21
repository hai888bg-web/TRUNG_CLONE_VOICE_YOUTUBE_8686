#!/bin/bash
# Double-click de chay tren Mac (cloud mode - nhe, khong can RAM manh)
cd "$(dirname "$0")"
export SETUPTOOLS_SCM_PRETEND_VERSION=0.1.0

echo "========================================"
echo "  NHAN BAN GIONG NOI - VoxCPM2 (Cloud)"
echo "========================================"
echo ""

if ! command -v uv >/dev/null 2>&1; then
  echo "Đang cài uv (trình quản lý Python)..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

echo "Đang chuẩn bị (lần đầu hơi lâu)..."
uv sync --no-editable --python 3.11

echo ""
echo "Đang khởi động (cloud mode - nhẹ, không tốn RAM)..."
echo "Trình duyệt sẽ tự mở: http://localhost:7860"
echo ""
uv run python app_cloud.py
