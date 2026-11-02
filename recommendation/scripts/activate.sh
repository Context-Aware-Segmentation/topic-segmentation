# Install the locked recommendation environment and activate it (bash or zsh):
#   source scripts/activate.sh
part="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
export UV_PROJECT_ENVIRONMENT="${MT_UV_ENV_HOME:-$HOME/.local/share/uv/mt}/news-recommendation"
uv sync --project "$part" --locked && source "$UV_PROJECT_ENVIRONMENT/bin/activate"
