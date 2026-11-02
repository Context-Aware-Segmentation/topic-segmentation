# Install the locked topic-segmentation environment and activate it (bash or zsh):
#   source scripts/activate.sh
part="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
export UV_PROJECT_ENVIRONMENT="${MT_UV_ENV_HOME:-$HOME/.local/share/uv/mt}/topic-segmentation"
export NLTK_DATA="$part/data/resources/nltk"  # the Punkt sentence splitter of the news and MIND data
uv sync --project "$part" --locked && source "$UV_PROJECT_ENVIRONMENT/bin/activate"
