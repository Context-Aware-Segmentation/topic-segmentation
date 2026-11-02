# Install the locked knowledge-graph environment and activate it (bash or zsh):
#   source scripts/activate.sh
part="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
export UV_PROJECT_ENVIRONMENT="${MT_UV_ENV_HOME:-$HOME/.local/share/uv/mt}/knowledge-graph"
# The clinical code also uses the MIMIC note rules of the segmentation part.
export PYTHONPATH="$(dirname "$part")/topic_segmentation/src"
uv sync --project "$part" --locked && source "$UV_PROJECT_ENVIRONMENT/bin/activate"
