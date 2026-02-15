import os
from typing import Optional


def _default_model_root() -> str:
    # <repo_root>/models when running from source checkout.
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return os.path.join(repo_root, "models")


def get_model_root() -> str:
    return os.environ.get("TRELLIS_MODEL_DIR", _default_model_root())


def local_repo_path(repo_id: str, model_root: Optional[str] = None) -> Optional[str]:
    parts = repo_id.split("/")
    if len(parts) < 2:
        return None
    root = model_root or get_model_root()
    candidate = os.path.join(root, parts[0], parts[1])
    if os.path.isdir(candidate):
        return candidate
    return None


def resolve_hf_path(path: str, model_root: Optional[str] = None) -> str:
    """
    Resolve a Hugging Face-style identifier to a local mirror path when available.

    Examples:
      microsoft/TRELLIS.2-4B -> <model_root>/microsoft/TRELLIS.2-4B
      microsoft/TRELLIS.2-4B/ckpts/foo -> <model_root>/microsoft/TRELLIS.2-4B/ckpts/foo
    """
    if not path:
        return path

    # Keep explicit local paths unchanged.
    if os.path.exists(path):
        return path

    parts = path.split("/")
    if len(parts) < 2:
        return path

    repo_dir = local_repo_path(f"{parts[0]}/{parts[1]}", model_root=model_root)
    if repo_dir is None:
        return path

    if len(parts) == 2:
        return repo_dir

    local_candidate = os.path.join(repo_dir, *parts[2:])
    if (
        os.path.exists(local_candidate)
        or os.path.exists(f"{local_candidate}.json")
        or os.path.exists(f"{local_candidate}.safetensors")
    ):
        return local_candidate

    return path
