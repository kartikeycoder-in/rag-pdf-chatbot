"""Report whether required LLM settings exist without exposing their values."""

from pathlib import Path

from dotenv import dotenv_values

ENV_PATH = Path(__file__).resolve().parents[1] / ".env"


def main() -> None:
    """Print only boolean presence checks for the three LLM variables."""
    values = dotenv_values(ENV_PATH)
    for name in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL"):
        print(f"{name}: {bool(values.get(name))}")


if __name__ == "__main__":
    main()
