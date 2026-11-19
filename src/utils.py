from pathlib import Path
import re
import sys
from typing import Any

from pydantic import ValidationError
from tqdm import tqdm

from .enums import Statics


class Utils:
    """Provide shared file, validation, serialization, and logging helpers."""

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """Tokenize text into normalized identifier-like terms.

        Args:
            text: Text to tokenize.

        Returns:
            Lowercase tokens longer than one character.
        """
        if not text:
            return []

        tokens: list[str] = []

        matches = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", text)

        for match in matches:
            parts = re.split(r"[_\s]+", match)
            for part in parts:
                clean = part.lower().strip()
                if len(clean) > 1 and clean:
                    tokens.append(clean)

        return tokens

    @staticmethod
    def get_path(
        path: Any,
        is_dir: bool = False,
        create: bool = False,
        pbar: tqdm | None = None,
    ) -> Path:
        """Convert a value to a path and optionally create it.

        Args:
            path: Path-like value to convert.
            is_dir: Whether ``path`` represents a directory.
            create: Whether to create the requested path.
            pbar: Optional progress bar to close on failure.

        Returns:
            The converted path.
        """
        try:
            path = Path(str(path))
            if create and is_dir:
                print("create dir:", path)
                path.mkdir(parents=True, exist_ok=True)

            if not is_dir and create:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()

            return path

        except OSError as e:

            if pbar is not None:
                pbar.close()

            Utils.write_err(f"File: {path}")
            Utils.write_err(f"Error: {e.strerror}")
            sys.exit(1)

    @staticmethod
    def write_log(msg: str) -> None:
        """Write an informational message to standard output.

        Args:
            msg: Message to write.
        """
        print(msg, file=sys.stdout)

    @staticmethod
    def write_err(msg: str) -> None:
        """Write an error message to standard error.

        Args:
            msg: Message to write.
        """
        print(msg, file=sys.stderr)

    @staticmethod
    def get_valid_max_chunk_size(max_chunk_size: Any) -> int:
        """Validate and normalize a requested maximum chunk size.

        Args:
            max_chunk_size: Value supplied by the caller.

        Returns:
            A validated chunk size.
        """
        try:

            min_size, max_size = (
                Statics.MIN_CHUNK_SIZE.value,
                Statics.GLOBAL_MAX_CHUNK_SIZE.value,
            )

            value = int(max_chunk_size)

            if value > max_size:
                raise ValueError(f"'max_chunk_size' greater than {max_size}")

            elif value < min_size:
                raise ValueError(
                    "'max_chunk_size' cannot be less than "
                    + f"min_chunk_size size {min_size}",
                )

            return value

        except ValueError as e:
            Utils.write_err(f"Error: {e}")
            sys.exit(1)

    @staticmethod
    def get_valid_k_value(k: Any) -> int:
        """Validate and normalize a positive result count.

        Args:
            k: Value supplied by the caller.

        Returns:
            A positive integer result count.
        """
        try:
            if isinstance(k, str) and not k.isdigit():
                raise ValueError()

            if not isinstance(k, int):
                raise ValueError()

            k_value = int(k)
            if k_value <= 0:
                raise ValueError()
            return k_value

        except (TypeError, ValueError):
            Utils.write_err(
                f"Error: 'k' must be a positive integer; received {k}",
            )
            sys.exit(1)

    @staticmethod
    def readfile(file_path: Path, pbar: tqdm | None = None) -> str:
        """Read a UTF-8 text file, reporting errors consistently.

        Args:
            file_path: File to read.
            pbar: Optional progress bar to close on failure.

        Returns:
            The file contents.
        """
        try:
            with open(file_path) as f:
                return f.read()

        except (OSError, UnicodeDecodeError) as e:
            if pbar is not None:
                pbar.close()
            Utils.write_err(f"File: {str(file_path)}")

            if isinstance(e, OSError):
                Utils.write_err(f"Error: {e.strerror}")
            else:
                Utils.write_err(f"Error: {e}")
            sys.exit(1)

    @staticmethod
    def writefile(
        file_path: str, content: str, pbar: tqdm | None = None
    ) -> None:
        """Write text content to a file, creating parent directories.

        Args:
            file_path: Destination file path.
            content: Text to write.
            pbar: Optional progress bar to close on failure.
        """
        try:
            path = Utils.get_path(file_path, create=True, pbar=pbar)

            with open(path, "w") as f:
                f.write(content)

        except (OSError, UnicodeDecodeError) as e:
            if pbar is not None:
                pbar.close()

            Utils.write_err(f"File: {file_path}")

            if isinstance(e, OSError):
                Utils.write_err(f"Error: {e.strerror}")
            else:
                Utils.write_err(f"Error: {e}")
            sys.exit(1)

    @staticmethod
    def get_pydantic_model(file_path: Path, obj: Any) -> Any:
        """Parse a JSON file into the requested Pydantic model.

        Args:
            file_path: JSON file to parse.
            obj: Pydantic model class used for validation.

        Returns:
            The validated model instance.
        """
        content = Utils.readfile(file_path)
        try:
            return obj.model_validate_json(content)

        except ValidationError as e:
            Utils.write_err(f"File: {str(file_path)}")
            Utils.write_err(
                f"Error: file content must be a valid '{obj.__name__}' Model"
            )
            Utils.write_err(str(e))

            sys.exit(1)
