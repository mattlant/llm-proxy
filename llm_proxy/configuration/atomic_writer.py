from __future__ import annotations

import os
import stat
import tempfile
from pathlib import Path


class AtomicConfigurationWriter:
    def write(self, path: Path, content: str) -> None:
        mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else None
        descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        temporary_path = Path(temporary)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            if mode is not None:
                os.chmod(temporary_path, mode)
            os.replace(temporary_path, path)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise
