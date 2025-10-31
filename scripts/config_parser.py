# yaml_setter.py
from pathlib import Path
import re

# --- ruamel round-trip: Pflicht für Format-Erhalt ---
from ruamel.yaml import YAML
from ruamel.yaml.scalarstring import DoubleQuotedScalarString as DQ  # optional, fürs Setzen mit "..."

_yaml = YAML()            # Round-Trip-Loader
_yaml.preserve_quotes = True
_yaml.width = 4096        # verhindert Zeilenumbrüche
_yaml.indent(sequence=2, offset=2)

_index_re = re.compile(r"([^\[\]]+)|\[(\-?\d+)\]")

class Config_Parser:
    def __init__(self, yaml_path, create_missing=False, preserve_formatting=True):
        self.yaml_path = Path(yaml_path)
        self.out_path = None
        self.create_missing = create_missing
        # Erzwingen: wir nutzen ruamel immer, damit nichts "aus Versehen" reflowt
        self.preserve_formatting = True

    def set_out_path(self, out_path):
        self.out_path = Path(out_path)

    # --- Pfad-Parsing ---
    def _parse_path(self, dotpath: str):
        parts = []
        for seg in dotpath.split("."):
            for m in _index_re.finditer(seg):
                key, idx = m.group(1), m.group(2)
                parts.append(key if key is not None else int(idx))
        return parts

    # --- Traversal/Setzen ---
    def _set_nested(self, obj, path_parts, value):
        cur = obj
        for i, p in enumerate(path_parts):
            last = (i == len(path_parts) - 1)

            if last:
                if isinstance(p, int):
                    if not isinstance(cur, list):
                        if not self.create_missing:
                            raise KeyError(f"Expected list at {path_parts[:i]}, got {type(cur).__name__}")
                        raise TypeError(f"Cannot assign by index into non-list at {path_parts[:i]}")
                    if p >= len(cur):
                        if not self.create_missing:
                            raise IndexError(f"Index {p} out of range at {path_parts[:i]}")
                        cur.extend([None] * (p - len(cur) + 1))
                    cur[p] = value
                else:
                    if not isinstance(cur, dict):
                        if not self.create_missing:
                            raise KeyError(f"Expected dict at {path_parts[:i]}, got {type(cur).__name__}")
                        raise TypeError(f"Cannot assign key into non-dict at {path_parts[:i]}")
                    cur[p] = value
                return

            nxt = path_parts[i + 1]
            if isinstance(p, int):
                if not isinstance(cur, list):
                    if not self.create_missing:
                        raise KeyError(f"Expected list at {path_parts[:i]}, got {type(cur).__name__}")
                    raise TypeError(f"Cannot descend by index into non-list at {path_parts[:i]}")
                if p >= len(cur):
                    if not self.create_missing:
                        raise IndexError(f"Index {p} out of range at {path_parts[:i]}")
                    cur.extend([None] * (p - len(cur) + 1))
                if cur[p] is None:
                    cur[p] = [] if isinstance(nxt, int) else {}
                cur = cur[p]
            else:
                if not isinstance(cur, dict):
                    if not self.create_missing:
                        raise KeyError(f"Expected dict at {path_parts[:i]}, got {type(cur).__name__}")
                    raise TypeError(f"Cannot descend by key into non-dict at {path_parts[:i]}")
                if p not in cur or cur[p] is None:
                    cur[p] = [] if isinstance(nxt, int) else {}
                cur = cur[p]

    # --- IO mit ruamel (Round-Trip) ---
    def _load_yaml(self, path: Path):
        with path.open("r", encoding="utf-8") as f:
            data = _yaml.load(f)
        return data

    def _dump_yaml(self, path: Path, data):
        with path.open("w", encoding="utf-8") as f:
            _yaml.dump(data, f)

    # --- API ---
    def set_yaml_values(self, changes: dict[str, object]):
        data = self._load_yaml(self.yaml_path)
        for dotkey, value in changes.items():
            # Optional: Strings mit Quotes erzwingen -> value = DQ(str(value))
            self._set_nested(data, self._parse_path(dotkey), value)
        dst = self.out_path if self.out_path else self.yaml_path
        self._dump_yaml(dst, data)
        return data

    def set_yaml_value(self, key: str, value: object):
        return self.set_yaml_values({key: value})
