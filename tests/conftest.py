import sys
import types
from pathlib import Path


repo_root = Path(__file__).resolve().parents[1]
package = types.ModuleType("molscribe")
package.__path__ = [str(repo_root / "molscribe")]
sys.modules.setdefault("molscribe", package)
