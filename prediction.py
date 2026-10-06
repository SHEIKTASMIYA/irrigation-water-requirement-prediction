from pathlib import Path
import joblib
import pandas._libs.arrays as pa

# Define project-root-safe paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "irrigation_model.pkl"

# Compatibility hook for unpickling pandas StringDtype/NDArrayBacked in pandas 2.3+ / Python 3.13
_orig_load_build = joblib.numpy_pickle.NumpyUnpickler.dispatch[b'b'[0]]

def _compat_load_build(unpickler):
    state = unpickler.stack[-1]
    inst = unpickler.stack[-2]
    # If unpickling an NDArrayBacked object with a legacy 2-element state tuple, adapt to 3-element format
    if isinstance(inst, pa.NDArrayBacked) and isinstance(state, tuple) and len(state) == 2:
        unpickler.stack[-1] = (state[0], state[1], {})
    return _orig_load_build(unpickler)

joblib.numpy_pickle.NumpyUnpickler.dispatch[b'b'[0]] = _compat_load_build


def load_model(path: Path = MODEL_PATH):
    """Load the trained irrigation prediction pipeline."""
    return joblib.load(path)


# Load the model instance
model = load_model()

if __name__ == "__main__":
    print("Irrigation prediction model loaded successfully!")