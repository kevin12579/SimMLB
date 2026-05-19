"""앙상블 + Isotonic Calibration"""
import numpy as np
import joblib
from pathlib import Path
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss

from src.common.logger import get_logger

logger = get_logger(__name__)


class EnsembleCalibrator:
    def __init__(self) -> None:
        self.weights: dict[str, float] = {}
        self.calibrator: IsotonicRegression | None = None

    def compute_weights(self, logloss_map: dict[str, float]) -> None:
        """log_loss가 낮을수록 높은 가중치 (softmax 역수)"""
        scores = np.array(list(logloss_map.values()))
        inv    = 1.0 / scores
        w      = inv / inv.sum()
        self.weights = {k: float(w[i]) for i, k in enumerate(logloss_map)}
        logger.info("Ensemble weights: %s", self.weights)

    def blend(self, prob_map: dict[str, np.ndarray]) -> np.ndarray:
        """가중 평균 앙상블 확률"""
        result = np.zeros(len(next(iter(prob_map.values()))))
        for name, probs in prob_map.items():
            result += self.weights.get(name, 0.5) * probs
        return result

    def fit_calibrator(self, raw_probs: np.ndarray, y_true: np.ndarray) -> None:
        self.calibrator = IsotonicRegression(out_of_bounds="clip")
        self.calibrator.fit(raw_probs, y_true)
        cal_probs = self.calibrator.predict(raw_probs)
        logger.info(
            "Calibration — before log_loss=%.4f, after=%.4f",
            log_loss(y_true, raw_probs),
            log_loss(y_true, cal_probs),
        )

    def predict(self, prob_map: dict[str, np.ndarray]) -> np.ndarray:
        blended = self.blend(prob_map)
        if self.calibrator is None:
            return blended
        return self.calibrator.predict(blended)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"weights": self.weights, "calibrator": self.calibrator}, path)
        logger.info("Calibrator saved to %s", path)

    @classmethod
    def load(cls, path: str) -> "EnsembleCalibrator":
        data = joblib.load(path)
        obj = cls()
        obj.weights     = data["weights"]
        obj.calibrator  = data["calibrator"]
        return obj
