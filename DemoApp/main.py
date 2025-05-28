from app import app  # noqa: F401
import logging
from ml_models.custom_models import KernelSVM, ANN, DecisionTree, KNNScratch, DecisionNode, rbf_kernel


# Thiết lập logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
