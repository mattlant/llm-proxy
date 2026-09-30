from llm_proxy.application.errors import ModelNotFoundError
from llm_proxy.application.failure_classification import FailureCategory, FailureClassifier
from llm_proxy.domain.errors import ProviderUnavailableError


def test_failure_classifier_has_protocol_neutral_categories() -> None:
    assert FailureClassifier.classify(ModelNotFoundError("model", "requested")).category is FailureCategory.MODEL_NOT_FOUND
    assert FailureClassifier.classify(ProviderUnavailableError("down")).category is FailureCategory.PROVIDER_UNAVAILABLE
