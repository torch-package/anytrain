def chat_model(
    model_id,
    *,
    prompt,
    completion,
    discount=0.0,
    outputs=("text",),
    inputs=("text",),
):
    return {
        "id": model_id,
        "architecture": {
            "input_modalities": list(inputs),
            "output_modalities": list(outputs),
        },
        "supported_parameters": ["temperature"],
        "pricing": {
            "prompt": str(prompt),
            "completion": str(completion),
            "discount": discount,
        },
    }


def models_requests(catalog):
    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"data": catalog}

    class FakeRequests:
        calls: list[dict[str, object]] = []

        @staticmethod
        def get(**kwargs: object):
            FakeRequests.calls.append(kwargs)
            return FakeResponse()

    return FakeRequests
