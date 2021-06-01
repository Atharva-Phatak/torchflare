# flake8: noqa

"""
@pytest.mark.skip(reason="Comet ML requires first import. Logic running properly but will fail CI Tests.")
def test_comet():
    params = {"bs": 16, "lr": 0.01}
    logger = CometLogger(
        project_name="dl-experiments",
        workspace="notsogenius",
        params=params,
        tags=["Dummy", "test"],
        api_token=os.environ.get("COMET_API_TOKEN"),
    )

    acc = 10
    f1 = 10
    loss = 100
    for epoch in range(10):
        d = {"acc": acc, "f1": f1, "loss": loss, "TTE": 5}
        acc += 10
        f1 += 10

        loss = loss / 10
        logger.on_epoch_end()
    logger.on_experiment_end()
"""
