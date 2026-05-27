from app import main


def test_main_runs_uvicorn_with_configured_server_settings(mocker) -> None:
    mocked_run = mocker.patch("app.main.uvicorn.run")

    main.main()

    mocked_run.assert_called_once_with(
        "app.main:app",
        host=main.config.server.host,
        port=main.config.server.port,
    )
