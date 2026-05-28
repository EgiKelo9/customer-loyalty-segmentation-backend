import sys
import importlib


def test_main_app_routes(monkeypatch):
    import app.database.main as db_main

    monkeypatch.setattr(db_main, "create_db", lambda: None)
    monkeypatch.setattr(db_main.Base.metadata, "create_all", lambda **kwargs: None)

    if "app.main" in sys.modules:
        del sys.modules["app.main"]

    main_module = importlib.import_module("app.main")
    app = main_module.app
    paths = {route.path for route in app.router.routes}

    assert "/" in paths
    assert "/api/v1/health/" in paths
    assert "/api/v1/auth/register" in paths
    assert "/api/v1/auth/login" in paths
    assert "/api/v1/auth/logout" in paths
    assert "/api/v1/segmentation/lrfm" in paths
    assert "/api/v1/segmentation/transactions" in paths
    assert "/api/v1/segmentation/transactions/upload" in paths
    assert "/api/v1/analytics/kpi" in paths
    assert "/api/v1/analytics/charts" in paths
    assert "/api/v1/analytics/customers" in paths
