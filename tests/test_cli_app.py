"""
Testes do pacote de nivel superior deepbridge.

Garante que o app de CLI e importado de verdade (e nao engolido por um
try/except que o transformaria em None silenciosamente).
"""

import deepbridge


def test_cli_app_is_not_none():
    """deepbridge.cli_app deve ser um objeto Typer, nunca None."""
    assert deepbridge.cli_app is not None


def test_cli_app_is_exported():
    """cli_app deve estar declarado em __all__."""
    assert 'cli_app' in deepbridge.__all__


def test_cli_app_has_registered_commands():
    """O app de CLI deve expor pelo menos um grupo de comandos."""
    assert deepbridge.cli_app.registered_groups
