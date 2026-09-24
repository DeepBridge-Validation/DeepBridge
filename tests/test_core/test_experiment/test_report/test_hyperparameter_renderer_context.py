"""Regressao: o contexto Jinja do relatorio de hyperparameter.

O bug: ``HyperparameterRenderer.render`` recebia ``report_type`` como parametro
Python mas nunca o colocava no contexto passado ao Jinja, enquanto
``templates/report_types/hyperparameter/index.html`` usa a variavel. O render
morria com ``'report_type' is undefined`` e o relatorio nunca funcionou, nem no
modo interativo nem no estatico.

Havia um segundo defeito no mesmo lugar: os ``{% include %}`` montavam o caminho
com ``report_type``, produzindo ``report_types/interactive/partials/...`` em vez
de ``report_types/hyperparameter/partials/...``. As duas variaveis tem
significados diferentes e o template usava uma pela outra:

- ``test_type``   -> 'hyperparameter', identifica o teste e o diretorio
- ``report_type`` -> 'interactive' ou 'static', identifica o estilo
"""

import re
from pathlib import Path

import pytest

import deepbridge


TEMPLATE = (
    Path(deepbridge.__file__).parent
    / 'templates'
    / 'report_types'
    / 'hyperparameter'
    / 'index.html'
)


def test_template_exists():
    assert TEMPLATE.is_file(), f'template ausente: {TEMPLATE}'


def test_includes_are_built_from_test_type_not_report_type():
    """Os include tem que apontar para o diretorio do teste.

    Com ``report_type`` o caminho virava ``report_types/interactive/partials/``,
    que nao existe.
    """
    content = TEMPLATE.read_text(encoding='utf-8')

    includes = re.findall(r'{%\s*include\s+(.+?)\s*%}', content)
    dynamic = [inc for inc in includes if '+' in inc]

    assert dynamic, 'o template deixou de montar include dinamicamente'
    for inc in dynamic:
        # Olhar so os identificadores: o literal 'report_types/' contem
        # 'report_type' como substring e daria falso positivo.
        expression = re.sub(r"'[^']*'", '', inc)
        assert 'test_type' in expression, (
            f'include monta caminho com a variavel errada: {inc}'
        )
        assert 'report_type' not in expression, (
            f'include voltou a usar report_type: {inc}'
        )


def test_every_partial_referenced_by_the_template_exists():
    """Cada partial incluido precisa existir em disco."""
    content = TEMPLATE.read_text(encoding='utf-8')
    partials = re.findall(r"/partials/([A-Za-z0-9_]+\.html)'", content)

    assert partials, 'nenhum partial referenciado'
    missing = [
        name
        for name in partials
        if not (TEMPLATE.parent / 'partials' / name).is_file()
    ]
    assert not missing, f'partials referenciados mas ausentes: {missing}'


@pytest.mark.parametrize('report_type', ['interactive', 'static'])
def test_renderer_puts_report_type_in_the_context(report_type):
    """O renderer precisa injetar report_type, senao o Jinja morre.

    O teste inspeciona o contexto real em vez de renderizar o relatorio
    inteiro, para nao depender de assets nem de um experimento completo.
    """
    from deepbridge.core.experiment.report.renderers.hyperparameter_renderer import (
        HyperparameterRenderer,
    )

    captured = {}

    class _Template:
        def render(self, **context):
            captured.update(context)
            return '<html></html>'

    renderer = HyperparameterRenderer.__new__(HyperparameterRenderer)
    renderer.data_transformer = type(
        'T', (), {'transform': staticmethod(lambda results, model_name: {})}
    )()
    renderer._load_template = lambda test_type, rtype: _Template()
    renderer._get_assets = lambda test_type: {
        'css_content': '',
        'js_content': '',
        'logo': '',
    }
    renderer._create_base_context = lambda data, test_type, assets: {
        'test_type': test_type
    }
    renderer._render_template = lambda template, context: template.render(
        **context
    )
    renderer._write_html = lambda html, path: path

    renderer.render(
        results={},
        file_path='/tmp/unused-hyperparameter-report.html',
        model_name='Model',
        report_type=report_type,
    )

    assert captured.get('report_type') == report_type
    assert captured.get('test_type') == 'hyperparameter'
