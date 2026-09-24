"""Spike da Fase 3: compara os dois caminhos de geracao de relatorio.

Antes de apagar qualquer renderer, precisamos saber o que cada caminho
produz hoje. Este script roda os cinco tipos de teste uma unica vez e
depois gera relatorio pelos dois caminhos:

  legado : ReportManager.generate_report(...)  -> interactive e static
  novo   : ReportGenerator.generate_*_report(...)

Guarda todo HTML gerado e imprime uma matriz do que funciona onde.
Nao altera nada no pacote.
"""

import json
import os
import sys
import traceback
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/mplconfig-spike')

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else './spike_out')
OUT.mkdir(parents=True, exist_ok=True)

TESTS = ['robustness', 'uncertainty', 'resilience', 'hyperparameters', 'fairness']


def build_dataset(n=1200, seed=42):
    """Dataset tabular pequeno com um atributo protegido binario."""
    rng = np.random.default_rng(seed)
    gender = rng.integers(0, 2, n)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    x3 = rng.normal(0, 1, n)
    age = rng.integers(20, 70, n)
    # sinal com um vies deliberado no atributo protegido, para o teste de fairness ter o que achar
    logit = 0.9 * x1 - 0.7 * x2 + 0.4 * x3 + 0.6 * gender + 0.01 * (age - 45)
    y = (1 / (1 + np.exp(-logit)) > rng.random(n)).astype(int)
    return pd.DataFrame(
        {'x1': x1, 'x2': x2, 'x3': x3, 'age': age, 'gender': gender, 'target': y}
    )


def main():
    from deepbridge.core.db_data import DBDataset
    from deepbridge.core.experiment import Experiment

    df = build_dataset()
    features = ['x1', 'x2', 'x3', 'age', 'gender']
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)

    model = RandomForestClassifier(n_estimators=60, random_state=42)
    model.fit(train_df[features], train_df['target'])

    dataset = DBDataset(
        train_data=train_df,
        test_data=test_df,
        target_column='target',
        features=features,
        model=model,
        categorical_features=['gender'],
        dataset_name='spike_fase3',
    )

    exp = Experiment(
        dataset=dataset,
        experiment_type='binary_classification',
        tests=TESTS,
        protected_attributes=['gender'],
        config={'verbose': False},
    )

    print('>> rodando os testes (config=quick)...', flush=True)
    try:
        exp.run_tests('quick')
    except Exception:
        print('!! run_tests falhou:')
        traceback.print_exc()

    matrix = {}

    for test_type in TESTS:
        key = 'hyperparameter' if test_type == 'hyperparameters' else test_type
        matrix[key] = {}

        # ---- pegar os resultados crus deste teste
        results = None
        for getter in (
            lambda: exp.test_results.get(test_type),
            lambda: exp.test_results.get(key),
            lambda: getattr(exp, f'get_{key}_results')(),
        ):
            try:
                r = getter()
                if r:
                    results = r
                    break
            except Exception:
                continue

        if not results:
            matrix[key]['_resultados'] = 'AUSENTE: nenhum resultado produzido'
            print(f'-- {key}: sem resultados, pulando')
            continue

        if hasattr(results, 'results'):
            results = results.results
        matrix[key]['_resultados'] = f'ok ({len(results)} chaves)' if isinstance(results, dict) else type(results).__name__

        # ---- caminho legado
        for style in ('interactive', 'static'):
            label = f'legado/{style}'
            path = OUT / f'{key}__legado_{style}.html'
            try:
                from deepbridge.core.experiment.report.report_manager import ReportManager

                ReportManager().generate_report(
                    test_type=key,
                    results=results,
                    file_path=str(path),
                    model_name='SpikeModel',
                    report_type=style,
                )
                size = path.stat().st_size if path.exists() else 0
                matrix[key][label] = f'OK {size // 1024} KB' if size else 'VAZIO'
            except NotImplementedError as e:
                matrix[key][label] = f'NAO IMPLEMENTADO: {e}'
            except Exception as e:
                matrix[key][label] = f'ERRO {type(e).__name__}: {str(e)[:110]}'

        # ---- caminho novo
        label = 'novo/ReportGenerator'
        path = OUT / f'{key}__novo.html'
        try:
            from deepbridge.core.experiment.report.api import ReportGenerator

            gen = ReportGenerator()
            method = getattr(gen, f'generate_{key}_report', None)
            if method is None:
                matrix[key][label] = 'METODO INEXISTENTE na API nova'
            else:
                method(results=results, output_path=path)
                size = path.stat().st_size if path.exists() else 0
                matrix[key][label] = f'OK {size // 1024} KB' if size else 'VAZIO'
        except Exception as e:
            matrix[key][label] = f'ERRO {type(e).__name__}: {str(e)[:110]}'

    # ---- relatorio
    print('\n' + '=' * 78)
    print('MATRIZ DE COBERTURA DOS DOIS CAMINHOS'.center(78))
    print('=' * 78)
    cols = ['legado/interactive', 'legado/static', 'novo/ReportGenerator']
    print(f'{"tipo":<16}' + ''.join(f'{c:<26}' for c in cols))
    print('-' * 78)
    for k, v in matrix.items():
        print(f'{k:<16}' + ''.join(f'{str(v.get(c, "-"))[:25]:<26}' for c in cols))
    print('-' * 78)
    for k, v in matrix.items():
        print(f'  {k}: resultados = {v.get("_resultados")}')

    (OUT / 'matriz.json').write_text(json.dumps(matrix, indent=2, ensure_ascii=False))
    print(f'\nHTML e matriz.json em: {OUT}')


if __name__ == '__main__':
    main()
