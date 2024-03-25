import pandas as pd
from nucml.datasets.utilities import _ame_parser


def test_get_ame_originals_clean_up_originals(tmp_path):
    _ame_parser._get_ame_originals(tmp_path)

    periodic = tmp_path / 'periodic_table.csv'
    mass = tmp_path / 'mass.txt'
    rct1 = tmp_path / 'rct1.txt'
    rct2 = tmp_path / 'rct2.txt'

    all_files = [periodic, mass, rct1, rct2]
    for path in all_files:
        assert path.is_file()

    _ame_parser._clean_up_originals(tmp_path)
    for path in all_files:
        assert not path.is_file()


def test_preprocess_ame_df():
    test_df = pd.DataFrame({
        'test': ['#dummy', 'rows#', 'being', 'inserted', '*'],
        'all_nan': ['*', '*', '*', '*', '*'],
        'Page_Feed': ['#dummy', 'rows#', 'being', 'inserted', '*'],
        'ignore': ['#dummy', 'rows#', 'being', 'inserted', '*']
    })

    _ame_parser._preprocess_ame_df(test_df)
    assert 'Page_Feed' not in test_df.columns
    assert 'ignore' not in test_df.columns

    for value in test_df.test.values():
        assert '#' not in value

    assert test_df.isnull().all_nan.sum() == len(test_df)
