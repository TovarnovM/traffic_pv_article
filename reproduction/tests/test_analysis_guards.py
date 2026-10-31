"""Standard-library-only entrypoint guards; safe to run on the host."""
import ast
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ANALYSIS=Path(__file__).resolve().parents[1]/'analysis'


class OutputGuards(unittest.TestCase):
    def test_jl_existing_outputs_rejected_before_read(self):
        stubs={name:mock.MagicMock() for name in ('numpy','pandas','table_verification')}
        with mock.patch.dict(sys.modules,stubs):
            spec=importlib.util.spec_from_file_location('jl_guard',ANALYSIS/'jl_decomposition.py')
            module=importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for name in ('episodes.csv.gz','JL_substitution.csv','existing.csv'):
                target=root/name
                target.write_bytes(b'preserve')
                with self.subTest(name=name), self.assertRaises(SystemExit):
                    module.main([str(root),'--output',str(target)])
                self.assertEqual(target.read_bytes(),b'preserve')
            stubs['pandas'].read_csv.assert_not_called()

    def test_plot_existing_directory_precedes_data_reads(self):
        tree=ast.parse((ANALYSIS/'plot_figures.py').read_text())
        # Execute the actual entrypoint through its first data read, stubbing imports.
        first_read=next(i for i,node in enumerate(tree.body)
                        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='g' for t in node.targets))
        tree.body=tree.body[:first_read+1]
        stubs={name:mock.MagicMock() for name in
               ('numpy','pandas','matplotlib','matplotlib.pyplot','matplotlib.colors','matplotlib.lines')}
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'existing';out.mkdir()
            marker=out/'keep';marker.write_bytes(b'preserve')
            with mock.patch.dict(sys.modules,stubs), mock.patch.object(sys,'argv',['plot_figures.py','missing-data',str(out)]):
                with self.assertRaises(FileExistsError):
                    exec(compile(tree,str(ANALYSIS/'plot_figures.py'),'exec'),{})
            stubs['pandas'].read_csv.assert_not_called()
            self.assertEqual(list(out.iterdir()),[marker])
            self.assertEqual(marker.read_bytes(),b'preserve')


if __name__=='__main__':
    unittest.main()
