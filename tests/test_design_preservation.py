"""Compare core algorithm ASTs with upstream e005f76, excluding observation only.

No torch, LLM server, or network is needed. The manifest is generated from the
original Git revision, not from the instrumented implementation under test.
"""
import ast
import hashlib
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = Path(__file__).with_name('fixtures') / 'original_design.json'


class RemoveObservation(ast.NodeTransformer):
    def visit_ImportFrom(self, node):
        return None if node.module == 'MAR.Experiment.trace' else node

    def visit_With(self, node):
        if len(node.items) == 1:
            expression = node.items[0].context_expr
            if isinstance(expression, ast.Call) and isinstance(expression.func, ast.Name) and expression.func.id in ('generation_phase', 'processing_stage'):
                return [self.visit(statement) for statement in node.body]
        return self.generic_visit(node)

    def visit_Expr(self, node):
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id in ('record_round', 'record_reuse'):
            return None
        return self.generic_visit(node)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id == 'execute_node':
            return ast.Call(func=ast.Attribute(value=self.visit(node.args[0]), attr='execute', ctx=ast.Load()), args=[self.visit(node.args[1])], keywords=[])
        if isinstance(node.func, ast.Name) and node.func.id == 'run_graph':
            return ast.Call(func=ast.Attribute(value=self.visit(node.args[0]), attr='run', ctx=ast.Load()), args=[], keywords=[self.visit(k) for k in node.keywords if k.arg != 'routing'])
        return self.generic_visit(node)

    def visit_For(self, node):
        if isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name) and node.iter.func.id == 'enumerate' and isinstance(node.target, ast.Tuple) and isinstance(node.target.elts[0], ast.Name) and node.target.elts[0].id == 'query_index':
            node.target = node.target.elts[1]
            node.iter = node.iter.args[0]
        return self.generic_visit(node)


def structural_value(value):
    # Python 3.12 adds empty type_params to class/function nodes. Ignore that
    # compatibility field so the fingerprint is stable on 3.10 through 3.12.
    if isinstance(value, ast.AST):
        return [type(value).__name__, {field: structural_value(item) for field, item in ast.iter_fields(value) if field != 'type_params'}]
    if isinstance(value, list):
        return [structural_value(item) for item in value]
    return value


def fingerprint(source):
    tree = RemoveObservation().visit(ast.parse(source))
    return hashlib.sha256(json.dumps(structural_value(tree), sort_keys=True).encode()).hexdigest()


class DesignPreservationTests(unittest.TestCase):
    def test_core_matches_original_except_observation(self):
        manifest = json.loads(MANIFEST.read_text())
        for filename, expected in manifest['core_ast_sha256'].items():
            with self.subTest(file=filename):
                self.assertEqual(fingerprint((ROOT / filename).read_text()), expected,
                                 'Core logic differs from original ' + manifest['revision'])

    def test_role_profiles_are_unchanged(self):
        manifest = json.loads(MANIFEST.read_text())
        actual = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in (ROOT / 'MAR/Roles').rglob('*.json')}
        self.assertEqual(actual, manifest['role_sha256'])

    def test_check_detects_algorithm_changes(self):
        original = 'def f(x):\n    return x * 6\n'
        self.assertNotEqual(fingerprint(original), fingerprint(original.replace('* 6', '* 3')))
        self.assertEqual(fingerprint(original), fingerprint('def f(x):\n    with generation_phase("primary_reasoning"):\n        return x * 6\n'))
