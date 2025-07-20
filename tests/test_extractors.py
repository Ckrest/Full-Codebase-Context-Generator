import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import LLM_Extreme_Context as lec


def test_extract_from_python(tmp_path):
    code = """
def foo():
    return 1


def bar(x):
    foo()
    return x
"""
    f = tmp_path / "sample.py"
    f.write_text(code)
    results = lec.extract_from_python(str(f))
    names = {r["name"] for r in results}
    assert names == {"foo", "bar"}
    bar_entry = next(r for r in results if r["name"] == "bar")
    assert "foo" in bar_entry["called_functions"]


def test_build_call_graph_called_by(tmp_path):
    code = """
def foo():
    pass

def bar():
    foo()
"""
    f = tmp_path / "sample.py"
    f.write_text(code)
    entries = lec.extract_from_python(str(f))
    graph = lec.build_call_graph(entries)
    out = tmp_path / "graph.json"
    lec.save_graph_json(graph, out)
    data = json.loads(out.read_text())
    foo_id = f"{f}::foo"
    bar_id = f"{f}::bar"
    node_map = {n["id"]: n for n in data["nodes"]}
    assert bar_id in node_map[foo_id]["called_by"]
    assert foo_id in node_map[bar_id]["calls"]


def test_method_call_extraction(tmp_path):
    code = """
class A:
    def foo(self):
        pass

    def bar(self):
        self.foo()
"""
    f = tmp_path / "sample.py"
    f.write_text(code)
    results = lec.extract_from_python(str(f))
    bar_entry = next(r for r in results if r["name"] == "bar")
    assert "foo" in bar_entry["called_functions"]


def test_qualified_names(tmp_path):
    code1 = """
def foo():
    pass

def bar():
    foo()
"""
    code2 = """
def foo():
    pass
"""
    f1 = tmp_path / "a.py"
    f2 = tmp_path / "b.py"
    f1.write_text(code1)
    f2.write_text(code2)
    entries = lec.extract_from_python(str(f1)) + lec.extract_from_python(str(f2))
    graph = lec.build_call_graph(entries)
    a_bar = f"{f1}::bar"
    a_foo = f"{f1}::foo"
    b_foo = f"{f2}::foo"
    assert graph.has_edge(a_bar, a_foo)
    assert not graph.has_edge(a_bar, b_foo)


def test_cross_file_import(tmp_path):
    utils = """
def clean():
    pass
"""
    main = """
from utils import clean

def run():
    clean()
"""
    utils_f = tmp_path / "utils.py"
    main_f = tmp_path / "main.py"
    utils_f.write_text(utils)
    main_f.write_text(main)
    entries = lec.extract_from_python(str(utils_f)) + lec.extract_from_python(str(main_f))
    graph = lec.build_call_graph(entries)
    run_id = f"{main_f}::run"
    clean_id = f"{utils_f}::clean"
    assert graph.has_edge(run_id, clean_id)


def test_edge_weights(tmp_path):
    code = """
def foo():
    pass

def bar():
    foo()
    foo()
"""
    f = tmp_path / "sample.py"
    f.write_text(code)
    entries = lec.extract_from_python(str(f))
    graph = lec.build_call_graph(entries)
    bar_id = f"{f}::bar"
    foo_id = f"{f}::foo"
    assert graph[bar_id][foo_id]["weight"] == 2


def test_extract_from_javascript(tmp_path):
    code = """
function foo(a) {
    return a + 1;
}

const bar = x => x * 2;

async function baz(y) {
    await foo(y);
}

const qux = async (z) => {
    return await foo(z);
};
"""
    f = tmp_path / "sample.js"
    f.write_text(code)
    results = lec.extract_from_javascript(str(f))
    names = {r["name"] for r in results}
    assert names == {"foo", "bar", "baz", "qux"}
    baz_entry = next(r for r in results if r["name"] == "baz")
    assert "foo" in baz_entry["called_functions"]


def test_extract_from_json(tmp_path):
    data = {"a": 1, "b": {"c": 2}}
    f = tmp_path / "config.json"
    f.write_text(json.dumps(data))
    results = lec.extract_from_json(str(f))
    assert results
    entry = results[0]
    assert entry["language"] == "json"
    assert "a" in entry["code"]


def test_extract_from_yaml(tmp_path):
    text = "a: 1\nb:\n  c: 2\n"
    f = tmp_path / "config.yaml"
    f.write_text(text)
    results = lec.extract_from_yaml(str(f))
    assert results
    entry = results[0]
    assert entry["language"] == "yaml"
    assert '"a"' in entry["code"]


def test_extract_from_txt(tmp_path):
    content = "hello world"
    f = tmp_path / "notes.txt"
    f.write_text(content)
    results = lec.extract_from_txt(str(f))
    assert results == [
        {
            "file_path": str(f),
            "language": "text",
            "type": "raw_text",
            "name": None,
            "code": content,
            "comments": [],
            "called_functions": [],
            "hash": lec.hash_content(content),
            "estimated_tokens": lec.estimate_tokens(content),
        }
    ]



def test_crawl_directory_extensions_and_gitignore(tmp_path):
    (tmp_path / "a.py").write_text("def foo():\n    pass\n")
    (tmp_path / "config.json").write_text("{\"a\": 1}")
    (tmp_path / "README.md").write_text("# hi")
    (tmp_path / "ignore.yml").write_text("a: b")
    (tmp_path / ".gitignore").write_text("ignore.yml\n")

    entries = lec.crawl_directory(str(tmp_path))
    processed = {Path(e["file_path"]).name for e in entries}

    assert {"a.py", "config.json", "README.md"} <= processed
    assert "ignore.yml" not in processed
