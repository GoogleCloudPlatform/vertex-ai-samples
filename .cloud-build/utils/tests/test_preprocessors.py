import nbformat
from utils import NotebookProcessors


def test_update_value():
    # Test that the content was updated with double quotes
    preprocessor = NotebookProcessors.UniqueStringsPreprocessor()

    content = 'PROJECT_ID = "your-project-id-unique"'
    new_content = preprocessor.update_unique_strings(content)

    assert new_content != content
    assert new_content.startswith('PROJECT_ID = "your-project-id-')
    assert new_content.endswith('"')


def test_update_unique_strings_single_quotes():
    # Test that single-quoted strings preserve matching single quote closures
    preprocessor = NotebookProcessors.UniqueStringsPreprocessor()

    content_hyphen = "PROJECT_ID = 'your-project-id-unique'"
    new_hyphen = preprocessor.update_unique_strings(content_hyphen)
    assert new_hyphen != content_hyphen
    assert new_hyphen.startswith("PROJECT_ID = 'your-project-id-")
    assert new_hyphen.endswith("'")

    content_underscore = "DATASET_ID = 'my_dataset_unique'"
    new_underscore = preprocessor.update_unique_strings(content_underscore)
    assert new_underscore != content_underscore
    assert new_underscore.startswith("DATASET_ID = 'my_dataset_")
    assert new_underscore.endswith("'")


def test_vertex_ai_install_preprocessor():
    wheel_path = "gs://test-bucket/wheels/google_cloud_aiplatform.whl"
    preprocessor = NotebookProcessors.VertexAIInstallPreprocessor(
        vertex_ai_wheel=wheel_path
    )

    # Content containing google-cloud-aiplatform
    cell_code = "pip install google-cloud-aiplatform\n"
    updated = preprocessor.update_vertex_ai_install(cell_code)
    expected_header = (
        f"gcloud storage cp {wheel_path} google-cloud-aiplatform.whl\n"
    )
    assert expected_header in updated
    assert "google-cloud-aiplatform.whl\n" in updated

    # Content without google-cloud-aiplatform
    unrelated_code = "pip install numpy\n"
    unchanged = preprocessor.update_vertex_ai_install(unrelated_code)
    assert unchanged == unrelated_code

    # Preprocessor contract test: preprocess() returns (notebook, resources)
    nb = nbformat.v4.new_notebook()
    nb.cells.append(nbformat.v4.new_code_cell(cell_code))
    res_nb, res_res = preprocessor.preprocess(
        nb, resources={"meta": "test"}
    )
    assert res_nb is nb
    assert res_res == {"meta": "test"}
    assert "google-cloud-aiplatform.whl" in res_nb.cells[0].source


def test_vertex_ai_install_alias_compatibility():
    # Verify backward compatibility alias exists and functions identically
    alias_cls = NotebookProcessors.VertexAIInstallProprocessor
    target_cls = NotebookProcessors.VertexAIInstallPreprocessor
    assert alias_cls is target_cls
