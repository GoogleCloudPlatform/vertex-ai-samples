import nbformat

from utils import NotebookProcessors


def test_update_value():
    # Test that the content was updated
    preprocessor = NotebookProcessors.UniqueStringsPreprocessor()

    content = 'PROJECT_ID = "your-project-id-unique"'

    new_content = preprocessor.update_unique_strings(content)

    assert new_content != content
    assert new_content.startswith('PROJECT_ID = "your-project-id-')
    assert new_content.endswith('"')


WHEEL = "gs://staging-bucket/google-cloud-aiplatform.whl"


def test_install_from_wheel_replaces_pinned_requirement():
    preprocessor = NotebookProcessors.VertexAIInstallProprocessor(
        vertex_ai_wheel=WHEEL
    )

    new_content = preprocessor.update_vertex_ai_install(
        "! pip install google-cloud-aiplatform==1.36.0"
    )

    assert new_content == (
        "! gcloud storage cp gs://staging-bucket/google-cloud-aiplatform.whl"
        " google-cloud-aiplatform.whl\n"
        "! pip install google-cloud-aiplatform.whl"
    )


def test_install_from_wheel_keeps_other_packages():
    preprocessor = NotebookProcessors.VertexAIInstallProprocessor(
        vertex_ai_wheel=WHEEL
    )

    new_content = preprocessor.update_vertex_ai_install(
        "! pip install google-cloud-aiplatform['tensorboard'] pandas"
    )

    assert new_content.endswith("! pip install google-cloud-aiplatform.whl pandas")


def test_install_from_wheel_is_skipped_without_the_package():
    preprocessor = NotebookProcessors.VertexAIInstallProprocessor(
        vertex_ai_wheel=WHEEL
    )

    content = "! pip install pandas"

    assert preprocessor.update_vertex_ai_install(content) == content


def test_install_from_wheel_returns_the_notebook():
    preprocessor = NotebookProcessors.VertexAIInstallProprocessor(
        vertex_ai_wheel=WHEEL
    )

    notebook = nbformat.v4.new_notebook()
    notebook.cells = [
        nbformat.v4.new_code_cell("! pip install google-cloud-aiplatform\n")
    ]

    new_notebook, resources = preprocessor.preprocess(notebook)

    assert new_notebook.cells[0].source.startswith("! gcloud storage cp")
    assert resources is None
