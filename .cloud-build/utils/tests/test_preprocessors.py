from utils import NotebookProcessors


def test_update_value():
    # Test that the content was updated
    preprocessor = NotebookProcessors.UniqueStringsPreprocessor()

    content = 'PROJECT_ID = "your-project-id-unique"'

    new_content = preprocessor.update_unique_strings(content)

    assert new_content != content
    assert new_content.startswith('PROJECT_ID = "your-project-id-')
    assert new_content.endswith('"')
