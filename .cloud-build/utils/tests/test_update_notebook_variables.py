from utils import UpdateNotebookVariables


def test_update_value():
    new_content = UpdateNotebookVariables.get_updated_value(
        content='asdf\nPROJECT_ID = "[your-project-id]" #@param {type:"string"} \nasdf',
        variable_name="PROJECT_ID",
        variable_value="sample-project",
    )
    assert (
        new_content
        == 'asdf\nPROJECT_ID = "sample-project" #@param {type:"string"} \nasdf'
    )


def test_update_value_single_quotes():
    new_content = UpdateNotebookVariables.get_updated_value(
        content="PROJECT_ID = '[your-project-id]'",
        variable_name="PROJECT_ID",
        variable_value="sample-project",
    )
    assert new_content == "PROJECT_ID = 'sample-project'"


def test_update_value_avoidance():
    new_content = UpdateNotebookVariables.get_updated_value(
        content="PROJECT_ID = shell_output[0] ",
        variable_name="PROJECT_ID",
        variable_value="sample-project",
    )
    assert new_content == "PROJECT_ID = shell_output[0] "


def test_region():
    new_content = UpdateNotebookVariables.get_updated_value(
        content='REGION = "[your-region]"  # @param {type:"string"}',
        variable_name="REGION",
        variable_value="us-central1",
    )
    assert new_content == 'REGION = "us-central1"  # @param {type:"string"}'


def test_region_equal_equals_ignore():
    # Tests that == is ignored
    new_content = UpdateNotebookVariables.get_updated_value(
        content='REGION == "[your-region]"  # @param {type:"string"}',
        variable_name="REGION",
        variable_value="us-central1",
    )
    assert new_content == 'REGION == "[your-region]"  # @param {type:"string"}'


def test_service_account():
    # Tests that == is ignored
    new_content = UpdateNotebookVariables.get_updated_value(
        content='SERVICE_ACCOUNT = "[your-service-account]"  # @param {type:"string"}',
        variable_name="SERVICE_ACCOUNT",
        variable_value="12345-compute@developer.gserviceaccount.com",
    )
    assert (
        new_content
        == 'SERVICE_ACCOUNT = "12345-compute@developer.gserviceaccount.com"  # @param {type:"string"}'
    )
