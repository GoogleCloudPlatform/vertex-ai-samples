**REQUIRED:** Add a summary of your PR here, typically including why the change is needed and what was changed. Include any design alternatives for discussion purposes.

<br>
--- YOUR PR SUMMARY GOES HERE ---
<br><br><br>

**REQUIRED:** Fill out the below checklists or remove if irrelevant
1. If you are opening a PR for `Official Notebooks` under the [notebooks/official](https://github.com/GoogleCloudPlatform/vertex-ai-samples/tree/main/notebooks/official) folder, follow this mandatory checklist:
- [ ] Use the [notebook template](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/notebook_template.ipynb) as a starting point.
- [ ] Follow the style and grammar rules outlined in the above notebook template.
- [ ] Verify the notebook runs successfully in Colab since the automated tests cannot guarantee this even when it passes.
- [ ] Passes all the required automated checks. You can locally test for formatting and linting with these [instructions](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/CONTRIBUTING.md#code-quality-checks).
- [ ] You have consulted with a tech writer to see if tech writer review is necessary. If so, the notebook has been reviewed by a tech writer, and they have approved it.
- [ ] This notebook has been added to the [CODEOWNERS](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/CODEOWNERS) file under the `Official Notebooks` section, pointing to the author or the author's team.
- [ ] The Jupyter notebook cleans up any artifacts it has created (datasets, ML models, endpoints, etc) so as not to eat up unnecessary resources.

<br>

2. If you are opening a PR for `Community Notebooks` under the [notebooks/community](https://github.com/GoogleCloudPlatform/vertex-ai-samples/tree/main/notebooks/community) folder:
- [ ] This notebook has been added to the [CODEOWNERS](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/community/CODEOWNERS) file under the `Community Notebooks` section, pointing to the author or the author's team.
- [ ] Passes all the required formatting and linting checks. You can locally test with these [instructions](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/CONTRIBUTING.md#code-quality-checks).

<br>

3. If you are opening a PR for `Community Content` under the [community-content](https://github.com/GoogleCloudPlatform/vertex-ai-samples/tree/main/community-content) folder:
- [ ] Make sure your main `Content Directory Name` is descriptive, informative, and includes some of the key products and attributes of your content, so that it is differentiable from other content
- [ ] The main content directory has been added to the [CODEOWNERS](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/community-content/CODEOWNERS) file under the `Community Content` section, pointing to the author or the author's team.
- [ ] Passes all the required formatting and linting checks. You can locally test with these [instructions](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/CONTRIBUTING.md#code-quality-checks).
