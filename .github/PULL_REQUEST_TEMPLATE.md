If you are opening a PR for `Notebooks` under the [official](https://github.com/GoogleCloudPlatform/vertex-ai-samples/tree/master/notebooks/official) folder, follow this mandatory checklist:
- [ ] Use the [notebook template](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/master/notebooks/notebook_template.ipynb) as a starting point.
- [ ] Follow the style and grammar rules outlined in the above notebook template.
- [ ] Verify the notebook runs successfully in Colab since the automated tests cannot guarantee this even when it passes.
- [ ] Passes all the required automated checks
- [ ] You have consulted with a tech writer to see if tech writer review is necessary. If so, the notebook has been reviewed by a tech writer, and they have approved it.
- [ ] This notebook has been added to the [CODEOWNERS](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/master/docs/CODEOWNERS) file under `# Official Notebooks` section, pointing to the author or the author's team.
- [ ] The Jupyter notebook cleans up any artifacts it has created (datasets, ML models, endpoints, etc) so as not to eat up unnecessary resources.


If you are opening a PR for `Notebooks` under the [community](https://github.com/GoogleCloudPlatform/vertex-ai-samples/tree/master/notebooks/community) folder:
- [ ] This notebook has been added to the [CODEOWNERS](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/master/docs/CODEOWNERS) file under the `# Community Notebooks` section, pointing to the author or the author's team.


If you are opening a PR for `Community Content`, and it will NOT be used on cloud.google.com/docs:
- [ ] Make sure your main `Content Directory Name` is descriptive, informative, and includes some of the key products and attributes of your content, so that it is differentiable from other content
- [ ] The main content directory has been added to the [CODEOWNERS](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/master/docs/CODEOWNERS) file under the `# Community Content` section, pointing to the author or the author's team.

If you are opening a PR for `Community Content`, and it will be used on cloud.google.com/docs:
- [ ] Make sure your main `Content Directory Name` is descriptive, informative, and includes some of the key products and attributes of your content, so that it is differentiable from other content
- [ ] Use the [notebook template](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/master/notebooks/notebook_template.ipynb) for all the notebooks in your content directory.
- [ ] Follow the style and grammar rules outlined in the above notebook template.
- [ ] Verify each notebook runs successfully in Colab since the automated tests cannot guarantee this even when it passes.
- [ ] Passes all the required automated checks
- [ ] You have consulted with a tech writer to see if tech writer review is necessary. If so, the content has been reviewed by a tech writer, and they have approved it.
- [ ] This content has been added (/moved) to the [CODEOWNERS](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/master/docs/CODEOWNERS) file under the `# Official Community Content` section (/from the `# Community Content` section), pointing to the author or the author's team.
- [ ] Each notebook cleans up any artifacts it has created (datasets, ML models, endpoints, etc) so as not to eat up unnecessary resources.
