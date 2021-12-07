# How to Contribute

We'd love to accept your patches and contributions to this project. There are
just a few small guidelines you need to follow.

## Contributor License Agreement

Contributions to this project must be accompanied by a Contributor License
Agreement. You (or your employer) retain the copyright to your contribution;
this simply gives us permission to use and redistribute your contributions as
part of the project. Head over to <https://cla.developers.google.com/> to see
your current agreements on file or to sign a new one.

You generally only need to submit a CLA once, so if you've already submitted one
(even if it was for a different project), you probably don't need to do it
again.

## Code Quality Checks

All notebooks in this project are checked for formatting and style, to ensure a
consistent experience. To test notebooks prior to submitting a pull request,
you can follow these steps.

From a command-line terminal (e.g. from Vertex Workbench or locally), install
the code analysis tools:

```shell
pip install --user -U nbqa black flake8 isort pyupgrade git+https://github.com/tensorflow/docs
```

Then, set an environment variable for your notebook (or directory):

```shell
export notebook="your-notebook.ipynb"
```

Finally, run this code block to check for errors. Each step will attempt to
automatically fix any issues. If the fixes can't be performed automatically,
then you will need to manually address them before submitting your PR.

```shell
nbqa black "$notebook"
nbqa pyupgrade "$notebook"
nbqa isort "$notebook"
python3 -m tensorflow_docs.tools.nbfmt --remove_outputs "$notebook"
nbqa flake8 "$notebook" --extend-ignore=W391,E501,F821,E402,F404,W503,E203,E722,W293,W291
```

## Code Reviews

All submissions, including submissions by project members, require review. We
use GitHub pull requests for this purpose. Consult
[GitHub Help](https://help.github.com/articles/about-pull-requests/) for more
information on using pull requests.

## Community Guidelines

This project follows [Google's Open Source Community
Guidelines](https://opensource.google/conduct/).
