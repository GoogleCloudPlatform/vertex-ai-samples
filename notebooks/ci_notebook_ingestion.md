# Administrative Howto notes on CI Notebook Ingestion


This readme covers administrative actions that are performed on an as-needed basis.

## Team: vertex-ai-owners

Members of the vertex-ai-owners (git team) have administrative privileges. 


### Viewing members

1. Goto the repo
2. From top-level menu, select: (Settings -> Collaborators and Teams)[https://github.com/GoogleCloudPlatform/vertex-ai-samples/settings/access]


### Adding a new member

If another member needs to be added:
   - Have the new member make a request to join the team.
   - vertex-ai-owners with the `Maintainer` tag may add the new member.

   
## Executing CI notebook ingestion checks on a PR
   
### Killing a stuck PR

If the CI notebook ingestion test is stuck (not terminating), you can kill the process by:

1. Goto the PR
2. Under checks, find the entry: vertex-ai-notebook-execution-test (python-docs-samples-tests) In progress —> Summary
3. Select Details
4. At bottom of details page, select: View more details on Google Cloud Build
5. In Cloud Build history page, select Cancel on the top menu bar.

### Restart a PR test

There are two ways to restart the CI notebook ingestion tests on an open PR.

1. In Cloud Build history page, select Rebuild on the top menu bar.
2. or, in a comment in the PR enter: /gcbrun

## Bypassing CI notebook ingestion checks on a PR

We strongly discourage this, unless there is a compelling reason that would impact the integrity of the quality process.

There are two ways of doing this. In both cases, you do:

1. Goto the repo
2. From top-level menu, select: (Settings -> Branches)[https://github.com/GoogleCloudPlatform/vertex-ai-samples/settings/branches]
3. Under Branch Protection Rules, select the `main` branch.

### Allowing a member to disable requirements for merging

Specific member(s) can be assigned the ability to override requirements and merge a PR, by:

1. Select Edit for the `main` branch in Branch Protection Rules.
2. Find the entry "Allow specified actors to bypass required pull requests".
3. Under this entry, add the member's git LDAP.
4. Select SAVE.
5. The "Squash and Merge" button will now be enabled on all PRs viewed by that member.

### Temporarily disable checks.

You can disable requirement checks temporarily on all PRs. 

1. Select Edit for the `main` branch in Branch Protection Rules.
2. Uncheck:
  - Require approvals
  - Require review from Code Owners
  - Require status checks to pass before merging
3. Select SAVE
4. Now all members will see a green "Squash and Merge" on all PRs viewed by that member.

To reverse, recheck the settings you unchecked above.

## Linting

To execute the identical lint image locally, from the CI notebook ingestion checks, do:

1. Goto the corresponding local folder in the repo.
2. Run: `docker run -v ${PWD}:/setup/app gcr.io/python-docs-samples-tests/notebook_linter:latest <your_notebooks>`

## Install dependency issues

Some packages (and combinations) have dependencies that fail on the virgin VM image used for the CI notebook ingestion test.

### TFDV

If the notebook installs and uses tensorflow_data_validation, install as follows:

! pip3 install -q {USER_FLAG} google-cloud-aiplatform \
                              tensorflow-data-validation \
                              protobuf==3.20.3

! pip3 install -q {USER_FLAG} cachetools==5.2.0




   
   
