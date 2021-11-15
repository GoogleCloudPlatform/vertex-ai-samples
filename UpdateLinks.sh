#!/bin/bash

export search_dir=community-content

notebook_filenames=$(find $search_dir -type f -name "*.ipynb")

export repo_name=vertex-ai-samples

export search_links=https:.*\.ipynb

export search_console=https://console\.cloud\.google\.com/ai/platform/notebooks/deploy-notebook?download_url=.*\.ipynb
export search_colab=https://colab\.research\.google\.com/github/GoogleCloudPlatform/.*/blob/master/.*\.ipynb
export search_github=https://github\.com/GoogleCloudPlatform/.*/blob/master/.*\.ipynb

export console_prefix=https://console.cloud.google.com/ai/platform/notebooks/deploy-notebook?download_url=https://github.com/GoogleCloudPlatform
export colab_prefix=https://colab.research.google.com/github/GoogleCloudPlatform
export github_prefix=https://github.com/GoogleCloudPlatform

for notebook_filename in $notebook_filenames
  do

    echo "Notebook: $notebook_filename"

    echo "        Existing Linkes"
    grep $search_links $notebook_filename
    grep $search_console $notebook_filename
    grep $search_colab $notebook_filename
    grep $search_github $notebook_filename

    export replace_console="$console_prefix/$repo_name/raw/master/$notebook_filename"
    export replace_colab="$colab_prefix/$repo_name/blob/master/$notebook_filename"
    export replace_github="$github_prefix/$repo_name/blob/master/$notebook_filename"

    echo "        New Linkes"
    echo "replace_console:  $replace_console"
    echo "replace_colab:    $replace_colab"
    echo "replace_github:   $replace_github"

    sed -i -e s,$search_console,$replace_console,g $notebook_filename
    sed -i -e s,$search_colab,$replace_colab,g $notebook_filename
    sed -i -e s,$search_github,$replace_github,g $notebook_filename

  done

find $search_dir -name '*.ipynb-e' -delete
